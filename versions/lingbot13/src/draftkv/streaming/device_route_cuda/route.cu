// FP32 stable device routing. CUB stability preserves frame/block tie order.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace {
constexpr int Threads=1024;
constexpr int Header=8, Fields=5;
using Scan=cub::BlockScan<int,Threads>;

__global__ void seed(const float* scores,float* keys,int* ids,int* take,
                     const int* counts,int n,int blocks,int regions,int sink,
                     int start,int current,int protected_start) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) {
    int frame=sink+i/regions;
    keys[i]=frame<protected_start?scores[i]:-INFINITY;
    ids[i]=i;
  }
  if(i<blocks) {
    int frame=i/regions;
    take[i]=(frame<sink || (frame>=protected_start && frame<start+current))?counts[i%regions]:0;
  }
}

__global__ void choose(const int* sorted,int* take,const int* counts,
                       int n,int regions,int sink,int protected_start,int target) {
  __shared__ Scan::TempStorage temp;
  int t=threadIdx.x;
  int carry=0;
  // Stable Top-K over arbitrarily many Spatial 8x16 blocks. Each wave
  // scans 1024 descriptors; the carried sum preserves exact global rank.
  for(int off=0;off<n;off+=Threads) {
    int i=off+t,id=i<n?sorted[i]:0;
    int count=i<n && sink+id/regions<protected_start?counts[id%regions]:0;
    int local,aggregate;
    Scan(temp).ExclusiveSum(count,local,aggregate);
    if(i<n && count)take[sink*regions+id]=max(0,min(count,target-carry-local));
    carry+=aggregate;
    __syncthreads();
  }
}

__global__ void prefix(const int* take,const int* counts,int64_t* meta,int blocks,int regions,int heads) {
  __shared__ Scan::TempStorage temp;
  int t=threadIdx.x;
  int logical_carry=0,physical_carry=0;
  // Scan all frames/blocks, not only the first 1024 descriptors.
  for(int off=0;off<blocks;off+=Threads) {
    int i=off+t;
    int n=i<blocks?take[i]:0;
    int full=n?counts[i%regions]:0;
    int logical,physical,logical_sum,physical_sum;
    Scan(temp).ExclusiveSum(n,logical,logical_sum);
    __syncthreads();
    Scan(temp).ExclusiveSum(full,physical,physical_sum);
    if(i<blocks) {
      meta[4*i]=int64_t(logical_carry)+logical;
      meta[4*i+1]=(int64_t(physical_carry)+physical)*heads*256;
      meta[4*i+2]=full;meta[4*i+3]=n;
    }
    logical_carry+=logical_sum;physical_carry+=physical_sum;
    __syncthreads();
  }
}

__global__ void offsets(const int* take,const int* before,const int64_t* meta,
                        int64_t* out,int* slots,int blocks,int regions,int capacity,
                        int total,int frame_tokens,bool chrono,int heads) {
  int block=blockIdx.x;
  int frame=block/regions,b=block%regions;
  int n=meta[4*block+3],full=meta[4*block+2],logical=meta[4*block];
  int64_t base=meta[4*block+1];
  for(int j=threadIdx.x;j<n;j+=blockDim.x) {
    out[logical+j]=base+int64_t(j)*heads*128;
    out[total+logical+j]=base+int64_t(full+j)*heads*128;
    int rank=meta[4*(frame*regions)];
    for(int other=0;other<regions;other++)
      rank+=min(take[frame*regions+other],before[(b*capacity+j)*regions+other]);
    slots[logical+j]=chrono?rank/frame_tokens:frame;
  }
}

__global__ void current_copy(const uint4* packed,uint4* payload,const int64_t* meta,
                             const int* starts,int start,int regions,int frame_tokens,int heads) {
  int local=blockIdx.x,absolute=start*regions+local,b=local%regions;
  int64_t source=int64_t(local/regions)*2*frame_tokens*heads*128+int64_t(starts[b])*heads*256;
  int64_t dest=meta[4*absolute+1];
  int vectors=meta[4*absolute+2]*heads*256/8;
  for(int i=threadIdx.x;i<vectors;i+=blockDim.x)payload[dest/8+i]=packed[source/8+i];
}

__global__ void descriptions(const int64_t* addresses,const int64_t* meta,
                            int64_t* table,int* frontier,int start,int regions,
                            uint64_t payload,int batch_bytes,int history_tokens,int heads) {
  if(threadIdx.x)return;
  int n=0;int64_t batch=0,total=0;
  for(int block=start*regions-1;block>=0;--block) {
    if(!meta[4*block+3])continue;
    int64_t bytes=meta[4*block+2]*heads*512;
    int64_t src=addresses[block],dst=payload+meta[4*block+1]*2;
    batch+=bytes;total+=bytes;
    int value=batch>=batch_bytes?int((meta[4*block]+63)/64):-1;
    if(value>=0)batch=0;
    int64_t* prev=table+Header+(n-1)*Fields;
    if(n && prev[4]<0 && src+bytes==prev[0] && dst+bytes==prev[1]) {
      prev[0]=src;prev[1]=dst;prev[2]+=bytes;prev[4]=value;
    } else {
      int64_t* row=table+Header+n*Fields;
      row[0]=src;row[1]=dst;row[2]=bytes;row[3]=0;row[4]=value;++n;
    }
  }
  if(n)table[Header+(n-1)*Fields+4]=0;
  table[0]=n;table[1]=total;table[2]=history_tokens;
  frontier[0]=(history_tokens+63)/64;
}
}

int64_t route_workspace(int64_t n) {
  size_t bytes=0;
  C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(
      nullptr,bytes,(float*)nullptr,(float*)nullptr,(int*)nullptr,(int*)nullptr,int(n)));
  return int64_t(bytes);
}

void device_route(at::Tensor scores,at::Tensor keys,at::Tensor sorted_keys,
                  at::Tensor ids,at::Tensor sorted_ids,at::Tensor workspace,
                  at::Tensor take,at::Tensor counts,at::Tensor before,at::Tensor starts,
                  at::Tensor meta,at::Tensor offset,at::Tensor slots,
                  at::Tensor addresses,at::Tensor table,at::Tensor frontier,
                  at::Tensor packed,at::Tensor payload,int64_t sink,int64_t start,
                  int64_t current,int64_t budget,int64_t protect,int64_t ft,
                  int64_t regions,int64_t capacity,int64_t batch,bool chrono,int64_t heads,bool index_only,bool skip_topk) {
  c10::cuda::CUDAGuard guard(scores.device());
  int n=(start-sink)*regions,blocks=(start+current)*regions;
  TORCH_CHECK(blocks>0 && blocks<=65535 && n<=blocks && scores.scalar_type()==at::kFloat,
              "device route capacity/dtype");
  TORCH_CHECK(scores.numel()==n && budget<=start-sink && protect<=budget,"device route score/budget geometry");
  TORCH_CHECK(!skip_topk || budget==protect,"selector bypass requires entirely protected/all-history context");
  auto stream=at::cuda::getCurrentCUDAStream(scores.get_device()).stream();
  seed<<<(max(blocks,n)+Threads-1)/Threads,Threads,0,stream>>>(scores.data_ptr<float>(),keys.data_ptr<float>(),ids.data_ptr<int>(),
      take.data_ptr<int>(),counts.data_ptr<int>(),n,blocks,regions,sink,start,current,start-protect);
  if(n && !skip_topk) {
    size_t bytes=workspace.numel();
    C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(workspace.data_ptr(),bytes,
        keys.data_ptr<float>(),sorted_keys.data_ptr<float>(),ids.data_ptr<int>(),sorted_ids.data_ptr<int>(),n,0,32,stream));
    choose<<<1,Threads,0,stream>>>(sorted_ids.data_ptr<int>(),take.data_ptr<int>(),counts.data_ptr<int>(),
        n,regions,sink,start-protect,(budget-protect)*ft);
  }
  prefix<<<1,Threads,0,stream>>>(take.data_ptr<int>(),counts.data_ptr<int>(),meta.data_ptr<int64_t>(),blocks,regions,heads);
  offsets<<<blocks,Threads,0,stream>>>(take.data_ptr<int>(),before.data_ptr<int>(),meta.data_ptr<int64_t>(),
      offset.data_ptr<int64_t>(),slots.data_ptr<int>(),blocks,regions,capacity,(sink+budget+current)*ft,ft,chrono,heads);
  if(!index_only) {
    current_copy<<<current*regions,Threads,0,stream>>>(reinterpret_cast<const uint4*>(packed.data_ptr()),
        reinterpret_cast<uint4*>(payload.data_ptr()),meta.data_ptr<int64_t>(),starts.data_ptr<int>(),start,regions,ft,heads);
    descriptions<<<1,1,0,stream>>>(addresses.data_ptr<int64_t>(),meta.data_ptr<int64_t>(),table.data_ptr<int64_t>(),
        frontier.data_ptr<int>(),start,regions,reinterpret_cast<uint64_t>(payload.data_ptr()),batch,(sink+budget)*ft,heads);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace {
// Dense group K/V destinations retain all selected tokens until every Q CTA
// finishes. Source block-major [K(group),V(group)] pages are never CPU packed.
__global__ void group_current(const uint4* src,uint4* dst,const int64_t* meta,
 const int* starts,int start,int regions,int ft,int total,int heads,int group) {
 int block=blockIdx.x, g=blockIdx.y, abs=start*regions+block;
 int full=meta[4*abs+2],n=meta[4*abs+3],logical=meta[4*abs];
 int64_t source=int64_t(block/regions)*2*ft*heads*128+int64_t(starts[block%regions])*heads*256+
                int64_t(g)*2*full*group*128;
 int64_t base=int64_t(g)*2*total*group*128+int64_t(logical)*group*128;
 // Bound work per CTA: a handful of giant block/head CTAs underused HBM.
 int end=min(n*group*128/8,int(blockIdx.z+1)*2048);
 for(int i=int(blockIdx.z)*2048+threadIdx.x;i<end;i+=blockDim.x) {
   dst[base/8+i]=src[source/8+i];
   dst[(base+int64_t(total)*group*128)/8+i]=src[(source+int64_t(full)*group*128)/8+i];
 }
}
__global__ void group_descriptions(const int64_t* addresses,const int64_t* meta,
 int64_t* workspace,int* dma_ready,int* consume_ready,uint64_t payload,int start,int regions,
 int total,int heads,int group,int splits,bool reverse,int stride) {
 if(threadIdx.x)return;
 int order=blockIdx.x%splits,g=blockIdx.x/splits;
 int64_t* table=workspace+int64_t(blockIdx.x)*stride;
 int rows=0;int64_t bytes=0;
 {
   int s=reverse?splits-1-order:order,unit=g*splits+s;
   int lo=int(int64_t(total)*s/splits),hi=int(int64_t(total)*(s+1)/splits);
   int first=rows;
   for(int b=start*regions-1;b>=0;b--) {
     int n=meta[4*b+3],full=meta[4*b+2],logical=meta[4*b];
     int a=max(lo,logical),z=min(hi,logical+n);
     if(a>=z)continue;
     for(int kv=0;kv<2;kv++) {
       int64_t src=addresses[b]+(int64_t(g)*2*full*group*128+int64_t(kv)*full*group*128+
                               int64_t(a-logical)*group*128)*2;
       int64_t dst=payload+(int64_t(g)*2*total*group*128+int64_t(kv)*total*group*128+
                           int64_t(a)*group*128)*2;
       int64_t count=int64_t(z-a)*group*256;
       auto prev=table+Header+(rows-1)*Fields;
       if(rows>first && prev[4]<0 && src+count==prev[0] && dst+count==prev[1]) {
         prev[0]=src;prev[1]=dst;prev[2]+=count;
       } else {
         auto row=table+Header+rows*Fields;
         row[0]=src;row[1]=dst;row[2]=count;row[3]=unit;row[4]=-1;rows++;
       }
       bytes+=count;
     }
   }
   if(rows==first) {
     auto row=table+Header+rows*Fields;
     row[0]=row[1]=row[2]=0;row[3]=unit;rows++;
   }
   table[Header+(rows-1)*Fields+4]=1;
   dma_ready[unit]=consume_ready[unit]=0;
 }
 table[0]=rows;table[1]=bytes;table[2]=total;table[3]=heads/group*splits;
}

// Independent units build in parallel; compact them in the same consumer
// order before the one small D2H table transfer. No CPU prefix/gather step.
__global__ void compact_group_descriptions(const int64_t* workspace,int64_t* table,
                                         int units,int stride,int total) {
 int unit=blockIdx.x;const auto source=workspace+int64_t(unit)*stride;
 int prefix=0;
 for(int i=0;i<unit;i++)prefix+=workspace[int64_t(i)*stride];
 for(int i=threadIdx.x;i<source[0]*Fields;i+=blockDim.x)
   table[Header+prefix*Fields+i]=source[Header+i];
 if(!unit && !threadIdx.x) {
   int rows=0;int64_t bytes=0;
   for(int i=0;i<units;i++) {
     rows+=workspace[int64_t(i)*stride];bytes+=workspace[int64_t(i)*stride+1];
   }
   table[0]=rows;table[1]=bytes;table[2]=total;table[3]=units;
 }
}
}
void build_groups(at::Tensor addresses,at::Tensor meta,at::Tensor starts,at::Tensor packed,
 at::Tensor payload,at::Tensor table,at::Tensor dma_ready,at::Tensor consume_ready,
 at::Tensor slots,int64_t start,int64_t current,int64_t regions,int64_t ft,
 int64_t total,int64_t heads,int64_t group,int64_t splits,bool reverse,at::Tensor workspace,int64_t region_capacity) {
 c10::cuda::CUDAGuard guard(payload.device());
 TORCH_CHECK(group>0 && heads%group==0 && splits>0 && splits<=total &&
              region_capacity==128,"invalid grouped 8x16 configuration");
 TORCH_CHECK(payload.numel()>=2*total*heads*128 && dma_ready.numel()==heads/group*splits &&
             consume_ready.numel()==dma_ready.numel(),"group buffer capacity");
 TORCH_CHECK(table.numel()>=Header+Fields*heads/group*(2*start*regions+splits*3),"descriptor capacity");
 int units=heads/group*splits;
 TORCH_CHECK(workspace.is_cuda() && workspace.device()==payload.device() &&
             workspace.scalar_type()==at::kLong && workspace.is_contiguous() &&
             workspace.dim()==2 && workspace.size(0)==units &&
             workspace.size(1)>=Header+Fields*(2*start*regions+1),"unit descriptor workspace");
 auto s=at::cuda::getCurrentCUDAStream(payload.get_device()).stream();
 if(current)group_current<<<dim3(current*regions,heads/group,(region_capacity*group*128/8+2047)/2048),256,0,s>>>(
   reinterpret_cast<const uint4*>(packed.data_ptr()),reinterpret_cast<uint4*>(payload.data_ptr()),
   meta.data_ptr<int64_t>(),starts.data_ptr<int>(),start,regions,ft,total,heads,group);
 group_descriptions<<<units,1,0,s>>>(addresses.data_ptr<int64_t>(),meta.data_ptr<int64_t>(),
   workspace.data_ptr<int64_t>(),dma_ready.data_ptr<int>(),consume_ready.data_ptr<int>(),
   reinterpret_cast<uint64_t>(payload.data_ptr()),start,regions,total,heads,group,splits,reverse,workspace.size(1));
 compact_group_descriptions<<<units,256,0,s>>>(workspace.data_ptr<int64_t>(),table.data_ptr<int64_t>(),
   units,workspace.size(1),total);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
