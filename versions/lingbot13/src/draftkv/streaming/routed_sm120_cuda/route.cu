// SM120-only exact FP32 route. CUB radix sorting is stable for equal keys.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <vector>
using T=at::Tensor;
__global__ void init_ids(int* ids,int n) {int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)ids[i]=i;}
__global__ void ranked_counts(const int* ids,const int* counts,int* c,int n,int r) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=counts[ids[i]%r];
}
__global__ void initialize_take(int* take,const int* counts,int n,int r,int sink,int start,int protect) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){int f=i/r;take[i]=(f<sink||f>=start-protect)?counts[i%r]:0;}
}
__global__ void choose_take(const int* ids,const int* prefix,const int* counts,int* take,int n,int r,int sink,int target) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n){int id=ids[i],c=counts[id%r],before=prefix[i]-c;take[sink*r+id]=max(0,min(c,target-before));}
}
__global__ void full_counts(const int* take,const int* counts,int* full,int n,int r) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)full[i]=take[i]?counts[i%r]:0;
}
__global__ void materialize(const int* take,const int* prefix,const int* fullprefix,
 const int* counts,const int* starts,const int* index,int64_t* offsets,int* slots,int64_t* sources,
 int r,int cap,int ft,int start,int total,int reserve,bool chrono) {
 int b=blockIdx.x, f=blockIdx.y,i=threadIdx.x;
 int id=f*r+b,n=take[id],count=counts[b],out=prefix[id];
 int64_t base=f<start ? int64_t(fullprefix[id])*2*12*128 : int64_t(reserve+(f-start)*ft)*2*12*128+int64_t(starts[b])*2*12*128;
 for(int j=i;j<n;j+=blockDim.x){
   int token=index[b*cap+j];int rank=0;
   if(chrono){
     for(int other=0;other<r;other++){
       int lo=0,hi=take[f*r+other];
       while(lo<hi){int mid=(lo+hi)/2;if(index[other*cap+mid]<token)lo=mid+1;else hi=mid;}
       rank+=lo;
     }
   }
   offsets[out+j]=base+int64_t(j)*12*128;
   offsets[total+out+j]=base+int64_t(count+j)*12*128;
   slots[out+j]=chrono?(prefix[f*r]+rank)/ft:f;
   if(sources)sources[out+j]=int64_t(f)*ft+token;
 }
}
__global__ void dma_table(const int* take,const int* prefix,const int* fullprefix,
 const int* counts,const int* starts,const int64_t* addresses,int64_t* table,
 int r,int start,int64_t dst,int batch_bytes){
 if(threadIdx.x||blockIdx.x)return;
 int n=0,acc=0,blocks=0,cropped=0;int64_t totalbytes=0;
 for(int id=start*r-1;id>=0;--id){
   int valid=take[id];if(!valid)continue;
   int b=id%r,count=counts[b];blocks++;cropped+=(valid<count);
   int64_t src=addresses[id/r]+int64_t(starts[b])*2*12*128*2;
   int64_t target=dst+int64_t(fullprefix[id])*2*12*128*2;
   int64_t bytes=int64_t(count)*2*12*128*2;
   acc+=bytes;totalbytes+=bytes;
   bool last=prefix[id]==0;
   int publish=(acc>=batch_bytes||last)?(last?0:(prefix[id]+63)/64):-1;
   if(publish>=0)acc=0;
   int64_t* prev=table+n*5;
   if(n&&prev[4]<0&&src+bytes==prev[0]&&target+bytes==prev[1]){
     prev[0]=src;prev[1]=target;prev[2]+=bytes;prev[4]=publish;
   }else{int64_t* row=table+(++n)*5;row[0]=src;row[1]=target;row[2]=bytes;row[3]=0;row[4]=publish;}
 }
 table[0]=n;table[1]=totalbytes;table[2]=blocks;table[3]=cropped;table[4]=0;
}
int64_t route_workspace_bytes(int n,int all){
 size_t sort=0,scan=0;cudaStream_t stream=at::cuda::getCurrentCUDAStream();
 if(n)cub::DeviceRadixSort::SortPairsDescending(nullptr,sort,(float*)nullptr,(float*)nullptr,(int*)nullptr,(int*)nullptr,n,0,32,stream);
 cub::DeviceScan::ExclusiveSum(nullptr,scan,(int*)nullptr,(int*)nullptr,std::max(all,n),stream);
 return std::max(sort,scan)+4096;
}
void gpu_route(T scores,T addresses,T counts,T starts,T index,std::vector<T> w,
 T offsets,T slots,T sources,T table,T payload,int64_t sink,int64_t start,int64_t current,
 int64_t protect,int64_t budget,int64_t ft,int64_t cap,int64_t reserve,int64_t batch,bool chrono){
 TORCH_CHECK(scores.is_cuda()&&scores.scalar_type()==at::kFloat&&scores.is_contiguous(),"FP32 CUDA scores required");
 TORCH_CHECK(w.size()==11,"route workspace ABI");
 auto stream=at::cuda::getCurrentCUDAStream(scores.get_device()).stream();
 int r=counts.numel(),all=(start+current)*r,n=(start-sink-protect)*r;
 TORCH_CHECK(scores.numel()==(start-sink)*r&&n>=0,"score extent");
 int *ids=w[1].data_ptr<int>(),*sorted=w[2].data_ptr<int>(),*rc=w[3].data_ptr<int>(),*rp=w[4].data_ptr<int>(),
 *take=w[5].data_ptr<int>(),*prefix=w[6].data_ptr<int>(),*full=w[7].data_ptr<int>(),*fp=w[8].data_ptr<int>();
 size_t bytes=w[10].nbytes();void* temp=w[10].data_ptr();
 initialize_take<<<(all+255)/256,256,0,stream>>>(take,counts.data_ptr<int>(),all,r,sink,start,protect);
 if(n){
   init_ids<<<(n+255)/256,256,0,stream>>>(ids,n);
   C10_CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(temp,bytes,scores.data_ptr<float>(),w[0].data_ptr<float>(),ids,sorted,n,0,32,stream));
   ranked_counts<<<(n+255)/256,256,0,stream>>>(sorted,counts.data_ptr<int>(),rc,n,r);
   bytes=w[10].nbytes();C10_CUDA_CHECK(cub::DeviceScan::InclusiveSum(temp,bytes,rc,rp,n,stream));
   choose_take<<<(n+255)/256,256,0,stream>>>(sorted,rp,counts.data_ptr<int>(),take,n,r,sink,(budget-protect)*ft);
 }
 bytes=w[10].nbytes();C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp,bytes,take,prefix,all,stream));
 full_counts<<<(all+255)/256,256,0,stream>>>(take,counts.data_ptr<int>(),full,all,r);
 bytes=w[10].nbytes();C10_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp,bytes,full,fp,all,stream));
 materialize<<<dim3(r,start+current),256,0,stream>>>(take,prefix,fp,counts.data_ptr<int>(),starts.data_ptr<int>(),index.data_ptr<int>(),
 offsets.data_ptr<int64_t>(),slots.data_ptr<int>(),sources.numel()?sources.data_ptr<int64_t>():nullptr,r,cap,ft,start,slots.numel(),reserve,chrono);
 dma_table<<<1,1,0,stream>>>(take,prefix,fp,counts.data_ptr<int>(),starts.data_ptr<int>(),addresses.data_ptr<int64_t>(),table.data_ptr<int64_t>(),r,start,(int64_t)payload.data_ptr(),batch);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
__global__ void rotate_fetch(__nv_bfloat16* p,const int64_t* offsets,const int* slots,const double* phase,int total){
 int pair=threadIdx.x%22,head=threadIdx.x/22;
 for(int token=blockIdx.x;token<total;token+=gridDim.x){
   int64_t addr=offsets[token]+head*128+pair*2;
   double a=__bfloat162float(p[addr]),b=__bfloat162float(p[addr+1]);
   int64_t f=int64_t(slots[token])*44+pair*2;double c=phase[f],d=phase[f+1];
   // Match the verified Triton implementation: no FP contraction, then FP32 -> BF16.
   double x=__dsub_rn(__dmul_rn(a,c),__dmul_rn(b,d));
   double y=__dadd_rn(__dmul_rn(a,d),__dmul_rn(b,c));
   p[addr]=__float2bfloat16_rn(__double2float_rn(x));p[addr+1]=__float2bfloat16_rn(__double2float_rn(y));
 }
}
void launch_route_rotate(T payload,T offsets,T slots,T phase,cudaStream_t stream){
 rotate_fetch<<<std::min<int64_t>(slots.numel(),4096),12*22,0,stream>>>((__nv_bfloat16*)payload.data_ptr(),offsets.data_ptr<int64_t>(),slots.data_ptr<int>(),phase.data_ptr<double>(),slots.numel());
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#include <cuda_fp16.h>
// PyTorch 2.8 Reduce.cuh for this fixed [F,R,1024,12,128] FP32 sum:
// output vectorization=4, block=(32,4), vt0=4. Its CTA reduction tree
// is evaluated locally per output, so gather/narrow/mean/max share one load.
// Keeping this tree also preserves FP16 ties at edge counts.
__global__ void pool_exact_kernel(const __nv_bfloat16* q,__half* out,const int* index,const int* counts,
 int ft,int r,int frames,int cap,int ctas,int64_t qt,int64_t qh){
 int output=blockIdx.x*blockDim.x+threadIdx.x;
 int total=frames*r*12*128;if(output>=total)return;
 int d=output%128,head=output/128%12,b=output/(128*12)%r,f=output/(128*12*r),count=counts[b];
 float partial[16],maximum=-__int_as_float(0x7f800000);
 for(int cta=0;cta<ctas;cta++){
   float warp[4];
   #pragma unroll
   for(int y=0;y<4;y++){
     float acc[4]={0,0,0,0};int idx=y+cta*4,stride=4*ctas;
     while(idx<cap){
       #pragma unroll
       for(int a=0;a<4;a++){
         int tokenidx=idx+a*stride;
         if(tokenidx<cap){
           float value=0;
           if(tokenidx<count){int token=index[b*cap+tokenidx];value=__half2float(__float2half_rn(__bfloat162float(q[(int64_t(f)*ft+token)*qt+head*qh+d])));maximum=fmaxf(maximum,value);}
           acc[a]=__fadd_rn(acc[a],value);
         }
       }
       idx+=4*stride;
     }
     warp[y]=__fadd_rn(__fadd_rn(__fadd_rn(acc[0],acc[1]),acc[2]),acc[3]);
   }
   partial[cta]=__fadd_rn(__fadd_rn(warp[0],warp[2]),__fadd_rn(warp[1],warp[3]));
 }
 float result;
 if(ctas==1)result=partial[0];
 else{
   float warp[4]={0,0,0,0};
   for(int y=0;y<4;y++)for(int i=y;i<ctas;i+=4)warp[y]=__fadd_rn(warp[y],partial[i]);
   result=__fadd_rn(__fadd_rn(warp[0],warp[2]),__fadd_rn(warp[1],warp[3]));
 }
 int offset=((head*frames+f)*r+b)*128+d;
 out[offset]=__float2half_rn(__fdiv_rn(result,float(count)));
 out[12*frames*r*128+offset]=__float2half_rn(maximum);
}
void pool_query_exact(T q,T out,T index,T counts,int64_t ft,int64_t cap){
 TORCH_CHECK(cap==1024&&q.scalar_type()==at::kBFloat16,"frozen Q pooling ABI");
 int r=counts.numel(),frames=q.size(1)/ft;
 const auto* prop=at::cuda::getCurrentDeviceProperties();
 int target=prop->multiProcessorCount*(prop->maxThreadsPerMultiProcessor/128);
 int grid=frames*r*12;
 int ctas=std::max(1,std::min((target+grid-1)/grid,16));
 auto stream=at::cuda::getCurrentCUDAStream(q.get_device()).stream();
 int outputs=frames*r*12*128;
 pool_exact_kernel<<<(outputs+127)/128,128,0,stream>>>((__nv_bfloat16*)q.data_ptr(),(__half*)out.data_ptr(),index.data_ptr<int>(),counts.data_ptr<int>(),ft,r,frames,cap,ctas,q.stride(1),q.stride(2));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void initialize_route_kernels(){
 cudaFuncAttributes attributes;
 C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,rotate_fetch));
}
