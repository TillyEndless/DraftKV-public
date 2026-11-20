#include "prepacked_params.h"
#include "prepacked_fwd_kernel.h"
#include <c10/cuda/CUDAException.h>
#include <map>
#include <string>

template<int M,int N,bool Split>
__global__ void offload_fa2(__grid_constant__ const PreparedSM120Params p) {
    using Traits=Flash_fwd_kernel_traits<128,M,N,4,false,false,cutlass::bfloat16_t>;
    if constexpr(Split) flash::compute_attn_1rowblock_splitkv<Traits,false,false,false,false,true,false,true,false>(p,0,blockIdx.z,blockIdx.x,blockIdx.y,p.num_splits);
    else flash::compute_attn_1rowblock<Traits,false,false,false,false,false,true,false,false>(p,0,blockIdx.z,blockIdx.x);
}
template<int N,int Log>
__global__ void offload_merge(__grid_constant__ const PreparedSM120Params p) {
    using Traits=Flash_fwd_kernel_traits<128,64,N,4,false,false,cutlass::bfloat16_t>;
    flash::combine_attn_seqk_parallel<Traits,4,Log,true>(p);
}
template<int M,int N,bool Split>
std::map<std::string,int64_t> resources() {
    using Traits=Flash_fwd_kernel_traits<128,M,N,4,false,false,cutlass::bfloat16_t>;
    constexpr int shared=Traits::kSmemSize;
    auto kernel=offload_fa2<M,N,Split>;
    if constexpr(shared>=48*1024) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    cudaFuncAttributes a;int active;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,128,shared));
    return {{"block_m",M},{"block_n",N},{"split",Split},{"registers_per_thread",a.numRegs},{"shared_bytes_per_cta",shared+a.sharedSizeBytes},{"active_cta_per_sm",active},{"local_bytes_per_thread",a.localSizeBytes}};
}
template<int M,int N,bool Split>
void launch(PreparedSM120Params &p,cudaStream_t stream) {
    using Traits=Flash_fwd_kernel_traits<128,M,N,4,false,false,cutlass::bfloat16_t>;
    constexpr int shared=Traits::kSmemSize;
    auto kernel=offload_fa2<M,N,Split>;
    if constexpr(shared>=48*1024) C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    kernel<<<dim3((p.seqlen_q+M-1)/M,Split?p.num_splits:1,p.h),128,shared,stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if constexpr(Split) {
        int blocks=(p.b*p.h*p.seqlen_q+3)/4;
        if(p.num_splits<=2) offload_merge<N,1><<<blocks,128,0,stream>>>(p);
        else if(p.num_splits<=4) offload_merge<N,2><<<blocks,128,0,stream>>>(p);
        else if(p.num_splits<=8) offload_merge<N,3><<<blocks,128,0,stream>>>(p);
        else offload_merge<N,4><<<blocks,128,0,stream>>>(p);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
void launch_prepared_fwd(PreparedSM120Params &p,cudaStream_t stream,int splits) {
    if(splits>1) launch<64,64,true>(p,stream); else launch<64,64,false>(p,stream);
}
std::map<std::string,int64_t> prepared_info(int splits) {
    return splits>1 ? resources<64,64,true>() : resources<64,64,false>();
}
