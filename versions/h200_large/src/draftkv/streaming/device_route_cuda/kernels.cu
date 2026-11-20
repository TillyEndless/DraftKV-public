#include "prepacked_params.h"
#include "prepacked_fwd_kernel.h"
#include <c10/cuda/CUDAException.h>
#include <map>
#include <string>

template<bool Cache,int Sleep> struct OptParams : PreparedParams {
    static constexpr bool cache_frontier=Cache;
    static constexpr int sleep_ns=Sleep;
};
template<int N,bool Cache,int Sleep,int M=64,bool Q=false,bool Share=false>
__global__ void b01_frontier_opt(__grid_constant__ const OptParams<Cache,Sleep> p) {
    using Traits=Flash_fwd_kernel_traits<128,M,N,4,Q,Share,cutlass::bfloat16_t>;
    flash::compute_attn_1rowblock<Traits,false,false,false,false,false,true,false,false>(p,0,blockIdx.z,blockIdx.x);
}
template<int N,bool Cache,int Sleep,int M=64,bool Q=false,bool Share=false>
void launch(PreparedParams &p,cudaStream_t stream) {
    OptParams<Cache,Sleep> op;
    static_cast<PreparedParams&>(op)=p;
    using Traits=Flash_fwd_kernel_traits<128,M,N,4,Q,Share,cutlass::bfloat16_t>;
    constexpr int smem=Traits::kSmemSize;
    if constexpr(smem>=48*1024) C10_CUDA_CHECK(cudaFuncSetAttribute(b01_frontier_opt<N,Cache,Sleep,M,Q,Share>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    b01_frontier_opt<N,Cache,Sleep,M,Q,Share><<<dim3((p.seqlen_q+M-1)/M,1,p.h),128,smem,stream>>>(op);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void launch_prepared_fwd(PreparedParams &p,cudaStream_t s) {
    launch<64,false,64,64,false,false>(p,s);
}

// Explicitly resolve this kernel before a producer-dependent stream wait exists.
// CUDA lazy module/function loading can synchronize the context on first use.
void preload_prepared_fwd() {
    cudaFuncAttributes attributes{};
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes,b01_frontier_opt<64,false,64,64,false,false>));
    using Traits=Flash_fwd_kernel_traits<128,64,64,4,false,false,cutlass::bfloat16_t>;
    if constexpr(Traits::kSmemSize>=48*1024)
        C10_CUDA_CHECK(cudaFuncSetAttribute(b01_frontier_opt<64,false,64,64,false,false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,Traits::kSmemSize));
}
