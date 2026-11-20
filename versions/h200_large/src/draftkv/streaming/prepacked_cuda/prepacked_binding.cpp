#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include "prepacked_params.h"
void launch_prepared_fwd(PreparedParams&, cudaStream_t);
void prepared_fwd(at::Tensor q, at::Tensor payload, at::Tensor offsets,
                  at::Tensor slots, at::Tensor phase, at::Tensor frontier,
                  at::Tensor out, at::Tensor lse, int64_t head_group,
                  int64_t first, int64_t last, bool chrono) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type()==at::kBFloat16 &&
                q.dim()==4 && q.size(0)==1 && q.size(2)==12 && q.size(3)==128, "Q must be CUDA BF16 [1,Q,12,128]");
    TORCH_CHECK(payload.is_cuda() && payload.is_contiguous() && payload.scalar_type()==at::kBFloat16, "invalid paired payload");
    TORCH_CHECK(head_group==3 || head_group==4 || head_group==6 || head_group==12, "invalid group");
    int b=1, sq=q.size(1), sk=offsets.size(2), h=12, d=128;
    TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type()==at::kLong && offsets.is_contiguous() && offsets.dim()==3 && offsets.size(0)==12/head_group && offsets.size(1)==2, "invalid offsets");
    TORCH_CHECK(slots.is_cuda() && slots.scalar_type()==at::kInt && slots.is_contiguous() && slots.numel()==sk, "invalid time map");
    TORCH_CHECK(phase.is_cuda() && phase.scalar_type()==at::kDouble && phase.is_contiguous() && phase.dim()==2 && phase.size(1)==44, "invalid phases");
    TORCH_CHECK(frontier.is_cuda() && frontier.scalar_type()==at::kInt && frontier.is_contiguous() && frontier.numel()==12/head_group, "invalid frontier");
    TORCH_CHECK(out.sizes()==q.sizes() && out.is_contiguous() && out.scalar_type()==at::kBFloat16 && lse.is_contiguous() && lse.scalar_type()==at::kFloat && lse.numel()==12*sq, "invalid output");
    for (auto t : {payload, offsets, slots, phase, frontier, out, lse}) TORCH_CHECK(t.device()==q.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(q.device());
    PreparedParams p{};
    p.kv_head_group=head_group;
    p.token_offsets=offsets.data_ptr<int64_t>(); p.time_slots=slots.data_ptr<int>();
    p.time_phase=phase.data_ptr<double>(); p.chrono=chrono;
    p.q_ptr=q.data_ptr(); p.k_ptr=payload.data_ptr(); p.v_ptr=payload.data_ptr();
    p.o_ptr=out.data_ptr(); p.softmax_lse_ptr=lse.data_ptr();
    p.q_batch_stride=q.stride(0); p.o_batch_stride=out.stride(0);
    p.q_row_stride=q.stride(1); p.o_row_stride=out.stride(1);
    p.q_head_stride=q.stride(2); p.o_head_stride=out.stride(2);
    p.k_row_stride=p.v_row_stride=head_group*128; p.k_head_stride=p.v_head_stride=128;
    p.b = b;
    p.h = h;
    p.h_k = h;
    p.h_h_k_ratio = 1;
    p.seqlen_q = sq;
    p.seqlen_k = sk;
    p.seqlen_q_rounded = ((sq + 127) / 128) * 128;
    p.seqlen_k_rounded = ((sk + 127) / 128) * 128;
    p.d = d;
    p.d_rounded = d;
    p.total_q = sq;

    p.scale_softmax = static_cast<float>((1.0 / std::sqrt(128.0)));
    p.scale_softmax_log2 = static_cast<float>((1.0 / std::sqrt(128.0)) * M_LOG2E);
    p.p_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0f;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.window_size_left = -1;
    p.window_size_right = -1;
    p.softcap = 0.0f;
    p.is_bf16 = true;
    p.is_causal = false;
    p.is_seqlens_k_cumulative = true;
    p.unpadded_lse = false;
    p.seqlenq_ngroups_swapped = false;

    p.streaming_first_tile=first; p.streaming_last_tile=last;
    p.streaming_group_tiles=4;
    p.streaming_frontier_ptr=last<first ? nullptr : frontier.data_ptr<int>();
    p.streaming_frontier_batch_stride=12/head_group;
    launch_prepared_fwd(p, at::cuda::getCurrentCUDAStream(q.get_device()).stream());
}
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
// A CPU submission plan: still actual pinned H2D on the copy engine, no GPU producer.
class PairedProducer {
    std::vector<at::Tensor> dk_,dv_,sk_,sv_;
    at::Tensor ready_;
    std::vector<int> index_,value_;
    int64_t bytes_=0;
    void write(cudaStream_t s,int index,int value) {
        CUresult rc=cuStreamWriteValue32(reinterpret_cast<CUstream>(s),
            reinterpret_cast<CUdeviceptr>(ready_.data_ptr<int>()+index),value,CU_STREAM_WRITE_VALUE_DEFAULT);
        TORCH_CHECK(rc==CUDA_SUCCESS,"stream write failed: ",int(rc));
    }
public:
    PairedProducer(std::vector<at::Tensor> dk,std::vector<at::Tensor> dv,
                 std::vector<at::Tensor> sk,std::vector<at::Tensor> sv,
                 at::Tensor ready,std::vector<int> index,std::vector<int> value)
      :dk_(std::move(dk)),dv_(std::move(dv)),sk_(std::move(sk)),sv_(std::move(sv)),
       ready_(ready),index_(std::move(index)),value_(std::move(value)) {
        const auto n=dk_.size();
        TORCH_CHECK(n && dv_.size()==n && sk_.size()==n && sv_.size()==n && index_.size()==n && value_.size()==n,"plan size mismatch");
        TORCH_CHECK(ready_.is_cuda() && ready_.scalar_type()==at::kInt && ready_.is_contiguous(),"invalid readiness vector");
        for(size_t i=0;i<n;i++) {
            TORCH_CHECK(dk_[i].device()==ready_.device() && dv_[i].device()==ready_.device(),"destination device mismatch");
            TORCH_CHECK(sk_[i].device().is_cpu() && sv_[i].device().is_cpu() && sk_[i].is_pinned() && sv_[i].is_pinned(),"pinned CPU sources required");
            TORCH_CHECK(dk_[i].is_contiguous() && dv_[i].is_contiguous() && sk_[i].is_contiguous() && sv_[i].is_contiguous(),"DMA must be contiguous");
            TORCH_CHECK(dk_[i].scalar_type()==at::kBFloat16 && dv_[i].scalar_type()==at::kBFloat16 && sk_[i].scalar_type()==at::kBFloat16 && sv_[i].scalar_type()==at::kBFloat16,"BF16 required");
            TORCH_CHECK(dk_[i].sizes()==sk_[i].sizes() && dv_[i].sizes()==sv_[i].sizes(),"copy shape mismatch");
            TORCH_CHECK(index_[i]>=0 && index_[i]<ready_.numel(),"readiness index out of bounds");
            bytes_+=sk_[i].nbytes();
        }
    }
    void initialize(int value) {
        c10::cuda::CUDAGuard guard(ready_.device());
        auto stream=at::cuda::getCurrentCUDAStream(ready_.get_device()).stream();
        for(int i=0;i<ready_.numel();i++) write(stream,i,value);
    }
    void enqueue(int begin,int end) {
        TORCH_CHECK(begin>=0 && end>=begin && end<=dk_.size(),"invalid plan interval");
        c10::cuda::CUDAGuard guard(ready_.device());
        auto stream=at::cuda::getCurrentCUDAStream(ready_.get_device()).stream();
        for(int i=begin;i<end;i++) {
            C10_CUDA_CHECK(cudaMemcpyAsync(dk_[i].data_ptr(),sk_[i].data_ptr(),sk_[i].nbytes(),cudaMemcpyHostToDevice,stream));

            if (value_[i]>=0) write(stream,index_[i],value_[i]);
        }
    }
    int64_t bytes() const{return bytes_;}
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("fwd", &prepared_fwd);
 pybind11::class_<PairedProducer>(m,"PairedProducer")
 .def(pybind11::init<std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>,std::vector<at::Tensor>,at::Tensor,std::vector<int>,std::vector<int>>())
 .def("initialize",&PairedProducer::initialize).def("enqueue",&PairedProducer::enqueue).def("bytes",&PairedProducer::bytes);
}
