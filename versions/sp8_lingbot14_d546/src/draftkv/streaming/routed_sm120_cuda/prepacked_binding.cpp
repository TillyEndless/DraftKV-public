#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <map>
#include <string>
#include "prepacked_params.h"
void launch_prepared_fwd(PreparedSM120Params&, cudaStream_t,int);
std::map<std::string,int64_t> prepared_info(int);
void prepared_fwd(at::Tensor q, at::Tensor payload, at::Tensor offsets,
                  at::Tensor slots, at::Tensor phase, at::Tensor frontier,
                  at::Tensor out, at::Tensor lse, int64_t head_group,
                  int64_t first, int64_t last, bool chrono, int64_t splits, at::Tensor partial, at::Tensor partial_lse) {
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
    PreparedSM120Params p{};
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
    TORCH_CHECK(splits==1 || splits==2 || splits==4,"SM120 experiment supports S1/S2/S4");
    if(splits>1) {
        TORCH_CHECK(partial.is_cuda() && partial_lse.is_cuda() && partial.device()==q.device() && partial_lse.device()==q.device(),"invalid split workspace device");
        TORCH_CHECK(partial.scalar_type()==at::kFloat && partial_lse.scalar_type()==at::kFloat && partial.is_contiguous() && partial_lse.is_contiguous(),"FP32 split workspace required");
        TORCH_CHECK(partial.numel()==splits*12*sq*128 && partial_lse.numel()==splits*12*sq,"invalid split workspace size");
        p.num_splits=splits; p.oaccum_ptr=partial.data_ptr(); p.softmax_lseaccum_ptr=partial_lse.data_ptr();
    }
    launch_prepared_fwd(p, at::cuda::getCurrentCUDAStream(q.get_device()).stream(),splits);
}
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include "route_producer.h"
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("fwd", &prepared_fwd);
 m.def("initialize_route_kernels",&initialize_route_kernels);
 m.def("pool_query_exact",&pool_query_exact);
 m.def("route_workspace_bytes",&route_workspace_bytes);
 m.def("gpu_route",&gpu_route);
 pybind11::class_<RouteTicket,std::shared_ptr<RouteTicket>>(m,"RouteTicket")
 .def("metrics",&RouteTicket::metrics,pybind11::arg("wait")=false,pybind11::call_guard<pybind11::gil_scoped_release>());
 pybind11::class_<RouteProducer>(m,"RouteProducer")
 .def(pybind11::init<int>()).def("submit",&RouteProducer::submit);

 m.def("info", &prepared_info);
}
