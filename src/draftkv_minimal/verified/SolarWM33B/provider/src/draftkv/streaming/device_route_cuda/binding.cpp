#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include "prepacked_params.h"
#ifdef DRAFTKV_NATIVE_ONLY
void preload_prepared_fwd() {}
void launch_prepared_fwd(PreparedParams&, cudaStream_t) {
    TORCH_CHECK(false,"this architecture requires HeadGroupPlan and installed native FlashAttention");
}
#else
void launch_prepared_fwd(PreparedParams&, cudaStream_t);
void preload_prepared_fwd();
#endif
void prepared_fwd(at::Tensor q, at::Tensor payload, at::Tensor offsets,
                  at::Tensor slots, at::Tensor phase, at::Tensor frontier,
                  at::Tensor out, at::Tensor lse, int64_t head_group,
                  int64_t first, int64_t last, bool chrono) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type()==at::kBFloat16 &&
                q.dim()==4 && q.size(0)==1 && q.size(3)==128, "Q must be CUDA BF16 [1,Q,H,128]");
    TORCH_CHECK(payload.is_cuda() && payload.is_contiguous() && payload.scalar_type()==at::kBFloat16, "invalid paired payload");
    TORCH_CHECK(head_group>0 && q.size(2)%head_group==0, "invalid group");
    int b=1, sq=q.size(1), sk=offsets.size(2), h=q.size(2), d=128;
    TORCH_CHECK(offsets.is_cuda() && offsets.scalar_type()==at::kLong && offsets.is_contiguous() && offsets.dim()==3 && offsets.size(0)==h/head_group && offsets.size(1)==2, "invalid offsets");
    TORCH_CHECK(slots.is_cuda() && slots.scalar_type()==at::kInt && slots.is_contiguous() && slots.numel()==sk, "invalid time map");
    TORCH_CHECK(phase.is_cuda() && phase.is_contiguous() && (!chrono ||
        (phase.scalar_type()==at::kDouble && phase.dim()==2 && phase.size(1)==44)), "invalid phases");
    TORCH_CHECK(frontier.is_cuda() && frontier.scalar_type()==at::kInt && frontier.is_contiguous() && frontier.numel()==h/head_group, "invalid frontier");
    TORCH_CHECK(out.sizes()==q.sizes() && out.is_contiguous() && out.scalar_type()==at::kBFloat16 && lse.is_contiguous() && lse.scalar_type()==at::kFloat && lse.numel()==h*sq, "invalid output");
    for (auto t : {payload, offsets, slots, phase, frontier, out, lse}) TORCH_CHECK(t.device()==q.device(), "device mismatch");
    c10::cuda::CUDAGuard guard(q.device());
    PreparedParams p{};
    p.kv_head_group=head_group;
    p.token_offsets=offsets.data_ptr<int64_t>(); p.time_slots=slots.data_ptr<int>();
    p.time_phase=chrono ? phase.data_ptr<double>() : nullptr; p.chrono=chrono;
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
    p.streaming_frontier_batch_stride=h/head_group;
    launch_prepared_fwd(p, at::cuda::getCurrentCUDAStream(q.get_device()).stream());
}
#include <c10/cuda/CUDAException.h>
#include <cuda.h>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <chrono>
#include <exception>

int64_t route_workspace(int64_t);
void device_route(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
 at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
 at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
 int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool,int64_t,bool,bool);

// Host only submits the GPU-produced descriptor rows. No score, block or token
// processing, no mapped-host load kernel. Tensor owners survive until DMA ends.
class AsyncProducer {
 struct Job {
   at::Tensor table,ready,payload,consumed;
   int64_t slots=0,first=0,stop=-1;
   bool windowed=false;
   int64_t admitted=0;
   cudaEvent_t description,begin,end;
   cudaStream_t stream;
   int device;
   int64_t rows=0,count=0,bytes=0;
   double wait_ms=0,submit_ms=0;
 };
 std::mutex mutex_;
 std::condition_variable cv_;
 std::deque<std::shared_ptr<Job>> queue_,finished_;
 std::shared_ptr<Job> windowed_;
 std::thread worker_;
 bool stopping_=false,working_=false;
 std::exception_ptr failure_;
 static double ms(std::chrono::steady_clock::time_point a) {
   return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count();
 }
 void run() {
   while(true) {
     std::shared_ptr<Job> j;
     {std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock,[&]{return stopping_ || !queue_.empty();});
      if(queue_.empty() && stopping_)return;
      j=queue_.front();queue_.pop_front();working_=true;}
     try {
       C10_CUDA_CHECK(cudaSetDevice(j->device));
       auto t=std::chrono::steady_clock::now();
       C10_CUDA_CHECK(cudaEventSynchronize(j->description));j->wait_ms=ms(t);
       auto rows=j->table.data_ptr<int64_t>();
       j->rows=rows[0];auto declared_bytes=rows[1];j->bytes=0;
       TORCH_CHECK(j->rows>=0 && 8+5*j->rows<=j->table.numel(),"invalid DMA description count");
       C10_CUDA_CHECK(cudaEventRecord(j->begin,j->stream));
       t=std::chrono::steady_clock::now();
       int64_t ordinal=0;bool unit_begin=true;
       std::vector<int64_t> previous(j->slots,-1);
       for(int64_t i=0;i<j->rows;i++) {
         auto r=rows+8+5*i;
         if(j->windowed && unit_begin) {
           // Never install a reuse wait whose consumer has not been queued.
           // cudaMemcpyAsync may block inside the driver; FA3's host launch
           // also needs that driver's lock, even after its first call.
           std::unique_lock<std::mutex> lock(mutex_);
           cv_.wait(lock,[&]{return stopping_ || ordinal<j->admitted;});
           TORCH_CHECK(!stopping_,"slot producer stopped before admission");
         }
         TORCH_CHECK(r[2]>=0 && r[3]>=0 && r[3]<j->ready.numel(),"invalid DMA readiness index");
         bool selected=ordinal>=j->first && (j->stop<0 || ordinal<j->stop);
         if(selected && unit_begin && j->slots && ordinal>=j->slots) {
           auto prior=previous[ordinal%j->slots];
           TORCH_CHECK(prior>=0 && prior<j->consumed.numel(),"invalid slot reuse predecessor");
           auto rc=cuStreamWaitValue32(reinterpret_cast<CUstream>(j->stream),
             reinterpret_cast<CUdeviceptr>(j->consumed.data_ptr<int>()+prior),1,CU_STREAM_WAIT_VALUE_GEQ);
           TORCH_CHECK(rc==CUDA_SUCCESS,"slot consumer wait failed: ",int(rc));
         }
         unit_begin=false;
         if(selected && r[2]) {
           ++j->count;j->bytes+=r[2];
           TORCH_CHECK(r[1]>=reinterpret_cast<int64_t>(j->payload.data_ptr()) &&
             r[1]+r[2]<=reinterpret_cast<int64_t>(j->payload.data_ptr())+j->payload.nbytes(),"invalid DMA span");
           C10_CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(r[1]),reinterpret_cast<void*>(r[0]),
                                          r[2],cudaMemcpyHostToDevice,j->stream));
         }
         if(selected && r[4]>=0) {
           auto rc=cuStreamWriteValue32(reinterpret_cast<CUstream>(j->stream),
              reinterpret_cast<CUdeviceptr>(j->ready.data_ptr<int>()+r[3]),r[4],CU_STREAM_WRITE_VALUE_DEFAULT);
           TORCH_CHECK(rc==CUDA_SUCCESS,"frontier publication failed: ",int(rc));
         }
         if(r[4]>=0) {
           if(j->slots)previous[ordinal%j->slots]=r[3];
           ++ordinal;unit_begin=true;
         }
       }
       TORCH_CHECK(j->bytes<=declared_bytes && (j->first!=0 || j->stop>=0 || j->bytes==declared_bytes),
         "DMA descriptor byte count mismatch");
       C10_CUDA_CHECK(cudaEventRecord(j->end,j->stream));j->submit_ms=ms(t);
     } catch(...) {std::lock_guard<std::mutex> lock(mutex_);failure_=std::current_exception();}
     {std::lock_guard<std::mutex> lock(mutex_);finished_.push_back(j);working_=false;cv_.notify_all();}
   }
 }
public:
 AsyncProducer():worker_([this]{run();}) {}
 ~AsyncProducer() {
   {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;cv_.notify_all();}
   worker_.join();
   for(auto &j:finished_) {
     cudaSetDevice(j->device);cudaEventSynchronize(j->end);
     cudaEventDestroy(j->description);cudaEventDestroy(j->begin);cudaEventDestroy(j->end);
   }
 }
 void submit_impl(at::Tensor gpu,at::Tensor cpu,at::Tensor ready,at::Tensor payload,int64_t stream,
                  at::Tensor consumed=at::Tensor(),int64_t slots=0,int64_t first=0,int64_t stop=-1,
                  bool windowed=false) {
   TORCH_CHECK(gpu.is_cuda() && gpu.scalar_type()==at::kLong && gpu.is_contiguous() &&
     cpu.device().is_cpu() && cpu.is_pinned() && cpu.sizes()==gpu.sizes() && cpu.scalar_type()==at::kLong,
     "invalid descriptor staging buffers");
   c10::cuda::CUDAGuard guard(gpu.device());
   auto j=std::make_shared<Job>();j->table=cpu;j->ready=ready;j->payload=payload;
   j->consumed=consumed;j->slots=slots;j->first=first;j->stop=stop;
   j->windowed=windowed;j->admitted=slots;
   j->device=gpu.get_device();j->stream=reinterpret_cast<cudaStream_t>(stream);
   C10_CUDA_CHECK(cudaEventCreateWithFlags(&j->description,cudaEventDisableTiming));
   C10_CUDA_CHECK(cudaEventCreate(&j->begin));C10_CUDA_CHECK(cudaEventCreate(&j->end));
   auto current=at::cuda::getCurrentCUDAStream(j->device).stream();
   C10_CUDA_CHECK(cudaMemcpyAsync(cpu.data_ptr(),gpu.data_ptr(),gpu.nbytes(),cudaMemcpyDeviceToHost,current));
   C10_CUDA_CHECK(cudaEventRecord(j->description,current));
   {std::lock_guard<std::mutex> lock(mutex_);
    if(failure_)std::rethrow_exception(failure_);
    if(windowed) {
      TORCH_CHECK(!windowed_,"previous windowed slot job was not collected");
      windowed_=j;
    }
    queue_.push_back(j);cv_.notify_all();}
 }
 void submit(at::Tensor gpu,at::Tensor cpu,at::Tensor ready,at::Tensor payload,int64_t stream) {
   submit_impl(gpu,cpu,ready,payload,stream);
 }
 void submit_slots(at::Tensor gpu,at::Tensor cpu,at::Tensor ready,at::Tensor payload,
                   at::Tensor consumed,int64_t stream,int64_t slots,int64_t first,int64_t stop) {
   TORCH_CHECK(slots>0 && first>=0 && (stop<0 || stop>first) && consumed.is_cuda() &&
     consumed.device()==ready.device() && consumed.scalar_type()==at::kInt &&
     consumed.is_contiguous() && consumed.numel()==ready.numel(),"invalid slot reuse contract");
   submit_impl(gpu,cpu,ready,payload,stream,consumed,slots,first,stop);
 }
 void submit_slots_windowed(at::Tensor gpu,at::Tensor cpu,at::Tensor ready,at::Tensor payload,
                            at::Tensor consumed,int64_t stream,int64_t slots) {
   TORCH_CHECK(slots>0 && consumed.is_cuda() && consumed.device()==ready.device() &&
     consumed.scalar_type()==at::kInt && consumed.is_contiguous() &&
     consumed.numel()==ready.numel(),"invalid windowed slot reuse contract");
   submit_impl(gpu,cpu,ready,payload,stream,consumed,slots,0,-1,true);
 }
 void admit_slots(int64_t units) {
   std::lock_guard<std::mutex> lock(mutex_);
   if(failure_)std::rethrow_exception(failure_);
   TORCH_CHECK(windowed_ && units>=windowed_->admitted &&
     units<=windowed_->ready.numel(),"invalid slot admission frontier");
   windowed_->admitted=units;cv_.notify_all();
 }
 void wait_submitted() {
   // Join host API submission, NOT DMA completion. All copy-engine work and
   // frontier writes must exist before a consumer-side allocator/module load
   // can synchronize the context. A stream wait on work not yet submitted
   // can otherwise deadlock with CUDA's runtime lock. Payload DMA continues
   // asynchronously while prep/native attention consume earlier units.
   std::unique_lock<std::mutex> lock(mutex_);
   cv_.wait(lock,[&]{return queue_.empty() && !working_;});
   if(failure_)std::rethrow_exception(failure_);
 }
 std::vector<std::vector<double>> collect() {
   std::deque<std::shared_ptr<Job>> done;
   {std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock,[&]{return queue_.empty() && !working_;});
    if(failure_)std::rethrow_exception(failure_);
    done.swap(finished_);windowed_.reset();}
   std::vector<std::vector<double>> result;
   for(auto &j:done) {
     C10_CUDA_CHECK(cudaSetDevice(j->device));C10_CUDA_CHECK(cudaEventSynchronize(j->end));
     float elapsed;C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed,j->begin,j->end));
     result.push_back({double(j->count),double(j->bytes),j->wait_ms,j->submit_ms,double(elapsed)});
     cudaEventDestroy(j->description);cudaEventDestroy(j->begin);cudaEventDestroy(j->end);
   }
   return result;
 }
};
void wait_ready(at::Tensor ready) {
 c10::cuda::CUDAGuard guard(ready.device());
 auto s=at::cuda::getCurrentCUDAStream(ready.get_device()).stream();
 auto rc=cuStreamWaitValue32(reinterpret_cast<CUstream>(s),reinterpret_cast<CUdeviceptr>(ready.data_ptr<int>()),0,CU_STREAM_WAIT_VALUE_EQ);
 TORCH_CHECK(rc==CUDA_SUCCESS,"frontier wait failed: ",int(rc));
}
void group_ready(at::Tensor ready,int64_t index,bool publish) {
 c10::cuda::CUDAGuard guard(ready.device());
 TORCH_CHECK(ready.is_cuda() && ready.scalar_type()==at::kInt && ready.is_contiguous() &&
             index>=0 && index<ready.numel(),"invalid group readiness");
 auto s=reinterpret_cast<CUstream>(at::cuda::getCurrentCUDAStream(ready.get_device()).stream());
 auto p=reinterpret_cast<CUdeviceptr>(ready.data_ptr<int>()+index);
 // DEFAULT write includes the system-scope memory barrier. H2D and write are
 // ordered on the same CUDA stream; wait precedes every consumer HBM/TMA read.
 auto rc=publish?cuStreamWriteValue32(s,p,1,CU_STREAM_WRITE_VALUE_DEFAULT):
                 cuStreamWaitValue32(s,p,1,CU_STREAM_WAIT_VALUE_GEQ);
 TORCH_CHECK(rc==CUDA_SUCCESS,"group readiness operation failed: ",int(rc));
}
void build_groups(at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor,
                  at::Tensor,at::Tensor,at::Tensor,int64_t,int64_t,int64_t,int64_t,
                  int64_t,int64_t,int64_t,int64_t,bool,at::Tensor,int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("fwd",&prepared_fwd);m.def("preload",&preload_prepared_fwd);m.def("route_workspace",&route_workspace);
 m.def("route",&device_route);m.def("wait_ready",&wait_ready);
 m.def("build_groups",&build_groups);m.def("group_ready",&group_ready);
 pybind11::class_<AsyncProducer>(m,"AsyncProducer")
   .def(pybind11::init<>()).def("submit",&AsyncProducer::submit)
   .def("submit_slots",&AsyncProducer::submit_slots)
   .def("submit_slots_windowed",&AsyncProducer::submit_slots_windowed)
   .def("admit_slots",&AsyncProducer::admit_slots)
   .def("wait_submitted",&AsyncProducer::wait_submitted,pybind11::call_guard<pybind11::gil_scoped_release>())
   .def("collect",&AsyncProducer::collect,pybind11::call_guard<pybind11::gil_scoped_release>());
}
