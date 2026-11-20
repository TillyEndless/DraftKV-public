#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <chrono>
#include <exception>
#include <vector>
class SlotAsyncProducer {
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
 bool batch_=false;
 bool stopping_=false,working_=false;
 std::exception_ptr failure_;
 std::thread worker_;
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
       std::vector<CUdeviceptr> dsts,srcs;
       std::vector<size_t> sizes;
       dsts.reserve(j->rows);srcs.reserve(j->rows);sizes.reserve(j->rows);
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
           if(batch_) {
             dsts.push_back(static_cast<CUdeviceptr>(r[1]));
             srcs.push_back(static_cast<CUdeviceptr>(r[0]));sizes.push_back(r[2]);
           } else {
             auto rc=cuMemcpyHtoDAsync(static_cast<CUdeviceptr>(r[1]),reinterpret_cast<const void*>(r[0]),
                                      r[2],reinterpret_cast<CUstream>(j->stream));
             TORCH_CHECK(rc==CUDA_SUCCESS,"slot driver DMA failed: ",int(rc));
           }
         }
         if(selected && r[4]>=0 && batch_ && !sizes.empty()) {
           CUmemcpyAttributes attr{};
           attr.srcAccessOrder=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
           attr.flags=CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
           size_t first=0,failed=0;
           auto rc=cuMemcpyBatchAsync(dsts.data(),srcs.data(),sizes.data(),sizes.size(),
             &attr,&first,1,&failed,reinterpret_cast<CUstream>(j->stream));
           TORCH_CHECK(rc==CUDA_SUCCESS,"slot batch DMA failed: ",int(rc)," at ",failed);
           dsts.clear();srcs.clear();sizes.clear();
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
 SlotAsyncProducer(bool batch):batch_(batch),worker_([this]{run();}) {}
 ~SlotAsyncProducer() {
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
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 pybind11::class_<SlotAsyncProducer>(m,"AsyncProducer")
   .def(pybind11::init<bool>()).def("submit",&SlotAsyncProducer::submit)
   .def("submit_slots",&SlotAsyncProducer::submit_slots)
   .def("submit_slots_windowed",&SlotAsyncProducer::submit_slots_windowed)
   .def("admit_slots",&SlotAsyncProducer::admit_slots)
   .def("wait_submitted",&SlotAsyncProducer::wait_submitted,pybind11::call_guard<pybind11::gil_scoped_release>())
   .def("collect",&SlotAsyncProducer::collect,pybind11::call_guard<pybind11::gil_scoped_release>());
}
