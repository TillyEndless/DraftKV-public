// Native scheduler: GPU-built compact descriptors -> copy engine. No host token work.
#include <condition_variable>
#include <thread>
#include <mutex>
#include <deque>
#include <memory>
#include <chrono>
#include <c10/cuda/CUDAStream.h>
using T=at::Tensor;
int64_t route_workspace_bytes(int,int);
void pool_query_exact(T,T,T,T,int64_t,int64_t);
void initialize_route_kernels();
void gpu_route(T,T,T,T,T,std::vector<T>,T,T,T,T,T,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool);
void launch_route_rotate(T,T,T,T,cudaStream_t);
using Clock=std::chrono::steady_clock;
inline double ms(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::milli>(b-a).count();}
inline void write_value(cudaStream_t stream,int* ptr,int value){
 CUresult rc=cuStreamWriteValue32((CUstream)stream,(CUdeviceptr)ptr,value,CU_STREAM_WRITE_VALUE_DEFAULT);
 TORCH_CHECK(rc==CUDA_SUCCESS,"route stream write: ",int(rc));
}
struct RouteTicket {
 std::vector<T> tensors;
 // origin, descriptor-ready, fetch-begin/end, consumer-begin/end, rotation-done, join
 cudaEvent_t ev[8]{};
 int device,splits,history;bool chrono;std::string error;
 double wait_ms=0,submit_ms=0;int64_t copies=0,bytes=0,blocks=0,cropped=0;
 bool submitted=false;std::mutex mutex;std::condition_variable cv;
 RouteTicket(int d):device(d){for(auto& e:ev)C10_CUDA_CHECK(cudaEventCreate(&e));}
 ~RouteTicket(){cudaSetDevice(device);for(auto e:ev)cudaEventDestroy(e);}
 std::map<std::string,double> metrics(bool wait){
   std::unique_lock<std::mutex> lock(mutex);
   if(wait)cv.wait(lock,[&]{return submitted;});else if(!submitted)return {};
   TORCH_CHECK(error.empty(),error);
   auto status=cudaEventQuery(ev[7]);
   if(wait&&status==cudaErrorNotReady)status=cudaEventSynchronize(ev[7]);
   if(status==cudaErrorNotReady)return {};
   C10_CUDA_CHECK(status);
   auto elapsed=[&](int a,int b){float value;C10_CUDA_CHECK(cudaEventElapsedTime(&value,ev[a],ev[b]));return double(value);};
   return {{"descriptor_d2h_ms",elapsed(0,1)},{"producer_cpu_wait_ms",wait_ms},{"producer_cpu_submit_ms",submit_ms},
     {"h2d_ms",elapsed(2,3)},{"fetch_attention_ms",elapsed(2,5)},{"gpu_critical_ms",elapsed(0,7)},
     {"dma_count",double(copies)},{"h2d_bytes",double(bytes)},{"selected_history_blocks",double(blocks)},{"cropped_blocks",double(cropped)}};
 }
};
class RouteProducer {
 int device_;cudaStream_t fetch_,compute_;std::thread thread_;
 std::mutex mutex_;std::condition_variable cv_;std::deque<std::shared_ptr<RouteTicket>> queue_;bool stop_=false;
 void loop(){
   cudaSetDevice(device_);
   while(true){
     std::shared_ptr<RouteTicket> t;
     {std::unique_lock<std::mutex> lock(mutex_);cv_.wait(lock,[&]{return stop_||!queue_.empty();});
      if(stop_&&queue_.empty())break;t=queue_.front();queue_.pop_front();}
     auto wait_begin=Clock::now();
     auto& a=t->tensors;
     try{
       C10_CUDA_CHECK(cudaEventSynchronize(t->ev[1]));
       auto submit_begin=Clock::now();t->wait_ms=ms(wait_begin,submit_begin);
       int64_t* table=a[1].data_ptr<int64_t>();int64_t n=table[0];
       TORCH_CHECK(n>=0&&n<a[1].size(0),"invalid GPU DMA descriptor count");
       t->bytes=table[1];t->blocks=table[2];t->cropped=table[3];t->copies=n;
       C10_CUDA_CHECK(cudaStreamWaitEvent(fetch_,t->ev[1],0));C10_CUDA_CHECK(cudaEventRecord(t->ev[2],fetch_));
       auto enqueue=[&](int begin,int end){for(int i=begin;i<end;++i){
         int64_t* row=table+(i+1)*5;
         TORCH_CHECK(row[0]&&row[1]&&row[2]>0&&row[3]==0,"invalid paired DMA row");
         C10_CUDA_CHECK(cudaMemcpyAsync((void*)row[1],(void*)row[0],row[2],cudaMemcpyHostToDevice,fetch_));
         if(row[4]>=0)write_value(fetch_,a[7].data_ptr<int>()+row[3],row[4]);
       }};
       int first=0;while(first<n){++first;if(table[first*5+4]>=0)break;}
       enqueue(0,first);
       if(t->chrono){
         enqueue(first,n);C10_CUDA_CHECK(cudaEventRecord(t->ev[3],fetch_));
         launch_route_rotate(a[3],a[4],a[5],a[6],fetch_);
         C10_CUDA_CHECK(cudaEventRecord(t->ev[6],fetch_));C10_CUDA_CHECK(cudaStreamWaitEvent(compute_,t->ev[6],0));
       }else C10_CUDA_CHECK(cudaStreamWaitEvent(compute_,t->ev[1],0));
       C10_CUDA_CHECK(cudaEventRecord(t->ev[4],compute_));
       {c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(compute_,device_));
        prepared_fwd(a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],12,0,(t->history ? (t->history-1)/64 : -1),false,t->splits,a[10],a[11]);}
       C10_CUDA_CHECK(cudaEventRecord(t->ev[5],compute_));
       if(!t->chrono){enqueue(first,n);C10_CUDA_CHECK(cudaEventRecord(t->ev[3],fetch_));}
       // Main model stream waits on flags that were initialized BEFORE origin.
       // Event waiting on a not-yet-recorded worker event would be a race.
       write_value(fetch_,a[12].data_ptr<int>(),1);write_value(compute_,a[12].data_ptr<int>()+1,1);
       t->submit_ms=ms(submit_begin,Clock::now());

     }catch(const std::exception& e){
       t->error=e.what();
       // Release dependent streams, then report the original failure at collection.
       write_value(fetch_,a[7].data_ptr<int>(),0);
       write_value(fetch_,a[12].data_ptr<int>(),1);write_value(compute_,a[12].data_ptr<int>()+1,1);
     }
     {std::lock_guard<std::mutex> lock(t->mutex);t->submitted=true;}t->cv.notify_all();
   }
 }
public:
 RouteProducer(int device):device_(device){
   c10::cuda::CUDAGuard guard(device_);C10_CUDA_CHECK(cudaStreamCreateWithFlags(&fetch_,cudaStreamNonBlocking));
   C10_CUDA_CHECK(cudaStreamCreateWithFlags(&compute_,cudaStreamNonBlocking));thread_=std::thread([this]{loop();});
 }
 ~RouteProducer(){
   {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}cv_.notify_all();if(thread_.joinable())thread_.join();
   cudaSetDevice(device_);cudaStreamSynchronize(fetch_);cudaStreamSynchronize(compute_);cudaStreamDestroy(fetch_);cudaStreamDestroy(compute_);
 }
 std::shared_ptr<RouteTicket> submit(std::vector<T> tensors,int splits,int history,bool chrono){
   TORCH_CHECK(tensors.size()==13,"producer ABI");c10::cuda::CUDAGuard guard(device_);
   auto t=std::make_shared<RouteTicket>(device_);t->tensors=std::move(tensors);t->splits=splits;t->history=history;t->chrono=chrono;
   auto& a=t->tensors;auto current=at::cuda::getCurrentCUDAStream(device_).stream();
   TORCH_CHECK(a[0].is_cuda()&&a[1].is_pinned()&&a[0].sizes()==a[1].sizes(),"pinned descriptor ABI");
   C10_CUDA_CHECK(cudaEventRecord(t->ev[0],current));
   C10_CUDA_CHECK(cudaMemcpyAsync(a[1].data_ptr(),a[0].data_ptr(),a[0].nbytes(),cudaMemcpyDeviceToHost,current));
   C10_CUDA_CHECK(cudaEventRecord(t->ev[1],current));
   {std::lock_guard<std::mutex> lock(mutex_);queue_.push_back(t);}cv_.notify_one();
   for(int i=0;i<2;i++){
     CUresult rc=cuStreamWaitValue32((CUstream)current,(CUdeviceptr)(a[12].data_ptr<int>()+i),1,CU_STREAM_WAIT_VALUE_EQ);
     TORCH_CHECK(rc==CUDA_SUCCESS,"route completion wait: ",int(rc));
   }
   C10_CUDA_CHECK(cudaEventRecord(t->ev[7],current));return t;
 }
};
