#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>

class KernelRAP;
queue& getQueue(){
    static queue qCPU(sycl::INTEL::host_selector{}, async_exception_handler);
    return qCPU;
}

sycl::event cpu_submitKernel(queue& q, sycl::buffer<ptype,1>& buf_a, sycl::buffer<ptype,1>& buf_b,
                      sycl::buffer<ptype,1>& buf_func, sycl::nd_range<1> size_range, size_t offset, size_t M, sycl::range<1> Rw){
    q = getQueue();
   return q.submit([&](handler &h) {

auto opt1 = buf_a.get_access<sycl::access::mode::read>(h);
auto opt2 = buf_b.get_access<sycl::access::mode::discard_write>(h);
auto func = buf_func.get_access<sycl::access::mode::read>(h);

#define UNROLL_FACTOR 8

h.parallel_for<KernelRAP>(size_range, [=](nd_item<1> item) {
  const auto idx = item.get_global_id(0);
  const auto id = idx + offset;
  int tmp, j;
  if (id <= M) {
    int tmpv[UNROLL_FACTOR];
    tmp = func[0];

    unsigned int i = 0;
    for(; (i + UNROLL_FACTOR)<=id;){
      #pragma unroll UNROLL_FACTOR
      for(j=0; j<UNROLL_FACTOR; ++i, ++j){
        tmpv[j] = opt1[id - i] + func[i];
      }
      for(j=0; j<UNROLL_FACTOR; ++j){
        tmp = sycl::max(tmpv[j], tmp);
      }
    }
    for(;i<=id; ++i){
      tmp = sycl::max(opt1[id - i] + func[i], tmp);
    }
    opt2[idx] = tmp; // without offset since the buffer has the offset
  }
});

});
}
