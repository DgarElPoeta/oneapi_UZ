#include "kernels.h"

class KernelMatmulCPU;

sycl::event cpu_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_a, sycl::buffer<ptype,2>& buf_b,
                       sycl::buffer<ptype,2>& buf_c, sycl::nd_range<2> size_range, uint64_t N){
    sycl::event kern_ev = q.submit([&](sycl::handler &h) {
      
      sycl::accessor a(buf_a, h, sycl::read_only);
      sycl::accessor b(buf_b, h, sycl::read_only);
      sycl::accessor c(buf_c, h, sycl::write_only);

      h.parallel_for<KernelMatmulCPU>(size_range, [=](sycl::nd_item<2> item){
        size_t i = item.get_global_id(0);
        size_t j = item.get_global_id(1);
        ptype sum = 0;
        for(size_t k = 0 ; k < N; k++){
          sum += a[{i,k}] * b[{k,j}];
        }

        c[{i,j}] = sum;
      });
    });
    return kern_ev;
}
