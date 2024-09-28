#include "kernels.h"

class KernelMataddCPU;

sycl::event cpu_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_a, sycl::buffer<ptype,2>& buf_b,
                       sycl::buffer<ptype,2>& buf_c, sycl::nd_range<2> size_range){
    sycl::event kern_ev = q.submit([&](sycl::handler &h) {
      
      sycl::accessor a(buf_a, h, sycl::read_only);
      sycl::accessor b(buf_b, h, sycl::read_only);
      sycl::accessor c(buf_c, h, sycl::write_only, sycl::no_init);

      h.parallel_for<KernelMataddCPU>(size_range, [=](sycl::nd_item<2> item){
        size_t i = item.get_global_id(0);
        size_t j = item.get_global_id(1);
        c[{i,j}] = a[{i,j}] + b[{i,j}];
      });
    });
    
    return kern_ev;
}
