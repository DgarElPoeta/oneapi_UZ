#include "kernels.h"

class KernelMataddCPU;

sycl::event cpu_submitKernel(queue& q, sycl::buffer<ptype,2>& buf_a, sycl::buffer<ptype,2>& buf_b,
                       sycl::buffer<ptype,2>& buf_c, sycl::nd_range<2> size_range){
    sycl::event kern_ev = q.submit([&](handler &h) {
      auto a = buf_a.get_access<sycl::access::mode::read>(h);
      auto b = buf_b.get_access<sycl::access::mode::read>(h);
      auto c = buf_c.get_access<sycl::access::mode::discard_write>(h);

      h.parallel_for<KernelMataddCPU>(size_range, [=](nd_item<2> item){
        auto i = item.get_global_id(0);
        auto j = item.get_global_id(1);
        c[{i,j}] = a[{i,j}] + b[{i,j}];
      });
    });
    return kern_ev;
}
