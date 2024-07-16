#include "kernels.h"

sycl::event cpu_submitKernel(queue& q, sycl::buffer<ptype,2>& buf_a, sycl::buffer<ptype,2>& buf_b,
                       sycl::buffer<ptype,2>& buf_c, sycl::nd_range<1> size_range){
    sycl::event kern_ev = q.submit([&](handler &h) {
      auto a = buf_a.get_access<sycl::access::mode::read>(h);
      auto b = buf_b.get_access<sycl::access::mode::read>(h);
      auto c = buf_c.get_access<sycl::access::mode::discard_write>(h);

      h.parallel_for(size_range, [=](nd_item<1> item){
        c[item] = a[item] + b[item];
      });
    });
    return kern_ev;
}
