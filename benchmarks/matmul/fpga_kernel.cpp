#include "kernels.h"
#include <sycl/ext/intel/fpga_extensions.hpp>
class KernelMatmulFPGA;

sycl::event fpga_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_a, sycl::buffer<ptype,2>& buf_b,
                       sycl::buffer<ptype,2>& buf_c, sycl::nd_range<2> size_range, uint64_t N){
    sycl::event kern_ev = q.submit([&](sycl::handler &h) {
      
      sycl::accessor a(buf_a, h, sycl::read_only);
      sycl::accessor b(buf_b, h, sycl::read_only);
      sycl::accessor c(buf_c, h, sycl::write_only, sycl::no_init);

      h.parallel_for<KernelMatmulFPGA>(size_range, [=](sycl::nd_item<2> item){
        size_t i = item.get_global_id(0);
        size_t j = item.get_global_id(1);
        ptype sum = 0;
        for(size_t k = 0 ; k < N; k++){
          ptype aik = a[{i,k}];
          ptype bkj = b[{k,j}];
          sum += aik * bkj;
        }

        c[{i,j}] = sum;
      });
    });
    
    return kern_ev;
}
