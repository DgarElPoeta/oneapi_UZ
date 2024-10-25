#include "kernels.h"

#include <sycl/ext/intel/fpga_extensions.hpp>

class KernelMatmulFPGA;

sycl::event fpga_submitKernel(
                              sycl::queue& q,
                              sycl::buffer<ptype,2>& buf_a,
                              sycl::buffer<ptype,2>& buf_b,
                              sycl::buffer<ptype,2>& buf_c, 
                              const sycl::nd_range<2> size_range, 
                              const uint64_t N
                            )
{

  sycl::event kern_ev = q.submit([&](sycl::handler &h)
  {
    
    const sycl::accessor a(buf_a, h, sycl::read_only);
    const sycl::accessor b(buf_b, h, sycl::read_only);
    const sycl::accessor c(buf_c, h, sycl::write_only, sycl::no_init);

    h.parallel_for<KernelMatmulFPGA>(size_range, [=](sycl::nd_item<2> item)
    {

      // Get the row global id
      const size_t i = item.get_global_id(0);
      
      // Get the column global id
      const size_t j = item.get_global_id(1);

      // Variable to store the sum of the product of the elements
      ptype sum = 0;

      // Multiply the elements of the matrices
      for(size_t k = 0 ; k < N; k++){
        sum += a[{i,k}] * b[{k,j}];
      }
      
      // Store the sum in the result matrix
      c[{i,j}] = sum;

    });
  
  });

  return kern_ev;

}
