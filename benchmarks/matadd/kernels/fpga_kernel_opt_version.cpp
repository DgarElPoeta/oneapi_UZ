#include "kernels.h"

#include <sycl/ext/intel/fpga_extensions.hpp>

class KernelMataddFPGA;

constexpr uint32_t SIMD_WORK_ITEMS = 16;

sycl::event fpga_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,2>& buf_a,
                              sycl::buffer<ptype,2>& buf_b, 
                              sycl::buffer<ptype,2>& buf_c,
                              const sycl::nd_range<2> size_range
                            )
{
  
  sycl::event kern_ev = q.submit([&](sycl::handler &h) 
  {
    
    const sycl::accessor a(buf_a, h, sycl::read_only);
    const sycl::accessor b(buf_b, h, sycl::read_only);
    const sycl::accessor c(buf_c, h, sycl::write_only, sycl::no_init);

    h.parallel_for<KernelMataddFPGA>(size_range, [=](sycl::nd_item<2> item)
    [
      [
        sycl::reqd_work_group_size(1,WORK_GROUP_SIZE),
        intel::num_simd_work_items(SIMD_WORK_ITEMS),
        intel::kernel_args_restrict
      ]
    ]
    {

      // Get the row global id
      const size_t i = item.get_global_id(0);

      // Get the column global id
      const size_t j = item.get_global_id(1);

      // Add the elements of the matrices
      c[{i,j}] = a[{i,j}] + b[{i,j}];

    });

  });
  
  return kern_ev;

}
