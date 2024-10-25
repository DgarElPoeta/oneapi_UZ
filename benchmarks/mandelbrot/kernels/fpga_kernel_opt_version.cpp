#include "kernels.h"

#include <sycl/ext/intel/fpga_extensions.hpp>

class KernelMandelFPGA;

constexpr uint32_t SIMD_WORK_ITEMS = 16;

sycl::event fpga_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,2>& buf_out, 
                              const sycl::nd_range<2> size_range, 
                              const uint64_t N, 
                              const size_t offset
                            )
{
  
  sycl::event kern_ev = q.submit([&](sycl::handler &h)
  {
    
    const sycl::accessor out(buf_out, h, sycl::write_only, sycl::no_init);

    h.parallel_for<KernelMandelFPGA>(size_range, [=](sycl::nd_item<2> item)
    [
      [
        sycl::reqd_work_group_size(1,WORK_GROUP_SIZE),
        intel::num_simd_work_items(SIMD_WORK_ITEMS),
        intel::kernel_args_restrict
      ]
    ]
    {

      // Get the row global id of work item inside the NDRange
      const size_t i = item.get_global_id(0);

      // Get the column global id of work item inside the NDRange
      const size_t j = item.get_global_id(1);
      
      // Get the row global id of work item inside the total work
      const uint64_t x = i + offset;

      // Get the column global id of work item inside the total work
      const uint64_t y = j;

      // Create complex number from the coordinates
      const ctype c = ctype(MINX + (x * (MAXX / N)), MINY + (y * (MAXY / N)));

      // Auxiliar complex number
      ctype z = 0;

      // Absolute value of z squared
      float abs_z_square = 0;

      // Number of iterations
      uint64_t iter;

      // Iterate until the maximum number of iterations or the series diverges
      for(iter = 0; iter < MAXITERATIONS && abs_z_square < 4.0f; ++iter){
        const float r = z.real();
        const float im = z.imag();
        abs_z_square = r*r + im*im;
        z = z*z + c;
      }

      // Store the number of iterations in the output buffer
      out[{i,j}] = iter-1;

    });

  });
  
  return kern_ev;

}
