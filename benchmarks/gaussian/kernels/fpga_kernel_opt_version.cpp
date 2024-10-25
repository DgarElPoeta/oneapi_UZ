#include "kernels.h"

#include <sycl/ext/intel/fpga_extensions.hpp>

class KernelGaussianFPGA;

constexpr uint32_t SIMD_WORK_ITEMS = 16;

sycl::event fpga_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,2>& buf_input, 
                              sycl::buffer<ftype,2>& buf_filter,
                              sycl::buffer<ptype,2>& buf_blurred,
                              const sycl::nd_range<2> size_range, 
                              const uint64_t N, 
                              const uint64_t offset
                            )
{
  
  sycl::event kern_ev = q.submit([&](sycl::handler &h)
  {

    const sycl::accessor input(buf_input, h, sycl::read_only);
    const sycl::accessor filter(buf_filter, h, sycl::read_only);
    const sycl::accessor blurred(buf_blurred, h, sycl::write_only, sycl::no_init);

    h.parallel_for<KernelGaussianFPGA>(size_range, [=](sycl::nd_item<2> item)
    [
      [
        sycl::reqd_work_group_size(1,WORK_GROUP_SIZE),
        intel::num_simd_work_items(SIMD_WORK_ITEMS),
        intel::kernel_args_restrict
      ]
    ]
    {

      // Get the row global id of work-item inside the work package
      const size_t i = item.get_global_id(0);

      // Get the column global id of work-item inside the work package
      const size_t j = item.get_global_id(1);

      // Get the row global id of work inside the total work
      const size_t xI = i + offset;

      // Get the middle of the filter
      constexpr size_t middle = filterDim / 2;
      
      // Acumulation of the blurred values of each pixel
      sycl::float3 blurredValue = sycl::float3{0.0};

      // Acumulate the blurred values of the filter
      #pragma unroll filterDim
      for(size_t x = 0; x < filterDim; x++){
        for(size_t y = 0; y < filterDim; y++){

          // Get the row position of the input matrix
          const int64_t r = (int64_t) xI + (int64_t) x - (int64_t) middle;

          // Get the column position of the input matrix
          const int64_t c = (int64_t) j + (int64_t) y - (int64_t) middle;

          // If the position is out of the input matrix, continue
          if(r >= 0 && r < N && c >= 0 && c < N){

            const size_t ir = r;
            const size_t ic = c;

            // Get the value of the input matrix
            sycl::float3 fin = input[{ir,ic}].convert<float>();

            // Get the weight of the filter
            ftype weight  = filter[{x,y}];

            // Acumulate the blurred value
            blurredValue += fin * weight;

          }

        }

      }
      
      // Pass the blurred value(float3) to the result blurred value(uchar3)
      ptype b;
      b.x() = (unsigned char) sycl::round(blurredValue.x());
      b.y() = (unsigned char) sycl::round(blurredValue.y());
      b.z() = (unsigned char) sycl::round(blurredValue.z());

      // Store the blurred value in the blurred matrix
      blurred[{i,j}] = b;

    });

  });
  
  return kern_ev;

}