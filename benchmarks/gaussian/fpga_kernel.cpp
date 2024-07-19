#include "kernels.h"
#include <sycl/ext/intel/fpga_extensions.hpp>
class KernelGaussianFPGA;

sycl::event fpga_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_input, 
                              sycl::buffer<float,2>& buf_filter,sycl::buffer<ptype,2>& buf_blurred,
                              sycl::nd_range<2> size_range, uint64_t size,uint64_t offset){
    sycl::event kern_ev = q.submit([&](sycl::handler &h) {

      accesor input(buf_input, h, sycl::read_only);
      accesor filter(buf_filter, h, sycl::read_only);
      accesor blurred(buf_blurred, h, sycl::write_only);

      h.parallel_for<KernelGaussianFPGA>(size_range, [=](sycl::nd_item<2> item){
        size_t i = item.get_global_id(0);
        size_t j = item.get_global_id(1);

        const size_t middle = filterSize / 2;

        const size_t xI = i + offset;

        sycl::float3 blurredValue = sycl::float3{0,0,0};

        for(size_t x = 0; x < filterSize; x++){
          for(size_t y = 0; y < filterSize; y++){
            int64_t r = (int64_t) xI + x - middle;
            int64_t c = (int64_t) j + y - middle; 
            if(r < size && r > 0 && c < size && c > 0){
              ptype p = input[r][c];
              float w = filter[x][y];
              blurredValue += p * w;
            }
          }
        }

        blurred[i][j] = (sycl::round(blurredValue)).convert<uchar>();


      });
    });
    
    return kern_ev;
}
