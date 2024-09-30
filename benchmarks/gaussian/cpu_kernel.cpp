#include "kernels.h"

class KernelGaussianCPU;

sycl::event cpu_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_input, 
                             sycl::buffer<float,2>& buf_filter,sycl::buffer<ptype,2>& buf_blurred,
                             sycl::nd_range<2> size_range, uint64_t N,uint64_t offset){
    sycl::event kern_ev = q.submit([&](sycl::handler &h) {

      sycl::accessor input(buf_input, h, sycl::read_only);
      sycl::accessor filter(buf_filter, h, sycl::read_only);
      sycl::accessor blurred(buf_blurred, h, sycl::write_only, sycl::no_init);

      h.parallel_for<KernelGaussianCPU>(size_range, [=](sycl::nd_item<2> item){
        const size_t i = item.get_global_id(0);
        const size_t j = item.get_global_id(1);

        constexpr size_t middle = filterDim / 2;

        const size_t xI = i + offset;

        sycl::float3 blurredValue = sycl::float3{0.0};

        size_t xfI = 0;
        size_t yfI = 0;
        size_t xIter = 0;
        size_t yIter = 0;
        uint64_t Nmx1 = middle + xI + 1 - N;
        uint64_t Nmy1 = middle + j + 1 - N;

        if (xI < middle) xfI = middle - xI;
        else if (Nmx1 <= middle) xIter = Nmx1;
        if (j < middle) yfI = middle - j;
        else if (Nmy1 <= middle) yIter = Nmy1;

        

        for(size_t x = xfI; x + xIter < filterDim; x++){
          for(size_t y = yfI; y + yIter < filterDim; y++){
            size_t r = xI + x - middle;
            size_t c = j + y - middle; 
            ptype p = input[r][c];
            sycl::float3 fp = p.convert<float>();
            float w = filter[x][y];
            blurredValue += fp * w;
          }
        }
        
        ptype r;
        r.x() = (unsigned char) sycl::round(blurredValue.x());
        r.y() = (unsigned char) sycl::round(blurredValue.y());
        r.z() = (unsigned char) sycl::round(blurredValue.z());

        blurred[i][j] = r;
      });
    });
    
    return kern_ev;
}
