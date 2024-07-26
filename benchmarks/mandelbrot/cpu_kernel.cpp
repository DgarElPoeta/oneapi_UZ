#include "kernels.h"

class KernelMandelCPU;

sycl::event fpga_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_out, sycl::nd_range<2> size_range, 
                              const uint64_t N, const size_t offset){
    sycl::event kern_ev = q.submit([&](sycl::handler &h) {
      
      sycl::accessor out(buf_out, h, sycl::write_only);

      h.parallel_for<KernelMandelCPU>(size_range, [=](sycl::nd_item<2> item){
        const size_t i = item.get_global_id(0);
        const size_t j = item.get_global_id(1);
        const size_t x = i + offset, y = j;
        const ctype c = ctype(MINX + (x * (MAXX / N)), MINY + (y * (MAXY / N)));
        ctype z = 0;
        float abs_z_square = 0;
        for(uint64_t iter = 0; iter < MAXITERATIONS && abs_z_square < 4.0f; ++iter){
          const float r = z.real();
          const float im = z.imag();
          abs_z_square = r*r + im*im;
          z = z*z + c;
        }
        if(iter != MAXITERATIONS) iter--;
        out[{i,j}] = iter;
      });
    });
    
    return kern_ev;
}
