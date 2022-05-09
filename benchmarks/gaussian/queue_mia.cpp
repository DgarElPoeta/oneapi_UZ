#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>

class KernelGaussian;

sycl::event submitKernel(queue& q, sycl::buffer<uchar4, 1>& buf_input, sycl::buffer<float, 1>& buf_filterWeight,
                       sycl::buffer<uchar4, 1>& buf_blurred, sycl::nd_range<1> size_range, size_t offset,
                       int rows, int cols, int filterWidth){
    std::cout << "kernel erroneo" << std::endl;
    return q.submit([&](handler &h) {

      auto input = buf_input.get_access<sycl::access::mode::read>(h);
      auto filterWeight = buf_filterWeight.get_access<sycl::access::mode::read>(h);
      auto blurred = buf_blurred.get_access<sycl::access::mode::discard_write>(h);

      //#define PACKED 0
      // TODO: PACKED is worse than unpacked here

#define UNROLL_FACTOR 4

      h.parallel_for<KernelGaussian>(size_range, [=](nd_item<1> item) {
        size_t tid = item.get_global_id(0);

        int r = tid / cols; // current row
        int c = tid % cols; // current column

        int middle = filterWidth / 2;

        float4 blur{0.f};
        int width = cols - 1;
        int height = rows - 1;

        int i_start = sycl::max(r-middle, 0);
        int i_end = sycl::min(r+middle, height);

        int j_start = sycl::max(c-middle, 0);
        int j_offset = - sycl::min(c-middle, 0);
        int j_end = sycl::min(c+middle, width);

        int i_pesos =  - sycl::min(r-middle, 0);

        uchar4 local_input[UNROLL_FACTOR];
        float local_weight[UNROLL_FACTOR];
        for(int i = i_start; i <= i_end; i++, i_pesos++){
          int j_pesos = j_offset;
          int j = j_start;
          for(; (j+UNROLL_FACTOR) <= j_end;){
            #pragma unroll UNROLL_FACTOR
            for(int k = 0; k < UNROLL_FACTOR; k++, j++, j_pesos++){
              local_input[k] = input[i*cols + j];
              local_weight[k] = filterWeight[i_pesos*filterWidth + j_pesos];
            }
            for(int k = 0; k < UNROLL_FACTOR; k++){
                blur += local_input[k].convert<float>() * local_weight[k];
            }
          }
          for(; j <= j_end; j++, j_pesos++){
            float4 pixel = input[i*cols + j].convert<float>();
            float weight = filterWeight[i_pesos*filterWidth + j_pesos];
            blur += pixel * weight;
          }
        }

        blurred[tid] = (cl::sycl::round(blur)).convert<uchar>();
      });
    });
}
