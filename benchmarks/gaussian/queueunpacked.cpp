#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>

class KernelGaussian;

sycl::event submitKernel(queue& q, sycl::buffer<uchar4, 1>& buf_input, sycl::buffer<float, 1>& buf_filterWeight,
                       sycl::buffer<uchar4, 1>& buf_blurred, sycl::nd_range<1> size_range, size_t offset,
                       int rows, int cols, int filterWidth){
    return q.submit([&](handler &h) {

      auto input = buf_input.get_access<sycl::access::mode::read>(h);
      auto filterWeight = buf_filterWeight.get_access<sycl::access::mode::read>(h);
      auto blurred = buf_blurred.get_access<sycl::access::mode::discard_write>(h);

      //#define PACKED 0
      // TODO: PACKED is worse than unpacked here

#define UNROLL_FACTOR 5

      h.parallel_for<KernelGaussian>(size_range, [=](nd_item<1> item) {
        size_t tid = item.get_global_id(0);

        int r = tid / cols; // current row
        int c = tid % cols; // current column

        int middle = filterWidth / 2;

        float4 blur{0.f};

        float blurX = 0.f; // will contained blurred value
        float blurY = 0.f; // will contained blurred value
        float blurZ = 0.f; // will contained blurred value

        int width = cols - 1;
        int height = rows - 1;

        int i_start = sycl::max(r-middle, 0);
        int i_end = sycl::min(r+middle, height);

        int j_start = sycl::max(c-middle, 0);
        int j_offset = - sycl::min(c-middle, 0);
        int j_end = sycl::min(c+middle, width);

        int i_pesos =  - sycl::min(r-middle, 0);

        //uchar4 local_input[UNROLL_FACTOR];

        float local_pixelX[UNROLL_FACTOR];
        float local_pixelY[UNROLL_FACTOR];
        float local_pixelZ[UNROLL_FACTOR];

        float local_weight[UNROLL_FACTOR];
        for(int i = i_start; i <= i_end; i++, i_pesos++){
          int j_pesos = j_offset;
          int j = j_start;
          for(; (j+UNROLL_FACTOR) <= j_end;){
            #pragma unroll UNROLL_FACTOR
            for(int k = 0; k < UNROLL_FACTOR; k++, j++, j_pesos++){
              //local_input[k] = input[i*cols + j];
              int idx = i*cols + j;
              local_pixelX[k] = (input[idx].x()); //s[0]);
              local_pixelY[k] = (input[idx].y()); //s[1]);
              local_pixelZ[k] = (input[idx].z()); //s[2]);

              local_weight[k] = filterWeight[i_pesos*filterWidth + j_pesos];
            }
            #pragma unroll UNROLL_FACTOR
            for(int k = 0; k < UNROLL_FACTOR; k++){
                //blur += local_input[k].convert<float>() * local_weight[k];

                blurX += local_pixelX[k] * local_weight[k];
                blurY += local_pixelY[k] * local_weight[k];
                blurZ += local_pixelZ[k] * local_weight[k];
            }
          }
          for(; j <= j_end; j++, j_pesos++){
            int idx = i*cols + j;
            float weight = filterWeight[i_pesos*filterWidth + j_pesos];
            blurX += (input[idx].x()) * weight;
            blurY += (input[idx].y()) * weight;
            blurZ += (input[idx].z()) * weight;
          }
        }

        //blurred[tid] = (cl::sycl::round(blur)).convert<uchar>();

        blurred[tid].x() = (unsigned char) cl::sycl::round(blurX);
        blurred[tid].y() = (unsigned char) cl::sycl::round(blurY);
        blurred[tid].z() = (unsigned char) cl::sycl::round(blurZ);
      });
    });
}
