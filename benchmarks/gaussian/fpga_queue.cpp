#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>

class KernelGaussian;

queue& getQueue(){
#ifdef FPGA_EMULATOR
    static queue qFPGA(sycl::INTEL::fpga_emulator_selector{}, async_exception_handler);
#else
    static queue qFPGA(sycl::INTEL::fpga_selector{}, async_exception_handler);
#endif
    return qFPGA;
}

sycl::event fpga_submitKernel(queue& q, sycl::buffer<uchar4, 1>& buf_input, sycl::buffer<float, 1>& buf_filterWeight,
                       sycl::buffer<uchar4, 1>& buf_blurred, sycl::nd_range<1> size_range, size_t offset,
                       int rows, int cols, int filterWidth){
    q = getQueue();
    return q.submit([&](handler &h) {

auto input = buf_input.get_access<sycl::access::mode::read>(h);
auto filterWeight = buf_filterWeight.get_access<sycl::access::mode::read>(h);
auto blurred = buf_blurred.get_access<sycl::access::mode::discard_write>(h);

#define PACKED 0
#define FILTERWIDTH 5
#define MIDDLE 2
// TODO: PACKED is worse than unpacked here

h.parallel_for<KernelGaussian>(size_range, [=](nd_item<1> item) {
  size_t tid = item.get_global_id(0);

#if INPUT_BUFFER_SENT_ALL
  // input buffer needs the offset always from 0..i-1, i..N-1
  tid += offset;
#endif

  int r = tid / cols; // current row
  int c = tid % cols; // current column

  int middle = filterWidth / 2;

#if PACKED
float4 blur{0.f};
#else
  float blurX = 0.f; // will contained blurred value
  float blurY = 0.f; // will contained blurred value
  float blurZ = 0.f; // will contained blurred value
#endif
  int width = cols - 1;
  int height = rows - 1;

  #pragma unroll FILTERWIDTH
  for (int i = -MIDDLE; i <= MIDDLE; ++i) // rows
  {
    for (int j = -MIDDLE; j <= MIDDLE; ++j) // columns
    {

      int h = r + i;
      int w = c + j;
      if (h > height || h < 0 || w > width || w < 0) {
        continue;
      }

      int idx = w + cols * h; // current pixel index

#if PACKED
float4 pixel = input[idx].convert<float>();
#else
      float pixelX = (input[idx].x()); //s[0]);
      float pixelY = (input[idx].y()); //s[1]);
      float pixelZ = (input[idx].z()); //s[2]);
#endif

      idx = (i + MIDDLE) * FILTERWIDTH + j + MIDDLE;
      float weight = filterWeight[idx];

#if PACKED
blur += pixel * weight;
#else
      blurX += pixelX * weight;
      blurY += pixelY * weight;
      blurZ += pixelZ * weight;
#endif
    }
  }

#if INPUT_BUFFER_SENT_ALL
  // output buffer starts with 0 always (the range has the real address)  0..i-1
  tid -= offset;
#endif

#if PACKED
blurred[tid] = (cl::sycl::round(blur)).convert<uchar>();
#else
blurred[tid].x() = (unsigned char) cl::sycl::round(blurX);
blurred[tid].y() = (unsigned char) cl::sycl::round(blurY);
blurred[tid].z() = (unsigned char) cl::sycl::round(blurZ);
#endif
});

});
}
