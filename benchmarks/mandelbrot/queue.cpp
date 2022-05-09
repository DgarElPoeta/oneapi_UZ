#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>

class KernelMandelbrot;

sycl::event submitKernel(queue& q, sycl::buffer<ptype,1>& buf_out, sycl::nd_range<1> size_range, size_t offset, float leftxF, float topyF,
                            float xstepF, float ystepF, uint max_iterations, uint numDevices, int bench, int width){
    return q.submit([&](handler &h) {

auto mandelbrotImage = buf_out.get_access<sycl::access::mode::discard_write>(h);

// TODO: mandelbrot should be with lws 256
// h.parallel_for(sycl::nd_range<1>(sycl::range<1>{size >> 2}, sycl::range<1>{lws}), [=](sycl::nd_item<1> ndItem) {
#if QUEUE_NDRANGE
h.parallel_for<class KernelMandelbrot>(size_range, [=](nd_item<1> it) {
  size_t tid = it.get_global_id(0);
#else // QUEUE_NDRANGE 0
h.parallel_for(R4, [=](item<1> it) {
  auto tid = it.get_id(0);
#endif // QUEUE_NDRANGE 0
    auto posx = leftxF;
    auto posy = topyF;
    auto stepSizeX = xstepF;
    auto stepSizeY = ystepF;
    auto maxIterations = max_iterations;

    // int tid = ndItem.get_global_id(0);
    // int tid = item.get_id(0);
//     int tid = item.get_linear_id(); // TODO: global?

// printf("tid %d\n", tid);
// static const CONSTANT char FMT[] = "[%d] r %d c %d idx %d\n";
// sycl::intel::experimental::printf(FMT, tid, r, c, idx2tid);

    int i = (tid + (offset >> 2)) % (width / 4);
    int j = (tid + (offset >> 2)) / (width / 4);
// static const CONSTANT char FMT[] = "[%d] x %d y %d\n";
// sycl::intel::experimental::printf(FMT, tid, i, j);
//     if (tid == 0){
//       if (cpu){
// static const CONSTANT char FMT[] = "CPU [%d] +off[%d] i %d j %d\n";
// sycl::intel::experimental::printf(FMT, tid, tid + (offset >> 2), i, j);
//       }else{
// static const CONSTANT char FMT[] = "GPU [%d] +off[%d] i %d j %d\n";
// sycl::intel::experimental::printf(FMT, tid, tid + (offset >> 2), i, j);
//       }
//     }

    int4 veci = {4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3};
    int4 vecj = {j, j, j, j};

    float4 x0;
    x0.s0() = (float) (posx + stepSizeX * (float) veci.s0());
    x0.s1() = (float) (posx + stepSizeX * (float) veci.s1());
    x0.s2() = (float) (posx + stepSizeX * (float) veci.s2());
    x0.s3() = (float) (posx + stepSizeX * (float) veci.s3());

    float4 y0;
    y0.s0() = (float) (posy + stepSizeY * (float) vecj.s0());
    y0.s1() = (float) (posy + stepSizeY * (float) vecj.s1());
    y0.s2() = (float) (posy + stepSizeY * (float) vecj.s2());
    y0.s3() = (float) (posy + stepSizeY * (float) vecj.s3());

    float4 x = x0;
    float4 y = y0;

    uint iter = 0;
    float4 tmp;
    int4 stay;
// int4 ccount = 0;
    int4 ccount{0, 0, 0, 0};

// uchar4 v{0, 100, 50, 100};
// uchar4 v{ (int)x0.s0(), (int)x0.s1(), (int)x0.s2(), (int)x0.s3() };
// uchar4 v{ (int)x0.s0(), 0, 0, (int)x0.s3() };
// uchar4 v{ 0, 0, 0, 0 };
// mandelbrotImage[tid] = v;
//
// uchar4 v1{ 255, 255, 255, 255 };
// mandelbrotImage[0] = v1;
//
// uchar4 v2{ 100, 100, 100, 100 };
// mandelbrotImage[4] = v2;
//
// uchar4 v3{ 100, 0, 120, 100 };
// mandelbrotImage[63] = v3;
// stay.s0() = (x.s0() * x.s0() + y.s0() * y.s0()) <= 4.0f;
// stay.s1() = (x.s1() * x.s1() + y.s1() * y.s1()) <= 4.0f;
// stay.s2() = (x.s2() * x.s2() + y.s2() * y.s2()) <= 4.0f;
// stay.s3() = (x.s3() * x.s3() + y.s3() * y.s3()) <= 4.0f;
    stay.s0() = (x.s0() * x.s0() + y.s0() * y.s0()) <= 4.0f;
    stay.s1() = (x.s1() * x.s1() + y.s1() * y.s1()) <= 4.0f;
    stay.s2() = (x.s2() * x.s2() + y.s2() * y.s2()) <= 4.0f;
    stay.s3() = (x.s3() * x.s3() + y.s3() * y.s3()) <= 4.0f;
    float4 savx = x;
    float4 savy = y;

// for (iter = 0; (stay.s0() | stay.s1() | stay.s2() | stay.s3()) && (iter < maxIterations); iter += 16)
    for (iter = 0; (stay.s0() | stay.s1() | stay.s2() | stay.s3()) && (iter < maxIterations); iter += 16) {
      x = savx;
      y = savy;

// Two iterations
      tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
      y = MUL_ADD(2.0f * x, y, y0);
      x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
      y = MUL_ADD(2.0f * tmp, y, y0);

// Two iterations
      tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
      y = MUL_ADD(2.0f * x, y, y0);
      x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
      y = MUL_ADD(2.0f * tmp, y, y0);

// Two iterations
      tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
      y = MUL_ADD(2.0f * x, y, y0);
      x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
      y = MUL_ADD(2.0f * tmp, y, y0);

// Two iterations
      tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
      y = MUL_ADD(2.0f * x, y, y0);
      x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
      y = MUL_ADD(2.0f * tmp, y, y0);

// Two iterations
      tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
      y = MUL_ADD(2.0f * x, y, y0);
      x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
      y = MUL_ADD(2.0f * tmp, y, y0);

// Two iterations
      tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
      y = MUL_ADD(2.0f * x, y, y0);
      x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
      y = MUL_ADD(2.0f * tmp, y, y0);

// Two iterations
      tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
      y = MUL_ADD(2.0f * x, y, y0);
      x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
      y = MUL_ADD(2.0f * tmp, y, y0);

// Two iterations
      tmp = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
      y = MUL_ADD(2.0f * x, y, y0);
      x = MUL_ADD(-y, y, MUL_ADD(tmp, tmp, x0));
      y = MUL_ADD(2.0f * tmp, y, y0);

// stay.s0() = (x.s0() * x.s0() + y.s0() * y.s0()) <= 4.0f;
// stay.s1() = (x.s1() * x.s1() + y.s1() * y.s1()) <= 4.0f;
// stay.s2() = (x.s2() * x.s2() + y.s2() * y.s2()) <= 4.0f;
// stay.s3() = (x.s3() * x.s3() + y.s3() * y.s3()) <= 4.0f;
      stay.s0() = (x.s0() * x.s0() + y.s0() * y.s0()) <= 4.0f;
      stay.s1() = (x.s1() * x.s1() + y.s1() * y.s1()) <= 4.0f;
      stay.s2() = (x.s2() * x.s2() + y.s2() * y.s2()) <= 4.0f;
      stay.s3() = (x.s3() * x.s3() + y.s3() * y.s3()) <= 4.0f;

// savx.s0() = (stay.s0() ? x.s0() : savx.s0());
// savx.s1() = (stay.s1() ? x.s1() : savx.s1());
// savx.s2() = (stay.s2() ? x.s2() : savx.s2());
// savx.s3() = (stay.s3() ? x.s3() : savx.s3());
// savy.s0() = (stay.s0() ? y.s0() : savy.s0());
// savy.s1() = (stay.s1() ? y.s1() : savy.s1());
// savy.s2() = (stay.s2() ? y.s2() : savy.s2());
// savy.s3() = (stay.s3() ? y.s3() : savy.s3());
      savx.s0() = (stay.s0() ? x.s0() : savx.s0());
      savx.s1() = (stay.s1() ? x.s1() : savx.s1());
      savx.s2() = (stay.s2() ? x.s2() : savx.s2());
      savx.s3() = (stay.s3() ? x.s3() : savx.s3());
      savy.s0() = (stay.s0() ? y.s0() : savy.s0());
      savy.s1() = (stay.s1() ? y.s1() : savy.s1());
      savy.s2() = (stay.s2() ? y.s2() : savy.s2());
      savy.s3() = (stay.s3() ? y.s3() : savy.s3());
      ccount += stay * 16;
    }
// Handle remainder
// if (!(stay.s0() & stay.s1() & stay.s2() & stay.s3()))
    if (!(stay.s0() & stay.s1() & stay.s2() & stay.s3())) {
      iter = 16;
      do {
        x = savx;
        y = savy;
// stay.s0() = ((x.s0() * x.s0() + y.s0() * y.s0()) <= 4.0f) && (ccount.s0() < maxIterations);
// stay.s1() = ((x.s1() * x.s1() + y.s1() * y.s1()) <= 4.0f) && (ccount.s1() < maxIterations);
// stay.s2() = ((x.s2() * x.s2() + y.s2() * y.s2()) <= 4.0f) && (ccount.s2() < maxIterations);
// stay.s3() = ((x.s3() * x.s3() + y.s3() * y.s3()) <= 4.0f) && (ccount.s3() < maxIterations);
        stay.s0() = ((x.s0() * x.s0() + y.s0() * y.s0()) <= 4.0f) && (ccount.s0() < maxIterations);
        stay.s1() = ((x.s1() * x.s1() + y.s1() * y.s1()) <= 4.0f) && (ccount.s1() < maxIterations);
        stay.s2() = ((x.s2() * x.s2() + y.s2() * y.s2()) <= 4.0f) && (ccount.s2() < maxIterations);
        stay.s3() = ((x.s3() * x.s3() + y.s3() * y.s3()) <= 4.0f) && (ccount.s3() < maxIterations);
        tmp = x;
        x = MUL_ADD(-y, y, MUL_ADD(x, x, x0));
        y = MUL_ADD(2.0f * tmp, y, y0);
        ccount += stay;
        iter--;
// savx.s0() = (stay.s0() ? x.s0() : savx.s0());
// savx.s1() = (stay.s1() ? x.s1() : savx.s1());
// savx.s2() = (stay.s2() ? x.s2() : savx.s2());
// savx.s3() = (stay.s3() ? x.s3() : savx.s3());
// savy.s0() = (stay.s0() ? y.s0() : savy.s0());
// savy.s1() = (stay.s1() ? y.s1() : savy.s1());
// savy.s2() = (stay.s2() ? y.s2() : savy.s2());
// savy.s3() = (stay.s3() ? y.s3() : savy.s3());
        savx.s0() = (stay.s0() ? x.s0() : savx.s0());
        savx.s1() = (stay.s1() ? x.s1() : savx.s1());
        savx.s2() = (stay.s2() ? x.s2() : savx.s2());
        savx.s3() = (stay.s3() ? x.s3() : savx.s3());
        savy.s0() = (stay.s0() ? y.s0() : savy.s0());
        savy.s1() = (stay.s1() ? y.s1() : savy.s1());
        savy.s2() = (stay.s2() ? y.s2() : savy.s2());
        savy.s3() = (stay.s3() ? y.s3() : savy.s3());
// while ((stay.s0() | stay.s1() | stay.s2() | stay.s3()) && iter);
      } while ((stay.s0() | stay.s1() | stay.s2() | stay.s3()) && iter);
    }
    x = savx;
    y = savy;

    float4 fc = convert_float4(ccount);
// if (i == 0 && j == 0){
//   static const CONSTANT char FMT[] = "[%d] %f %f %f %f\n";
//   sycl::intel::experimental::printf(FMT, tid, fc.s0(), fc.s1(), fc.s2(), fc.s3());
// }
    fc.s0() = (float) ccount.s0() + 1 - native_log2(native_log2(x.s0() * x.s0() + y.s0() * y.s0()));
    fc.s1() = (float) ccount.s1() + 1 - native_log2(native_log2(x.s1() * x.s1() + y.s1() * y.s1()));
    fc.s2() = (float) ccount.s2() + 1 - native_log2(native_log2(x.s2() * x.s2() + y.s2() * y.s2()));
    fc.s3() = (float) ccount.s3() + 1 - native_log2(native_log2(x.s3() * x.s3() + y.s3() * y.s3()));

    float c = fc.s0() * 2.0f * 3.1416f / 256.0f;
    uchar4 color[4];
    color[0].s0() = ((1.0f + native_cos(c)) * 0.5f) * 255;
    color[0].s1() = ((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[0].s2() = ((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[0].s3() = 0xff;
    if (ccount.s0() == maxIterations) {
      color[0].s0() = 0;
      color[0].s1() = 0;
      color[0].s2() = 0;
    }
    if (bench) {
      color[0].s0() = ccount.s0() & 0xff;
      color[0].s1() = (ccount.s0() & 0xff00) >> 8;
      color[0].s2() = (ccount.s0() & 0xff0000) >> 16;
      color[0].s3() = (ccount.s0() & 0xff000000) >> 24;
    }
// if (i == 0 && j == 0){
//   static const CONSTANT char FMT[] = "[%d] (%d,%d)\n";
//   sycl::intel::experimental::printf(FMT, tid, row, col);
// sycl::intel::experimental::printf(FMT, tid, color[0].s0(), color[0].s1(), color[0].s2(), color[0].s3());
// }
    mandelbrotImage[4 * tid + 0] = color[0];

    c = fc.s1() * 2.0f * 3.1416f / 256.0f;
    color[1].s0() = ((1.0f + native_cos(c)) * 0.5f) * 255;
    color[1].s1() = ((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[1].s2() = ((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[1].s3() = 0xff;
    if (ccount.s1() == maxIterations) {
      color[1].s0() = 0;
      color[1].s1() = 0;
      color[1].s2() = 0;
    }
    if (bench) {
      color[1].s0() = ccount.s1() & 0xff;
      color[1].s1() = (ccount.s1() & 0xff00) >> 8;
      color[1].s2() = (ccount.s1() & 0xff0000) >> 16;
      color[1].s3() = (ccount.s1() & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid + 1] = color[1];
    c = fc.s2() * 2.0f * 3.1416f / 256.0f;
    color[2].s0() = ((1.0f + native_cos(c)) * 0.5f) * 255;
    color[2].s1() = ((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[2].s2() = ((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[2].s3() = 0xff;
    if (ccount.s2() == maxIterations) {
      color[2].s0() = 0;
      color[2].s1() = 0;
      color[2].s2() = 0;
    }
    if (bench) {
      color[2].s0() = ccount.s2() & 0xff;
      color[2].s1() = (ccount.s2() & 0xff00) >> 8;
      color[2].s2() = (ccount.s2() & 0xff0000) >> 16;
      color[2].s3() = (ccount.s2() & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid + 2] = color[2];
    c = fc.s3() * 2.0f * 3.1416f / 256.0f;
    color[3].s0() = ((1.0f + native_cos(c)) * 0.5f) * 255;
    color[3].s1() = ((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[3].s2() = ((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255;
    color[3].s3() = 0xff;
    if (ccount.s3() == maxIterations) {
      color[3].s0() = 0;
      color[3].s1() = 0;
      color[3].s2() = 0;
    }
    if (bench) {
      color[3].s0() = ccount.s3() & 0xff;
      color[3].s1() = (ccount.s3() & 0xff00) >> 8;
      color[3].s2() = (ccount.s3() & 0xff0000) >> 16;
      color[3].s3() = (ccount.s3() & 0xff000000) >> 24;
    }
    mandelbrotImage[4 * tid + 3] = color[3];
  });

});
}
