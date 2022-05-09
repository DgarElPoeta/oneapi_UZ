Mandelbrot* mandelbrot = reinterpret_cast<Mandelbrot*>(opts->p_problem);

//------------------------------------------------

//cl_uint* verificationOutput = reinterpret_cast<uint*>(mandelbrot->out);
uint* out_ptr = reinterpret_cast<uint*>(mandelbrot->out);
cl_float posx =  mandelbrot->leftxF;
cl_float posy = mandelbrot->topyF;
cl_float stepSizeX = mandelbrot->xstepF;
cl_float stepSizeY = mandelbrot->ystepF;
cl_int maxIterations = mandelbrot->max_iterations;
cl_int width = mandelbrot->width;
cl_int height = mandelbrot->width;
cl_int bench = mandelbrot->bench;

int tid;

for (tid = (offset >> 2); tid < ((size+offset) >> 2); tid++) {
  int i = tid  % (width / 4);
  int j = tid  / (width / 4);

  int4 veci = { 4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3 };
  int4 vecj = { j, j, j, j };

  float4 x0;
  x0.s0() = (float)(posx + stepSizeX * (float)veci.s0());
  x0.s1() = (float)(posx + stepSizeX * (float)veci.s1());
  x0.s2() = (float)(posx + stepSizeX * (float)veci.s2());
  x0.s3() = (float)(posx + stepSizeX * (float)veci.s3());
  float4 y0;
  y0.s0() = (float)(posy + stepSizeY * (float)vecj.s0());
  y0.s1() = (float)(posy + stepSizeY * (float)vecj.s1());
  y0.s2() = (float)(posy + stepSizeY * (float)vecj.s2());
  y0.s3() = (float)(posy + stepSizeY * (float)vecj.s3());

  float4 x = x0;
  float4 y = y0;

  cl_int iter = 0;
  float4 tmp;
  int4 stay;
  int4 ccount = { 0, 0, 0, 0 };

  stay.s0() = (x.s0() * x.s0() + y.s0() * y.s0()) <= 4.0f;
  stay.s1() = (x.s1() * x.s1() + y.s1() * y.s1()) <= 4.0f;
  stay.s2() = (x.s2() * x.s2() + y.s2() * y.s2()) <= 4.0f;
  stay.s3() = (x.s3() * x.s3() + y.s3() * y.s3()) <= 4.0f;
  float4 savx = x;
  float4 savy = y;

  for (iter = 0; (stay.s0() | stay.s1() | stay.s2() | stay.s3()) && (iter < maxIterations); iter += 16) {
    x = savx;
    y = savy;

    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;

    // Two iterations
    tmp = x * x + x0 - y * y;
    y = 2.0f * x * y + y0;
    x = tmp * tmp + x0 - y * y;
    y = 2.0f * tmp * y + y0;

    stay.s0() = (x.s0() * x.s0() + y.s0() * y.s0()) <= 4.0f;
    stay.s1() = (x.s1() * x.s1() + y.s1() * y.s1()) <= 4.0f;
    stay.s2() = (x.s2() * x.s2() + y.s2() * y.s2()) <= 4.0f;
    stay.s3() = (x.s3() * x.s3() + y.s3() * y.s3()) <= 4.0f;

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
  if (!(stay.s0() & stay.s1() & stay.s2() & stay.s3())) {
    iter = 16;
    do {
      x = savx;
      y = savy;
      stay.s0() = ((x.s0() * x.s0() + y.s0() * y.s0()) <= 4.0f) && (ccount.s0() < maxIterations);
      stay.s1() = ((x.s1() * x.s1() + y.s1() * y.s1()) <= 4.0f) && (ccount.s1() < maxIterations);
      stay.s2() = ((x.s2() * x.s2() + y.s2() * y.s2()) <= 4.0f) && (ccount.s2() < maxIterations);
      stay.s3() = ((x.s3() * x.s3() + y.s3() * y.s3()) <= 4.0f) && (ccount.s3() < maxIterations);
      tmp = x;
      x = x * x + x0 - y * y;
      y = 2.0f * tmp * y + y0;
      ccount += stay;
      iter--;
      savx.s0() = (stay.s0() ? x.s0() : savx.s0());
      savx.s1() = (stay.s1() ? x.s1() : savx.s1());
      savx.s2() = (stay.s2() ? x.s2() : savx.s2());
      savx.s3() = (stay.s3() ? x.s3() : savx.s3());
      savy.s0() = (stay.s0() ? y.s0() : savy.s0());
      savy.s1() = (stay.s1() ? y.s1() : savy.s1());
      savy.s2() = (stay.s2() ? y.s2() : savy.s2());
      savy.s3() = (stay.s3() ? y.s3() : savy.s3());
    } while ((stay.s0() | stay.s1() | stay.s2() | stay.s3()) && iter);
  }
  x = savx;
  y = savy;
  float4 fc = convert_float4(ccount);

  fc.s0() = (float)ccount.s0() + 1 - native_log2(native_log2(x.s0() * x.s0() + y.s0() * y.s0()));
  fc.s1() = (float)ccount.s1() + 1 - native_log2(native_log2(x.s1() * x.s1() + y.s1() * y.s1()));
  fc.s2() = (float)ccount.s2() + 1 - native_log2(native_log2(x.s2() * x.s2() + y.s2() * y.s2()));
  fc.s3() = (float)ccount.s3() + 1 - native_log2(native_log2(x.s3() * x.s3() + y.s3() * y.s3()));

  float c = fc.s0() * 2.0f * 3.1416f / 256.0f;
  u_uchar4 color[4];
  color[0].ch.s0 = (unsigned char)(((1.0f + native_cos(c)) * 0.5f) * 255);
  color[0].ch.s1 =
      (unsigned char)(((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255);
  color[0].ch.s2 = (unsigned char)(((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255);
  color[0].ch.s3 = 0xff;
  if (ccount.s0() == maxIterations) {
    color[0].ch.s0 = 0;
    color[0].ch.s1 = 0;
    color[0].ch.s2 = 0;
  }
  if (bench) {
    color[0].ch.s0 = ccount.s0() & 0xff;
    color[0].ch.s1 = (ccount.s0() & 0xff00) >> 8;
    color[0].ch.s2 = (ccount.s0() & 0xff0000) >> 16;
    color[0].ch.s3 = (ccount.s0() & 0xff000000) >> 24;
  }
  out_ptr[4 * tid] = color[0].num;

  c = fc.s1() * 2.0f * 3.1416f / 256.0f;
  color[1].ch.s0 = (unsigned char)(((1.0f + native_cos(c)) * 0.5f) * 255);
  color[1].ch.s1 =
      (unsigned char)(((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255);
  color[1].ch.s2 = (unsigned char)(((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255);
  color[1].ch.s3 = 0xff;
  if (ccount.s1() == maxIterations) {
    color[1].ch.s0 = 0;
    color[1].ch.s1 = 0;
    color[1].ch.s2 = 0;
  }
  if (bench) {
    color[1].ch.s0 = ccount.s1() & 0xff;
    color[1].ch.s1 = (ccount.s1() & 0xff00) >> 8;
    color[1].ch.s2 = (ccount.s1() & 0xff0000) >> 16;
    color[1].ch.s3 = (ccount.s1() & 0xff000000) >> 24;
  }
  out_ptr[4 * tid + 1] = color[1].num;

  c = fc.s2() * 2.0f * 3.1416f / 256.0f;
  color[2].ch.s0 = (unsigned char)(((1.0f + native_cos(c)) * 0.5f) * 255);
  color[2].ch.s1 =
      (unsigned char)(((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255);
  color[2].ch.s2 = (unsigned char)(((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255);
  color[2].ch.s3 = 0xff;
  if (ccount.s2() == maxIterations) {
    color[2].ch.s0 = 0;
    color[2].ch.s1 = 0;
    color[2].ch.s2 = 0;
  }
  if (bench) {
    color[2].ch.s0 = ccount.s2() & 0xff;
    color[2].ch.s1 = (ccount.s2() & 0xff00) >> 8;
    color[2].ch.s2 = (ccount.s2() & 0xff0000) >> 16;
    color[2].ch.s3 = (ccount.s2() & 0xff000000) >> 24;
  }
  out_ptr[4 * tid + 2] = color[2].num;

  c = fc.s3() * 2.0f * 3.1416f / 256.0f;
  color[3].ch.s0 = (unsigned char)(((1.0f + native_cos(c)) * 0.5f) * 255);
  color[3].ch.s1 =
      (unsigned char)(((1.0f + native_cos(2.0f * c + 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255);
  color[3].ch.s2 = (unsigned char)(((1.0f + native_cos(c - 2.0f * 3.1416f / 3.0f)) * 0.5f) * 255);
  color[3].ch.s3 = 0xff;
  if (ccount.s3() == maxIterations) {
    color[3].ch.s0 = 0;
    color[3].ch.s1 = 0;
    color[3].ch.s2 = 0;
  }
  if (bench) {
    color[3].ch.s0 = ccount.s3() & 0xff;
    color[3].ch.s1 = (ccount.s3() & 0xff00) >> 8;
    color[3].ch.s2 = (ccount.s3() & 0xff0000) >> 16;
    color[3].ch.s3 = (ccount.s3() & 0xff000000) >> 24;
  }
  out_ptr[4 * tid + 3] = color[3].num;
}
