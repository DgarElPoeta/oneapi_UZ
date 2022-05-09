// you have to use `size` and `offset`
// does are part of N

Mandelbrot* mandelbrot = reinterpret_cast<Mandelbrot*>(opts->p_problem);
auto N = mandelbrot->size; // all problem, mandelbrot->size is different than `size`
auto R = sycl::range<1>(N);
auto Rsize = sycl::range<1>(size);
// printf("size: %d  size >> 2: %d  size / 4: %f\n", size, size >> 2, (float)size / 4.0);
auto R4 = sycl::range<1>(size >> 2); // func
if (debug) {
  std::cout << "Rw(opt2): (" << size << ")\n";
  std::cout << "R(opt1,func): (" << N << ")\n";
}
#if USM
auto mandelbrotImage = mandelbrot->out + offset;
#else
sycl::buffer<ptype, 1> buf_out((mandelbrot->out + offset), Rsize); // offset should be done inside
#endif
// sycl::buffer<ptype, 1> buf_a(mandelbrot->a, R);
// sycl::buffer<ptype, 1> buf_out((mandelbrot->out + offset), R); // offset should be done inside
// sycl::buffer<ptype, 1> buf_func((mandelbrot->func), R);

float leftxF = mandelbrot->leftxF;
float topyF = mandelbrot->topyF;
float xstepF = mandelbrot->xstepF;
float ystepF = mandelbrot->ystepF;
uint max_iterations = mandelbrot->max_iterations;
uint numDevices = mandelbrot->numDevices;
int bench = mandelbrot->bench;

int width = mandelbrot->width;

#if QUEUE_NDRANGE
sycl::range<1> range_lws(atoi(getenv("LWS")));
sycl::nd_range<1> size_range(R4, range_lws);
#endif

auto lws = opts->lws;