//
// Created by radon on 17/09/20.
//

#include "mandelbrot.h"
#include "mandelbrot/queue.cpp"

using namespace std::chrono;

using std::ostream;

inline ostream&
operator<<(ostream& os, uchar4& t)
{
  os << "(" << (int)t[0] << "," << (int)t[1] << "," << (int)t[2] << "," << (int)t[3] << ")";
  return os;
}

//typedef sycl::cl_uchar4 cl_uchar4;

#include "schedulers/st.cpp"
#include "schedulers/dyn.cpp"
#include "schedulers/hg.cpp"

void process(bool cpu, Options* opts, int thr_id) {
  if (opts->algo == Algo::Dynamic) {
    process_dynamic(cpu, opts);
  } else if (opts->algo == Algo::HGuided) {
    process_hguided(cpu, opts);
  } else {
    process_static(cpu, opts, thr_id);
  }
}


void
transform_image(uchar4* out, int width, int height)
{
  for (auto i = 0; i < height; ++i) {
    for (auto j = 0; j < width; ++j) {
      int cur_pixel = i * width + j;
      // pixel = cl_char4
      // r = s[0] g = s[1] b = s[2]
      Pixel p = out[cur_pixel];
      if (p[0] > 250 && p[1] < 100 && p[2] < 100) {
        p[0] = 0xff;
        p[1] = 0xff;
        p[2] = 0xff;
        out[cur_pixel] = p;
      }
    }
  }
}

void
mandelbrotRefFloat(cl_uint* verificationOutput,
                   cl_float posx,
                   cl_float posy,
                   cl_float stepSizeX,
                   cl_float stepSizeY,
                   cl_int maxIterations,
                   cl_int width,
                   cl_int height,
                   cl_int bench)
{
  int tid;

  for (tid = 0; tid < (height * width / 4); tid++) {
    int i = tid % (width / 4);
    int j = tid / (width / 4);

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
    verificationOutput[4 * tid] = color[0].num;

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
    verificationOutput[4 * tid + 1] = color[1].num;

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
    verificationOutput[4 * tid + 2] = color[2].num;

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
    verificationOutput[4 * tid + 3] = color[3].num;
  }
}

auto
check_mandelbrot(uchar4* out,
                 float leftxF,
                 float topyF,
                 float xstepF,
                 float ystepF,
                 uint max_iterations,
                 int width,
                 int height,
                 int bench,
                 float threshold)
{

  int max = 10;
  // std::cout << "out:\n";
  // for (uint i = 0; i < max; ++i) {
  //   std::cout << out[i] << " ";
  // }
  // std::cout << "\n";

  auto size_bytes = width * height * sizeof(uchar4);
  uint* res = (uint*)malloc(size_bytes);

  uint* out_ptr = reinterpret_cast<uint*>(out);
  mandelbrotRefFloat(res, leftxF, topyF, xstepF, ystepF, max_iterations, width, height, bench);

  cl_uchar4* res_ptr = reinterpret_cast<cl_uchar4*>(res);

  // std::cout << "res:\n";
  // for (uint i = 0; i < max; ++i) {
  //   std::cout << res_ptr[i] << " ";
  // }
  // std::cout << "\n";

  int i, j;
  int counter = 0;

  auto pos = -1;

  for (j = 0; j < height; j++) {
    for (i = 0; i < width; i++) {
      u_uchar4 temp_ver, temp_out;
      temp_ver.num = res[j * width + i];
      temp_out.num = out_ptr[j * width + i];

      unsigned char threshold = 2;

      if (((temp_ver.ch.s0 - temp_out.ch.s0) > threshold) ||
          ((temp_out.ch.s0 - temp_ver.ch.s0) > threshold) ||

          ((temp_ver.ch.s1 - temp_out.ch.s1) > threshold) ||
          ((temp_out.ch.s1 - temp_ver.ch.s1) > threshold) ||

          ((temp_ver.ch.s2 - temp_out.ch.s2) > threshold) ||
          ((temp_out.ch.s2 - temp_ver.ch.s2) > threshold) ||

          ((temp_ver.ch.s3 - temp_out.ch.s3) > threshold) ||
          ((temp_out.ch.s3 - temp_ver.ch.s3) > threshold)) {
        if (pos == -1) { // capture the first error
          pos = i + j * width;
          printf("error en %d\n", pos);
          printf("%d, %d, %d, %d : %d, %d, %d, %d\n", temp_ver.ch.s0,temp_ver.ch.s1,temp_ver.ch.s2,temp_ver.ch.s3,temp_out.ch.s0,temp_out.ch.s1,temp_out.ch.s2,temp_out.ch.s3);
        }
        counter++;
      }
    }
  }

  int numPixels = height * width;
  double ratio = (double)counter / numPixels;

  // if( ratio > threshold){
  //   pos =
  // }
  return pos == -1;
}

auto
verify(Mandelbrot* mandelbrot, float threshold){
  return check_mandelbrot(mandelbrot->out,
                          mandelbrot->leftxF,
                          mandelbrot->topyF,
                          mandelbrot->xstepF,
                          mandelbrot->ystepF,
                          mandelbrot->max_iterations,
                          mandelbrot->width,
                          mandelbrot->width,
                          mandelbrot->bench,
                          threshold);
}

int usage() {
  std::cout
      << "usage: <cpu|gpu|cpugpu> <static|dynamic> <num pkgs (dyn)|gpu proportion (st|hg)> <side size> <max iterations>\n"
      << "DEBUG=y\n"
      << "CHECK=y\n"
      << "MIN_CHUNK_MULTIPLIER=1,1 (cpu,gpu)\n";
  return 1;
}

int main(int argc, char *argv[]) {
  argc--;
  if (argc < 5) {
    return usage();
  }

  std::chrono::high_resolution_clock::time_point tStart = std::chrono::high_resolution_clock::now();
  std::string type;
  std::string n;
  Mode mode = Mode::CPU;
  bool check = false;
  float cpu_prop = 0.0;
  int pkg_size = 0; // old dyn
  int num_pkgs = 0; // dyn
  bool usm = false;

  // num cpp threads
  int num_cpp_threads = 1;
  if(argc >= 6) num_cpp_threads = atoi(argv[6]);

  // max iterations
  auto max_iterations = atoi(argv[5]);

  // size arg
  size_t size = atoi(argv[4]);
  size_t size_param = size;

  // gpu proportion arg
  cpu_prop = 1 - atof(argv[3]);

  // scheduler arg
  Algo algo;
  std::string algo_str = argv[2];
  if (algo_str == "static") {
    algo = Algo::Static;
  }
  else if (algo_str == "dynamic") {
    algo = Algo::Dynamic;
    num_pkgs = atof(argv[3]);
  }
  else if (algo_str == "hguided") {
    algo = Algo::HGuided;
  }
  else {
    return usage();
  }

  // devices arg
  std::string mode_str = argv[1];
  if (mode_str == "cpugpu") {
    mode = Mode::CPUGPU;
  }
  else if (mode_str == "cpu") {
    mode = Mode::CPU;
    if (algo == Algo::Static) {
      cpu_prop = 1.0;
    }
  }
  else if (mode_str == "gpu") {
    mode = Mode::GPU;
    if (algo == Algo::Static) {
      cpu_prop = 0.0;
    }
  }
  else {
    return usage();
  }

  // env variables
  char *debug_str = getenv("DEBUG");
  bool debug = (debug_str != NULL && std::string(debug_str) == "y");

  char *check_str = getenv("CHECK");
  check = (check_str != NULL && std::string(check_str) == "y");

  char *K_str = getenv("HGUIDED_K");
  float K = 2.0;
  if (K_str != nullptr) {
    float K_ = std::stof(K_str);
    //if (K_ > 1.0 && K_ < 10) {
      K = K_;
    //}
  }

  Mandelbrot mandelbrot;
  Options opts;

  mandelbrot.setup(size, max_iterations);
  std::vector<ptype> out;
  ptype* out_ptr;
  out = std::vector<ptype>(mandelbrot.size);
  out_ptr = out.data();
  mandelbrot.out = out_ptr;
  std::mutex m;
  size = mandelbrot.size;
  size_t offset = 0;
  //  int pkg_size = 1024 * 100;
  int pkgid = 1;

  // Options opts;
  opts.K = K;
  opts.usm = usm;
  opts.debug = debug;
  opts.p_total_size = size;
  opts.p_rest_size = &size;
  //  opts.pkg_size = pkg_size;
  opts.num_pkgs = num_pkgs;
  opts.cpu_prop = cpu_prop;
  opts.p_offset = &offset;
  opts.p_pkgid = &pkgid;
  opts.m = &m;
  opts.ptr1 = nullptr;
  opts.tStart = tStart;
  opts.algo = algo;
  opts.mode = mode;
  opts.num_cpp_threads = num_cpp_threads;

  int min_multiplier[2] = {1, 1};
  std::string multiplier("");
  char *val = getenv("MIN_CHUNK_MULTIPLIER");
  if (val != nullptr) {
    multiplier = std::string(val);
    std::stringstream ss(multiplier);
    std::string item;
    auto i = 0;
    while (std::getline(ss, item, ',')) {
      min_multiplier[i] = std::stoi(item);
      i++;
    }
  }
  opts.min_multiplier_gpu = min_multiplier[0];
  opts.min_multiplier_cpu = min_multiplier[1];

  // std::mutex m_profiling_acc;
  // opts.m_profiling_acc = &m_profiling_acc;
  double cpu_end = 0.0, gpu_end = 0.0, compute_cpu = 0.0, compute_gpu = 0.0, profiling_q_cpu = 0.0,
      profiling_q_gpu = 0.0;
  opts.cpu_end = &cpu_end;
  opts.gpu_end = &gpu_end;
  opts.compute_cpu = &compute_cpu;
  opts.compute_gpu = &compute_gpu;
  opts.profiling_q_cpu = &profiling_q_cpu;
  opts.profiling_q_gpu = &profiling_q_gpu;

  if (debug) {
    std::cout << "N: " << mandelbrot.size << " cpu prop: " << opts.cpu_prop << " min pkg size: " << opts.pkg_size_multiple <<
              " min_multiplier(hg: cpu,gpu) " << opts.min_multiplier_cpu << ","
              << opts.min_multiplier_gpu << "\n";
    std::cout << "width: " << mandelbrot.width << "\n";
  }

  // Matmul
//  char* filter_size_str = getenv("FILTER_SIZE");
//  if (filter_size_str == NULL){
//    std::cout << "needs FILTER_SIZE\n";
//    return usage();
//  }
//  int filterSize = atoi(filter_size_str);
  int imageWidth, imageHeight;
  imageWidth = imageHeight = size;

  int problem_size = mandelbrot.size;
  *opts.p_rest_size = problem_size;
  int lws = 256;
  int gws = problem_size >> 2;

  // struct timeval start, end;
  // gettimeofday(&start, NULL);
  auto tLaunchStart = std::chrono::high_resolution_clock::now();
  opts.tLaunchStart = tLaunchStart;
  opts.p_problem = &mandelbrot;
  opts.lws = lws;
  opts.pkg_size_multiple = lws;
  opts.setup();

  if (mode == Mode::CPU) {
    std::vector<std::thread> vecOfThreads;
    for(int i=1; i<num_cpp_threads; i++){
      vecOfThreads.push_back(std::thread(process, true, &opts, i));
    }
    tLaunchStart = std::chrono::high_resolution_clock::now();
    opts.tLaunchStart = tLaunchStart;
    process(true, &opts, 0);
    for (std::thread & th : vecOfThreads){
      if (th.joinable()) th.join();
    }
  } else if (mode == Mode::GPU) {
    process(false, &opts, 0);
  } else {
    std::vector<std::thread> vecOfThreads;
    for(int i=0; i<num_cpp_threads; i++){
      vecOfThreads.push_back(std::thread(process, true, &opts, i));
    }
    tLaunchStart = std::chrono::high_resolution_clock::now();
    opts.tLaunchStart = tLaunchStart;
    process(false, &opts, 0);
    for (std::thread & th : vecOfThreads){
      if (th.joinable()) th.join();
    }
  }

  auto tLaunchEnd = std::chrono::high_resolution_clock::now();
  auto diffLaunchEnd = (tLaunchEnd - tLaunchStart).count();
  auto diffLaunchEndMs = diffLaunchEnd / 1e6;

  // gettimeofday(&end, NULL);
  // double time_taken;
  // time_taken = (end.tv_sec - start.tv_sec) * 1e6;
  // time_taken = (time_taken + (end.tv_usec - start.tv_usec)) * 1e-6;
  // cout << "Time taken by queue is : " << std::fixed << time_taken << std::setprecision(6) << " sec " << "\n";
  // std::cout << "Sample values on GPU and CPU\n";
  printf("More Options:\n");
  printf("  size: %lu\n", size_param);
  printf("  max_iterations: %d\n", max_iterations);
  // printf("Runtime init timestamp: %.10s\n", std::to_string(opts.tStart.time_since_epoch().count()).c_str());
  printf("Program init timestamp: %.3f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(tStart.time_since_epoch()).count() / 1000.0));
  printf("Runtime init timestamp: %.3f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(tLaunchStart.time_since_epoch()).count() / 1000.0));
  printf("Kernel: mandelbrot\n");
  printf("OpenCL gws: (%lu) lws: ()\n", mandelbrot.size);
  printf("Memory mode: %s\n", usm ? "usm" : "normal");

  if (mode == Mode::GPU || mode == Mode::CPUGPU) {
    printf("Device id: 0\n");
    {
      sycl::queue q(sycl::accelerator_selector{});
      printf("Selected device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    }
    cout << "works: " << opts.mChunksGPU.size() << " works_size: " << opts.worksizeGPU << "\n";
    printf("duration increments:\n");
    printf(" completeKernel: %d ms. (and Read)\n", (int)std::round(*opts.profiling_q_gpu));
    printf(" total: %d ms.\n", (int)std::round(*opts.compute_gpu));
    printf("duration offsets from init:\n");
    printf(" deviceEnd: %d ms.\n", (int)std::round(*opts.gpu_end));

    cout << "chunks (mOffset+mSize:ts_ms+duration_ms+duration_read_ms)";
    cout << "type-chunks,";
    for (auto chunk : opts.mChunksGPU) {
      cout << chunk.offset << "+" << chunk.size << ":" << chunk.ts_ms << "+" << chunk.duration_ms
           << "+" << chunk.duration_read_ms << ",";
    }
    cout << "\n";
  }
  if (mode == Mode::CPU || mode == Mode::CPUGPU) {
    printf("Device id: 1\n");
    {
      sycl::queue q(cpu_selector{});
      printf("Selected device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    }
    cout << "works: " << opts.mChunksCPU.size() << " works_size: " << opts.worksizeCPU << "\n";
    printf("duration increments:\n");
    printf(" completeKernel: %d ms. (and Read)\n", (int)std::round(*opts.profiling_q_cpu));
    printf(" total: %d ms.\n", (int)std::round(*opts.compute_cpu));
    printf("duration offsets from init:\n");
    printf(" deviceEnd: %d ms.\n", (int)std::round(*opts.cpu_end));

    cout << "chunks (mOffset+mSize:ts_ms+duration_ms+duration_read_ms)";
    cout << "type-chunks,";
    for (auto chunk : opts.mChunksCPU) {
      cout << chunk.offset << "+" << chunk.size << ":" << chunk.ts_ms << "+" << chunk.duration_ms
           << "+" << chunk.duration_read_ms << ",";
    }
    cout << "\n";
  }

  printf("scheduler: ");
  if (algo == Algo::Static){
    printf("Static\n");
  } else if (algo == Algo::Dynamic) {
    printf("Dynamic\n");
  } else if (algo == Algo::HGuided) {
    printf("HGuided\n");
    printf("scheduler parameters:\n");
    printf(" K: %.1f\n", opts.K);
    printf(" minChunkMultiplier (gpu,cpu): %d,%d\n", opts.min_multiplier_gpu, opts.min_multiplier_cpu);
  }
  // printf("chunks: %d\n", *opts.p_pkgid - 1);
  printf("chunks: %lu\n", opts.mChunksCPU.size() + opts.mChunksGPU.size());
  printf("duration offsets from init:\n");
  printf(" schedulerEnd: %d ms.\n", (int)std::round(diffLaunchEndMs));

  if (check) {
    if (verify(&mandelbrot, THRESHOLD)) {
      std::cout << "Success\n";
    } else {
      std::cout << "Failure\n";
    }
  }

  char *image_str = getenv("IMAGE");
  bool image = (image_str != NULL && std::string(image_str) == "y");
  if (image){
    transform_image((uchar4*)mandelbrot.out, mandelbrot.width, mandelbrot.width);
    write_bmp_file((Pixel*)mandelbrot.out, mandelbrot.width, mandelbrot.width, "mandelbrot.bmp");
  }

  // printf("forcing the computation and read: %d...%d\n", out_ptr[0][0], out_ptr[mandelbrot.size - 1][0]);
  cout << "Output values: " << out_ptr[0] << "..." << out_ptr[size - 1] << "\n";
  if (usm){
    sycl::free(out_ptr, opts.gpuQ);
  }

  return 0;
}
