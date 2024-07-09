//
// Created by radon on 17/09/20.
//

#include "gaussian.h"
#include "gaussian/queue.cpp"

using namespace std::chrono;

using sycl::uchar4;
//typedef sycl::cl_uchar4 cl_uchar4;

inline ostream &
operator<<(ostream &os, uchar4 &t) {
  os << "(" << (int) t.x() << "," << (int) t.y() << "," << (int) t.z() << "," << (int) t.w() << ")";
  return os;
}

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
Gaussian::fill_image() {
  srand(0);

  int channels = 4;
  auto total = _total_size * channels;
  // #pragma omp parallel for num_threads(omp_get_max_threads())
  for (auto i = 0; i < total; i++) {
    int mod = i % channels;
    int value = rand() % 256;
    switch (mod) {
      case 0:_a[i / channels].x() = value;
        break;
      case 1:_a[i / channels].y() = value;
        break;
      case 2:_a[i / channels].z() = value;
        break;
      case 3:_a[i / channels].w() = 0;
        break;
    }
  }
}

void
Gaussian::fill_blurred(vector<uchar4> &blurred) {
  fill_blurred(blurred.data());
}
void
Gaussian::fill_blurred(uchar4* blurred) {
  int channels = 4;
  auto total = _total_size * channels;
#pragma omp parallel for num_threads(omp_get_max_threads())
  for (auto i = 0; i < total; i++) {
    int mod = i % channels;
    switch (mod) {
      case 0:blurred[i / channels][0] = 0;
        break;
      case 1:blurred[i / channels][1] = 0;
        break;
      case 2:blurred[i / channels][2] = 0;
        break;
      case 3:blurred[i / channels][3] = 0;
        break;
    }
  }
}

void
Gaussian::fill_filter() {
  const float sigma = 2.f;

  const int half = _filter_width / 2;
  float sum = 0.f;

  int r;
  for (r = -half; r <= half; ++r) {
    int c;
    for (c = -half; c <= half; ++c) {
      float weight = expf(-(float) (c * c + r * r) / (2.0f * sigma * sigma));
      int idx = (r + half) * _filter_width + c + half;
      _b[idx] = weight;
      sum += weight;
    }
  }

  float normal = 1.0f / sum;

  for (r = -half; r <= half; ++r) {
    int c;
    for (c = -half; c <= half; ++c) {
      int idx = (r + half) * _filter_width + c + half;
      _b[idx] *= normal;
    }
  }
}

/*
void
Gaussian::omp_gaussian_blur() {
  int rows = _height;
  int cols = _width;
  int filterWidth = _filter_width;
//#pragma GCC diagnostic ignored "-Wignored-attributes"

  uchar4* input = _a;
  float* filterWeight = _b;
  uchar4* blurred = _c;
//#pragma GCC diagnostic pop

  int total_size = _total_size;

  // auto num_threads = omp_get_max_threads();
  auto num_threads = 8;
  auto part = total_size / num_threads;

  // ANNOTATE_SITE_BEGIN(Site1);

#pragma omp parallel for num_threads(num_threads) schedule(static, part)
  for (int i = 0; i < total_size; i++) {
    // ANNOTATE_ITERATION_TASK(Task1);
    // int tid = get_global_id(0);
    int tid = i;

    if (tid < total_size) {
      int r = tid / cols; // current row
      int c = tid % cols; // current column

      int middle = filterWidth / 2;
      float blurX = 0.0f; // will contained blurred value
      float blurY = 0.0f; // will contained blurred value
      float blurZ = 0.0f; // will contained blurred value
      int width = cols - 1;
      int height = rows - 1;

      for (int i = -middle; i <= middle; ++i) // rows
      {
        // #pragma omp simd
        for (int j = -middle; j <= middle; ++j) // columns
        {
          // Clamp filter to the image border
          // int h=min(max(r+i, 0), height);
          // int w=min(max(c+j, 0), width);

          int h = r + i;
          int w = c + j;
          if (h > height || h < 0 || w > width || w < 0) {
            continue;
          }

          // Blur is a product of current pixel value and weight of that pixel.
          // Remember that sum of all weights equals to 1, so we are averaging
          // sum of all pixels by their weight.
          int idx = w + cols * h; // current pixel index
          float pixelX = input[idx].x();
          float pixelY = input[idx].y();
          float pixelZ = input[idx].z();

          idx = (i + middle) * filterWidth + j + middle;
          float weight = filterWeight[idx];

          blurX += pixelX * weight;
          blurY += pixelY * weight;
          blurZ += pixelZ * weight;
        }
      }
      blurred[tid].x() = (unsigned char) round(blurX);
      blurred[tid].y() = (unsigned char) round(blurY);
      blurred[tid].z() = (unsigned char) round(blurZ);
    }
  } // omp

  // ANNOTATE_SITE_END();
}
*/
bool
Gaussian::compare_gaussian_blur(float threshold) {
  int rows = _height;
  int cols = _width;
  int filterWidth = _filter_width;
//#pragma GCC diagnostic ignored "-Wignored-attributes"
  uchar4* input = _a;
  float* filterWeight = _b;
  uchar4* blurred = _c;
//#pragma GCC diagnostic pop

  int total_size = _total_size;

  auto num_threads = omp_get_max_threads();
  auto ok = true;
  vector<bool> oks(num_threads, true);

#pragma omp parallel num_threads(num_threads)
  {
    bool showable = true;
#pragma omp for reduction(& : ok)
    for (int i = 0; i < total_size; i++) {
      // int tid = get_global_id(0);
      int tid = i;

      if (tid < total_size) {
        int r = tid / cols; // current row
        int c = tid % cols; // current column

        int middle = filterWidth / 2;
        float blurX = 0.0f; // will contained blurred value
        float blurY = 0.0f; // will contained blurred value
        float blurZ = 0.0f; // will contained blurred value
        int width = cols - 1;
        int height = rows - 1;

        for (int i = -middle; i <= middle; ++i) // rows
        {
          for (int j = -middle; j <= middle; ++j) // columns
          {
            // Clamp filter to the image border
            // int h=min(max(r+i, 0), height);
            // int w=min(max(c+j, 0), width);

            int h = r + i;
            int w = c + j;
            if (h > height || h < 0 || w > width || w < 0) {
              continue;
            }

            // Blur is a product of current pixel value and weight of that
            // pixel. Remember that sum of all weights equals to 1, so we are
            // averaging sum of all pixels by their weight.
            int idx = w + cols * h; // current pixel index
            //std::cout << "Pixel:" << idx << ": " << h << ", " << w << std::endl;
            float pixelX = input[idx].x();
            float pixelY = input[idx].y();
            float pixelZ = input[idx].z();

            idx = (i + middle) * filterWidth + j + middle;
            //std::cout << "Peso:" << idx << ": " << (i + middle) << ", " << j + middle << std::endl;
            float weight = filterWeight[idx];

            blurX += pixelX * weight;
            blurY += pixelY * weight;
            blurZ += pixelZ * weight;
          }
        }

        auto diffX = abs(blurX - (float) blurred[tid].x());
        auto diffY = abs(blurY - (float) blurred[tid].y());
        auto diffZ = abs(blurZ - (float) blurred[tid].z());

        if (diffX >= threshold || diffY >= threshold || diffZ >= threshold) {
          if (showable) {
#pragma omp critical
            {
              cout << "i: " << tid << " blurred: " << blurred[tid] << " float calculated: (" << blurX
                   << "," << blurY << "," << blurZ << ")\n";
              cout << "   " << tid + 1 << " blurred[+1]: " << blurred[tid + 1] << "\n";
              cout << "   " << tid + 2 << " blurred[+2]: " << blurred[tid + 2] << "\n";
            }
            showable = false;
            ok = ok & false;
          }
        }
      }
    }
  } // omp
  return ok;
}

int usage() {
  std::cout
      << "usage: <cpu|gpu|cpugpu> <static|dynamic> <num pkgs (dyn)|gpu proportion (st|hg)> <size> <filter size>\n"
      << "DEBUG=y\n"
      << "CHECK=y\n"
      << "MIN_CHUNK_MULTIPLIER=1,1 (cpu,gpu)\n";
  return 1;
}

int main(int argc, char *argv[]) {
  argc--;
  if (argc < 5){
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

  // num cpp threads
  int num_cpp_threads = 1;
  if(argc >= 6) num_cpp_threads = atoi(argv[6]);

  // filter arg
  const int filterSize = atoi(argv[5]);
  if (filterSize % 2 == 0) {
    std::cout << "filterSize should be an odd number\n";
    return 1;
  }

  // size arg
  const size_t N = atoi(argv[4]);
  size_t size = N;

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

  int imageWidth, imageHeight;
  imageWidth = imageHeight = size;

  Gaussian gaussian(imageWidth, imageHeight, filterSize);
  std::vector<uchar4> a;
  std::vector<float> b;
  std::vector<uchar4> c;
  uchar4* a_ptr;
  float* b_ptr;
  uchar4* c_ptr;
  a_ptr = (uchar4*)malloc(gaussian._total_size * sizeof(uchar4));
  b_ptr = (float*)malloc(gaussian._filter_total_size * sizeof(float));
  c_ptr = (uchar4*)malloc(gaussian._total_size * sizeof(uchar4));
  gaussian.set_buffers(a_ptr, b_ptr, c_ptr);
  gaussian.build();

  std::mutex m;
  size_t offset = 0;
  int pkgid = 1;

  Options opts;
  opts.K = K;
  opts.usm = false;
  opts.debug = debug;
  opts.p_total_size = size;
  opts.p_rest_size = &size;
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

  double cpu_end = 0.0, gpu_end = 0.0, compute_cpu = 0.0, compute_gpu = 0.0, profiling_q_cpu = 0.0,
      profiling_q_gpu = 0.0;
  opts.cpu_end = &cpu_end;
  opts.gpu_end = &gpu_end;
  opts.compute_cpu = &compute_cpu;
  opts.compute_gpu = &compute_gpu;
  opts.profiling_q_cpu = &profiling_q_cpu;
  opts.profiling_q_gpu = &profiling_q_gpu;

  if (debug) {
    std::cout << "N: " << N << " cpu prop: " << opts.cpu_prop << " min pkg size: " << opts.pkg_size << " filter size: "
              << filterSize << " min_multiplier(hg: cpu,gpu) " << opts.min_multiplier_cpu << ","
              << opts.min_multiplier_gpu << "\n";
  }
  int problem_size = gaussian._total_size;
  opts.p_total_size = problem_size;
  *opts.p_rest_size = problem_size;
  int lws = 128;
  int gws = problem_size;

  auto tLaunchStart = std::chrono::high_resolution_clock::now();
  opts.tLaunchStart = tLaunchStart;
  opts.p_problem = &gaussian;
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

  printf("More Options:\n");
  printf("  size: %lu\n", N);
  printf("  filter size: %d\n", filterSize);
  printf("Program init timestamp: %.3f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(tStart.time_since_epoch()).count() / 1000.0));
  printf("Runtime init timestamp: %.3f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(tLaunchStart.time_since_epoch()).count() / 1000.0));
  printf("Kernel: gaussian_blur\n");
  printf("OpenCL gws: (%lu) lws: ()\n", gaussian._total_size);

  if (mode == Mode::GPU || mode == Mode::CPUGPU) {
    printf("Device id: 0\n");
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
  printf("chunks: %lu\n", opts.mChunksCPU.size() + opts.mChunksGPU.size());
  printf("duration offsets from init:\n");
  printf(" schedulerEnd: %d ms.\n", (int)std::round(diffLaunchEndMs));

  if (check) {
    if (gaussian.compare_gaussian_blur(THRESHOLD)){
      std::cout << "Success\n";
    } else {
      std::cout << "Failure\n";
    }
  }

  char *image_str = getenv("IMAGE");
  bool image = (image_str != NULL && std::string(image_str) == "y");
  if (image){
    write_bmp_file((uchar4*)a_ptr, gaussian._width, gaussian._width, "in.bmp");
    write_bmp_file((uchar4*)c_ptr, gaussian._width, gaussian._width, "out.bmp");
  }

  cout << "Output values: " << c_ptr[0] << "..." << c_ptr[gaussian._total_size - 1] << "\n";

  free(a_ptr);
  free(b_ptr);
  free(c_ptr);
  return 0;
}
