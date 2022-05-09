//
// Created by radon on 17/09/20.
//

#include "binomial.h"
#include "binomial/queue.cpp"

using namespace std::chrono;

#if !defined(USM)
#error USM should be 0 or 1
#endif

//typedef sycl::cl_uchar4 cl_uchar4;

inline ostream &
operator<<(ostream &os, cl_uchar4 &t) {
  os << "(" << (int) t.s[0] << "," << (int) t.s[1] << "," << (int) t.s[2] << "," << (int) t.s[3] << ")";
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

float*
binomial_option(float* randArray, int samplesPerVectorWidth, int numSamples, int numSteps)
{
  float* stepsArray = (float*)malloc((numSteps + 1) * sizeof(float4));

  float* refOutput = (float*)malloc(samplesPerVectorWidth * sizeof(float4));

  // Iterate for all samples
  for (int bid = 0; bid < numSamples; ++bid) {
    float s[4];
    float x[4];
    float vsdt[4];
    float puByr[4];
    float pdByr[4];
    float optionYears[4];

    float inRand[4];

    for (int i = 0; i < 4; ++i) {
      inRand[i] = randArray[bid + i];
      s[i] = (1.0f - inRand[i]) * 5.0f + inRand[i] * 30.f;
      x[i] = (1.0f - inRand[i]) * 1.0f + inRand[i] * 100.f;
      optionYears[i] = (1.0f - inRand[i]) * 0.25f + inRand[i] * 10.f;
      float dt = optionYears[i] * (1.0f / (float)numSteps);
      vsdt[i] = VOLATILITY * sqrtf(dt);
      float rdt = RISKFREE * dt;
      float r = expf(rdt);
      float rInv = 1.0f / r;
      float u = expf(vsdt[i]);
      float d = 1.0f / u;
      float pu = (r - d) / (u - d);
      float pd = 1.0f - pu;
      puByr[i] = pu * rInv;
      pdByr[i] = pd * rInv;
    }
    /**
     * Compute values at expiration date:
     * Call option value at period end is v(t) = s(t) - x
     * If s(t) is greater than x, or zero otherwise...
     * The computation is similar for put options...
     */
    for (int j = 0; j <= numSteps; j++) {
      for (int i = 0; i < 4; ++i) {
        float profit = s[i] * expf(vsdt[i] * (2.0f * j - numSteps)) - x[i];
        stepsArray[j * 4 + i] = profit > 0.0f ? profit : 0.0f;
      }
    }

    /**
     * walk backwards up on the binomial tree of depth numSteps
     * Reduce the price step by step
     */
    for (int j = numSteps; j > 0; --j) {
      for (int k = 0; k <= j - 1; ++k) {
        for (int i = 0; i < 4; ++i) {
          int index_k = k * 4 + i;
          int index_k_1 = (k + 1) * 4 + i;
          stepsArray[index_k] = pdByr[i] * stepsArray[index_k_1] + puByr[i] * stepsArray[index_k];
        }
      }
    }

    // Copy the root to result
    refOutput[bid] = stepsArray[0];
  }

  free(stepsArray);

  return refOutput;
}

#define DEBUG_BIN 0

auto
verify(float* in_ptr,
               float* out_ptr,
               uint samplesPerVectorWidth,
               uint samples,
               uint steps,
               float threshold)
{
  float* res = binomial_option(in_ptr, samplesPerVectorWidth, samples, steps);

  if (DEBUG_BIN) {
    std::cout << "in:\n";
    for (uint i = 0; i < samples; ++i) {
      std::cout << in_ptr[i] << " ";
    }
    std::cout << "\n";
    std::cout << "out:\n";
    for (uint i = 0; i < samples; ++i) {
      std::cout << out_ptr[i] << " ";
    }
    std::cout << "\n";

    std::cout << "res:\n";
    for (uint i = 0; i < samples; ++i) {
      std::cout << res[i] << " ";
    }
    std::cout << "\n";
  }
  auto pos = -1;
  for (uint i = 0; i < samples; ++i) {
    auto diff = abs(res[i] - out_ptr[i]);
    if (std::isnan(out_ptr[i]) || diff >= threshold) {
      cout << "diff: " << i << " ver[i]: " << res[i] << " out[i]: " << out_ptr[i] << "\n";
      pos = i;
      break;
    // } else {
    //   cout << "[" << i << "] ver[i]: " << res[i] << " out[i]: " << out_ptr[i] << "\n";
    }
  }
  free(res);
  return pos == -1;
}

// bool verify(int size, int M, ptype *a, ptype *b, ptype *func) {
//   std::vector<ptype> b2(size, 0);
//   bool verification_passed = true;
//
//   bool debug = false;
//
//   for (size_t i = 0; i<size; ++i){
//     int tmp,j;
//     // int myid = get_global_id(0);
//     if(i <= M){
//       tmp = func[0];
//       for(j=0; j<=i; j++){
//         tmp = std::max(a[i-j] + func[j], tmp);
//         b2[i] = tmp;
//       }
//     }
//   }
//
//   // for (int i=0; i<size; ++i){
//   //   printf("[%d] = %d != %d\n", i, b[i], b2[i]);
//   // }
//
//   int show_errors = 5;
//   for (size_t i = 0; i<size; ++i){
//     const int kernel_value = b[i];
//     const int host_value = b2[i];
//     if (kernel_value != host_value) {
//       fprintf(stderr, "VERIFICATION FAILED for element %ld: %d != %d (kernel != host value)\n", i, kernel_value, host_value);
//       show_errors--;
//       verification_passed = false;
//       if (show_errors == 0){
//         break;
//       }
//     }
//   }
//   return verification_passed;
// }

int usage() {
  std::cout
      << "usage: <cpu|gpu|cpugpu> <static|dynamic> <num pkgs (dyn)|gpu proportion (st|hg)> <size> \n"
      << "DEBUG=y\n"
      << "CHECK=y\n"
      << "MIN_CHUNK_MULTIPLIER=1,1 (cpu,gpu)\n";
  return 1;
}

int main(int argc, char *argv[]) {

#if QUEUE_NDRANGE
  throw runtime_error("binomial: QUEUE_NDRANGE 1 is unsupported (fixed lws for this benchmark)");
#endif

  argc--;
  if (argc < 4) {
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
  if(argc >= 5) num_cpp_threads = atoi(argv[5]);

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

  Binomial binomial;
  uint samples = size;
  samples = (samples / 4) ? (samples / 4) * 4 : 4; // convierte a multiplo de 4
  uint steps = 254;
  size_t steps1 = steps + 1;

  if(steps1 % num_cpp_threads != 0) std::cout << "STEPS" << std::endl;

  size_t samplesPerVectorWidth = samples / 4;
  size_t gws = steps1 * samplesPerVectorWidth;
  size_t lws = steps1;
  size = gws;
  binomial.size = gws / steps1;
  binomial.workgroups = samplesPerVectorWidth;
  binomial.steps1 = steps1;
  binomial.steps = steps;
  std::vector<ptype> a;
  std::vector<ptype> b;
  ptype* a_ptr;
  ptype* b_ptr;
  a = std::vector<ptype>(binomial.workgroups);
  b = std::vector<ptype>(binomial.workgroups); // set 0 in loop
  a_ptr = a.data();
  b_ptr = b.data();
  binomial.a = a_ptr;
  binomial.b = b_ptr;

  srand(0);
  for (auto i = 0; i < binomial.workgroups; i++) {
    float f1 = (float)rand() / (float)RAND_MAX; // rand
    float f2 = (float)rand() / (float)RAND_MAX; // rand
    float f3 = (float)rand() / (float)RAND_MAX; // rand
    float f4 = (float)rand() / (float)RAND_MAX; // rand
    float4 f{f1, f2, f3, f4};
    a_ptr[i] = f;
    b_ptr[i] = float4{0.0f};
  }

  std::mutex m;
  size_t offset = 0;
  //  int pkg_size = 1024 * 100;
  int pkgid = 1;

  Options opts;
  opts.K = K;
  opts.usm = usm;
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
    std::cout << "N: " << binomial.size << " cpu prop: " << opts.cpu_prop << " min pkg size: " << opts.pkg_size <<
              " min_multiplier(hg: cpu,gpu) " << opts.min_multiplier_cpu << ","
              << opts.min_multiplier_gpu << "\n";
    std::cout << "samples: " << samples << " steps1: " << steps1 << " samplesPerVectorWidth: " << samplesPerVectorWidth << "\n";
  }

  int problem_size = binomial.size;
  *opts.p_rest_size = problem_size;
  // Removed in Binomial:
  // int lws = 128;
  // int gws = problem_size;

  // struct timeval start, end;
  // gettimeofday(&start, NULL);
  auto tLaunchStart = std::chrono::high_resolution_clock::now();
  opts.tLaunchStart = tLaunchStart;
  opts.p_problem = &binomial;
  // Important for binomial
  opts.pkg_size_multiple = lws; // dynamic
  opts.lws = lws; // static and hguided
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
  printf("  samples: %lu\n", size_param);
  printf("Runtime init timestamp: %.10s\n", std::to_string(opts.tStart.time_since_epoch().count()).c_str());
  printf("Kernel: binomial_options\n");
  printf("OpenCL gws: (%lu) lws: (%lu)\n", binomial.size, binomial.steps1);
  printf("Memory mode: %s\n", usm ? "usm" : "normal");

  if (mode == Mode::GPU || mode == Mode::CPUGPU) {
    printf("Device id: 0\n");
    {
      sycl::queue q(gpu_selector{});
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
  printf("chunks: %lu\n", opts.mChunksCPU.size() + opts.mChunksGPU.size());
  printf("duration offsets from init:\n");
  printf(" schedulerEnd: %d ms.\n", (int)std::round(diffLaunchEndMs));

  if (check) {
    if (verify((float*)a_ptr, (float*)b_ptr, binomial.workgroups, samples, steps, THRESHOLD)) {
      std::cout << "Success\n";
    } else {
      std::cout << "Failure\n";
    }
  }

  // printf("forcing the computation and read: %f...%f\n", b_ptr[0][0], b_ptr[binomial.workgroups - 1][0]);
  float* out = (float*)b_ptr;
  cout << "Output values: " << out[0] << "..." << out[samples - 1] << "\n";
  if (usm){
    sycl::free(a_ptr, opts.gpuQ);
    sycl::free(b_ptr, opts.gpuQ);
  }

  return 0;
}
