//
// Created by radon on 17/09/20.
//

#include "matmul.h"
#include "matmul/queue.cpp"

using namespace std::chrono;

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

int usage() {
  std::cout
      << "usage: <cpu|gpu|cpugpu> <static|dynamic> <num pkgs (dyn)|gpu proportion (st|hg)> <size side> \n"
      << "DEBUG=y\n"
      << "CHECK=y\n"
      << "MIN_CHUNK_MULTIPLIER=1,1 (cpu,gpu)\n";
  return 1;
}

void print_mat_cell(int v){
  printf("%4d ", v);
}
void print_mat_cell(float v){
  printf("%4f ", v);
}
void print_mat(std::string name, ptype *m, int N) {
  printf("%s:\n", name.c_str());
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      print_mat_cell(m[i * N + j]);
    }
    printf("\n");
  }
}

bool verify(int N, ptype *a, ptype *b, ptype *c) {
  std::vector<ptype> c2(N * N, 0);
  bool verification_passed = true;

  bool debug = false;

  #pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      int sum = 0;
      for (size_t k = 0; k < N; ++k) {
        sum += a[i * N + k] * b[k * N + j];
      }
      c2[i * N + j] = sum;
    }
  }

  if (debug) {
    print_mat("a", a, N);
    print_mat("b", b, N);
    print_mat("c", c, N);
    print_mat("c2", c2.data(), N);
  }

  int show_errs = 10;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      const int kernel_value = c[i * N + j];
      const int host_value = c2[i * N + j];
      if (kernel_value != host_value) {
        fprintf(stderr, "VERIFICATION FAILED for element %ld,%ld: %d != %d\n", i, j, kernel_value, host_value);
        verification_passed = false;
        show_errs --;
        if (show_errs == 0){
          return verification_passed;
        }
      }
    }
  }
  return verification_passed;
}

int main(int argc, char *argv[]) {
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

  // num cpp threads
  int num_cpp_threads = 1;
  if(argc >= 5) num_cpp_threads = atoi(argv[5]);

  // size arg
  const size_t N = atoi(argv[4]);

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

  Matmul matmul;
  matmul.size = N;
  std::vector<ptype> a;
  std::vector<ptype> b;
  std::vector<ptype> c;
  a = std::vector<ptype>(N * N);
  b = std::vector<ptype>(N * N);
  c = std::vector<ptype>(N * N);
  ptype* a_ptr;
  ptype* b_ptr;
  ptype* c_ptr;
  a_ptr = a.data();
  b_ptr = b.data();
  c_ptr = c.data();
  matmul.a = a_ptr;
  matmul.b = b_ptr;
  matmul.c = c_ptr;

  srand(0);
  auto nMax = 10;
  auto nMin = 0;
  for (auto i = 0; i < N*N; i++) {
    a_ptr[i] = rand() % ((nMax + 1) - nMin) + nMin;
    b_ptr[i] = rand() % ((nMax + 1) - nMin) + nMin;
    c_ptr[i] = 0;
  }

  std::mutex m;
  size_t size = N;
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
    std::cout << "N: " << N << " cpu prop: " << opts.cpu_prop << " min pkg size: " << opts.pkg_size <<
              " min_multiplier(hg: cpu,gpu) " << opts.min_multiplier_cpu << ","
              << opts.min_multiplier_gpu << "\n";
  }

  int imageWidth, imageHeight;
  imageWidth = imageHeight = size;

  int problem_size = matmul.size;
  *opts.p_rest_size = problem_size;
  int lws = 8; // remember, this is to the first dimension
  int gws = problem_size;

  auto tLaunchStart = std::chrono::high_resolution_clock::now();
  opts.tLaunchStart = tLaunchStart;
  opts.p_problem = &matmul;
  opts.lws = lws;
  opts.pkg_size_multiple = lws;
  opts.setup();

  //unsigned int hw_n = std::thread::hardware_concurrency();
  //std::cout << hw_n << " concurrent threads are supported.\n";
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
  printf("Program init timestamp: %.3f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(tStart.time_since_epoch()).count() / 1000.0));
  printf("Runtime init timestamp: %.3f\n", (std::chrono::duration_cast<std::chrono::milliseconds>(tLaunchStart.time_since_epoch()).count() / 1000.0));
  printf("Kernel: matmul\n");
  printf("OpenCL gws: (%lu,%lu) lws: ()\n", matmul.size, matmul.size);

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
  // printf("chunks: %d\n", *opts.p_pkgid - 1);
  printf("chunks: %lu\n", opts.mChunksCPU.size() + opts.mChunksGPU.size());
  printf("duration offsets from init:\n");
  printf(" schedulerEnd: %d ms.\n", (int)std::round(diffLaunchEndMs));

  if (check) {
    if (verify(N, a_ptr, b_ptr, c_ptr)) {
      std::cout << "Success\n";
    } else {
      std::cout << "Failure\n";
    }
  }

  cout << "Output values: " << c_ptr[0] << "..." << c_ptr[matmul.size - 1] << "\n";
  return 0;
}
