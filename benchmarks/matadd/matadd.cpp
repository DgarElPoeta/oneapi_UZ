//
// Created by radon on 17/09/20.
//

#include "matadd.h"
#include "benchmarks.h"

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

template <typename T>
void process(bool cpu, Options<T>& opts, uint32_t thr_id) {
  
  if (opts.algo == Algo::Dynamic) {
    process_dynamic<T>(cpu, opts,thr_id);
  } else if (opts.algo == Algo::HGuided) {
    process_hguided<T>(cpu, opts,thr_id);
  } else {
    process_static<T>(cpu, opts, thr_id);
  }
}

int usage() {
  std::cout
      << "usage: <cpu|gpu|fpgaemu|fpgahw|cpu_gpu|cpu_fpgaemu|cpu_fpgahw> <static|dynamic|hguided> <num pkgs (dyn)|cpu proportion (st|hg)> <side size> [num_cpp_threads]\n"
      << "DEBUG=y\n"
      << "CHECK=y\n"
      << "MIN_PKG_MULTIPLIER=1,1 (cpu,acc)\n";
  return 1;
}

void print_mat(std::string name, std::vector<ptype>& m, uint64_t N) {
  printf("%s:\n", name.c_str());
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      printf("%f ",m[i * N + j]);
    }
    printf("\n");
  }
}

bool verify(uint64_t N, std::vector<ptype>& a, std::vector<ptype>& b, std::vector<ptype>& c) {
  std::vector<ptype> c2(N * N, 0);
  bool verification_passed = true;

  constexpr float threshold = 0.00001;

  #pragma omp parallel for
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      c2[i * N + j] = a[i * N + j] + b[i * N + j];
    }
  }

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      const auto kernel_value = c[i * N + j];
      const auto host_value = c2[i * N + j];
      const auto difference = (kernel_value >= host_value) ? kernel_value - host_value : host_value - kernel_value;
      if (difference > threshold) {
        fprintf(stderr, "VERIFICATION FAILED for element %ld,%ld: |%f-%f| = %f > %f = threshold\n", i, j, kernel_value, host_value, difference, threshold);
        verification_passed = false;
        break;
      }
    }
    if (!verification_passed) { break; }
  }
  return verification_passed;
}

int main(int argc, char *argv[]) {

  // Initial timepoint of program
  std::chrono::high_resolution_clock::time_point tpStart = std::chrono::high_resolution_clock::now();

  // -------------------------------------------------------------------------------------------------
  // Arguments comprobation and initialization

  argc--;
  if (argc < 4) {
    return usage();
  }


  // Heterogeneous execution mode arg
  std::string mode_str = argv[1]; // string with the mode
  Mode mode; // Heterogeneous execution mode
  if (!hashMode(mode_str, mode)) {
    return usage();
  }


  // Scheduler algorithm arg
  std::string algo_str = argv[2]; // string with the algorithm
  Algo algo; // Scheduler algorithm
  if (!hashAlgo(algo_str, algo)) {
    return usage();
  }

  
  // Proportion args
  float cpu_prop; // Work proportion assigned to CPU. Used in Static and HGuided Algorithms
  size_t num_pkgs; // Number of packages in which the total work is divided in Dynamic Algorithm
  if (algo == Algo::Dynamic) {
    num_pkgs = atof(argv[3]);
  } else {
    if (mode == Mode::CPU_GPU || mode == Mode::CPU_FPGA) {
      cpu_prop = atof(argv[3]);
    }
    else if(mode == Mode::CPU){
      cpu_prop = 1.0;
    }
    else{
      cpu_prop = 0.0;
    }
  }


  // Problem dimension arg
  const uint64_t N = atoi(argv[4]); // Problem dimension


  // Number of cpp threads arg
  uint32_t num_cpp_threads = 1; // Number of cpp threads
  if (argc >= 5) {
    num_cpp_threads = atoi(argv[5]);
  }

  // -------------------------------------------------------------------------------------------------


  // -------------------------------------------------------------------------------------------------
  // Environment variables

  // DEBUG environment variable
  char *debug_str = getenv("DEBUG"); // string with the debug value
  bool debug = (debug_str != NULL && std::string(debug_str) == "y"); // Debug mode activated(true) or deactivated(false)

  // CHECK environment variable
  char *check_str = getenv("CHECK"); // string with the check value
  bool check = (check_str != NULL && std::string(check_str) == "y"); // Result verification activated(true) or deactivated(false)

  // Processing capabilities of CPU and Accelerator. Used in HGuided Algorithm
  uint32_t min_multiplier[2] = {1, 1};

  // MIN_CHUNK_MULTIPLIER environment variable
  char *multipliers_str = getenv("MIN_CHUNK_MULTIPLIER");
  std::string multiplier("");
  if (multipliers_str != nullptr) {
    multiplier = std::string(multipliers_str);
    std::stringstream ss(multiplier);
    std::string item;
    auto i = 0;
    while (std::getline(ss, item, ',')) {
      min_multiplier[i] = std::stoi(item);
      i++;
    }
  }

  // K environment variable
  char *K_str = getenv("HGUIDED_K");
  float K = 2.0;
  if (K_str != nullptr) {
    float K_ = std::stof(K_str);
      K = K_;
  }

  // -------------------------------------------------------------------------------------------------


  
  // -------------------------------------------------------------------------------------------------
  // Initialization of Options data type

  Options<Matadd> opts;

  opts.mode = mode;
  opts.algo = algo;
  opts.debug = debug;
  opts.usm = false;
  opts.cpuProp = cpu_prop;
  opts.numPkgs = num_pkgs;
  opts.K = K;

  opts.minMultiplierCPU = min_multiplier[0];
  opts.minMultiplierAcc = min_multiplier[1];

  opts.wPkgsCPU = vector<WorkPackages>();
  opts.wPkgsAcc = vector<WorkPackages>();

  opts.workSizeCPU = 0;
  opts.workSizeAcc = 0;

  opts.accDeviceDesc = "";
  opts.cpuDeviceDesc = "";
  
  opts.numCppThreads = num_cpp_threads;

  opts.pTotalSize = N;

  opts.pWork = 0;
  opts.pPkg = 0;

  opts.tpStart = tpStart;

  opts.tCPUEnd = 0;
  opts.tAccEnd = 0;

  opts.tComputeKernelCPU = 0;
  opts.tComputeKernelAcc = 0;
  opts.tSubmitKernelCPU = 0;
  opts.tSubmitKernelAcc= 0;

  // Initialization of Matadd data type of Opts
  opts.pData = Matadd();
  opts.pData.size = N;
  opts.pData.a = std::vector<ptype>(N*N);
  opts.pData.b = std::vector<ptype>(N*N);
  opts.pData.c = std::vector<ptype>(N*N,0.0);


  // -------------------------------------------------------------------------------------------------
  

  // -------------------------------------------------------------------------------------------------
  // Initialization of elements in matrices in Matadd data type of Opts with random values

  srand(0);
  auto nMax = 10;
  auto nMin = 0;
  for (auto i = 0; i < N*N; i++) {
    opts.pData.a[i] = rand() % ((nMax + 1) - nMin) + nMin;
    opts.pData.b[i] = rand() % ((nMax + 1) - nMin) + nMin;
  }

  // -------------------------------------------------------------------------------------------------


  // -------------------------------------------------------------------------------------------------
  // Load scheduler processes

  auto timePoint = std::chrono::high_resolution_clock::now();
  opts.tpSchedulerStart = timePoint;
  if (mode == Mode::CPU) {

    // We reserve the first timepoint for the CPU actual thread
    timePoint = std::chrono::high_resolution_clock::now();
    opts.tpCPUStart.push_back(timePoint);

    // Thread vector
    std::vector<std::thread> vecOfThreads;

    // Creation of CPU threads
    for(size_t i=1; i<num_cpp_threads; i++){
      
      // Push start time of every thread created
      timePoint = std::chrono::high_resolution_clock::now();
      opts.tpCPUStart.push_back(timePoint);

      // Create thread with CPU scheduler process
      vecOfThreads.push_back(std::thread(process<Matadd>, true, std::ref(opts), i));
    }

    // Start time of CPU actual thread
    timePoint = std::chrono::high_resolution_clock::now();
    opts.tpCPUStart[0] = timePoint;

    // CPU scheduler process
    process<Matadd>(true, opts, 0);

    // Join of all CPU threads
    for (std::thread & th : vecOfThreads){
      if (th.joinable()) th.join();
    }

  } else if (mode == Mode::GPU || mode == Mode::FPGA) {
    
    // Start time of accelerator scheduler process
    timePoint = std::chrono::high_resolution_clock::now();
    opts.tpAccStart = timePoint;

    // Accelerator scheduler process
    process<Matadd>(false, opts, 0);

  } else {
    
    // Thread vector
    std::vector<std::thread> vecOfThreads;

    // Creation of CPU threads
    for(size_t i=0; i<num_cpp_threads; i++){

      // Push start time of every thread created
      timePoint = std::chrono::high_resolution_clock::now();
      opts.tpCPUStart.push_back(timePoint);

      // Create thread with CPU scheduler process
      vecOfThreads.push_back(std::thread(process<Matadd>, true, std::ref(opts), i));
    }

    // Start time of accelerator scheduler process
    timePoint = std::chrono::high_resolution_clock::now();
    opts.tpAccStart = timePoint;
    process<Matadd>(false, opts, 0);

    // Join of all CPU threads
    for (std::thread & th : vecOfThreads){
      if (th.joinable()) th.join();
    }
  }

  // -------------------------------------------------------------------------------------------------


  auto tpSchedulerEnd = std::chrono::high_resolution_clock::now();
  auto diffScheduler = (tpSchedulerEnd - opts.tpSchedulerStart).count();
  auto tScheduler = diffScheduler / 1e9;
  
  // Execution summary
  std::cout << "\n\n\n";
  std::cout << "---------------------------------------------------------------------------------\n";
  std::cout << "Execution summary\n";
  std::cout << "\n\n";

  // Type of benchamrk
  std::cout << "Benchmark: matadd\n";
  std::cout << "Matrices size: " << N << "," << N << "\n";
  std::cout << "Problem size: " << N << ". (an entire row is considered the work item)\n";
  std::cout << "\n\n";

  // Type of scheduler
  std::cout << "Scheduler: ";
  if (algo == Algo::Static){
    std::cout << "Static\n";

  } else if (algo == Algo::Dynamic) {
    std::cout << "Dynamic\n";
  } else if (algo == Algo::HGuided) {
    std::cout << "HGuided\n";
    std::cout << "scheduler parameters:\n";
    std::cout << " K: " << opts.K << "\n";
    std::cout << " minChunkMultiplier (gpu,cpu): (" << opts.minMultiplierAcc << "," << opts.minMultiplierCPU << ")\n";
  }
  std::cout << "\n\n";

  // Heterogeneus execution mode
  std::cout << "Mode: ";
  switch(mode){
    case Mode::CPU:
      std::cout << "CPU\n";
      break;
    case Mode::GPU:
      std::cout << "GPU\n";
      break;
    case Mode::FPGA:
#ifdef FPGA_EMULATOR
      std::cout << ("FPGA Emulator\n");
#else
      std::cout << "FPGA\n";
#endif
      break;
    case Mode::CPU_GPU:
      std::cout << "CPU + GPU\n";
      break;
    case Mode::CPU_FPGA:
#ifdef FPGA_EMULATOR
      std::cout << ("CPU + FPGA Emulator\n");
#else
      std::cout << "CPU + FPGA\n";
#endif
      break;
  }
  std::cout << "\n\n";

  
  std::cout << "Time spent on load scheduler: " << tScheduler << " s\n";
  std::cout << "Total work packages: " << opts.wPkgsCPU.size() + opts.wPkgsAcc.size() << "\n";

  // Accelerator device
  if (mode == Mode::GPU || mode == Mode::FPGA || mode == Mode::CPU_GPU || mode == Mode::CPU_FPGA) {
    std::cout << "Accelerator device: " << opts.accDeviceDesc << "\n";
    std::cout << "Number of work packages: " << opts.wPkgsAcc.size() << ", number of total work items : " << opts.workSizeAcc << "\n";
    std::cout << "Time spent on kernels:\n";
    std::cout << "\tSubmitting and waiting for resources availability: " << opts.tSubmitKernelAcc << " s\n";
    std::cout << "\tComputing: " << opts.tComputeKernelAcc << " s\n";
    std::cout << "Time spent on device: " << opts.tAccEnd << " s\n";

    std::cout << "Work packages summary:\n";
    uint32_t i = 1;
    for (auto pkg : opts.wPkgsAcc) {
      std::cout << "\tPackage " << i++ << " -> size: " << pkg.size << ", offset: " << pkg.offset << ", computation time: " << pkg.tCompute 
                << " s, time since start of program: " << pkg.tSinceStart << " s\n";
    }
  }

  std::cout << ("\n\n");

  // CPU device
  if (mode == Mode::CPU || mode == Mode::CPU_GPU || mode == Mode::CPU_FPGA) {
    std::cout << "CPU device: " << opts.cpuDeviceDesc << "\n";
    std::cout << "Number of work packages: " << opts.wPkgsCPU.size() << ", number of total work items : " << opts.workSizeCPU << "\n";
    std::cout << "Time spent on kernels:\n";
    std::cout << "\tSubmitting and waiting for resources availability: " << opts.tSubmitKernelCPU << " s\n";
    std::cout << "\tComputing: " << opts.tComputeKernelCPU << " s\n";
    std::cout << "Time spent on device: " << opts.tCPUEnd << " s\n";

    std::cout << "Work packages summary:\n";
    uint32_t i = 1;
    for (auto pkg : opts.wPkgsCPU) {
      std::cout << "\tPackage " << i++ << " -> size: " << pkg.size << ", offset: " << pkg.offset << ", computation time: " << pkg.tCompute 
                << " s, time since start of program: " << pkg.tSinceStart << " s\n";
    }
  }
  std::cout << "\n\n";

  if (check) {
    if (verify(N, opts.pData.a, opts.pData.b, opts.pData.c)) {
      std::cout << "Success\n";
    } else {
      std::cout << "Failure\n";
      print_mat("A", opts.pData.a, N);
      print_mat("B", opts.pData.b, N);
      print_mat("C", opts.pData.c, N);
    }
  }

  //cout << "Output values: " << c_ptr[0] << "..." << c_ptr[matadd.size - 1] << "\n";
  return 0;
}
