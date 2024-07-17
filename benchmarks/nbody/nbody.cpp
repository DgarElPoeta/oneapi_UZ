//
// Created by radon on 17/09/20.
//
#include <random>
#include <thread>
#include <iostream>

#include "nbody.h"
#include "benchmarks.h"

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
      << "usage: <cpu|gpu|fpga|cpu_gpu|cpu_fpga> <static|dynamic|hguided> <num pkgs (dyn)|cpu proportion (st|hg)> <problem size> [num_cpp_threads]\n"
      << "Environment variables:\n"
      << "DEBUG=y   to print messages during execution\n"
      << "CHECK=y   to evaluate the correctnes of the results\n"
      << "PRINT=y   to print the the data of the problem\n"
      << "MIN_PKG_MULTIPLIER=<uint>,<uint> (cpu,acc)   to specify the multiplier for the min package of each device in HGuided algorithm\n"
      << "K=<float>   to specify the K value in HGuided algorithm\n";
  return 1;
}

void print_ptype_v(std::string name, std::vector<ptype>& v, uint64_t N) {
  std::cout << name << "\n";
  for (size_t i = 0; i < N; ++i) {
    std::cout << v[i].x() << " " << v[i].y() << " " << v[i].z() << "\n";
  }
  std::cout << "\n";
}

void print_float_v(std::string name, std::vector<float>& v, uint64_t N) {
  std::cout << name << "\n";
  for (size_t i = 0; i < N; ++i) {
    std::cout << v[i] << "\n";
  }
  std::cout << "\n";
}



bool verify(Nbody& nbody) {
  uint64_t N = nbody.size;
  std::vector<ptype> post_out(N);
  std::vector<ptype> vel_out(N);
  bool verification_passed = true;

  constexpr float threshold = 0.00001;

  for (size_t i = 0; i < N; ++i) {
      ptype myPos = nbody.pos_in[i];
      ptype myVel = nbody.vel_in[i];
      ptype acc{0.0f};
      for(size_t = 0; j < N; j++){
        if(i != j){
          ptype p = nbody.pos_in[j];
          ptype r = p - myPos;
          float m = nbody.body_mass[j];
          float distSqr = sycl::dot(r,r) + SofteningSquared;
          float dist = sycl::sqrt(distSqr);
          float invDist = 1.0f / dist;
          float invDistCube = invDist * invDist * invDist;
          acc += m * r * invDistCube;
        }
      }
      acc *= G;
      ptype newVel = myVel + acc * DT;
      ptype newPos = myPos + myVel * DT + 0.5f * acc * DT * DT;
      post_out[i] = newPos;
      vel_out[i] = newVel;
  }

  for (size_t i = 0; i < N; ++i) {
      ptype pos_kernel_value = nbody.pos_out[i];
      ptype pos_host_value = post_out[i];
      ptype vel_kernel_value = nbody.vel_out[i];
      ptype vel_host_value = vel_out[i];

      float value1,value2, difference = 0.0f;
      std::string component = "";
      auto pos_difference_x = (pos_kernel_value.x() > pos_host_value.x()) ? pos_kernel_value.x() - pos_host_value.x() : pos_host_value.x() - pos_kernel_value.x();
      auto pos_difference_y = (pos_kernel_value.y() > pos_host_value.y()) ? pos_kernel_value.y() - pos_host_value.y() : pos_host_value.y() - pos_kernel_value.y();
      auto pos_difference_z = (pos_kernel_value.z() > pos_host_value.z()) ? pos_kernel_value.z() - pos_host_value.z() : pos_host_value.z() - pos_kernel_value.z();
      
      auto vel_difference_x = (vel_kernel_value.x() > vel_host_value.x()) ? vel_kernel_value.x() - vel_host_value.x() : vel_host_value.x() - vel_kernel_value.x();
      auto vel_difference_y = (vel_kernel_value.y() > vel_host_value.y()) ? vel_kernel_value.y() - vel_host_value.y() : vel_host_value.y() - vel_kernel_value.y();
      auto vel_difference_z = (vel_kernel_value.z() > vel_host_value.z()) ? vel_kernel_value.z() - vel_host_value.z() : vel_host_value.z() - vel_kernel_value.z();
      if(pos_difference_x > threshold){
        value1 = pos_kernel_value.x();
        value2 = pos_host_value.x();
        difference = pos_difference_x;
        component = "x component of position";
      }
      else if(pos_difference_y > threshold){
        value1 = pos_kernel_value.y();
        value2 = pos_host_value.y();
        difference = pos_difference_y;
        component = "y component of position";
      }
      else if(pos_difference_z > threshold){
        value1 = pos_kernel_value.z();
        value2 = pos_host_value.z();
        difference = pos_difference_z;
        component = "z component of position";
      }
      else if(vel_difference_x > threshold){
        value1 = vel_kernel_value.x();
        value2 = vel_host_value.x();
        difference = vel_difference_x;
        component = "x component of velocity";
      }
      else if(vel_difference_y > threshold){
        value1 = vel_kernel_value.y();
        value2 = vel_host_value.y();
        difference = vel_difference_y;
        component = "y component of velocity";
      }
      else if(vel_difference_z > threshold){
        value1 = vel_kernel_value.z();
        value2 = vel_host_value.z();
        difference = vel_difference_z;
        component = "z component of velocity";
      }

      if (difference > threshold) {
        fprintf(stderr, "VERIFICATION FAILED for body %ld: %s, |%f-%f| = %f > %f = threshold\n", i, component.c_str(), value1, value2, difference, threshold);
        verification_passed = false;
        break;
      }
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

  // PRINT environment variable
  char *print_str = getenv("PRINT"); // string with the print value
  bool print = (print_str != NULL && std::string(print_str) == "y"); // Print problem data activated(true) or deactivated(false)

  // Processing capabilities of CPU and Accelerator. Used in HGuided Algorithm
  uint32_t min_multiplier[2] = {1, 1};

  // MIN_PKG_MULTIPLIER environment variable
  char *multipliers_str = getenv("MIN_PKG_MULTIPLIER");
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

  Options<Nbody> opts;

  opts.mode = mode;
  opts.algo = algo;
  opts.debug = debug;
  opts.usm = false;
  opts.cpuProp = cpu_prop;
  opts.numPkgs = num_pkgs;
  opts.K = K;

  opts.minMultiplierCPU = min_multiplier[0];
  opts.minMultiplierAcc = min_multiplier[1];

  opts.accDeviceDesc = "";
  opts.cpuDeviceDesc = "";
  
  opts.numCppThreads = num_cpp_threads;

  opts.pTotalSize = N;

  uint64_t pWork = 0;
  uint64_t pPkg = 0;
  uint64_t pPkgCPU = 0;
  uint64_t pPkgAcc = 0;
  opts.pWork = &pWork;
  opts.pPkg = &pPkg;
  opts.pPkgCPU = &pPkgCPU;
  opts.pPkgAcc = &pPkgAcc;


  opts.tpStart = tpStart;

  opts.tCPUEnd = 0;
  opts.tAccEnd = 0;

  opts.tComputeKernelCPU = 0;
  opts.tComputeKernelAcc = 0;
  opts.tSubmitKernelCPU = 0;
  opts.tSubmitKernelAcc= 0;
  
  opts.setupWorkPkgs();

  // Initialization of Nbody data type of Opts
  opts.pData = Nbody();
  opts.pData.size = N;
  opts.pData.pos_in = std::vector<ptype>(N);
  opts.pData.vel_in = std::vector<ptype>(N);
  opts.pData.pos_out = std::vector<ptype>(N);
  opts.pData.vel_out = std::vector<ptype>(N);
  opts.pData.body_mass = std::vector<float>(N);


  // -------------------------------------------------------------------------------------------------
  

  // -------------------------------------------------------------------------------------------------
  // Initialization of the bodies position, velocity and mass in Nbody data type of Opts with random values

  constexpr float posMin = -20.0f, posMax = 20.0f;
  constexpr float velMin = -1.0f, velMax = 1.0f;
  constexpr float massMin = 1.0f, massMax = 10.0f;
  std::random_device dev;
  std::mt19937 gen(dev());

  std::uniform_real_distribution<float> pos_dis(posMin, posMax);
  std::uniform_real_distribution<float> vel_dis(velMin, velMax);
  std::uniform_real_distribution<float> mass_dis(massMin, massMax);

  for (size_t i = 0; i < N; i++) {
    opts.pData.pos_in[i] = pytpe{pos_dis(gen),pos_dis(gen),pos_dis(gen)};
    opts.pData.vel_in[i] = ptype{vel_dis(gen),vel_dis(gen),vel_dis(gen)};
    opts.pData.pos_out[i] = pytpe{0.0f,0.0f,0.0f};
    opts.pData.vel_out[i] = ptype{0.0f,0.0f,0.0f};
    opts.pData.body_mass[i] = mass_dis(gen);
  }

  // -------------------------------------------------------------------------------------------------


  // -------------------------------------------------------------------------------------------------
  // Load scheduler processes

  auto timePoint = std::chrono::high_resolution_clock::now();
  opts.tpSchedulerStart = timePoint;
  if (mode == Mode::CPU) {

    opts.tpCPUStart = std::vector<std::chrono::high_resolution_clock::time_point>(num_cpp_threads);

    // Thread vector
    std::vector<std::thread> vecOfThreads(num_cpp_threads-1);

    // Creation of CPU threads
    for(size_t i=1; i<num_cpp_threads; i++){
      
      // Push start time of every thread created
      timePoint = std::chrono::high_resolution_clock::now();
      opts.tpCPUStart[i] = timePoint;

      // Create thread with CPU scheduler process
      vecOfThreads[i-1] = std::thread(process<Nbody>, true, std::ref(opts), i);
    }

    // Start time of CPU actual thread
    timePoint = std::chrono::high_resolution_clock::now();
    opts.tpCPUStart[0] = timePoint;

    // CPU scheduler process
    process<Nbody>(true, opts, 0);

    // Join of all CPU threads
    for (std::thread & th : vecOfThreads){
      if (th.joinable()) th.join();
    }

  } else if (mode == Mode::GPU || mode == Mode::FPGA) {
    
    // Start time of accelerator scheduler process
    timePoint = std::chrono::high_resolution_clock::now();
    opts.tpAccStart = timePoint;

    // Accelerator scheduler process
    process<Nbody>(false, opts, 0);

  } else {

    opts.tpCPUStart = std::vector<std::chrono::high_resolution_clock::time_point>(num_cpp_threads);
    
    // Thread vector
    std::vector<std::thread> vecOfThreads(num_cpp_threads);

    // Creation of CPU threads
    for(size_t i=0; i<num_cpp_threads; i++){

      // Push start time of every thread created
      timePoint = std::chrono::high_resolution_clock::now();
      opts.tpCPUStart[i] = timePoint;

      // Create thread with CPU scheduler process
      vecOfThreads[i] = std::thread(process<Nbody>, true, std::ref(opts), i);
    }

    // Start time of accelerator scheduler process
    timePoint = std::chrono::high_resolution_clock::now();
    opts.tpAccStart = timePoint;
    process<Nbody>(false, opts, 0);

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
  std::cout << "Benchmark: nbody\n";
  std::cout << "Problem size: " << N << ". (each body is considered the work unit)\n";
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
    std::cout << " minPkgMultiplier (cpu,acc): (" << opts.minMultiplierCPU << "," << opts.minMultiplierAcc << ")\n";
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

  std::cout << "Load scheduler summary:\n";
  std::cout << "Time since start of program: " << (opts.tpSchedulerStart - tpStart).count() / 1e9 << " s\n";
  std::cout << "Time spent on load scheduler: " << tScheduler << " s\n";
  std::cout << "Total work packages: " << pPkgCPU + pPkgAcc << "\n";
  std::cout << "\n\n";

  // Accelerator device
  if (mode == Mode::GPU || mode == Mode::FPGA || mode == Mode::CPU_GPU || mode == Mode::CPU_FPGA) {
    std::cout << "Accelerator device: " << opts.accDeviceDesc << "\n";
    std::cout << "Number of work packages: " << pPkgAcc << ", number of total work items : " << opts.workSizeAcc << "\n";
    std::cout << "Time spent on kernels:\n";
    std::cout << "\tSubmitting and waiting for resources availability: " << opts.tSubmitKernelAcc << " s\n";
    std::cout << "\tComputing: " << opts.tComputeKernelAcc << " s\n";
    std::cout << "Time spent on device: " << opts.tAccEnd << " s\n";
    
    if (pPkgAcc > 0){
      std::cout << "Work packages summary:\n";
      for (size_t i = 0; i < pPkgAcc; i++) {
        WorkPackages pkg = opts.wPkgsAcc[i];
        std::cout << "\tPackage " << i+1 << " -> size: " << pkg.size << ", offset: " << pkg.offset << ", computation time: " << pkg.tCompute 
                  << " s, time since start of program: " << pkg.tSinceStart << " s\n";
      }
    }
  }

  std::cout << ("\n\n");

  // CPU device
  if (mode == Mode::CPU || mode == Mode::CPU_GPU || mode == Mode::CPU_FPGA) {
    std::cout << "CPU device: " << opts.cpuDeviceDesc << "\n";
    std::cout << "Number of work packages: " << pPkgCPU << ", number of total work items : " << opts.workSizeCPU << "\n";
    std::cout << "Time spent on kernels:\n";
    std::cout << "\tSubmitting and waiting for resources availability: " << opts.tSubmitKernelCPU << " s\n";
    std::cout << "\tComputing: " << opts.tComputeKernelCPU << " s\n";
    std::cout << "Time spent on device: " << opts.tCPUEnd << " s\n";

    if (pPkgCPU > 0){
      std::cout << "Work packages summary:\n";
      for (size_t i = 0; i < pPkgCPU; i++) {
        WorkPackages pkg = opts.wPkgsCPU[i];
        std::cout << "\tPackage " << i+1 << " -> size: " << pkg.size << ", offset: " << pkg.offset << ", computation time: " << pkg.tCompute 
                  << " s, time since start of program: " << pkg.tSinceStart << " s\n";
      }
    }
  }
  std::cout << "\n";

  if (check) {
    std::cout << "Verificating the correctness of the results...\n";
    if (verify(opts.pData)) {
      std::cout << "Verification completed: success\n";
    } else {
      std::cout << "Verification completed: failure\n";
    }
  }

  if(print){
    print_ptype_v("pos_in", opts.pData.pos_in, N);
    print_ptype_v("vel_in", opts.pData.vel_in, N);
    print_ptype_v("pos_out", opts.pData.pos_out, N);
    print_ptype_v("vel_out", opts.pData.vel_out, N);
    print_float_v("body_mass", opts.pData.body_mass, N);
  }

  std::cout << "---------------------------------------------------------------------------------\n";
  return 0;
}
