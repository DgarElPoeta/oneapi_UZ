//
// Created by radon on 17/09/20.
//
#include <random>
#include <thread>
#include <iostream>
#include <iomanip>

#include "nbody.h"
#include "benchmarks.h"

#include "schedulers/st.cpp"
#include "schedulers/dyn.cpp"
#include "schedulers/hg.cpp"


// Function that processes the scheduler algorithm
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

// Function that prints the usage of the program and returns 1
int usage(const std::string &name) {
  std::cerr
      << "Usage:\n" << name << " <cpu|fpga|cpu_fpga> <static|dynamic|hguided> <num pkgs (dyn)|cpu proportion (st|hg)> <problem size> \n"
      << "\nEnvironment variables:\n"
      << "DEBUG=y   to print messages during execution\n"
      << "CHECK=y   to evaluate the correctnes of the results\n"
      << "PRINT=y   to print the the data of the problem\n"
      << "MIN_PKG_MULTIPLIER=<uint>,<uint> (cpu,acc)   to specify the multiplier for the min package of each device in HGuided algorithm\n"
      << "K=<float>   to specify the K value in HGuided algorithm\n"
      << "NUM_CPU_THREADS=<uint>   to specify the number of threads in CPU mode\n"
      << "\n";
  return 1;
}

// Function that writes in standar output the vector of ptype
void print_ptype_v(std::string name, std::vector<ptype>& v, uint64_t N) {
  std::cout << name << "\n";
  for (size_t i = 0; i < N; ++i) {
    std::cout << v[i].x() << " " << v[i].y() << " " << v[i].z() << "\n";
  }
  std::cout << "\n";
}

// Function tha writes in standar output the vector of mtype
void print_mtype_v(std::string name, std::vector<mtype>& v, uint64_t N) {
  std::cout << name << "\n";
  for (size_t i = 0; i < N; ++i) {
    std::cout << v[i] << "\n";
  }
  std::cout << "\n";
}

// Function that verifies the correctness of the results
bool verify(Nbody& nbody) {
  uint64_t N = nbody.size;
  bool verification_passed = true;

  constexpr float threshold = 0.00001;

  for (size_t i = 0; i < N; ++i) {
      ptype myPos = nbody.pos_in[i];
      ptype myVel = nbody.vel_in[i];
      ptype acc{0.0f};
      for(size_t j = 0; j < N; j++){
        ptype p = nbody.pos_in[j];
        ptype r = p - myPos;
        mtype m = nbody.body_mass[j];
        float distSqr = sycl::dot(r,r) + SofteningSquared;
        float dist = sycl::sqrt(distSqr);
        float invDist = 1.0f / dist;
        float invDistCube = invDist * invDist * invDist;
        acc += m * r * invDistCube;
      }
      acc *= G;
      ptype newVel = myVel + acc * DT;
      ptype newPos = myPos + myVel * DT + 0.5f * acc * DT * DT;
      ptype pos_host_value = newPos;
      ptype vel_host_value = newVel;
      ptype pos_kernel_value = nbody.pos_out[i];
      ptype vel_kernel_value = nbody.vel_out[i];

      float kernel_value = 0.0 ,host_value = 0.0, difference = 0.0f;
      std::string component = "";
      auto pos_difference_x = (pos_kernel_value.x() > pos_host_value.x()) ? pos_kernel_value.x() - pos_host_value.x() : pos_host_value.x() - pos_kernel_value.x();
      auto pos_difference_y = (pos_kernel_value.y() > pos_host_value.y()) ? pos_kernel_value.y() - pos_host_value.y() : pos_host_value.y() - pos_kernel_value.y();
      auto pos_difference_z = (pos_kernel_value.z() > pos_host_value.z()) ? pos_kernel_value.z() - pos_host_value.z() : pos_host_value.z() - pos_kernel_value.z();
      
      auto vel_difference_x = (vel_kernel_value.x() > vel_host_value.x()) ? vel_kernel_value.x() - vel_host_value.x() : vel_host_value.x() - vel_kernel_value.x();
      auto vel_difference_y = (vel_kernel_value.y() > vel_host_value.y()) ? vel_kernel_value.y() - vel_host_value.y() : vel_host_value.y() - vel_kernel_value.y();
      auto vel_difference_z = (vel_kernel_value.z() > vel_host_value.z()) ? vel_kernel_value.z() - vel_host_value.z() : vel_host_value.z() - vel_kernel_value.z();
      
      if(pos_difference_x > threshold){
        kernel_value = pos_kernel_value.x();
        host_value = pos_host_value.x();
        difference = pos_difference_x;
        component = "x component of position";
      }
      else if(pos_difference_y > threshold){
        kernel_value = pos_kernel_value.y();
        host_value = pos_host_value.y();
        difference = pos_difference_y;
        component = "y component of position";
      }
      else if(pos_difference_z > threshold){
        kernel_value = pos_kernel_value.z();
        host_value = pos_host_value.z();
        difference = pos_difference_z;
        component = "z component of position";
      }
      else if(vel_difference_x > threshold){
        kernel_value = vel_kernel_value.x();
        host_value = vel_host_value.x();
        difference = vel_difference_x;
        component = "x component of velocity";
      }
      else if(vel_difference_y > threshold){
        kernel_value = vel_kernel_value.y();
        host_value = vel_host_value.y();
        difference = vel_difference_y;
        component = "y component of velocity";
      }
      else if(vel_difference_z > threshold){
        kernel_value = vel_kernel_value.z();
        host_value = vel_host_value.z();
        difference = vel_difference_z;
        component = "z component of velocity";
      }
      std::string svalue = "";
      std::string value = "";

      if (kernel_value == 0.0 && host_value == 0.0) ;
      else if(kernel_value == 0.0){
        difference = (host_value < 0.0) ? difference/ -host_value : difference/host_value;
        svalue = "/ |host_value|";
        value = "/ |" + std::to_string(host_value) + "|";
      }
      else{
        difference = (kernel_value < 0.0) ? difference/ -kernel_value : difference/kernel_value;
        svalue = "/ |kernel_value|";
        value = "/ |" + std::to_string(kernel_value) + "|";
      }
      if (difference > threshold) {
        std::cerr << "VERIFICATION FAILED for body (" << i << "), " << component << "\n\tThe next statement isn't true: |kernel_value - host_value| " << svalue << " < threshold\n\t-> |" << kernel_value 
                  << " - " << host_value << "| " << value << " = " << difference << " > " << threshold << "\n";
        verification_passed = false;
        break;
      }

      if (!verification_passed) { break; }
  }

  return verification_passed;
}

int main(int argc, char *argv[]) {

  // Initial timepoint of program
  std::chrono::high_resolution_clock::time_point tpStart = std::chrono::high_resolution_clock::now();

  { // Program scope

  // -------------------------------------------------------------------------------------------------
  // Arguments comprobation and initialization

  // Get the name of the program
  std::string name = argv[0];

  argc--;
  if (argc != 4) {
    std::cerr << "\nNumber of arguments is different than expected\n\n";
    return usage(name);
  }


  // Heterogeneous execution mode arg
  std::string mode_str = argv[1]; // string with the mode
  Mode mode; // Heterogeneous execution mode
  if (!hashMode(mode_str, mode)) {
    std::cerr << "\nInvalid mode\n\n";
    return usage(name);
  }


  // Scheduler algorithm arg
  std::string algo_str = argv[2]; // string with the algorithm
  Algo algo; // Scheduler algorithm
  if (!hashAlgo(algo_str, algo)) {
    std::cerr << "\nInvalid algorithm\n";
    return usage(name);
  }

  
  // Proportion args
  float cpu_prop; // Work proportion assigned to CPU. Used in Static and HGuided Algorithms
  size_t num_pkgs; // Number of packages in which the total work is divided in Dynamic Algorithm
  if (algo == Algo::Dynamic) {
    int64_t pkgs = atoi(argv[3]);
    if(pkgs <= 0){
      std::cerr << "\nNumber of packages must be greater than 0\n\n";
      return usage(name);
    }
    num_pkgs = pkgs;
  } else {
    if (mode == Mode::CPU_FPGA) {
      cpu_prop = atof(argv[3]);
      if(cpu_prop < 0 || cpu_prop > 1){
        std::cerr << "\nCpu proportion must be between 0 and 1\n\n";
        return usage(name);
      }
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
  if (N == 0 || ((N/WORK_GROUP_SIZE) * WORK_GROUP_SIZE != N)) {
    std::cerr << "\nProblem size must be greater than 0 and multiple of " << WORK_GROUP_SIZE << "\n\n";
    return usage(name);
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

  // Multipliers of work package size of CPU and Accelerator. Used in HGuided Algorithm
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
      int32_t mult = std::stoi(item);
      if(mult <= 0){
        std::cerr << "\nMin package multiplier must be greater than 0\n\n";
        return usage(name);
      }
      min_multiplier[i] = mult;
      i++;
    }
  }

  // K environment variable
  char *K_str = getenv("K");
  float K = 2.0;
  if (K_str != nullptr) {
    float K_ = std::stof(K_str);
    K = K_;
  }

  // Number of cpu threads environment variable
  uint32_t num_cpu_threads = 1; // Number of cpu threads
  char *NCT_str = getenv("NUM_CPU_THREADS");
  if (NCT_str != nullptr) {
    int nct = atoi(NCT_str);
    if(nct <= 0){
      std::cerr << "\nNumber of CPU threads must be greater than 0\n\n";
      return usage(name);
    }
    num_cpu_threads = nct;
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
  
  opts.numCPUThreads = num_cpu_threads;

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

  opts.firstCPU = true;

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
  opts.pData.pos_out = std::vector<ptype>(N,ptype{0.0});
  opts.pData.vel_out = std::vector<ptype>(N,ptype{0.0});
  opts.pData.body_mass = std::vector<mtype>(N);


  // -------------------------------------------------------------------------------------------------
  

  // -------------------------------------------------------------------------------------------------
  // Initialization of the bodies position, velocity and mass in Nbody data type of Opts with random values

  constexpr float posMin = -20.0f, posMax = 20.0f;
  constexpr float velMin = -1.0f, velMax = 1.0f;
  constexpr mtype massMin = 1.0f, massMax = 10.0f;
  std::random_device dev;
  std::mt19937 gen(dev());

  std::uniform_real_distribution<float> pos_dis(posMin, posMax);
  std::uniform_real_distribution<float> vel_dis(velMin, velMax);
  std::uniform_real_distribution<mtype> mass_dis(massMin, massMax);

  for (size_t i = 0; i < WORK_GROUP_SIZE; i++) {

    opts.pData.pos_in[i] = ptype{pos_dis(gen),pos_dis(gen),pos_dis(gen)};
    opts.pData.vel_in[i] = ptype{vel_dis(gen),vel_dis(gen),vel_dis(gen)};
    opts.pData.body_mass[i] = mass_dis(gen);
  }

  for(size_t i = WORK_GROUP_SIZE; i < N; i*=2){
    size_t chunk_size = std::min(i, N - i);
    std::memcpy(opts.pData.pos_in.data() + i, opts.pData.pos_in.data(), chunk_size * sizeof(ptype));
    std::memcpy(opts.pData.vel_in.data() + i, opts.pData.vel_in.data(), chunk_size * sizeof(ptype));
    std::memcpy(opts.pData.body_mass.data() + i, opts.pData.body_mass.data(), chunk_size * sizeof(mtype));
  }

  // -------------------------------------------------------------------------------------------------


  // -------------------------------------------------------------------------------------------------
  // Load scheduler processes

  std::cout << "Starting load scheduler...\n";

  auto timePoint = std::chrono::high_resolution_clock::now();
  opts.tpSchedulerStart = timePoint;
  if (mode == Mode::CPU) {

    opts.tpCPUStart = std::vector<std::chrono::high_resolution_clock::time_point>(num_cpu_threads);

    // Thread vector
    std::vector<std::thread> vecOfThreads(num_cpu_threads-1);

    // Creation of CPU threads
    for(size_t i=1; i<num_cpu_threads; i++){
      
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
      th.join();
    }

  } else if (mode == Mode::FPGA) {
    
    // Start time of accelerator scheduler process
    timePoint = std::chrono::high_resolution_clock::now();
    opts.tpAccStart = timePoint;

    // Accelerator scheduler process
    process<Nbody>(false, opts, 0);

  } else {

    opts.tpCPUStart = std::vector<std::chrono::high_resolution_clock::time_point>(num_cpu_threads);
    
    // Thread vector
    std::vector<std::thread> vecOfThreads(num_cpu_threads);

    // Creation of CPU threads
    for(size_t i=0; i<num_cpu_threads; i++){

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
      th.join();
    }
  }

  // -------------------------------------------------------------------------------------------------


  auto tpSchedulerEnd = std::chrono::high_resolution_clock::now();
  auto diffScheduler = (tpSchedulerEnd - opts.tpSchedulerStart).count();
  auto tScheduler = diffScheduler / 1e9;
  
  // Execution summary
  std::cout << std::fixed << std::setprecision(10) << "\n\n\n";
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
    std::cout << "Scheduler parameters:\n";
    std::cout << " CPU proportion: " << cpu_prop << "\n";

  } else if (algo == Algo::Dynamic) {
    std::cout << "Dynamic\n";
    std::cout << "Scheduler parameters:\n";
    std::cout << " Number of packages: " << num_pkgs << "\n";
  } else if (algo == Algo::HGuided) {
    std::cout << "HGuided\n";
    std::cout << "Scheduler parameters:\n";
    std::cout << " CPU proportion: " << cpu_prop << "\n";
    std::cout << " K: " << opts.K << "\n";
    std::cout << " min_pkg_multiplier (cpu,acc): (" << opts.minMultiplierCPU << "," << opts.minMultiplierAcc << ")\n";
  }
  std::cout << "\n\n";

  // Heterogeneus execution mode
  std::cout << "Mode: ";
  switch(mode){
    case Mode::CPU:
      std::cout << "CPU\n";
      std::cout << "Number of CPU threads: " << num_cpu_threads << "\n";
      break;
    case Mode::FPGA:
#ifdef FPGA_EMULATOR
      std::cout << ("FPGA Emulator\n");
#else
      std::cout << "FPGA\n";
#endif
      break;
    case Mode::CPU_FPGA:
#ifdef FPGA_EMULATOR
      std::cout << ("CPU + FPGA Emulator\n");
#else
      std::cout << "CPU + FPGA\n";
#endif
      std::cout << "Number of CPU threads: " << num_cpu_threads << "\n";
      break;
  }
  std::cout << "\n\n";

  std::cout << "Load scheduler summary:\n";
  std::cout << "Time since start of program: " << (opts.tpSchedulerStart - tpStart).count() / 1e9 << " s\n";
  std::cout << "Time spent on load scheduler: " << tScheduler << " s\n";
  std::cout << "Total work packages: " << pPkgCPU + pPkgAcc << "\n";
  std::cout << "\n\n";

  // Accelerator device
  if (mode == Mode::FPGA || mode == Mode::CPU_FPGA) {
    std::cout << "Accelerator device: " << opts.accDeviceDesc << "\n";
    std::cout << "Number of work packages: " << pPkgAcc << ", number of total work items : " << opts.workSizeAcc << "\n";
    std::cout << "Time between first kernel submitted and last kernel that completed execution: " << (opts.tpLastAcc - opts.tpFirstAcc).count() / 1e9 << " s\n";
    std::cout << "Time spent on device: " << opts.tAccEnd << " s\n";
    
    if (pPkgAcc > 0){
      std::cout << "Work packages summary:\n";
      for (size_t i = 0; i < pPkgAcc; i++) {
        WorkPackages pkg = opts.wPkgsAcc[i];
        std::cout << "\tPackage " << i+1 << " -> size: " << pkg.size << ", offset: " << pkg.offset << ", computation time: " << pkg.tCompute 
                  << " s, total kernel time: " << pkg.tTotalKernel
                  << " s, total event time: " << pkg.tTotalEvent
                  << " s, DTH time: " << pkg.tDTH
                  << " s, total time: " << pkg.tTotal
                  << " s, time since start of program: " << pkg.tSinceStart << " s\n";
      }
    }
    std::cout << ("\n\n");
  }


  // CPU device
  if (mode == Mode::CPU || mode == Mode::CPU_FPGA) {
    std::cout << "CPU device: " << opts.cpuDeviceDesc << "\n";
    std::cout << "Number of work packages: " << pPkgCPU << ", number of total work items : " << opts.workSizeCPU << "\n";
    std::cout << "Time between first kernel submitted and last kernel that completed execution: " << (opts.tpLastCPU - opts.tpFirstCPU).count() / 1e9 << " s\n";
    std::cout << "Time spent on device: " << opts.tCPUEnd << " s\n";

    if (pPkgCPU > 0){
      std::cout << "Work packages summary:\n";
      for (size_t i = 0; i < pPkgCPU; i++) {
        WorkPackages pkg = opts.wPkgsCPU[i];
        std::cout << "\tPackage " << i+1 << " -> size: " << pkg.size << ", offset: " << pkg.offset << ", computation time: " << pkg.tCompute 
                  << " s, total kernel time: " << pkg.tTotalKernel
                  << " s, total event time: " << pkg.tTotalEvent
                  << " s, DTH time: " << pkg.tDTH
                  << " s, total time: " << pkg.tTotal
                  << " s, time since start of program: " << pkg.tSinceStart << " s\n";
      }
    }
    std::cout << "\n\n";
  }

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
    print_mtype_v("body_mass", opts.pData.body_mass, N);
  }

  } // End of program scope 

  std::chrono::high_resolution_clock::time_point tpEnd = std::chrono::high_resolution_clock::now();
  std::cout << "\nTotal time elapsed in program: " << (tpEnd - tpStart).count() / 1e9 << " s\n";
  std::cout << "---------------------------------------------------------------------------------\n";

  return 0;
}
