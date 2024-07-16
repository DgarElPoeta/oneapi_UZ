//
// Created by radon on 9/10/20.
//

#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include <string>
#include <vector>
#include <cstdint>
#include <mutex>


// Types of scheduler algorithms
enum class Algo {
  Static, Dynamic, HGuided
};

bool hashAlgo(std::string sAlgo, Algo& algo){
  if (sAlgo == "static") {
    algo = Algo::Static;
  } else if (sAlgo == "dynamic") {
    algo = Algo::Dynamic;
  } else if (sAlgo == "hguided") {
    algo = Algo::HGuided;
  } else {
    return false;
  }
  return true;
}



// Types of heterogeneous execution modes
enum class Mode {
  CPU, GPU, FPGA,CPU_GPU, CPU_FPGA
};

bool hashMode(std::string sMode, Mode& mode){
  if (sMode == "cpu") {
    mode = Mode::CPU;
  } else if (sMode == "gpu") {
    mode = Mode::GPU;
  } else if (sMode == "fpga") {
    mode = Mode::FPGA;
  } else if (sMode == "cpu_gpu") {
    mode = Mode::CPU_GPU;
  } else if (sMode == "cpu_fpga") {
    mode = Mode::CPU_FPGA;
  } else {
    return false;
  }
  return true;
}

struct WorkPackages
{
  uint64_t offset;
  uint64_t size;
  double tCompute;
  double tSinceStart;
  WorkPackages() {}
  WorkPackages(uint64_t _offset, uint64_t _size, double _tCompute, double _tSinceStart)
  {
    offset = _offset;
    size = _size;
    tCompute = _tCompute;
    tSinceStart = _tSinceStart;
  }
};


template <typename T> struct Options {

  Algo algo; // type of load scheduler algorithm
  Mode mode; // heterogeneous execution mode

  // Debug mode activated(true) or deactivated(false)
  bool debug;

  // USM activated(true) or deactivated(false)
  bool usm;

  // Proportion of work asigned to CPU. Used in Static and HGuided algorithms
  float cpuProp;

  // Number of packages in which the total work is divided. Used in Dynamic algorithm
  uint64_t numPkgs;

  // Coefficient of workload decrement. Used in HGuided Algorithm
  float K; 

  // Processing capabilities of CPU and Accelerator. Used in HGuided Algorithm
  uint32_t minMultiplierCPU;
  uint32_t minMultiplierAcc;

  // Work group size
  uint64_t wgs;

  // The package size must be a multiple of the value of this variable
  uint64_t sizeMultiple;

  // Work packages
  std::vector<WorkPackages> wPkgsCPU; // Work packages assigned to CPU
  std::vector<WorkPackages> wPkgsAcc; // Work packages assigned to Accelerator
  uint64_t workSizeCPU; // Work size assigned to CPU
  uint64_t workSizeAcc; // work size assigned to Accelerator

  // Description of accelerator device and cpu device.
  string accDeviceDesc, cpuDeviceDesc;

  // Number of threads
  uint32_t numCppThreads;

  // Mutexes used for synchronization in critical sections
  std::mutex mWork; // mutex used when a load scheduler process requires work packages. Used in Dynamic an HGuided Algorithm.
  std::mutex mCPU; // mutex used when a load scheduler process related to CPU needs to update profiling times.

  // Sizes of the problem. Used in Dynamic and HGuided Algorithm
  uint64_t pTotalSize; // Total size of the problem
  uint64_t* pWork; // Size of the problem solved at the moment

  // Number of packages solved at the moment
  uint64_t* pPkg;
  uint64_t* pPkgCPU;
  uint64_t* pPkgAcc;

  // Benchmark start timepoint
  std::chrono::high_resolution_clock::time_point tpStart;
  // Load scheduler start timepoint
  std::chrono::high_resolution_clock::time_point tpSchedulerStart;
  // CPU threads start timepoint
  std::vector<std::chrono::high_resolution_clock::time_point> tpCPUStart;
  // Accelerator thread start timepoint
  std::chrono::high_resolution_clock::time_point tpAccStart;

  // Time spent in seconds for the solving of the problem in the devices
  double tCPUEnd; // Time spent in the CPU
  double tAccEnd; // Time spent in the accelerator

  // Time spent in seconds for the different parts of the solving in the devices
  double tComputeKernelCPU; // Time spent on the computation of the CPU thread's kernel
  double tComputeKernelAcc; // Time spent on the computation of the accelerator's kernel
  double tSubmitKernelAcc; // Time spent between the submission and the start of the computation of the accelerator's kernel
  double tSubmitKernelCPU; // Time spent between the submission and the start of the computation of the CPU thread's kernel

  T pData; // Benchmark data type with contains the data. Eg. Matmul, Gaussian

  Options() : usm(false), wgs(128), sizeMultiple(wgs), mWork(), mCPU(){
  }

  void setupWorkPkgs(){
    
    constexpr size_t dim = 20;
    size_t pCPU = dim, pAcc = dim;
    if(algo == Algo::Static){
      pCPU = 1 * numCppThreads;
      pAcc = 1;
      wPkgsCPU.reserve(pCPU);
      wPkgsAcc.reserve(pAcc);
    }
    else{
      wPkgsCPU.reserve(120);
      wPkgsAcc.reserve(120);
    }
    wPkgsCPU.clear();
    wPkgsAcc.clear();
    wPkgsCPU.resize(pCPU);
    wPkgsAcc.resize(pAcc);
    workSizeCPU = 0;
    workSizeAcc = 0;
  }

  void
  saveWorkPackages(bool cpu, uint64_t index, uint64_t offset, uint64_t size, double tCompute)
  {
    auto tp = std::chrono::high_resolution_clock::now();
    double tSinceStart = (tp - tpStart).count() / 1e9;
    if (cpu){
      if(index >= wPkgsCPU.size()) wPkgsCPU.resize(wPkgsCPU.size() + 20);
      wPkgsCPU[index] = WorkPackages(offset, size, tCompute, tSinceStart);
    } else {

      if (index >= wPkgsAcc.size()) wPkgsAcc.resize(wPkgsAcc.size() + 20);
      wPkgsAcc[index] = WorkPackages(offset, size, tCompute, tSinceStart);
    }
  }
};

#endif //BENCHMARKS_H
