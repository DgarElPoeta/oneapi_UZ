//
// Created by radon on 9/10/20.
//

#include "benchmarks.h"
#include "schedulers.h"
#include "kernels.h"
#include "queues.h"

template <typename T>
void process_static(bool cpu, Options<T>& opts, uint32_t thr_id) {

  bool debug = opts.debug;
  auto tpStart = opts.tpStart;
  auto tpDevInit = cpu ? opts.tpCPUStart[thr_id] : opts.tpAccStart;
  std::string device_type;
  { // SYCL scope

    sycl::queue q;

    if(cpu){
      q = CPU_QUEUE;
      opts.cpuDeviceDesc = q.get_device().get_info<sycl::info::device::name>();
    }
    else{
      switch(opts.mode) {
        case Mode::GPU:
        case Mode::CPU_GPU:
          q = GPU_QUEUE;
          break;
        default:
#ifdef FPGA_EMULATOR
          q = FPGAEMU_QUEUE;
#else
          q = FPGAHW_QUEUE;
#endif
          break;
      }
      opts.accDeviceDesc = q.get_device().get_info<sycl::info::device::name>();
    }

    device_type = cpu ? opts.cpuDeviceDesc + " Thread " + std::to_string(thr_id) : opts.accDeviceDesc;

    uint64_t total_size = opts.pTotalSize;
    
    float cpu_prop = opts.cpuProp;
    uint64_t pkg_size_multiple = opts.sizeMultiple;

    uint64_t size_accelerator = total_size * (1.0f - cpu_prop);
    size_accelerator = (cpu_prop == 0.0f) ? total_size : (size_accelerator / pkg_size_multiple) * pkg_size_multiple;
    uint64_t size_CPU = total_size - size_accelerator;

    uint32_t num_cpu_threads = opts.numCPUThreads;
    uint64_t eThread = ((size_CPU / num_cpu_threads) < pkg_size_multiple) ?  size_CPU / pkg_size_multiple : num_cpu_threads;

    uint64_t offset_CPU = size_accelerator;
    if(size_CPU > 0 ){
      if(eThread == 0 && thr_id == 0){
        size_CPU = total_size;
        *(opts.pPkgCPU) = 1;
      }
      else if(eThread < num_cpu_threads){
        if(thr_id + 1 == eThread && size_CPU != eThread * pkg_size_multiple){
          size_CPU = size_CPU - eThread * pkg_size_multiple;
          offset_CPU += eThread * pkg_size_multiple;
          *(opts.pPkgCPU) = eThread+1;
        }
        else if(thr_id < eThread){
          size_CPU = pkg_size_multiple;
          offset_CPU += thr_id * pkg_size_multiple;
          if(thr_id == 0) *(opts.pPkgCPU) = eThread;
        }
        else size_CPU = 0;
      }
      else{
        uint64_t total_pkg = size_CPU / pkg_size_multiple;
        uint64_t pkg_per_thread = total_pkg / num_cpu_threads;
        uint64_t pkg_1more = total_pkg - pkg_per_thread * num_cpu_threads;
        if(thr_id < pkg_1more){
          size_CPU = (pkg_per_thread + 1) * pkg_size_multiple;
          offset_CPU += thr_id * (pkg_per_thread + 1) * pkg_size_multiple;
        }
        else if(size_CPU != total_pkg * pkg_size_multiple && thr_id+1 == num_cpu_threads){
          size_CPU -= (total_pkg - pkg_per_thread) * pkg_size_multiple;
          offset_CPU += (total_pkg - pkg_per_thread) * pkg_size_multiple;
        }
        else{
          size_CPU = pkg_per_thread * pkg_size_multiple;
          offset_CPU += pkg_1more * pkg_size_multiple + thr_id * pkg_per_thread * pkg_size_multiple;
        }
        if(thr_id == 0) *(opts.pPkgCPU) = num_cpu_threads;

      }
    }
    if(!cpu){
      if(size_accelerator == 0) *(opts.pPkgAcc) = 0;
      else *(opts.pPkgAcc) = 1;
    }
    
    uint64_t size = ((cpu) ? size_CPU : size_accelerator);
    uint64_t offset = ((cpu) ? offset_CPU: 0);

    uint64_t wgs = opts.wgs;

    DEVICE_DEBUG("selected");
    if (size > 0) {

      // Buffers and kernels management variables
      size_t CK = 0; // Index of the current kernel
      constexpr size_t num_kernels = 1; // Total number of kernels that can be active at the same time
      sycl::event submit_event[num_kernels];  // Array with the events of the kernels
      
      // Include the file that defines the buffers used in the kernel.
      #include "buffers_sycl.cpp"

      // Include the file that setups the buffers and sycl variables used in the kernel.
      #include "setup_sycl.cpp"

      auto tpBefore = std::chrono::high_resolution_clock::now();
      if(cpu){
          std::lock_guard<std::mutex> lk(opts.mCPU);
          if(opts.firstCPU){
            opts.tpFirstCPU = tpBefore;
            opts.firstCPU = false;
          }
      }
      else{
        opts.tpFirstAcc = tpBefore;
      }
      auto diffBefore = (tpBefore - tpStart).count();
      auto tBefore = diffBefore / 1e9;
      std::string aux = std::to_string(tBefore) + " < size : " + std::to_string(size) + " offset : " + std::to_string(offset);
      DEVICE_DEBUG(aux);

      // Include the file that invokes the kernel
      #include "kernel_sycl.cpp"

      submit_event[CK].wait();

      // Time point after wait for kernel completion
      auto tpAfter = std::chrono::high_resolution_clock::now();
      
      // Time point before data transfer from device to host
      auto tpStartDTH = std::chrono::high_resolution_clock::now();
      auto diffStartDTH = (tpStartDTH - tpStart).count();
      auto tStartDTH = diffStartDTH / 1e9;
      aux = std::to_string(tStartDTH) + " : Start of data transfer from device to host";
      DEVICE_DEBUG(aux);
      
      size_t eventIndex = CK;
      {
        // Include the file that defines the host_accessors used for the data transmissions from device to host.
        #include "DTH.cpp"
      }
      // Time point after data transfer from device to host
      auto tpEndDTH = std::chrono::high_resolution_clock::now();
      auto diffEndDTH = (tpEndDTH - tpStart).count();
      auto tEndDTH = diffEndDTH / 1e9;
      aux = std::to_string(tEndDTH) + " : End of data transfer from device to host";
      DEVICE_DEBUG(aux);
      if(cpu){
        std::lock_guard<std::mutex> lk(opts.mCPU);
        opts.tpLastCPU = tpEndDTH;
      }
      else{
        opts.tpLastAcc = tpEndDTH;
      }

      //cl_ulong time_start, time_end, time_submit;
      auto time_submit = submit_event[CK].get_profiling_info<sycl::info::event_profiling::command_submit>();
      auto time_start = submit_event[CK].get_profiling_info<sycl::info::event_profiling::command_start>();
      auto time_end = submit_event[CK].get_profiling_info<sycl::info::event_profiling::command_end>();

      double tTotalKernel = (time_end - time_submit) / 1e9;
      double tCompute = (time_end - time_start) / 1e9;
      double tTotalEvent = (tpAfter - tpBefore).count() / 1e9;
      double tDTH = (tpEndDTH - tpStartDTH).count() / 1e9;
      double tTotal = tDTH + tTotalEvent;

      auto diffAfter = (tpAfter - tpStart).count();
      auto tAfter = diffAfter / 1e9;
      auto bandwidth =  size*N / tCompute;
      aux = std::to_string(tAfter) + " > Kernel times (Total : " + 
                  std::to_string(tTotalKernel) + " s, Compute: " + std::to_string(tCompute) + 
                  " s. Bandwidth: " + std::to_string(bandwidth) + " u/s";
      DEVICE_DEBUG(aux);

      if (cpu) {
        std::lock_guard<std::mutex> lk(opts.mCPU);
        opts.saveWorkPackages(cpu, thr_id, offset, size, tCompute, tTotalKernel, tTotalEvent, tDTH, tTotal);
        opts.workSizeCPU += size;
      } else {
        opts.saveWorkPackages(cpu, 0, offset, size, tCompute, tTotalKernel, tTotalEvent, tDTH, tTotal);
        opts.workSizeAcc += size;
      }
      
    } // size > 0

  } // SYCL scope

  // Time point before end of device process
  auto tpDevEnd = std::chrono::high_resolution_clock::now();

  auto diffSinceStart = (tpDevEnd - tpStart).count(); // Time elapsed since start of program in nanoseconds
  auto tSinceStart = diffSinceStart / 1e9; // Time elapsed since start of program in seconds

  auto diffDevice = (tpDevEnd - tpDevInit).count(); // Time elapsed since start of device process in nanoseconds
  auto tDevice = diffDevice / 1e9; // Time elapsed since start of device process in milliseconds

  std::string aux = std::to_string(tSinceStart) + " end [+" + std::to_string(tDevice) + " s.]";
  DEVICE_DEBUG(aux);

  (cpu) ? opts.tCPUEnd = tDevice : opts.tAccEnd = tDevice;
}
