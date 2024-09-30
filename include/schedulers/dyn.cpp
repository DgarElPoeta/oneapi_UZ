#include "benchmarks.h"
#include "schedulers.h"
#include "kernels.h"
#include "queues.h"

template <typename T>
void process_dynamic(bool cpu, Options<T>& opts, uint32_t thr_id) {

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
    
    DEVICE_DEBUG("selected");

    uint64_t total_size = opts.pTotalSize;
    uint64_t pkg_size = total_size / opts.numPkgs;
    uint64_t pkg_size_multiple = opts.sizeMultiple;
    if(pkg_size < pkg_size_multiple) pkg_size = pkg_size_multiple;
    else pkg_size = ( (pkg_size / pkg_size_multiple)) * pkg_size_multiple;  

    bool work = true;

    uint64_t wgs = opts.wgs;

    // Buffers and kernels management variables
    size_t CK = 0; // Index of the current kernel
    constexpr size_t num_kernels = 2; // Total number of kernels that can be active at the same time
    uint64_t sent_kernels = 0; // Total number of kernels that have been sent
    uint64_t active_kernels = 0; // Number of active kernels
    sycl::event submit_event[num_kernels]; // Array with the events of the kernels

    // Include the file that defines the buffers used in the kernels.
    #include "buffers_sycl.cpp"

    std::vector<uint64_t> sizeV(num_kernels), offsetV(num_kernels), pkgDevV(num_kernels), pkgV(num_kernels);
    std::vector<std::chrono::high_resolution_clock::time_point> tpV(num_kernels);
    while (work) {
      uint64_t size = 0;
      uint64_t offset = 0;
      uint64_t pkg = 0;
      uint64_t pkgdevid = 0;
      bool firstCPU = false;
      {
        std::lock_guard<std::mutex> lk(opts.mWork);
        uint64_t pWork = *(opts.pWork);
        uint64_t rest_size = opts.pTotalSize - pWork;
        pkg = *(opts.pPkg);
        if(cpu) pkgdevid = *(opts.pPkgCPU);
        else pkgdevid = *(opts.pPkgAcc);
        
        if (rest_size > 0) {
          offset = pWork;
          if (rest_size >= pkg_size) {
            size = pkg_size;
          } else {
            size = rest_size;
            work = false;
          }
          *(opts.pWork) += size;
          pkgV[CK] = pkg;
          *(opts.pPkg) = pkg+1;
          pkgDevV[CK] = pkgdevid;
          if (cpu) {
            *(opts.pPkgCPU) = pkgdevid+1;
            if(opts.firstCPU) {
              firstCPU = true;
              opts.firstCPU = false;
            }
          }
          else *(opts.pPkgAcc) = pkgdevid+1;
          sizeV[CK] = size;
          offsetV[CK] = offset;
        } else {
          work = false;
          continue;
        }
      }

      // Include the file that setups the buffers and sycl variables used in the kernel.
      #include "setup_sycl.cpp"

      auto tpBefore = std::chrono::high_resolution_clock::now();
      tpV[CK] = tpBefore;
      if(firstCPU){
        opts.tpFirstCPU = tpBefore;
      }
      else if(!cpu && pkgdevid == 0){
        opts.tpFirstAcc = tpBefore;
      }
      auto diffBefore = (tpBefore - tpStart).count();
      auto tBefore = diffBefore / 1e9;
      std::string aux = std::to_string(tBefore) + " < [" + std::to_string(pkg) + "] (" + std::to_string(pkgdevid) + ") size : " + std::to_string(size) + " offset : " + std::to_string(offset);
      DEVICE_DEBUG(aux);

      // Include the file that invokes the kernel
      #include "kernel_sycl.cpp"
      
      active_kernels++; // Increments the number of active buffers

      if(++CK == num_kernels) CK = 0; // Increments the buffer index

      // If the number of sent buffers is less than the total number of buffers, continue
      if(++sent_kernels < num_kernels) continue;

      submit_event[CK].wait();

      active_kernels--; // Decrements the number of active buffers

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

      auto time_submit = submit_event[CK].get_profiling_info<sycl::info::event_profiling::command_submit>();
      auto time_start = submit_event[CK].get_profiling_info<sycl::info::event_profiling::command_start>();
      auto time_end = submit_event[CK].get_profiling_info<sycl::info::event_profiling::command_end>();

      double tTotalKernel = (time_end - time_submit) / 1e9;
      double tCompute = (time_end - time_start) / 1e9;
      double tTotalEvent = (tpAfter - tpV[CK]).count() / 1e9;
      double tDTH = (tpEndDTH - tpStartDTH).count() / 1e9;
      double tTotal = tDTH + tTotalEvent;

      auto diffAfter = (tpAfter - tpStart).count();
      auto tAfter = diffAfter / 1e9;
      auto bandwidth =  sizeV[CK] / tCompute;

      aux = std::to_string(tAfter) + " > [" + std::to_string(pkgV[CK]) +"] Kernel times (Total : " + 
                  std::to_string(tTotalKernel) + " s, Compute: " + std::to_string(tCompute) + 
                  " s. Bandwidth: " + std::to_string(bandwidth) + " u/s";
      DEVICE_DEBUG(aux);

      if (cpu) {
        std::lock_guard<std::mutex> lk(opts.mCPU);
        opts.saveWorkPackages(cpu, pkgDevV[CK], offsetV[CK], sizeV[CK], tCompute, tTotalKernel, tTotalEvent, tDTH, tTotal);
        opts.workSizeCPU += sizeV[CK];
      } else {
        opts.saveWorkPackages(cpu, pkgDevV[CK], offsetV[CK], sizeV[CK], tCompute, tTotalKernel, tTotalEvent, tDTH, tTotal);
        opts.workSizeAcc += sizeV[CK];
      }

    } // continue next packages
    if(sent_kernels < num_kernels) CK = 0;
    else CK = (CK + 1) % num_kernels;
    for(size_t i=0; i<active_kernels; i++){
      size_t eventIndex = (CK+i) % num_kernels;
      submit_event[eventIndex].wait();
      auto tpAfter = std::chrono::high_resolution_clock::now();
      auto tpStartDTH = std::chrono::high_resolution_clock::now();
      auto diffStartDTH = (tpStartDTH - tpStart).count();
      auto tStartDTH = diffStartDTH / 1e9;
      std::string aux = std::to_string(tStartDTH) + " : Start of data transfer from device to host";
      DEVICE_DEBUG(aux);
      
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

      uint64_t pkgdevid = pkgDevV[eventIndex], size = sizeV[eventIndex], offset = offsetV[eventIndex];

      auto tpBefore = tpV[eventIndex];

      auto time_submit = submit_event[eventIndex].get_profiling_info<sycl::info::event_profiling::command_submit>();
      auto time_start = submit_event[eventIndex].get_profiling_info<sycl::info::event_profiling::command_start>();
      auto time_end = submit_event[eventIndex].get_profiling_info<sycl::info::event_profiling::command_end>();

      double tTotalKernel = (time_end - time_submit) / 1e9;
      double tCompute = (time_end - time_start) / 1e9;
      double tTotalEvent = (tpAfter - tpBefore).count() / 1e9;
      double tDTH = (tpEndDTH - tpStartDTH).count() / 1e9;
      double tTotal = tDTH + tTotalEvent;

      auto diffAfter = (tpAfter - tpStart).count();
      auto tAfter = diffAfter / 1e9;
      auto bandwidth =  size / tCompute;

      aux = std::to_string(tAfter) + " >[" + std::to_string(pkgV[CK]) +"] Kernel times (Total : " + 
                  std::to_string(tTotal) + " s, Compute: " + std::to_string(tCompute) + 
                  " s. Bandwidth: " + std::to_string(bandwidth) + " u/s";
      DEVICE_DEBUG(aux);
      
      if (cpu) {
        std::lock_guard<std::mutex> lk(opts.mCPU);
        opts.saveWorkPackages(cpu, pkgdevid, offset, size, tCompute, tTotalKernel, tTotalEvent, tDTH, tTotal);
        opts.workSizeCPU += size;
      } else {
        opts.saveWorkPackages(cpu, pkgdevid, offset, size, tCompute, tTotalKernel, tTotalEvent, tDTH, tTotal);
        opts.workSizeAcc += size;
      }
    }
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
