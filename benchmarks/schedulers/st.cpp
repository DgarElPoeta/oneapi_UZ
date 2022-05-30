//
// Created by radon on 9/10/20.
//

#include "schedulers.h"

void my_list_devices(){
    // Loop through available platforms
    for (auto const& this_platform : sycl::platform::get_platforms() ) {
        std::cout << "Found platform: "
            << this_platform.get_info<sycl::info::platform::name>() << "\n";
        // Loop through available devices in this platform
        for (auto const& this_device : this_platform.get_devices() ) {
            std::cout << " Device: "
                << this_device.get_info<sycl::info::device::name>() << "\n";
        }
        std::cout << "\n";
    }
}

void process_static(bool cpu, Options* opts, int thr_id) {

  char *kernel_reps_str = getenv("KERNEL_REPS");
  int  kernel_reps = 1;
  if (kernel_reps_str != nullptr) {
    int kernel_reps_ = std::stoi(kernel_reps_str);
    kernel_reps = kernel_reps_;
  }
  //my_list_devices();
  string device_type = ((cpu) ? "CPU" : "GPU");
  bool debug = opts->debug;
  auto tStart = opts->tStart;
  auto tDevInit = opts->tLaunchStart;
  { // SYCL scope

    //- float *ptr1 = opts->ptr1;
    //- std::mutex *m = opts->m;
    //- size_t *p_offset = opts->p_offset;
    //- int pkg_size = opts->pkg_size;
    //- int *pkgid = opts->p_pkgid;
    //- auto lws = opts->lws; // TODO: LWS common

    size_t *p_rest_size = opts->p_rest_size;
    size_t size = *opts->p_rest_size;
    float cpu_prop = opts->cpu_prop;
    size_t pSizeCpu = size * cpu_prop;
    auto pkg_size_multiple = opts->pkg_size_multiple;
    size_t sizeCpu = pSizeCpu;
    bool trimmed = false;
    int num_cpp_threads = opts->num_cpp_threads;

    // let's approximate to multiple of 128
    if (cpu_prop != 1.0) {
      size_t multiples = pSizeCpu / pkg_size_multiple;
      sizeCpu = (multiples * pkg_size_multiple);
      if (sizeCpu != pSizeCpu){
        trimmed = true;
      }
    }

    size_t sizeGpu = size - sizeCpu;
    size_t offset = 0;

    sizeCpu = sizeCpu / num_cpp_threads;

    size = ((cpu) ? sizeCpu : sizeGpu);
    offset = ((cpu) ? sizeGpu + sizeCpu*thr_id : 0);

    if(cpu && thr_id+1 == num_cpp_threads && offset+sizeCpu != *p_rest_size) size = *p_rest_size - offset;

    DEVICE_DEBUG("selected");
    if(trimmed) DEVICE_DEBUG("trimmed");

    if (size > 0) {

      int DB = 0;
      #if BENCHMARK_MATADD == 1 || BENCHMARK_MATMUL == 1
        std::unique_ptr<sycl::buffer<ptype, 2>> buf_a[1];
        std::unique_ptr<sycl::buffer<ptype, 2>> buf_b[1];
        std::unique_ptr<sycl::buffer<ptype, 2>> buf_c[1];
      #elif BENCHMARK_RAP == 1
        std::unique_ptr<sycl::buffer<ptype, 1>> buf_a[1];
        std::unique_ptr<sycl::buffer<ptype, 1>> buf_b[1];
        std::unique_ptr<sycl::buffer<ptype, 1>> buf_func[1];
      #elif BENCHMARK_NBODY == 1
        std::unique_ptr<sycl::buffer<ptype, 1>> buf_pos_in[1];
        std::unique_ptr<sycl::buffer<ptype, 1>> buf_vel_in[1];
        std::unique_ptr<sycl::buffer<ptype, 1>> buf_pos_out[1];
        std::unique_ptr<sycl::buffer<ptype, 1>> buf_vel_out[1];
      #elif BENCHMARK_GAUSSIAN == 1
        std::unique_ptr<sycl::buffer<uchar4, 1>> buf_input[1];
        std::unique_ptr<sycl::buffer<float, 1>> buf_filterWeight[1];
        std::unique_ptr<sycl::buffer<uchar4, 1>> buf_blurred[1];
      #endif

      auto tBefore = std::chrono::high_resolution_clock::now();
      auto diffBefore = (tBefore - tStart).count();
      auto diffBeforeS = diffBefore / 1e9;
      DEVICE_DEBUG(diffBeforeS << " size: " << size << " offset: " << offset);

      sycl::event submit_event;
      queue q;
      cl::sycl::property_list prop_list =
          cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};

      if(!cpu){
        queue q = fpga_QUEUE;
      }
      else {
        queue q = cpu_QUEUE;
      }
      //queue q = queue(sycl::host_selector{}, async_exception_handler);
      //DEBUG( "Device is: " << q.get_device().get_info<sycl::info::device::name>().c_str() );

      #if BENCHMARK_MATADD == 1
      #include "../matadd/setup_sycl.cpp"
      #elif BENCHMARK_MATMUL == 1
      #include "../matmul/setup_sycl.cpp"
      #elif BENCHMARK_RAP == 1
      #include "../rap/setup_sycl.cpp"
      #elif BENCHMARK_NBODY == 1
      #include "../nbody/setup_sycl.cpp"
      #elif BENCHMARK_GAUSSIAN == 1
      #include "../gaussian/setup_sycl.cpp"
      #elif BENCHMARK_RAY == 1
      #include "../ray/setup_sycl.cpp"
      #elif BENCHMARK_MANDELBROT == 1
      #include "../mandelbrot/setup_sycl.cpp"
      #elif BENCHMARK_BINOMIAL == 1
      #include "../binomial/setup_sycl.cpp"
      #endif


      tBefore = std::chrono::high_resolution_clock::now();
      for(int kerns=0; kerns<kernel_reps; kerns++){
      submit_event =
        #if BENCHMARK_MATADD == 1
        ((cpu) ? cpu_submitKernel(q, *buf_a[DB], *buf_b[DB], *buf_c[DB], size_range, offset)
         : fpga_submitKernel(q, *buf_a[DB], *buf_b[DB], *buf_c[DB], size_range, offset));
        #elif BENCHMARK_MATMUL == 1
        ((cpu) ? cpu_submitKernel(q, *buf_a[DB], *buf_b[DB], *buf_c[DB], size_range, offset, N)
         : fpga_submitKernel(q, *buf_a[DB], *buf_b[DB], *buf_c[DB], size_range, offset, N));
        #elif BENCHMARK_RAP == 1
        ((cpu) ? cpu_submitKernel(q, *buf_a[DB], *buf_b[DB], *buf_func[DB], size_range, offset, M, Rw)
         : fpga_submitKernel(q, *buf_a[DB], *buf_b[DB], *buf_func[DB], size_range, offset, M, Rw));
        #elif BENCHMARK_NBODY == 1
        ((cpu) ? cpu_submitKernel(q, *buf_pos_in[DB], *buf_vel_in[DB], *buf_pos_out[DB], *buf_vel_out[DB], size_range, offset, numBodies, epsSqr, deltaTime)
         : fpga_submitKernel(q, *buf_pos_in[DB], *buf_vel_in[DB], *buf_pos_out[DB], *buf_vel_out[DB], size_range, offset, numBodies, epsSqr, deltaTime));
        #elif BENCHMARK_GAUSSIAN == 1
        ((cpu) ? cpu_submitKernel(q, *buf_input[DB], *buf_filterWeight[DB], *buf_blurred[DB], size_range, offset, rows, cols, filterWidth)
         : fpga_submitKernel(q, *buf_input[DB], *buf_filterWeight[DB], *buf_blurred[DB], size_range, offset, rows, cols, filterWidth));
        #elif BENCHMARK_RAY == 1
        ((cpu) ? cpu_submitKernel(q, buf_prim, buf_pixels, width, height, size_range, offset, camera_x, camera_y, camera_z, viewport_x, viewport_y, prim_ptr, n_primitives)
         : fpga_submitKernel(q, buf_prim, buf_pixels, width, height, size_range, offset, camera_x, camera_y, camera_z, viewport_x, viewport_y, prim_ptr, n_primitives));
        #elif BENCHMARK_MANDELBROT == 1
        ((cpu) ? cpu_submitKernel(q, buf_out, size_range, offset, leftxF, topyF, xstepF, ystepF, max_iterations, numDevices, bench, width)
         : fpga_submitKernel(q, buf_out, size_range, offset, leftxF, topyF, xstepF, ystepF, max_iterations, numDevices, bench, width));
        #elif BENCHMARK_BINOMIAL == 1
        ((cpu) ? cpu_submitKernel(q, buf_a, buf_b, workgroups, steps, steps1)
         : fpga_submitKernel(q, buf_a, buf_b, workgroups, steps, steps1));
        #endif
      }
      submit_event.wait();

      auto tAfter = std::chrono::high_resolution_clock::now();
      //- pkgdevid++;

      cl_ulong time_start, time_end, time_submit;
      double profilingQueueMs = 0.0f;
      auto profiling = false;
      if (profiling && !cpu) {
        time_submit = submit_event.get_profiling_info<sycl::info::event_profiling::command_submit>();
        time_start = submit_event.get_profiling_info<sycl::info::event_profiling::command_start>();
        time_end = submit_event.get_profiling_info<sycl::info::event_profiling::command_end>();
        profilingQueueMs = (time_end - time_start) / 1e6;

        DEVICE_DEBUG( " command queue [+" << profilingQueueMs / 1e3 << " s.]" );
        DEBUG("submit->start " << (time_start - time_submit) / 1e6 / 1e3);
        DEBUG("start->end " << (time_end - time_start) / 1e6 / 1e3);
        DEBUG("submit->end  " << (time_end - time_submit) / 1e6 / 1e3);
      }


      auto diffAfter = (tAfter - tStart).count();
      // Compute is Duration data movement/computation
      auto diffCompute = (tAfter - tBefore).count();
      auto diffAfterS = diffAfter / 1e9;
      auto diffComputeMs = diffCompute / 1e6;
      auto bandwidthMS = (float) size / ((float) diffCompute / 1e6);
      DEVICE_DEBUG( diffAfterS << " > (Compute: " << diffComputeMs / 1e3
                    << " s. Bandwidth: " << bandwidthMS << " u/ms)" );

      if (cpu) {
        auto compute_cpu = *opts->compute_cpu;
        auto profiling_q_cpu = *opts->profiling_q_cpu;
        *opts->compute_cpu = compute_cpu + diffComputeMs;
        *opts->profiling_q_cpu = profiling_q_cpu + profilingQueueMs;
        opts->saveChunk(cpu, offset, size, diffComputeMs);
        opts->worksizeCPU += size;
      } else {
        auto compute_gpu = *opts->compute_gpu;
        *opts->compute_gpu = compute_gpu + diffComputeMs;
        auto profiling_q_gpu = *opts->profiling_q_gpu;
        *opts->profiling_q_gpu = profiling_q_gpu + profilingQueueMs;
        opts->saveChunk(cpu, offset, size, diffComputeMs);
        opts->worksizeGPU += size;
      }
    } // size > 0
  } // SYCL scope

  auto tDevEnd = std::chrono::high_resolution_clock::now();
  auto diffDevEnd = (tDevEnd - tStart).count();
  auto diffDevEndMs = diffDevEnd / 1e6;

  auto diffDevDuration = (tDevEnd - tDevInit).count();
  auto diffDevDurationMs = diffDevDuration / 1e6;

  (cpu) ? *opts->cpu_end = diffDevDurationMs : *opts->gpu_end = diffDevDurationMs;
  DEVICE_DEBUG( diffDevEndMs / 1e3 << " end [+" << diffDevDurationMs / 1e3 <<" s.]");
}
