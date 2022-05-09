#include "schedulers.h"

void process_dynamic(bool cpu, Options* opts) {

  char *kernel_reps_str = getenv("KERNEL_REPS");
  int  kernel_reps = 1;
  if (kernel_reps_str != nullptr) {
    int kernel_reps_ = std::stoi(kernel_reps_str);
    kernel_reps = kernel_reps_;
  }

  string device_type = ((cpu) ? "CPU" : "GPU");
  bool debug = opts->debug;
  auto tStart = opts->tStart;
  auto tDevInit = opts->tLaunchStart;
  { // SYCL scope

    //- float *ptr1 = opts->ptr1;
    std::mutex *m = opts->m;
    int num_pkgs = opts->num_pkgs;
    int pkg_size = opts->pkg_size;

    size_t *p_rest_size = opts->p_rest_size;
    size_t *p_offset = opts->p_offset;
    int *pkgid = opts->p_pkgid;

    int total_size = opts->p_total_size;
    int pkg_size_calc = total_size / num_pkgs;
    auto lws = opts->lws;
    auto pkg_size_multiple = opts->pkg_size_multiple;
    pkg_size = ((int) (pkg_size_calc / pkg_size_multiple)) * pkg_size_multiple;
    // total_size % pkg_size_calc;

    DEVICE_DEBUG("selected");

    queue q;
    cl::sycl::property_list prop_list =
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
    if(!cpu){
      q = fpga_QUEUE;
    }
    else{
      q = cpu_QUEUE;
    }

    DEBUG( "Device is " << q.get_device().get_info<sycl::info::device::name>().c_str());
    bool cont = true;

    // int min_split = psize * 0.5 > 256 ? psize * 0.5 : 256;
    int min_split = pkg_size;

    int pkgdevid = 1;

    int DB = 0;
    int num_buffers = 2;
    int sent_buffers = 1;

    #if BENCHMARK_MATADD == 1 || BENCHMARK_MATMUL == 1
      std::unique_ptr<sycl::buffer<ptype, 2>> buf_a[num_buffers];
      std::unique_ptr<sycl::buffer<ptype, 2>> buf_b[num_buffers];
      std::unique_ptr<sycl::buffer<ptype, 2>> buf_c[num_buffers];
    #elif BENCHMARK_RAP == 1
      std::unique_ptr<sycl::buffer<ptype, 1>> buf_a[num_buffers];
      std::unique_ptr<sycl::buffer<ptype, 1>> buf_b[num_buffers];
      std::unique_ptr<sycl::buffer<ptype, 1>> buf_func[num_buffers];
    #elif BENCHMARK_NBODY == 1
      std::unique_ptr<sycl::buffer<ptype, 1>> buf_pos_in[num_buffers];
      std::unique_ptr<sycl::buffer<ptype, 1>> buf_vel_in[num_buffers];
      std::unique_ptr<sycl::buffer<ptype, 1>> buf_pos_out[num_buffers];
      std::unique_ptr<sycl::buffer<ptype, 1>> buf_vel_out[num_buffers];
    #elif BENCHMARK_GAUSSIAN == 1
      std::unique_ptr<sycl::buffer<uchar4, 1>> buf_input[num_buffers];
      std::unique_ptr<sycl::buffer<float, 1>> buf_filterWeight[num_buffers];
      std::unique_ptr<sycl::buffer<uchar4, 1>> buf_blurred[num_buffers];
    #else
      num_buffers = 1;
    #endif

    sycl::event submit_event[2];

    while (cont) {
      size_t size = 0;
      size_t offset = 0;
      int pkg = 0;
      {
        std::lock_guard<std::mutex> lk(*m);
        size_t rest_size = *p_rest_size;
        offset = *p_offset;
        pkg = *pkgid;
        *pkgid = pkg + 1;
        if (rest_size > 0) {
          if (rest_size >= min_split) {
            size = min_split;
            *p_rest_size -= min_split;
            *p_offset += min_split;
          } else {
            size = rest_size;
            *p_rest_size = 0;
            *p_offset += rest_size;
          }
        } else {
          cont = false;
          continue;
        }
      }

      auto tBefore = std::chrono::high_resolution_clock::now();
      auto diffBefore = (tBefore - tStart).count();
      auto diffBeforeS = diffBefore / 1e9;
      DEVICE_DEBUG(diffBeforeS << " <[" << pkg << "] (" << pkgdevid << ") size: " << size << " offset: " << offset);

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
      submit_event[DB] =
        #if BENCHMARK_MATADD == 1
        submitKernel(q, *buf_a[DB], *buf_b[DB], *buf_c[DB], size_range, offset);
        #elif BENCHMARK_MATMUL == 1
        submitKernel(q, *buf_a[DB], *buf_b[DB], *buf_c[DB], size_range, offset, N);
        #elif BENCHMARK_RAP == 1
        submitKernel(q, *buf_a[DB], *buf_b[DB], *buf_func[DB], size_range, offset, M, Rw);
        #elif BENCHMARK_NBODY == 1
        submitKernel(q, *buf_pos_in[DB], *buf_vel_in[DB], *buf_pos_out[DB], *buf_vel_out[DB], size_range, offset, numBodies, epsSqr, deltaTime);
        #elif BENCHMARK_GAUSSIAN == 1
        submitKernel(q, *buf_input[DB], *buf_filterWeight[DB], *buf_blurred[DB], size_range, offset, rows, cols, filterWidth);
        #elif BENCHMARK_RAY == 1
        submitKernel(q, buf_prim, buf_pixels, width, height, size_range, offset, camera_x, camera_y, camera_z, viewport_x, viewport_y, prim_ptr, n_primitives);
        #elif BENCHMARK_MANDELBROT == 1
        submitKernel(q, buf_out, size_range, offset, leftxF, topyF, xstepF, ystepF, max_iterations, numDevices, bench, width);
        #elif BENCHMARK_BINOMIAL == 1
        submitKernel(q, buf_a, buf_b, workgroups, steps, steps1);
        #endif
      }
      DB = (DB+1) % num_buffers;
      if(sent_buffers < num_buffers){
        auto compute_gpu = *opts->compute_gpu;
        *opts->compute_gpu = compute_gpu + 0;
        auto profiling_q_gpu = *opts->profiling_q_gpu;
        *opts->profiling_q_gpu = profiling_q_gpu + 0;
        opts->saveChunk(cpu, offset, size, 0);
        opts->worksizeGPU += size;

        sent_buffers++;
        continue;
      }
      submit_event[DB].wait();

      auto tAfter = std::chrono::high_resolution_clock::now();
      pkgdevid++;

      cl_ulong time_start, time_end;
      double profilingQueueMs = 0.0f;
      auto profiling = false;
      if (profiling && !cpu) {
        time_start = submit_event[DB].get_profiling_info<sycl::info::event_profiling::command_start>();
        time_end = submit_event[DB].get_profiling_info<sycl::info::event_profiling::command_end>();
        profilingQueueMs = (time_end - time_start) / 1e6;

        DEVICE_DEBUG( " command queue [+" << profilingQueueMs / 1e3 << " s.]" );
      }

      auto diffAfter = (tAfter - tStart).count();
      auto diffCompute = (tAfter - tBefore).count();
      auto diffAfterS = diffAfter / 1e9;
      auto diffComputeMs = diffCompute / 1e6;
      auto bandwidthMS = (float) size / ((float) diffCompute / 1e6);

      DEVICE_DEBUG(diffAfterS << " >[" << pkg << "] (Compute: " << diffComputeMs / 1e3
                    << " s. Bandwidth: " << bandwidthMS << " u/ms)");

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

    } // continue next packages
    if(!cpu){
      for(int i=1; i<num_buffers; i++)
        submit_event[(DB+i) % num_buffers].wait();
      /*auto compute_gpu = *opts->compute_gpu;
      *opts->compute_gpu = compute_gpu + 0;
      auto profiling_q_gpu = *opts->profiling_q_gpu;
      *opts->profiling_q_gpu = profiling_q_gpu + 0;
      opts->saveChunk(cpu, offset, size, 0);
      opts->worksizeGPU += size;*/
    }
  }
  auto tDevEnd = std::chrono::high_resolution_clock::now();
  auto diffDevEnd = (tDevEnd - tStart).count();
  auto diffDevEndMs = diffDevEnd / 1e6;

  auto diffDevDuration = (tDevEnd - tDevInit).count();
  auto diffDevDurationMs = diffDevDuration / 1e6;

  (cpu) ? *opts->cpu_end = diffDevDurationMs : *opts->gpu_end = diffDevDurationMs;
  DEVICE_DEBUG( diffDevEndMs / 1e3 << " end [+" << diffDevDurationMs / 1e3 <<" s.]");
}
