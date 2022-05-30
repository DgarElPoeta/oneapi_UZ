//
// Created by radon on 9/10/20.
//
#include "schedulers.h"

void process_hguided(bool cpu, Options* opts) {

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

    float *ptr1 = opts->ptr1;
    std::mutex *m = opts->m;
    size_t *p_rest_size = opts->p_rest_size;
    size_t *p_offset = opts->p_offset;
    int pkg_size = opts->pkg_size;
    int *pkgid = opts->p_pkgid;
    float cpu_prop = opts->cpu_prop;
    int minMultiplier = cpu ? opts->min_multiplier_cpu : opts->min_multiplier_gpu;
    int num_cpp_threads = opts->num_cpp_threads;

    DEVICE_DEBUG("selected");

    // queue q = (cpu) ? queue(cpu_selector{}, prop_list) : queue(gpu_selector{}, prop_list);
    queue q;
    cl::sycl::property_list prop_list =
        cl::sycl::property_list{cl::sycl::property::queue::enable_profiling()};
    if(!cpu){
      q = fpga_QUEUE;
    }
    else{
      q = cpu_QUEUE;
    }

    DEBUG( "GPU is " << q.get_device().get_info<sycl::info::device::name>().c_str());

    bool cont = true;

    // int min_split = psize * 0.5 > 256 ? psize * 0.5 : 256;
    auto lws = opts->lws;
    auto pkg_size_multiple = opts->pkg_size_multiple;

    int pkgdevid = 1;
    float K = opts->K; //2.0; // HGuided
    float computePower = (cpu) ? cpu_prop / num_cpp_threads : (1.0 - cpu_prop);

    // int min_split = computePower * pkg_size;
    int min_split = pkg_size_multiple * minMultiplier;

    int DB = 0;
    #define num_buffers 2
    int sent_buffers = 1;
    sycl::event submit_event[num_buffers];

    #if BENCHMARK_MATADD == 1 || BENCHMARK_MATMUL == 1
      std::unique_ptr<sycl::buffer<ptype, 2>> buf_a[num_buffers];
      std::unique_ptr<sycl::buffer<ptype, 2>> buf_b[num_buffers];
      std::unique_ptr<sycl::buffer<ptype, 2>> buf_c[num_buffers];
    #elif BENCHMARK_RAP == 1
      std::unique_ptr<sycl::buffer<ptype, 1>> buf_a[1];
      std::unique_ptr<sycl::buffer<ptype, 1>> buf_b[1];
      std::unique_ptr<sycl::buffer<ptype, 1>> buf_func[1];
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
      sent_buffers = num_buffers;
    #endif

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

            // Work.hpp (splitWOrkLikeHGuided)
            size_t total = rest_size;
            size_t ret = (float) rest_size * computePower / K;
            size_t mult = ret / pkg_size_multiple; // LWS
            size_t rem = static_cast<int>(ret) % pkg_size_multiple;
            if (rem) {
              mult++;
            }
            ret = mult * pkg_size_multiple;
            if (ret < min_split) {
              ret = min_split;
            }
            if (total < ret) {
              ret = total;
            }
            // End Work.hpp

            size = ret;
            *p_rest_size -= ret;
            *p_offset += ret;
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
//          next_offset = offset + size;

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
      // Compute is Duration data movement/computation
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
//            offset += size;

    } // continue next packages
    if(!cpu){
      for(int i=1; i<num_buffers; i++)
        submit_event[(DB+i) % num_buffers].wait();
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
