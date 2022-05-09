//
// Created by radon on 9/10/20.
//

#ifndef VECTOR_ADD_BENCHMARKS_H
#define VECTOR_ADD_BENCHMARKS_H

#include <CL/sycl.hpp>
//#include "defs.h"

#include <iostream>
#include <iomanip>
#include <ctime>
#include <sys/time.h>
//#include <cstdlib>
//#include "sycl_exceptions.hpp"

#include <chrono>
#include <array>
#include <mutex>
#include <condition_variable>
#include <string>
#include <thread>

#include "io.hpp"

enum class Algo {
  Static, Dynamic, HGuided
};

enum class Mode {
  CPU, GPU, CPUGPU
};

struct Chunk
{
  size_t offset;
  size_t size;
  size_t ts_ms;
  size_t duration_ms;
  size_t duration_read_ms;
  Chunk() {}
  Chunk(size_t _offset, size_t _size, size_t _ts_ms, size_t _duration_ms, size_t _duration_read_ms)
  {
    offset = _offset;
    size = _size;
    ts_ms = _ts_ms;
    duration_ms = _duration_ms;
    duration_read_ms = _duration_read_ms;
  }
};


struct Options {
  Algo algo;
  Mode mode;
  bool debug;
  int num_pkgs; // dynamic
  int pkg_size; // old dynamic
  float cpu_prop;
  std::chrono::high_resolution_clock::time_point tStart;
  std::chrono::high_resolution_clock::time_point tLaunchStart;

  vector<Chunk> mChunksCPU;
  vector<Chunk> mChunksGPU;
  size_t worksizeCPU;
  size_t worksizeGPU;
  float *ptr1;
  std::mutex *m;

  size_t p_total_size; // To avoid barrier
  int num_cpp_threads;


  size_t *p_rest_size;
  size_t *p_offset;
  int *p_pkgid;

  int min_multiplier_cpu; // hguided
  int min_multiplier_gpu;

  // std::mutex *m_profiling_acc;
  // Profiling times:
  double *cpu_end;
  double *gpu_end;
  double *compute_cpu;
  double *compute_gpu;
  double *profiling_q_cpu;
  double *profiling_q_gpu;

  void* p_problem; // eg. Matmul*, Gaussian*
  size_t lws; // usually this means the min package or multiple of size
  int pkg_size_multiple; // it will split and give `size` and `offset`
  bool usm;
  sycl::queue gpuQ; // for usm

  float K; // hguided

  Options() : usm(false), lws(128), pkg_size_multiple(lws){
  }
  void setup(){
    int reserve = 120;
    if (algo == Algo::Static){
      reserve = 2;
    }
    if (mode == Mode::CPU || mode == Mode::CPUGPU){
      mChunksCPU.reserve(reserve);
    }
    if (mode == Mode::GPU || mode == Mode::CPUGPU){
      mChunksGPU.reserve(reserve);
    }
    worksizeCPU = 0;
    worksizeGPU = 0;
  }
  void
  saveChunk(bool cpu, size_t offset, size_t size, size_t duration_ms)
  {
    // auto t2 = std::chrono::system_clock::now().time_since_epoch();
    auto t2 = std::chrono::high_resolution_clock::now().time_since_epoch();
    size_t diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - tStart.time_since_epoch()).count();
    // size_t duration_ms = diff_ms - ts_ms;
    // printf("saveWriteChunk %lu = %lu ms\n", size, diff_ms);
    // return move(Chunk(offset, size, ts_ms, duration_ms, 0));
    if (cpu){
      mChunksCPU.push_back(Chunk(offset, size, diff_ms, duration_ms, 0));
    } else {
      mChunksGPU.push_back(Chunk(offset, size, diff_ms, duration_ms, 0));
    }
  }
};

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

using std::ostream;
using std::vector;
using std::runtime_error;
using std::cout;

// using sycl::cl_float4;
// using sycl::cl_uchar4;
using sycl::device_selector;
using sycl::cpu_selector;
using sycl::host_selector;
using sycl::gpu_selector;
using sycl::queue;
// To do Profiling
// using dpc::queue;
using sycl::handler;
using sycl::id;
using sycl::item;
using sycl::nd_item;

static void async_exception_handler(cl::sycl::exception_list exceptions)
// auto async_exception_handler = [] (cl::sycl::exception_list exceptions) {
{
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    }
    catch (cl::sycl::exception const &e) {
      std::cout << "Async Exception: " << e.what() << std::endl;
      std::terminate();
    }
  }
};
#endif //VECTOR_ADD_BENCHMARKS_H
