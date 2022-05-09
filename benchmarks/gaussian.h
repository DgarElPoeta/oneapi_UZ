//
// Created by radon on 17/09/20.
//

#ifndef VECTOR_ADD_THREADS_H
#define VECTOR_ADD_THREADS_H

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

// Gaussian
#include <memory>
#include <ostream>
#include <string>
#include <vector>

#include <cmath>
#include <cstdlib>
#include <omp.h>

#include <stdexcept>

// #include "../dpc_common.hpp"
#include "io.hpp"

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

using sycl::float4;
using sycl::uchar4;
using sycl::uchar;
#define BENCHMARK_GAUSSIAN 1

using std::ostream;
using std::vector;
using std::runtime_error;
using std::cout;

//using sycl::cl_float4;
//using sycl::cl_uchar4;
using sycl::device_selector;
using sycl::cpu_selector;
using sycl::gpu_selector;
using sycl::queue;
// To do Profiling
// using dpc::queue;
using sycl::handler;
using sycl::id;
using sycl::item;
using sycl::nd_item;

#define THRESHOLD 0.51
//#define THRESHOLD 0.10

// I was thinking about extrapolating borders to do a proper gaussian filter for an image,
// but the algo will be changed from the originally defined by AMD, so I skip it
//#define EXTRAPOLATE_BORDERS 1

// To mimic the EngineCL behavior, we send the whole image, if not we will have new borders per every package sent
// but we will need to calculate the proper offset inside the kernel (not as in the range)
#define INPUT_BUFFER_SENT_ALL 1

// here it comes the threshold when operating with floats and doing roundings
template<int T>
float
round_to_decimal(float f)
{
  auto inc = pow(10, T);
  return round((f * inc + 0.5) / inc);
}

class Gaussian
{
 public:
  Gaussian(int width, int height, int filter_width)
      : _width(width)
      , _height(height)
      , _total_size(width * height)
      , _filter_width(filter_width)
      , _filter_total_size(filter_width * filter_width)
  {
    if (filter_width % 2 == 0) {
      throw runtime_error("filter_width should be odd (1, 3, etc)");
    }
  }
  void set_buffers(uchar4* a, float* b, uchar4* c){
    _a = a;
    _b = b;
    _c = c;
  }
  void build(){
    fill_image();
    fill_blurred(_c);
    fill_filter();
  }

  void fill_image();
  void fill_blurred(vector<uchar4>& blurred);
  void fill_blurred(uchar4* blurred);
  void fill_filter();
  //void omp_gaussian_blur();

  bool compare_gaussian_blur(float threshold = THRESHOLD);
//  bool compare_gaussian_blur_2loops(float threshold = THRESHOLD);
//  string get_kernel_str();

  // private:
  int _width;
  int _height;
  size_t _total_size;
  int _filter_width;
  int _filter_total_size;
//#pragma GCC diagnostic ignored "-Wignored-attributes"
//   vector<cl_uchar4> _a; // image
//   vector<cl_float> _b;  // filter
//   vector<cl_uchar4> _c; // blurred
  uchar4* _a;
  float* _b;
  uchar4* _c;
//#pragma GCC diagnostic pop
  // shared_ptr<vector<cl_uchar4>> _a; // image
  // shared_ptr<vector<cl_float>> _b; // filter
  // shared_ptr<vector<cl_uchar4>> _c; // blurred
};

//using namespace std;
//
// enum class Algo {
//   Static, Dynamic, HGuided
// };
//
// enum class Mode {
//   CPU, GPU, CPUGPU
// };

// struct Options {
//   Algo algo;
//   bool debug;
//   int num_pkgs; // dynamic
//   int pkg_size; // old dynamic
//   float cpu_prop;
//   std::chrono::high_resolution_clock::time_point tStart;
//   std::chrono::high_resolution_clock::time_point tLaunchStart;
//
//   float *ptr1;
//   std::mutex *m;
//   int *p_rest_size;
//   int *p_offset;
//   int *p_pkgid;
//
//   int min_multiplier_cpu; // hguided
//   int min_multiplier_gpu;
//
//   // std::mutex *m_profiling_acc;
//   // Profiling times:
//   double *cpu_end;
//   double *gpu_end;
//   double *compute_cpu;
//   double *compute_gpu;
//   double *profiling_q_cpu;
//   double *profiling_q_gpu;
// };

//using namespace cl::sycl;

#endif //VECTOR_ADD_THREADS_H
