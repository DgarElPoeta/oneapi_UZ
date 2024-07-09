#ifndef VECTOR_ADD_MATMUL_H
#define VECTOR_ADD_MATMUL_H

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

#define BENCHMARK_MATMUL 1

using std::ostream;
using std::vector;
using std::runtime_error;
using std::cout;

// using sycl::cl_float4;
// using sycl::cl_uchar4;
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

typedef float ptype;

struct Matmul {
  ptype* a;
  ptype* b;
  ptype* c;
  size_t size;
};
//using namespace cl::sycl;

#endif //VECTOR_ADD_MATMUL_H

