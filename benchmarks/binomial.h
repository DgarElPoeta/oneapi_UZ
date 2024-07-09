#ifndef VECTOR_ADD_RAP_H
#define VECTOR_ADD_RAP_H

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

// Really important:
using sycl::float4;
#define BENCHMARK_BINOMIAL 1

using std::ostream;
using std::vector;
using std::runtime_error;
using std::cout;

using sycl::float4;
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

/**
 * \RISKFREE 0.02f
 * \brief risk free interest rate.
 */
#define RISKFREE 0.02f

/**
 * \VOLATILITY 0.30f
 * \brief Volatility factor for Binomial Option Pricing.
 */
#define VOLATILITY 0.30f

typedef float4 ptype;

struct Binomial {
  ptype* a;
  ptype* b;
  size_t size; // gws, all work-items
  size_t workgroups; // workgroups = samplesPerVectorWidth
  size_t steps1; // lws
  size_t steps;
};
//using namespace cl::sycl;

#endif //VECTOR_ADD_RAP_H

