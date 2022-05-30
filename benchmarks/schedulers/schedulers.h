//
// Created by radon on 9/10/20.
//

#ifndef VECTOR_ADD_SCHEDULERS_H
#define VECTOR_ADD_SCHEDULERS_H

#include "../benchmarks.h"
#include <CL/sycl/INTEL/fpga_extensions.hpp>

#define DEBUG(x) do { \
  if (debug) { std::cout << x << std::endl; } \
} while (0)

#define DEVICE_DEBUG(x) do { \
  if (debug) { std::cout << device_type << ": " << x << std::endl; } \
} while (0)

#define PRINT_TIME do { \
   auto tBefore = std::chrono::high_resolution_clock::now();\
   auto diffBefore = (tBefore - tStart).count();\
   auto diffBeforeS = diffBefore / 1e9;\
   std::cout << std::this_thread::get_id() << "->"<< diffBeforeS << std::endl;\
} while (0)

//#define cpu_QUEUE queue(sycl::INTEL::host_selector{}, async_exception_handler, prop_list);
#define cpu_QUEUE queue(sycl::host_selector(), async_exception_handler)

//#define fpga_QUEUE queue(sycl::host_selector{}, async_exception_handler, prop_list);
#define fpga_QUEUE queue(sycl::INTEL::fpga_emulator_selector{}, async_exception_handler)
//#define fpga_QUEUE queue(sycl::INTEL::fpga_selector{}, async_exception_handler, prop_list);


#endif //VECTOR_ADD_SCHEDULERS_H
