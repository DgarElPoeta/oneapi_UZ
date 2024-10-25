#ifndef KERNELS_H
#define KERNELS_H

#include "mandelbrot.h"

#include <sycl/sycl.hpp>

constexpr uint32_t WORK_GROUP_SIZE = 128;

sycl::event cpu_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,2>& buf_out, 
                              const sycl::nd_range<2> size_range, 
                              const uint64_t N, 
                              const size_t offset
                            );

sycl::event fpga_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,2>& buf_out, 
                              const sycl::nd_range<2> size_range, 
                              const uint64_t N, 
                              const size_t offset
                            );

#endif //KERNELS_H