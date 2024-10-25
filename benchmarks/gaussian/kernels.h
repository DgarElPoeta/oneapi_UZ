#ifndef KERNELS_H
#define KERNELS_H

#include "gaussian.h"

#include <sycl/sycl.hpp>

constexpr uint32_t WORK_GROUP_SIZE = 128;

sycl::event cpu_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,2>& buf_input, 
                              sycl::buffer<ftype,2>& buf_filter,
                              sycl::buffer<ptype,2>& buf_blurred,
                              const sycl::nd_range<2> size_range, 
                              const uint64_t N, 
                              const uint64_t offset
                            );

sycl::event fpga_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,2>& buf_input, 
                              sycl::buffer<ftype,2>& buf_filter,
                              sycl::buffer<ptype,2>& buf_blurred,
                              const sycl::nd_range<2> size_range, 
                              const uint64_t N, 
                              const uint64_t offset
                            );

#endif //KERNELS_H