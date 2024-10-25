#ifndef KERNELS_H
#define KERNELS_H

#include "rap.h"

#include <sycl/sycl.hpp>

constexpr uint32_t WORK_GROUP_SIZE = 128;

sycl::event cpu_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,1>& buf_a, 
                              sycl::buffer<ptype,1>& buf_b,
                              sycl::buffer<ptype,1>& buf_func, 
                              const sycl::nd_range<1> size_range, 
                              const uint64_t offset,
                              const uint64_t M
                            );

sycl::event fpga_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,1>& buf_a, 
                              sycl::buffer<ptype,1>& buf_b,
                              sycl::buffer<ptype,1>& buf_func, 
                              const sycl::nd_range<1> size_range, 
                              const uint64_t offset,
                              const uint64_t M
                            );

#endif //KERNELS_H