#ifndef KERNELS_H
#define KERNELS_H

#include "matmul.h"

#include <sycl/sycl.hpp>

constexpr uint32_t WORK_GROUP_SIZE = 256;

sycl::event cpu_submitKernel(
                              sycl::queue& q,
                              sycl::buffer<ptype,2>& buf_a,
                              sycl::buffer<ptype,2>& buf_b,
                              sycl::buffer<ptype,2>& buf_c, 
                              const sycl::nd_range<2> size_range, 
                              const uint64_t N
                            );

sycl::event fpga_submitKernel(
                              sycl::queue& q,
                              sycl::buffer<ptype,2>& buf_a,
                              sycl::buffer<ptype,2>& buf_b,
                              sycl::buffer<ptype,2>& buf_c, 
                              const sycl::nd_range<2> size_range, 
                              const uint64_t N
                            );

#endif //KERNELS_H