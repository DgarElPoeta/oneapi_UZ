#ifndef KERNELS_H
#define KERNELS_H
#include <sycl/sycl.hpp>
#include "matmul.h"

sycl::event cpu_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_a, sycl::buffer<ptype,2>& buf_b,
                       sycl::buffer<ptype,2>& buf_c, sycl::nd_range<2> size_range, uint64_t N);

sycl::event fpga_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_a, sycl::buffer<ptype,2>& buf_b,
                       sycl::buffer<ptype,2>& buf_c, sycl::nd_range<2> size_range, uint64_t N);

#endif //KERNELS_H