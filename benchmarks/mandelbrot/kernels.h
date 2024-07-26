#ifndef KERNELS_H
#define KERNELS_H
#include <sycl/sycl.hpp>
#include "mandelbrot.h"

sycl::event cpu_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_out, sycl::nd_range<2> size_range, 
                             const uint64_t N, const size_t offset);

sycl::event fpga_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_out, sycl::nd_range<2> size_range, 
                              const uint64_t N, const size_t offset);

#endif //KERNELS_H