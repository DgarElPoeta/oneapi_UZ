#ifndef KERNELS_H
#define KERNELS_H
#include "gaussian.h"

sycl::event cpu_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_input, 
                             sycl::buffer<float,2>& buf_filter,sycl::buffer<ptype,2>& buf_blurred,
                             sycl::nd_range<2> size_range, uint64_t N,uint64_t offset);

sycl::event fpga_submitKernel(sycl::queue& q, sycl::buffer<ptype,2>& buf_input, 
                              sycl::buffer<float,2>& buf_filter,sycl::buffer<ptype,2>& buf_blurred,
                              sycl::nd_range<2> size_range, uint64_t N,uint64_t offset);

#endif //KERNELS_H