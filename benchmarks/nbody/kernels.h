#ifndef KERNELS_H
#define KERNELS_H

#include <sycl/sycl.hpp>
#include "nbody.h"

sycl::event cpu_submitKernel(sycl::& q, sycl::buffer<ptype,1>& buf_pos_in, sycl::buffer<ptype,1>& buf_vel_in,
                       sycl::buffer<ptype,1>& buf_pos_out, sycl::buffer<pytpe,1>& buf_vel_out,
                       sycl::buffer<float,1>& body_mass, sycl::nd_range<1> size_range, 
                       uint64_t num_bodies, uint64_t offset);

sycl::event fpga_submitKernel(sycl::queue& q, sycl::buffer<ptype,1>& buf_pos_in, sycl::buffer<ptype,1>& buf_vel_in,
                       sycl::buffer<ptype,1>& buf_pos_out, sycl::buffer<pytpe,1>& buf_vel_out,
                       sycl::buffer<float,1>& body_mass, sycl::nd_range<1> size_range, 
                       uint64_t num_bodies, uint64_t offset);

#endif //KERNELS_H
