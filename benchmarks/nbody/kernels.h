#ifndef KERNELS_H
#define KERNELS_H

#include "nbody.h"

#include <sycl/sycl.hpp>

constexpr uint32_t WORK_GROUP_SIZE = 128;

sycl::event cpu_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,1>& buf_pos_in, 
                              sycl::buffer<ptype,1>& buf_vel_in,
                              sycl::buffer<ptype,1>& buf_pos_out, 
                              sycl::buffer<ptype,1>& buf_vel_out,
                              sycl::buffer<mtype,1>& body_mass, 
                              const sycl::nd_range<1> size_range, 
                              const uint64_t num_bodies,
                              const uint64_t offset
                            );

sycl::event fpga_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,1>& buf_pos_in, 
                              sycl::buffer<ptype,1>& buf_vel_in,
                              sycl::buffer<ptype,1>& buf_pos_out, 
                              sycl::buffer<ptype,1>& buf_vel_out,
                              sycl::buffer<mtype,1>& body_mass, 
                              const sycl::nd_range<1> size_range, 
                              const uint64_t num_bodies,
                              const uint64_t offset
                            );

#endif //KERNELS_H
