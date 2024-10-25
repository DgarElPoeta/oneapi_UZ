#include "kernels.h"

#include <sycl/ext/intel/fpga_extensions.hpp>

class KernelNbodyFPGA;

constexpr uint32_t UNROLL_FACTOR = 16;

constexpr uint32_t NUM_SIMD_WORK_ITEMS = 8;

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
                            )
{

  sycl::event kern_ev = q.submit([&](sycl::handler &h)
  {
    
    const sycl::accessor pos(buf_pos_in, h, sycl::read_only);
    const sycl::accessor vel(buf_vel_in, h, sycl::read_only);
    const sycl::accessor mass(body_mass, h, sycl::read_only);
    const sycl::accessor newPosition(buf_pos_out, h, sycl::write_only, sycl::no_init);
    const sycl::accessor newVelocity(buf_vel_out, h, sycl::write_only, sycl::no_init);

    h.parallel_for<KernelNbodyFPGA>(size_range, [=](sycl::nd_item<1> item)
    [
      [
        sycl::reqd_work_group_size(WORK_GROUP_SIZE),
        intel::num_simd_work_items(NUM_SIMD_WORK_ITEMS),
        intel::kernel_args_restrict
      ]
    ]
    {
      
      // Get the global id of the work package
      const size_t tid = item.get_global_id(0);

      // Get the global id of the total work
      const size_t gid = tid + offset;

      // Get the position of the current body
      const ptype myPos = pos[gid];

      // Get the velocity of the current body
      const ptype myVel = vel[gid];

      // Variable to store the acceleration of the current body
      ptype acc{0.0f};

      size_t i = 0;

      // Unrolled loop over the max number of bodies which is a multiple of UNROLL_FACTOR
      for(; (i + UNROLL_FACTOR) < num_bodies;){
        
        #pragma unroll UNROLL_FACTOR
        for(size_t j = 0; j < UNROLL_FACTOR; j++, i++){

          // Get the position of the other body
          const ptype p = pos[i];

          // Get the mass of the other body
          const mtype m = mass[i];

          // Calculate the distance between the two bodies
          const ptype r = p - myPos;

          // Calculate the distance squared
          const float distSqr = sycl::dot(r,r) + SofteningSquared;

          // Calculate the distance
          const float dist = sycl::sqrt(distSqr);

          // Calculate the inverse distance
          const float invDist = 1.0f / dist;

          // Calculate the inverse distance cubed
          const float invDistCube = invDist * invDist * invDist;

          // Add the acceleration due to the other body to the total acceleration
          acc += m * r * invDistCube;

        }
        
      }


      // Loop over the rest of the bodies
      for(; i< num_bodies; i++){

        // Get the position of the other body
        const ptype p = pos[i];

        // Get the mass of the other body
        const mtype m = mass[i];

        // Calculate the distance between the two bodies
        const ptype r = p - myPos;

        // Calculate the distance squared
        const float distSqr = sycl::dot(r,r) + SofteningSquared;

        // Calculate the distance
        const float dist = sycl::sqrt(distSqr);

        // Calculate the inverse distance
        const float invDist = 1.0f / dist;

        // Calculate the inverse distance cubed
        const float invDistCube = invDist * invDist * invDist;

        // Add the acceleration due to the other body to the total acceleration
        acc += m * r * invDistCube;
        
      }

      // Multiply the acceleration by the gravitational constant
      acc *= G;

      // Calculate the new position of the body
      const ptype newPos = myPos + myVel * DT + 0.5f * acc * DT * DT;
      
      // Calculate the new velocity of the body
      const ptype newVel = myVel + acc * DT;

      // Store the new position and velocity
      newPosition[tid] = newPos;
      newVelocity[tid] = newVel;

    });

  });

  return kern_ev;

}

