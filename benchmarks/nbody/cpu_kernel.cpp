#include "kernels.h"
class KernelNbodyCPU;

sycl::event cpu_submitKernel(sycl::queue& q, sycl::buffer<ptype,1>& buf_pos_in, sycl::buffer<ptype,1>& buf_vel_in,
                       sycl::buffer<ptype,1>& buf_pos_out, sycl::buffer<ptype,1>& buf_vel_out,
                       sycl::buffer<float,1>& body_mass, sycl::nd_range<1> size_range, 
                       uint64_t num_bodies, uint64_t offset){
    sycl::event kern_ev = q.submit([&](sycl::handler &h) {
      sycl::accessor pos(buf_pos_in, h, sycl::read_only);
      sycl::accessor vel(buf_vel_in, h, sycl::read_only);
      sycl::accessor mass(body_mass, h, sycl::read_only);
      sycl::accessor newPosition(buf_pos_out, h, sycl::write_only, sycl::no_init);
      sycl::accessor newVelocity(buf_vel_out, h, sycl::write_only, sycl::no_init);

      h.parallel_for<KernelNbodyCPU>(size_range, [=](sycl::nd_item<1> item){
        size_t tid = item.get_global_id(0);
        size_t gid = tid + offset;

        ptype myPos = pos[gid];
        ptype myVel = vel[gid];
        ptype acc{0.0f};
        for(size_t i = 0; i < num_bodies; i++){
          ptype p = pos[i];
          ptype r = p - myPos;
          float m = mass[i];
          float distSqr = sycl::dot(r,r) + SofteningSquared;
          float dist = sycl::sqrt(distSqr);
          float invDist = 1.0f / dist;
          float invDistCube = invDist * invDist * invDist;
          acc += m * r * invDistCube;
        }
        acc *= G;
        ptype newVel = myVel + acc * DT;
        ptype newPos = myPos + myVel * DT + 0.5f * acc * DT * DT;

        newPosition[tid] = newPos;
        newVelocity[tid] = newVel;
      });
    });
    return kern_ev;
}
