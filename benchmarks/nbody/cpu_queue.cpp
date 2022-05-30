#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>

class KernelNBody;
queue& getQueue(){
    static queue qCPU(sycl::INTEL::host_selector{}, async_exception_handler);
    return qCPU;
}

sycl::event cpu_submitKernel(queue& q, sycl::buffer<float4,1>& buf_pos_in, sycl::buffer<float4,1>& buf_vel_in,
                       sycl::buffer<float4,1>& buf_pos_out, sycl::buffer<float4,1>& buf_vel_out,
						 sycl::nd_range<1> size_range, size_t offset, size_t numBodies, float epsSqr, float deltaTime){
    q = getQueue();
    sycl::event kern_ev = q.submit([&](handler &h) {

auto pos = buf_pos_in.get_access<sycl::access::mode::read>(h);
auto vel = buf_vel_in.get_access<sycl::access::mode::read>(h);
auto newPosition = buf_pos_out.get_access<sycl::access::mode::discard_write>(h);
auto newVelocity = buf_vel_out.get_access<sycl::access::mode::discard_write>(h);

#define COMPACT 1
#define UNROLL_FACTOR 128

h.parallel_for<KernelNBody>(size_range, [=](nd_item<1> item) {
  size_t tid = item.get_global_id(0);
  const auto gid = tid + offset;

  float4 myPos = pos[gid];
  float4 acc{0.0f};
  bool shown = false;

  unsigned int i = 0;
  for (; (i + UNROLL_FACTOR) < numBodies;) {
    #pragma unroll UNROLL_FACTOR
    for (int j = 0; j < UNROLL_FACTOR; j++, i++) {
      float4 p = pos[i];
      float4 r;

      // TODO: review
#if COMPACT
      r = p - myPos;
#else
      r.x() = p.x() - myPos.x();
      r.y() = p.y() - myPos.y();
      r.z() = p.z() - myPos.z();
#endif

      float distSqr = r.x() * r.x() + r.y() * r.y() + r.z() * r.z();

      float invDist = 1.0f / sycl::sqrt(distSqr + epsSqr);
      float invDistCube = invDist * invDist * invDist;
      float s = p.w() * invDistCube;

      // accumulate effect of all particles
      // acc.xyz += s * r.xyz;
#if COMPACT
      acc += s * r;
#else
      acc.x() += s * r.x();
      acc.y() += s * r.y();
      acc.z() += s * r.z();
#endif
    }
  }
#if COMPACT
  acc.w() = 0.0f;
#endif

  for (; i < numBodies; i++) {
    float4 p = pos[i];
    float4 r;

#if COMPACT
    r = p - myPos;
#else
    r.x() = p.x() - myPos.x();
    r.y() = p.y() - myPos.y();
    r.z() = p.z() - myPos.z();
#endif

    float distSqr = r.x()* r.x()+ r.y() * r.y() + r.z() * r.z();

    float invDist = 1.0f / sycl::sqrt(distSqr + epsSqr);
    float invDistCube = invDist * invDist * invDist;
    float s = p.w() * invDistCube;

    // accumulate effect of all particles
#if COMPACT
    acc += s * r;
#else
    acc.x() += s * r.x();
    acc.y() += s * r.y();
    acc.z() += s * r.z();
#endif
  }

  float4 oldVel = vel[gid];

  // updated position and velocity
  float4 newPos;

  // newPos.xyz = myPos.xyz + oldVel.xyz * deltaTime + acc.xyz * 0.5f * deltaTime * deltaTime;
#if COMPACT
  newPos = myPos + oldVel * deltaTime + acc * 0.5f * deltaTime * deltaTime;
#else
  newPos.x() = myPos.x() + oldVel.x() * deltaTime + acc.x() * 0.5f * deltaTime * deltaTime;
  newPos.y() = myPos.y() + oldVel.y() * deltaTime + acc.y() * 0.5f * deltaTime * deltaTime;
  newPos.z() = myPos.z() + oldVel.z() * deltaTime + acc.z() * 0.5f * deltaTime * deltaTime;
#endif
  newPos.w() = myPos.w();

  float4 newVel;

  // newVel.xyz = oldVel.xyz + acc.xyz * deltaTime;
#if COMPACT
  newVel = oldVel + acc * deltaTime;
#else
  newVel.x() = oldVel.x() + acc.x() * deltaTime;
  newVel.y() = oldVel.y() + acc.y() * deltaTime;
  newVel.z() = oldVel.z() + acc.z() * deltaTime;
#endif
  newVel.w() = oldVel.w();

  // write to global memory
  newPosition[gid - offset] = newPos;
  newVelocity[gid - offset] = newVel;
});

});

sycl::event update_host_pos_event;
update_host_pos_event = q.submit([&](handler &h) {
  sycl::accessor accessor_pos(buf_pos_out, h, sycl::read_only);
  h.update_host(accessor_pos);
});

sycl::event update_host_vel_event;
update_host_vel_event = q.submit([&](handler &h) {
  sycl::accessor accessor_vel(buf_vel_out, h, sycl::read_only);
  h.update_host(accessor_vel);
});

  return kern_ev;
}
