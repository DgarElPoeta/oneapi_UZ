#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>

class KernelMatadd;
#define MAX_WG_SIZE 16

sycl::event submitKernel(queue& q, sycl::buffer<ptype,1>& buf_a, sycl::buffer<ptype,1>& buf_b, int workgroups, int steps, int steps1){
    return q.submit([&](handler &h) {
  auto randArray = buf_a.get_access<sycl::access::mode::read>(h);
  auto output = buf_b.get_access<sycl::access::mode::discard_write>(h);

  h.parallel_for_work_group(sycl::range<1>(workgroups), sycl::range<1>(steps1), [=](sycl::group<1>grp) {

    float4 callA[255];
    float4 callB[254];

    sycl::private_memory<float4> puByr(grp);
    sycl::private_memory<float4> pdByr(grp);

    int numSteps = steps;
    grp.parallel_for_work_item([&](sycl::h_item<1> item) {
      auto tid = item.get_local_id(0);
      auto bid = (item.get_global_id(0) / steps1);

      float4 inRand = randArray[bid];

      float4 s = (1.0f - inRand) * 5.0f + inRand * 30.f;
      float4 x = (1.0f - inRand) * 1.0f + inRand * 100.f;
      float4 optionYears = (1.0f - inRand) * 0.25f + inRand * 10.f;
      float4 dt = optionYears * (1.0f / (float)numSteps);
      float4 vsdt = VOLATILITY * sqrt(dt);
      float4 rdt = RISKFREE * dt;
      float4 r = exp(rdt);
      float4 rInv = 1.0f / r;
      float4 u = exp(vsdt);
      float4 d = 1.0f / u;
      float4 pu = (r - d) / (u - d);
      float4 pd = 1.0f - pu;
      puByr(item) = pu * rInv;
      pdByr(item) = pd * rInv;

      float4 profit = s * exp(vsdt * (2.0f * tid - (float)numSteps)) - x;
      callA[tid].x() = profit.x() > 0 ? profit.x() : 0.0f;
      callA[tid].y() = profit.y() > 0 ? profit.y() : 0.0f;
      callA[tid].z() = profit.z() > 0 ? profit.z() : 0.0f;
      callA[tid].w() = profit.w() > 0 ? profit.w() : 0.0f;
    });

    for (int j = numSteps; j > 0; j -= 2) {
      grp.parallel_for_work_item([&](sycl::h_item<1> item) {
        auto tid = item.get_local_id(0);

        if (tid < j) {
          callB[tid] = puByr(item) * callA[tid] + pdByr(item) * callA[tid + 1];
        }
      });

      grp.parallel_for_work_item([&](sycl::h_item<1> item) {
        auto tid = item.get_local_id(0);

        if (tid < j - 1) {
          callA[tid] = puByr(item) * callB[tid] + pdByr(item) * callB[tid + 1];
        }
      });
    }

    grp.parallel_for_work_item([&](sycl::h_item<1> item) {
      auto tid = item.get_local_id(0);
      auto bid = (item.get_global_id(0) / steps1); // no offset_workgroups because we have buffer offsets

      if (tid == 0){
        output[bid] = callA[0];
       }
    });

  });

});
}
