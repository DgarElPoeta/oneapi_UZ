#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>

class KernelMatadd;
#define MAX_WG_SIZE 16

queue& getQueue(){
#ifdef FPGA_EMULATOR
    static queue qFPGA(sycl::INTEL::fpga_emulator_selector{}, async_exception_handler);
#else
    static queue qFPGA(sycl::INTEL::fpga_selector{}, async_exception_handler);
#endif
    return qFPGA;
}

sycl::event fpga_submitKernel(queue& q, sycl::buffer<ptype,2>& buf_a, sycl::buffer<ptype,2>& buf_b,
                       sycl::buffer<ptype,2>& buf_c, sycl::nd_range<2> size_range, size_t offset){
    q = getQueue();
    sycl::event kern_ev = q.submit([&](handler &h) {
      auto a = buf_a.get_access<sycl::access::mode::read>(h);
      auto b = buf_b.get_access<sycl::access::mode::read>(h);
      auto c = buf_c.get_access<sycl::access::mode::discard_write>(h);

      h.parallel_for<KernelMatadd>(size_range, [=](nd_item<2> item)
              [[intel::kernel_args_restrict
              intel::max_work_group_size(1, 1, MAX_WG_SIZE),
              cl::reqd_work_group_size(1,1,MAX_WG_SIZE),
              intel::num_simd_work_items(MAX_WG_SIZE)]]{
        size_t i = item.get_global_id(0);
        size_t j = item.get_global_id(1);

        c[{i,j}] = a[{i,j}] + b[{i,j}];
      });
    });
/*
    sycl::event update_host_event;
    update_host_event = q.submit([&](handler &h) {
    sycl::accessor accessor_c(buf_c, h, sycl::read_only);
      h.update_host(accessor_c);
    });
*/
    return kern_ev;
}
