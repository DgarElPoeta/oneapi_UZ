#include <CL/sycl/INTEL/fpga_extensions.hpp>
#include <CL/sycl.hpp>

class KernelMatmul;

#define M 4096
#define P 4096
//#define N 4096
#define BL 16

queue& getQueue(){
#ifdef FPGA_EMULATOR
    static queue qFPGA(sycl::INTEL::fpga_emulator_selector{}, async_exception_handler);
#else
    static queue qFPGA(sycl::INTEL::fpga_selector{}, async_exception_handler);
#endif
    return qFPGA;
}

sycl::event fpga_submitKernel(queue& q,
                         sycl::buffer<ptype,2>& buf_a,
                         sycl::buffer<ptype,2>& buf_b,
                         sycl::buffer<ptype,2>& buf_c,
                         sycl::nd_range<2> size_range,
                         size_t offset,
                         int N){
    q = getQueue();
    auto e_c = q.submit([&](handler &h) {


auto A = buf_a.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(h);
auto B = buf_b.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(h);
auto C = buf_c.get_access<sycl::access::mode::discard_write>(h);

/*
auto A = buf_a.get_access<sycl::access::mode::read>(h);
auto B = buf_b.get_access<sycl::access::mode::read>(h);
auto C = buf_c.get_access<sycl::access::mode::write>(h);
*/

sycl::accessor<float, 2, sycl::access::mode::read_write, sycl::access::target::local> A_local(sycl::range<2>{16, 16}, h);
sycl::accessor<float, 2, sycl::access::mode::read_write, sycl::access::target::local> B_local(sycl::range<2>{16, 16}, h);
//#define UNROLL_FACTOR 8

h.parallel_for<KernelMatmul>(size_range, [=](nd_item<2> item)
  [[
  intel::kernel_args_restrict,
  intel::max_work_group_size(1, BL, BL),
  cl::reqd_work_group_size(1,BL,BL),
  intel::num_simd_work_items(BL)
  ]]
{
  int global_x = item.get_global_id(0);
  int global_y = item.get_global_id(1);

  int block_x = item.get_group().get_id(0);
  int block_y = item.get_group().get_id(1);

  int local_x = item.get_local_id(0);
  int local_y = item.get_local_id(1);

  int a_start = N * BL * block_x;
  int a_end   = a_start + N - 1;
  int b_start = BL * block_y;

  float sum = 0.0f;

  for (int offset = 0; offset <= N-1; offset += BL){

      A_local[local_x][local_y] = A[global_x][offset + local_y];
      B_local[local_x][local_y] = B[offset + local_x][global_y];

      item.barrier(sycl::access::fence_space::local_space);

      //#pragma unroll
      for (int k = 0; k < BL; ++k){
          float aaa = A_local[local_x][k];
          float bbb = B_local[k][local_y];
          sum += aaa * bbb;
          //sum += A_local[local_x][k] * B_local[local_y][k];
      }

      item.barrier(sycl::access::fence_space::local_space);

  }

  C[item.get_global_id(0)][item.get_global_id(1)] = sum;

});




});

  return e_c;
}
