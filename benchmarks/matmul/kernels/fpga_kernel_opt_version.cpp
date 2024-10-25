#include "kernels.h"

#include <sycl/ext/intel/fpga_extensions.hpp>

class KernelMatmulFPGA;

constexpr uint32_t SIMD_WORK_ITEMS = 16;

sycl::event fpga_submitKernel(
                              sycl::queue& q,
                              sycl::buffer<ptype,2>& buf_a,
                              sycl::buffer<ptype,2>& buf_b,
                              sycl::buffer<ptype,2>& buf_c, 
                              const sycl::nd_range<2> size_range, 
                              const uint64_t N
                            )
{

  sycl::event kern_ev = q.submit([&](sycl::handler &h)
  {
    
    const sycl::accessor a(buf_a, h, sycl::read_only);
    const sycl::accessor b(buf_b, h, sycl::read_only);
    const sycl::accessor c(buf_c, h, sycl::write_only, sycl::no_init);

    const sycl::local_accessor<ptype, 2> a_local(sycl::range<2>{WORK_GROUP_SIZE, WORK_GROUP_SIZE}, h);
    const sycl::local_accessor<ptype, 2> b_local(sycl::range<2>{WORK_GROUP_SIZE, WORK_GROUP_SIZE}, h);

    h.parallel_for<KernelMatmulFPGA>(size_range, [=](sycl::nd_item<2> item)
    [
      [
        sycl::reqd_work_group_size(WORK_GROUP_SIZE,WORK_GROUP_SIZE),
        intel::num_simd_work_items(SIMD_WORK_ITEMS),
        intel::kernel_args_restrict,
      ]
    ]
    {
      
      // Get the row global id
      const size_t global_x = item.get_global_id(0);

      // Get the column global id
      const size_t global_y = item.get_global_id(1);

      // Get the row local id
      const size_t local_x = item.get_local_id(0);

      // Get the column local id
      const size_t local_y = item.get_local_id(1);
      
      // Get the group whom the item belongs to
      const sycl::group<2> itemGroup = item.get_group();

      // Variable to store the sum of the product of the elements
      ptype sum(0.0);
      
      // Go through the matrix in tiles
      for (size_t offset = 0; offset < N; offset += WORK_GROUP_SIZE){
        
        // Load the elements of the matrices into local memory
        a_local[local_x][local_y] = a[global_x][offset + local_y];
        b_local[local_x][local_y] = b[offset + local_x][global_y];

        // Wait for all work-items in the group to finish loading the elements
        group_barrier(itemGroup);

        // Multiply the elements of the matrices of the tile
        for (size_t k = 0; k < WORK_GROUP_SIZE; ++k){
          const ptype aaa = a_local[local_x][k];
          const ptype bbb = b_local[k][local_y];
          sum += aaa * bbb;
        }

        // Wait for all work-items in the group to finish multiplying the elements
        group_barrier(itemGroup);

      }

      // Store the sum in the result matrix
      c[{global_x, global_y}] = sum;

    });
  
  });

  return kern_ev;

}
