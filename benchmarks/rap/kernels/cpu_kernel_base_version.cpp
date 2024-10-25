#include "kernels.h"

class KernelRapCPU;

sycl::event cpu_submitKernel(
                              sycl::queue& q, 
                              sycl::buffer<ptype,1>& buf_a, 
                              sycl::buffer<ptype,1>& buf_b,
                              sycl::buffer<ptype,1>& buf_func, 
                              const sycl::nd_range<1> size_range, 
                              const uint64_t offset,
                              const uint64_t M
                            )
{

  sycl::event kern_ev = q.submit([&](sycl::handler &h)
  {
    
    const sycl::accessor a(buf_a, h, sycl::read_only);
    const sycl::accessor func(buf_func, h, sycl::read_only);
    const sycl::accessor b(buf_b, h, sycl::write_only, sycl::no_init);

    h.parallel_for<KernelRapCPU>(size_range, [=](sycl::nd_item<1> item)
    {
      
      // Get the global id of the work-item inside the NDRange
      const size_t gid = item.get_global_id(0);

      // Get the global id of the work-item inside the total work
      const size_t id = gid + offset;
      
      // Check if the id is inside the range
      if(id <= M){
        
        // Auxiliar variable to store the maximum value
        ptype tmp = func[0];

        // Loop to obtain the max value of a[id-x] + func[x] for x in [0,id] 
        for(size_t i = 0 ; i <= id ; i++){
          ptype aux = a[id-i] + func[i];
          tmp = sycl::max(aux,tmp);
        }

        // Store the result
        b[gid] = tmp;

      }

    });

  });
  
  return kern_ev;

}
