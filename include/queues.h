#ifndef QUEUES_H
#define QUEUES_H

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <iostream>

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
      std::cout << "Async Exception: " << e.what() << std::endl;
      std::terminate();
    }
  }
};

const sycl::property_list plist = sycl::property_list{sycl::property::queue::enable_profiling()};

#define DEFAULT_QUEUE sycl::queue(sycl::default_selector_v, exception_handler,plist)
#define CPU_QUEUE sycl::queue(sycl::cpu_selector_v, exception_handler,plist)
#define GPU_QUEUE sycl::queue(sycl::gpu_selector_v, exception_handler,plist)
#define FPGAEMU_QUEUE sycl::queue(sycl::ext::intel::fpga_emulator_selector_v, exception_handler,plist)
#define FPGAHW_QUEUE sycl::queue(sycl::ext::intel::fpga_selector_v, exception_handler,plist)


#endif //QUEUES_H