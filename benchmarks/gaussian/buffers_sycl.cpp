std::unique_ptr<sycl::buffer<ptype, 2>> buf_input[num_kernels];
std::unique_ptr<sycl::buffer<float, 2>> buf_filter[num_kernels];
std::unique_ptr<sycl::buffer<ptype, 2>> buf_blurred[num_kernels];