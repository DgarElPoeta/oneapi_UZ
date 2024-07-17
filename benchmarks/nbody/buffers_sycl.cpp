std::unique_ptr<sycl::buffer<ptype, 1>> buf_pos_in[num_kernels];
std::unique_ptr<sycl::buffer<ptype, 1>> buf_vel_in[num_kernels];
std::unique_ptr<sycl::buffer<float, 1>> buf_mass[num_kernels];
std::unique_ptr<sycl::buffer<ptype, 1>> buf_pos_out[num_kernels];
std::unique_ptr<sycl::buffer<ptype, 1>> buf_vel_out[num_kernels];