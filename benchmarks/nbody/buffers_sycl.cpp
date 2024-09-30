std::unique_ptr<sycl::buffer<ptype, 1>> buf_pos_in[1];
std::unique_ptr<sycl::buffer<ptype, 1>> buf_vel_in[1];
std::unique_ptr<sycl::buffer<float, 1>> buf_mass[1];
std::unique_ptr<sycl::buffer<ptype, 1>> buf_pos_out[num_kernels];
std::unique_ptr<sycl::buffer<ptype, 1>> buf_vel_out[num_kernels];

// Get size of the data vectors
auto N = opts.pData.size;

// Define range of the input data
sycl::range<1> range_read = sycl::range<1>(N);

// Get the pointers values
ptype* pos_in = opts.pData.pos_in.data();
ptype* vel_in = opts.pData.vel_in.data();
float* mass = opts.pData.body_mass.data();

// Set the buffers
buf_pos_in[0].reset(new sycl::buffer<ptype, 1>(pos_in, range_read));
buf_vel_in[0].reset(new sycl::buffer<ptype, 1>(vel_in, range_read));
buf_mass[0].reset(new sycl::buffer<float, 1>(mass, range_read));



