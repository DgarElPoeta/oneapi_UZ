// Get side size of the matrices
const auto N = opts.pData.size;

std::vector<sycl::buffer<ptype, 1>> buf_pos_in(1,sycl::buffer<ptype, 1>(opts.pData.pos_in.data(),sycl::range(N)));
std::vector<sycl::buffer<ptype, 1>> buf_vel_in(1,sycl::buffer<ptype, 1>(opts.pData.vel_in.data(),sycl::range(N)));
std::vector<sycl::buffer<mtype, 1>> buf_mass(1,sycl::buffer<mtype, 1>(opts.pData.body_mass.data(),sycl::range(N)));
std::vector<sycl::buffer<ptype, 1>> buf_pos_out(num_kernels,sycl::buffer<ptype, 1>(opts.pData.pos_out.data(),sycl::range(0)));
std::vector<sycl::buffer<ptype, 1>> buf_vel_out(num_kernels,sycl::buffer<ptype, 1>(opts.pData.vel_out.data(),sycl::range(0)));