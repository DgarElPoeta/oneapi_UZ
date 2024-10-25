// Get total problem size
const auto N = opts.pData.size;

// Get M size
const auto M = opts.pData.M;

std::vector<sycl::buffer<ptype, 1>> buf_a(1,sycl::buffer<ptype, 1>(opts.pData.a.data(),sycl::range(N)));
std::vector<sycl::buffer<ptype, 1>> buf_func(1,sycl::buffer<ptype, 1>(opts.pData.func.data(),sycl::range(N)));
std::vector<sycl::buffer<ptype, 1>> buf_b(num_kernels,sycl::buffer<ptype, 1>(opts.pData.b.data(),sycl::range(0)));