std::vector<sycl::buffer<ptype, 2>> buf_a(num_kernels,sycl::buffer<ptype, 2>(opts.pData.a.data(),sycl::range(0,0)));
std::vector<sycl::buffer<ptype, 2>> buf_b(num_kernels,sycl::buffer<ptype, 2>(opts.pData.b.data(),sycl::range(0,0)));
std::vector<sycl::buffer<ptype, 2>> buf_c(num_kernels,sycl::buffer<ptype, 2>(opts.pData.c.data(),sycl::range(0,0)));