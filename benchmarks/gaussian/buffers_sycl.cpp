// Get side size of the matrices
const auto N = opts.pData.size;

std::vector<sycl::buffer<ptype, 2>> buf_input(1,sycl::buffer<ptype, 2>(opts.pData.input.data(),sycl::range(N,N)));
std::vector<sycl::buffer<ftype, 2>> buf_filter(1,sycl::buffer<ftype, 2>(opts.pData.filter.data(),sycl::range(filterDim,filterDim)));
std::vector<sycl::buffer<ptype, 2>> buf_blurred(num_kernels,sycl::buffer<ptype, 2>(opts.pData.blurred.data(),sycl::range(0,0)));