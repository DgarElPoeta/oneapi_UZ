std::unique_ptr<sycl::buffer<ptype, 2>> buf_a[num_kernels];
std::unique_ptr<sycl::buffer<ptype, 2>> buf_b[1];
std::unique_ptr<sycl::buffer<ptype, 2>> buf_c[num_kernels];

// Get side size of the matrices
auto N = opts.pData.size;

// Define range of the matrix b.
sycl::range<2> range_b = sycl::range<2>(N,N);
ptype* b = opts.pData.b.data();

// Set buffer for matrix b.
buf_b[0].reset(new sycl::buffer<ptype, 2>(b, range_b));