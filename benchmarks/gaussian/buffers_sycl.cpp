std::unique_ptr<sycl::buffer<ptype, 2>> buf_input[1];
std::unique_ptr<sycl::buffer<float, 2>> buf_filter[1];
std::unique_ptr<sycl::buffer<ptype, 2>> buf_blurred[num_kernels];

// Get side size of the matrices
auto N = opts.pData.size;

// Define range of the input image.
sycl::range<2> range_input = sycl::range<2>(N,N);

// Define range of the filter.
sycl::range<2> range_filter = sycl::range<2>(filterDim, filterDim);

// Get the pointers value
ptype* input = opts.pData.input.data();
float* filter = opts.pData.filter.data();

// Set the buffers.
buf_input[0].reset(new sycl::buffer<ptype, 2>(input, range_input));
buf_filter[0].reset(new sycl::buffer<float, 2>(filter, range_filter));