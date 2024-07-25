// IMPORTANT NOTE: we mimic the EngineCL behavior here (send complete read buffers):
// But it can be easily modified in EngineCL to support partial read buffers

// Get side size of the image
auto N = opts.pData.size;

/*
 * We define the range of the input image.
 */
sycl::range<2> range_input = sycl::range<2>(N,N); 

/*
 * We define the range of the gaussian filter.
 */
sycl::range<2> range_filter = sycl::range<2>(filterDim, filterDim); 


/*
 * We define the range of the global work size.
 */
sycl::range<2> range_gws = sycl::range<2>(size,N); 

if(wgs > N) {
    wgs = N;
}

/*
 * We define the range of the local work size.
 */
sycl::range<2> range_lws(1,wgs);

/*
 * We define the nd_range of the problem. It combines the range of the global work size and 
 * the range of the local work size.
 */
sycl::nd_range<2> size_range(range_gws, range_lws);

// Get the offset pointers values

ptype* input = opts.pData.input.data();
float* filter = opts.pData.filter.data();
ptype* blurred = opts.pData.blurred.data() + offset*N;

// Create the buffers

buf_input[CK].reset(new sycl::buffer<ptype, 2>(input, range_input));
buf_filter[CK].reset(new sycl::buffer<float, 2>(filter, range_filter));
buf_blurred[CK].reset(new sycl::buffer<ptype, 2>(blurred, range_gws));

submit_event[CK] =
((cpu) ? cpu_submitKernel(q, *buf_input[CK], *buf_filter[CK], *buf_blurred[CK], size_range, N, offset) 
    : fpga_submitKernel(q, *buf_input[CK], *buf_filter[CK], *buf_blurred[CK], size_range, N, offset) );