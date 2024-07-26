// IMPORTANT NOTE: we mimic the EngineCL behavior here (send complete read buffers):
// But it can be easily modified in EngineCL to support partial read buffers

// Get side size of the matrices
auto N = opts.pData.size;

auto max_iterations = opts.max_iterations;

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

ptype* out = opts.pData.image.data() + offset*N;

// Create the buffers

buf_out[CK].reset(new sycl::buffer<ptype, 2>(out, range_gws));

submit_event[CK] =
((cpu) ? cpu_submitKernel(q, *buf_out[CK], size_range, N, offset)
    : fpga_submitKernel(q, *buf_out[CK], size_range, N, offset));