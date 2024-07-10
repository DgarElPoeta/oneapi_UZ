// IMPORTANT NOTE: we mimic the EngineCL behavior here (send complete read buffers):
// But it can be easily modified in EngineCL to support partial read buffers

// Get side size of the matrices
auto N = opts.pData.size;

/*
 * We define the range of the global work size.
 */
sycl::range<2> range_gws = sycl::range<2>(size,N); 

/*
 * We define the range of the matrix b.
 */
sycl::range<2> range_b = sycl::range<2>(N,N);

if(wgs > N) {
    wgs = N;
}

/*
 * We define the range of the local work size.
 */
sycl::range<2> range_lws(1,N);

/*
 * We define the nd_range of the problem. It combines the range of the global work size and 
 * the range of the local work size.
 */
sycl::nd_range<2> size_range(range_gws, range_lws);

// Get the offset pointers values

ptype* a = opts.pData.a.data() + offset*N;
ptype* b = opts.pData.b.data() + offset*N;
ptype* c = opts.pData.c.data() + offset*N;

// Create the buffers

buf_a[CK].reset(new sycl::buffer<ptype, 2>(a, range_gws));
buf_b[CK].reset(new sycl::buffer<ptype, 2>(b, range_b));
buf_c[CK].reset(new sycl::buffer<ptype, 2>(c, range_gws));

submit_event[CK] =
((cpu) ? cpu_submitKernel(q, *buf_a[CK], *buf_b[CK], *buf_c[CK], size_range, N)
    : fpga_submitKernel(q, *buf_a[CK], *buf_b[CK], *buf_c[CK], size_range, N));