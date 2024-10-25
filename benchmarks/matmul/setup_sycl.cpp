// Define range of the global work size.
const sycl::range<2> range_gws = sycl::range<2>(size,N); 

// Define range of the local work size.
const sycl::range<2> range_lws(wgs,wgs);

/*
 * We define the nd_range of the problem. It combines the range of the global work size and 
 * the range of the local work size.
 */
const sycl::nd_range<2> size_range(range_gws, range_lws);

// Get the offset pointers values
const ptype* a = opts.pData.a.data() + offset*N;
ptype* c = opts.pData.c.data() + offset*N;

// Set the buffers
buf_a[CK] = sycl::buffer<ptype, 2>(a, range_gws);
buf_c[CK] = sycl::buffer<ptype, 2>(c, range_gws);