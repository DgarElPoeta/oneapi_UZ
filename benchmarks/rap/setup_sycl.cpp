// Define range of the global work size.
const sycl::range<1> range_gws = sycl::range<1>(size); 

// Define range of the local work size.
const sycl::range<1> range_lws(wgs);

/*
 * We define the nd_range of the problem. It combines the range of the global work size and 
 * the range of the local work size.
 */
const sycl::nd_range<1> size_range(range_gws, range_lws);

// Get the offset pointers values
ptype* b = opts.pData.b.data() + offset;

// Set the buffers
buf_b[CK] = sycl::buffer<ptype, 1>(b, range_gws);