// Define the range of the global work size.
sycl::range<2> range_gws = sycl::range<2>(size,N); 

if(wgs > N) {
    wgs = N;
}

// Define the range of the local work size.
sycl::range<2> range_lws(1,wgs);

/*
 * We define the nd_range of the problem. It combines the range of the global work size and 
 * the range of the local work size.
 */
sycl::nd_range<2> size_range(range_gws, range_lws);

// Get the offset pointers values
ptype* blurred = opts.pData.blurred.data() + offset*N;

// Set the buffer
buf_blurred[CK].reset(new sycl::buffer<ptype, 2>(blurred, range_gws));