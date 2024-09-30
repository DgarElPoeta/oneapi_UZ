// Get size of the matrix
auto N = opts.pData.size;

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
ptype* out = opts.pData.image.data() + offset*N;

// Set the buffers
buf_out[CK].reset(new sycl::buffer<ptype, 2>(out, range_gws));