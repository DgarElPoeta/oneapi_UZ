// Get side size of the matrices
auto N = opts.pData.size;


// Define range of the global work size.
sycl::range<2> range_gws = sycl::range<2>(size,N); 

if(wgs > N) {
    wgs = N;
}

// Define range of the local work size.
sycl::range<2> range_lws(1,wgs);

/*
 * We define the nd_range of the problem. It combines the range of the global work size and 
 * the range of the local work size.
 */
sycl::nd_range<2> size_range(range_gws, range_lws);

// Get the offset pointers values
ptype* a = opts.pData.a.data() + offset*N;
ptype* b = opts.pData.b.data() + offset*N;
ptype* c = opts.pData.c.data() + offset*N;

// Set the buffers
buf_a[CK].reset(new sycl::buffer<ptype, 2>(a, range_gws));
buf_b[CK].reset(new sycl::buffer<ptype, 2>(b, range_gws));
buf_c[CK].reset(new sycl::buffer<ptype, 2>(c, range_gws));