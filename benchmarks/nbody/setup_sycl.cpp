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
ptype* pos_out = opts.pData.pos_out.data() + offset;
ptype* vel_out = opts.pData.vel_out.data() + offset;

// Set the buffers
buf_pos_out[CK] = sycl::buffer<ptype, 1>(pos_out, range_gws);
buf_vel_out[CK] = sycl::buffer<ptype, 1>(vel_out, range_gws);
