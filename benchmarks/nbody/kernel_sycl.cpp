// IMPORTANT NOTE: we mimic the EngineCL behavior here (send complete read buffers):
// But it can be easily modified in EngineCL to support partial read buffers

// Get side size of the matrices
auto N = opts.pData.size;

/*
 * We define the range of the global work size.
 */
sycl::range<1> range_gws = sycl::range<1>(size); 

/*
 * We define the range of the matrix b.
 */
sycl::range<1> range_read = sycl::range<1>(N);

if(wgs > N) {
    wgs = N;
}

/*
 * We define the range of the local work size.
 */
sycl::range<1> range_lws(wgs);

/*
 * We define the nd_range of the problem. It combines the range of the global work size and 
 * the range of the local work size.
 */
sycl::nd_range<1> size_range(range_gws, range_lws);

// Get the offset pointers values

ptype* pos_in = opts.pData.pos_in.data();
ptype* vel_in = opts.pData.vel_in.data();
float* mass = opts.pData.body_mass.data();
ptype* pos_out = opts.pData.pos_out.data() + offset;
ptype* vel_out = opts.pData.vel_out.data() + offset;

// Create the buffers

buf_pos_in[CK].reset(new sycl::buffer<ptype, 1>(pos_in, range_read));
buf_vel_in[CK].reset(new sycl::buffer<ptype, 1>(vel_in, range_read));
buf_mass[CK].reset(new sycl::buffer<float, 1>(mass, range_read));
buf_pos_out[CK].reset(new sycl::buffer<ptype, 1>(pos_out, range_gws));
buf_vel_out[CK].reset(new sycl::buffer<ptype, 1>(vel_out, range_gws));

submit_event[CK] =
((cpu) ? cpu_submitKernel(q, *buf_pos_in[CK], *buf_vel_in[CK], *buf_pos_out[CK], *buf_vel_out[CK], *buf_mass[CK], size_range, N, offset)
    : fpga_submitKernel(q, *buf_pos_in[CK], *buf_vel_in[CK], *buf_pos_out[CK], *buf_vel_out[CK], *buf_mass[CK], size_range, N, offset));