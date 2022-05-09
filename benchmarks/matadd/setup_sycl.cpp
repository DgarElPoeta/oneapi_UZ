// IMPORTANT NOTE: we mimic the EngineCL behavior here (send complete read buffers):
// But it can be easily modified in EngineCL to support partial read buffers

Matadd* matadd = reinterpret_cast<Matadd*>(opts->p_problem);
auto N = matadd->size;
auto R = sycl::range<2>(size, N); // partially, read and write
auto Rb = sycl::range<2>(N, N); // read all (buffer b)

// auto Rinput = sycl::range<2>(matadd->size);
// auto Rfilter = sycl::range<2>(matadd->_filter_total_size);
if (debug) {
std::cout << "R(a,c): (" << size << "," << N << ")\n";
std::cout << "Rb: (" << N << "," << N << ")\n";
}

sycl::range<2> range_lws(1, atoi(getenv("LWS"))); // gpu 32, cpu 128
sycl::nd_range<2> size_range(R, range_lws);

const size_t offsetN = offset * N;
// IMPORTANT: read matadd note
// sycl::buffer<ptype, 2> buf_a((matadd->a + offset * N), R); // offset should be done inside
// sycl::buffer<ptype, 2> buf_b(matadd->b + offset * N, Rb);

/*sycl::buffer<ptype, 2> buf_a(matadd->a + offset * N, R);
sycl::buffer<ptype, 2> buf_b(matadd->b + offset * N, R);
sycl::buffer<ptype, 2> buf_c((matadd->c + offset * N), R);*/

/*buf_a[DB] = new sycl::buffer<ptype, 2>(matadd->a + offset * N, R);
buf_b[DB] = new sycl::buffer<ptype, 2>(matadd->b + offset * N, R);
buf_c[DB] = new sycl::buffer<ptype, 2>((matadd->c + offset * N), R);*/

buf_a[DB].reset(new sycl::buffer<ptype, 2>(matadd->a + offset * N, R));
buf_b[DB].reset(new sycl::buffer<ptype, 2>(matadd->b + offset * N, R));
buf_c[DB].reset(new sycl::buffer<ptype, 2>((matadd->c + offset * N), R));


int rows = N;
int cols = N;
