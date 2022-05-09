
Matmul* matmul = reinterpret_cast<Matmul*>(opts->p_problem);
auto N = matmul->size;
auto R = sycl::range<2>(size, N);
auto Rb = sycl::range<2>(N, N);

if (debug) {
  std::cout << "R(a,c): (" << size << "," << N << ")\n";
  std::cout << "Rb: (" << N << "," << N << ")\n";
}
// matmul gpu is much worse (11s) compared with ecl (3s)
//sycl::range<2> range_lws(atoi(getenv("LWS")), atoi(getenv("LWS2"))); // 1, 64
sycl::range<2> range_lws(16, 16); // 1, 64
sycl::nd_range<2> size_range(R, range_lws);

/*
sycl::buffer<ptype, 2> buf_a((matmul->a + (offset * N)), R); // offset should be done inside
sycl::buffer<ptype, 2> buf_b(matmul->b, Rb);
sycl::buffer<ptype, 2> buf_c((matmul->c + (offset * N)), R);^*/

buf_a[DB].reset(new sycl::buffer<ptype, 2>((matmul->a + (offset * N)), R));
buf_b[DB].reset(new sycl::buffer<ptype, 2>(matmul->b, Rb));
buf_c[DB].reset(new sycl::buffer<ptype, 2>((matmul->c + (offset * N)), R));

int rows = N;
int cols = N;
