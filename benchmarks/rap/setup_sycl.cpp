// you have to use `size` and `offset`
// does are part of N

Rap* rap = reinterpret_cast<Rap*>(opts->p_problem);
auto N = rap->size; // all problem, rap->size is different than `size`
auto Rw = sycl::range<1>(size);
auto R = sycl::range<1>(N); // func

if (debug) {
  std::cout << "Rw(opt2): (" << size << ")\n";
  std::cout << "Rdb(opt1,func): (" << size+offset << ")\n";
  std::cout << "R(opt1,func): (" << N << ")\n";
}

/*sycl::buffer<ptype, 1> buf_a(rap->a, R);
sycl::buffer<ptype, 1> buf_b((rap->b + offset), Rw); // offset should be done inside
sycl::buffer<ptype, 1> buf_func((rap->func), R);*/

auto Rdb = sycl::range<1>(size+offset);
buf_a[DB].reset(new sycl::buffer<ptype, 1>(rap->a, Rdb));
buf_b[DB].reset(new sycl::buffer<ptype, 1>((rap->b + offset), Rw));
buf_func[DB].reset(new sycl::buffer<ptype, 1>((rap->func), Rdb));

auto M = rap->M;

sycl::range<1> range_lws(atoi(getenv("LWS")));
sycl::nd_range<1> size_range(Rw, range_lws);
