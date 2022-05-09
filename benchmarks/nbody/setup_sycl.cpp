// you have to use `size` and `offset`
// does are part of N

Nbody* nbody = reinterpret_cast<Nbody*>(opts->p_problem);
auto N = nbody->size; // all problem, rap->size is different than `size`
auto Rw = sycl::range<1>(size);
auto R = sycl::range<1>(N); // func

if (debug) {
  std::cout << "Rw(opt2): (" << size << ")\n";
  std::cout << "R(opt1,func): (" << N << ")\n";
}
/*
sycl::buffer<ptype, 1> buf_pos_in(nbody->pos_in, R);
sycl::buffer<ptype, 1> buf_vel_in(nbody->vel_in, R);
sycl::buffer<ptype, 1> buf_pos_out((nbody->pos_out + offset), Rw); // offset should be done inside
sycl::buffer<ptype, 1> buf_vel_out((nbody->vel_out + offset), Rw);
*/

buf_pos_in[DB].reset(new sycl::buffer<ptype, 1>(nbody->pos_in, R));
buf_vel_in[DB].reset(new sycl::buffer<ptype, 1>(nbody->vel_in, R));
buf_pos_out[DB].reset(new sycl::buffer<ptype, 1>((nbody->pos_out + offset), Rw));
buf_vel_out[DB].reset(new sycl::buffer<ptype, 1>((nbody->vel_out + offset), Rw));

sycl::range<1> range_lws(atoi(getenv("LWS")));
sycl::nd_range<1> size_range(Rw, range_lws);

auto numBodies = nbody->size;
auto epsSqr = nbody->espSqr;
auto deltaTime = nbody->delT;
