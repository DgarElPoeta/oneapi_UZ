// you have to use `size` and `offset`
// does are part of N


Binomial* binomial = reinterpret_cast<Binomial*>(opts->p_problem);
auto steps1 = binomial->steps1;
auto steps = binomial->steps;
auto offset_workgroups = offset;// / steps1;
auto workgroups = size;// / steps1;
// auto N = binomial->size; // all problem, binomial->size is different than `size`
// auto Rw = sycl::range<1>(size);
auto R = sycl::range<1>(workgroups); // func

if (debug) {
  std::cout << "R(a,b): (" << workgroups << ")\n";
  // std::cout << "R(opt1,func): (" << N << ")\n";
}
// printf("size: %d offset %d\n", size, offset);
#if USM
auto randArray = (binomial->a + offset_workgroups);
auto output = (binomial->b + offset_workgroups);
#else
sycl::buffer<ptype, 1> buf_a((binomial->a + offset_workgroups), R);
sycl::buffer<ptype, 1> buf_b((binomial->b + offset_workgroups), R);
#endif
// sycl::buffer<ptype, 1> buf_b((binomial->b + offset), Rw); // offset should be done inside
// sycl::buffer<ptype, 1> buf_func((binomial->func), R);
// sycl::buffer<float4, 1> bufgpuv1((v1.data()), R);
// sycl::buffer<float4, 1> bufgpuv2((v2.data()), R);

// la unidad minima tiene que ser samplesPerVectorWidth

// auto workgroups = size / steps1; // samplesPerVectorWidth;
