// you have to use `size` and `offset`
// does are part of N

Ray* ray = reinterpret_cast<Ray*>(opts->p_problem);
data_t* data = ray->data;

auto pixels_ptr = ray->pixels_ptr;
auto width = data->width;
auto height = data->height;
auto camera_x = data->camera_x;
auto camera_y = data->camera_y;
auto camera_z = data->camera_z;
auto viewport_x = data->viewp_w;
auto viewport_y = data->viewp_h;
auto prim_ptr = ray->prim_ptr;
auto n_primitives = ray->n_primitives;

auto R = sycl::range<1>(size);

#if USE_LOCAL_MEM == 1
auto Nprim = n_primitives * sizeof(Primitive)/sizeof(float);
auto Rprim = sycl::range<1>(Nprim); // X floats are 1 primitive
sycl::buffer<float, 1> buf_prim((float*)prim_ptr, Rprim); // bring 1 primitive
#else
auto Rprim = sycl::range<1>(n_primitives);
sycl::buffer<Primitive, 1> buf_prim(prim_ptr, Rprim);
#endif
sycl::buffer<Pixel, 1> buf_pixels((pixels_ptr + offset), R);


if (debug) {
std::cout << "R: (" << size << ")\n";
}

#if QUEUE_NDRANGE
size_t lws = atoi(getenv("LWS"));
sycl::range<1> range_lws(lws);
sycl::nd_range<1> size_range(R, range_lws);
#else
size_t lws = 128;
#endif

size_t workgroups = size / lws;
size_t gws = size;
