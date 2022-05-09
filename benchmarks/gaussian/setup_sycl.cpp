Gaussian* gaussian = reinterpret_cast<Gaussian*>(opts->p_problem);
auto R = sycl::range<1>(size);
auto Rinput = sycl::range<1>(gaussian->_total_size);
auto Rfilter = sycl::range<1>(gaussian->_filter_total_size);

sycl::range<1> range_lws(atoi(getenv("LWS")));
sycl::nd_range<1> size_range(R, range_lws);

/*
#if INPUT_BUFFER_SENT_ALL
sycl::buffer<uchar4, 1> buf_input((gaussian->_a), Rinput); // offset should be done inside
#else
sycl::buffer<uchar4, 1> buf_input((gaussian->_a + offset), R);
#endif

sycl::buffer<float, 1> buf_filterWeight(gaussian->_b, Rfilter);
sycl::buffer<uchar4, 1> buf_blurred((gaussian->_c + offset), R);
*/

buf_input[DB].reset(new sycl::buffer<uchar4, 1> ((gaussian->_a), Rinput));
buf_filterWeight[DB].reset(new sycl::buffer<float, 1> (gaussian->_b, Rfilter));
buf_blurred[DB].reset(new sycl::buffer<uchar4, 1> ((gaussian->_c + offset), R));

int rows = gaussian->_width;
int cols = gaussian->_width;
int filterWidth = gaussian->_filter_width;
