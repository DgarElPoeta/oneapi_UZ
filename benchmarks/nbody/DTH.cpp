sycl::host_accessor pos_result(buf_pos_out[eventIndex], sycl::read_only);
sycl::host_accessor vel_result(buf_vel_out[eventIndex], sycl::read_only);