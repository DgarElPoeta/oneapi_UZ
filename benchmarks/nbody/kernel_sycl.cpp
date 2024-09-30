submit_event[CK] =
((cpu) ? cpu_submitKernel(q, *buf_pos_in[0], *buf_vel_in[0], *buf_pos_out[CK], *buf_vel_out[CK], *buf_mass[0], size_range, N, offset)
    : fpga_submitKernel(q, *buf_pos_in[0], *buf_vel_in[0], *buf_pos_out[CK], *buf_vel_out[CK], *buf_mass[0], size_range, N, offset));