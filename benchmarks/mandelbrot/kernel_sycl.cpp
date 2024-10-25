submit_event[CK] =
((cpu) ? cpu_submitKernel(q, buf_out[CK], size_range, N, offset)
    : fpga_submitKernel(q, buf_out[CK], size_range, N, offset));