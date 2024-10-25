submit_event[CK] =
((cpu) ? cpu_submitKernel(q, buf_input[0], buf_filter[0], buf_blurred[CK], size_range, N, offset) 
    : fpga_submitKernel(q, buf_input[0], buf_filter[0], buf_blurred[CK], size_range, N, offset) );