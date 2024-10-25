submit_event[CK] =
((cpu) ? cpu_submitKernel(q, buf_a[CK], buf_b[CK], buf_c[CK], size_range)
    : fpga_submitKernel(q, buf_a[CK], buf_b[CK], buf_c[CK], size_range));