submit_event[CK] =
((cpu) ? cpu_submitKernel(q, *buf_a[CK], *buf_b[0], *buf_c[CK], size_range, N)
    : fpga_submitKernel(q, *buf_a[CK], *buf_b[0], *buf_c[CK], size_range, N));