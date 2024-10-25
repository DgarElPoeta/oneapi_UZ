submit_event[CK] =
((cpu) ? cpu_submitKernel(q, buf_a[0], buf_b[CK], buf_func[0], size_range, offset, M)
    : fpga_submitKernel(q, buf_a[0], buf_b[CK], buf_func[0], size_range, offset, M));