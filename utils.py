Dim = int | tuple[int, int]


def v(x: Dim, i: int) -> int:
    return x if type(x) == int else x[i]


def conv_sz(
    h_in: int, w_in: int,
    stride: Dim, padding: Dim, kernel_size: Dim, dilation: Dim = 1
) -> tuple[int, int]:
    return (
        (h_in + 2 * v(padding, 0) - v(dilation, 0) * (v(kernel_size, 0) - 1) - 1) // v(stride, 0) + 1,
        (w_in + 2 * v(padding, 1) - v(dilation, 1) * (v(kernel_size, 1) - 1) - 1) // v(stride, 1) + 1
    )


def conv_t_sz(
    h_in: int, w_in: int,
    stride: Dim, padding: Dim, kernel_size: Dim,
    dilation: Dim = 1, out_padding: Dim = 1
) -> tuple[int, int]:
    return (
        (h_in - 1) * v(stride, 0) - 2 * v(padding, 0) + v(dilation, 0) * (v(kernel_size, 0) - 1) + v(out_padding, 0) + 1,
        (w_in - 1) * v(stride, 1) - 2 * v(padding, 1) + v(dilation, 1) * (v(kernel_size, 1) - 1) + v(out_padding, 1) + 1
    )
