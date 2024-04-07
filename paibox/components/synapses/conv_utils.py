import numpy as np
from functools import partial
from itertools import repeat
from typing import Iterable
from numpy.typing import NDArray

from paibox.exceptions import ShapeError
from paibox.types import SpikeType, SynOutType, WeightType

from .conv_types import SizeAnyType, Size1Type, Size2Type, Size3Type, _Order2d, _Order3d


def _ntuple(x, n: int):
    if isinstance(x, Iterable):
        return tuple(x)

    return tuple(repeat(x, n))


_single = partial(_ntuple, n=1)
_pair = partial(_ntuple, n=2)
_triple = partial(_ntuple, n=3)
_quadruple = partial(_ntuple, n=4)


def _fm_ndim1_check(fm_shape: SizeAnyType, fm_order: _Order2d) -> Size2Type:
    if len(fm_shape) < 1 or len(fm_shape) > 2:
        raise ShapeError(f"expected shape of 1 or 2, but got {len(fm_shape)}.")

    if len(fm_shape) == 1:
        channels, l = (1,) + fm_shape
    else:
        if fm_order == "CL":
            channels, l = fm_shape
        else:
            l, channels = fm_shape

    return channels, l


def _fm_ndim2_check(fm_shape: SizeAnyType, fm_order: _Order3d) -> Size3Type:
    if len(fm_shape) < 2 or len(fm_shape) > 3:
        raise ShapeError(f"expected shape of 2 or 3, but got {len(fm_shape)}.")

    if len(fm_shape) == 2:
        channels, h, w = (1,) + fm_shape
    else:
        if fm_order == "CHW":
            channels, h, w = fm_shape
        else:
            h, w, channels = fm_shape

    return channels, h, w


def _conv1d_unroll(
    in_shape: Size1Type,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type,
    # padding: Size1Type,
    # fm_order: str,
) -> WeightType:
    """Unroll the convolution kernel of 1d convolution into a matrix.

    XXX: The case where the input feature map is in 'LC' order is not considered for the time being.
    """
    cout, cin, kl = kernel.shape
    il = in_shape[0]
    ol = out_shape[0]

    w_unrolled = np.zeros((cin * il, cout * ol), dtype=kernel.dtype)
    zeros_image = np.zeros((cin * il, cout, ol), dtype=kernel.dtype)

    for i in range(ol):
        for ch_idx in np.ndindex(kernel.shape[:2]):
            # [0] -> o_ch, [1] -> i_ch
            zeros_image[
                i * stride[0] + ch_idx[1] * il : i * stride[0] + ch_idx[1] * il + kl,
                ch_idx[0],
                i,
            ] = kernel[ch_idx[0], ch_idx[1], :]

        # if fm_order == "CL":
        # (cin*il, cout) -> (cout, cin*il)
        temp = zeros_image[:, :, i].T
        # else:
        #     # (cin*il, cout) -> (cout, il, cin)
        #     temp = zeros_image[:, :, i].reshape(cin, il, cout).transpose()

        for o_ch in range(cout):
            w_unrolled[:, i + o_ch * ol] = temp[o_ch].ravel()

    return w_unrolled


def _conv2d_unroll(
    in_shape: Size2Type,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    # padding: Size2Type,
    # fm_order: str,
) -> WeightType:
    """Unroll the convolution kernel of 2d convolution into a matrix.

    XXX: The case where the input feature map is in 'HWC' order is not considered for the time being.
    """
    cout, cin, kh, kw = kernel.shape
    ih, iw = in_shape
    oh, ow = out_shape
    in_size = ih * iw
    out_size = oh * ow

    w_unrolled = np.zeros((cin * in_size, cout * out_size), dtype=kernel.dtype)
    zeros_image = np.zeros((cin * ih, cout * iw, out_size), dtype=kernel.dtype)

    for i in range(oh):
        for j in range(ow):
            for ch_idx in np.ndindex(kernel.shape[:2]):
                # [0] -> o_ch, [1] -> i_ch
                zeros_image[
                    i * stride[0]
                    + ch_idx[1] * ih : i * stride[0]
                    + ch_idx[1] * ih
                    + kh,
                    j * stride[1]
                    + ch_idx[0] * iw : j * stride[1]
                    + ch_idx[0] * iw
                    + kw,
                    i * ow + j,
                ] = kernel[ch_idx[0], ch_idx[1], :, :]

            temp = np.swapaxes(
                zeros_image[:, :, i * ow + j].reshape(cin * ih, cout, iw),
                0,
                1,
                # .transpose(1, 0, 2)
            )
            # else:
            #     # (cin*ih, cout, iw) -> (cout, cin, ih, iw)
            #     temp = (
            #         zeros_image[:, :, i * ow + j]
            #         .reshape(cin, ih, cout, iw)
            #         .transpose(2, 1, 3, 0)
            #     )

            for o_ch in range(cout):
                w_unrolled[:, i * ow + j + o_ch * out_size] = temp[o_ch].ravel()

    return w_unrolled


def _pool2d_kernel_unroll(
    channels: int,
    in_shape: Size2Type,
    out_shape: Size2Type,
    ksize: Size2Type,
    stride: Size2Type,
    # padding: Size2Type,
    # fm_order: str,
) -> WeightType:
    kh, kw = ksize
    ih, iw = in_shape
    oh, ow = out_shape
    in_size = ih * iw
    out_size = oh * ow

    w_unrolled = np.zeros((channels * in_size, channels * out_size), dtype=np.bool_)

    for i in range(oh):
        for j in range(ow):
            zeros_image = np.zeros((channels * ih, iw * channels), dtype=np.bool_)
            for i_ch in range(channels):
                zeros_image[
                    (i * stride[0] + i_ch * ih) : (i * stride[0] + i_ch * ih) + kh,
                    (j * stride[1] + i_ch * iw) : (j * stride[1] + i_ch * iw) + kw,
                ] = 1

            temp = zeros_image.reshape((channels * ih, channels, iw)).transpose(1, 0, 2)

            for o_ch in range(channels):
                w_unrolled[:, i * ow + j + o_ch * oh * ow] = temp[o_ch].ravel()

    return w_unrolled


def _func_pool2d(
    x_chw: SpikeType,
    out_shape: Size2Type,
    ksize: Size2Type,
    stride: Size2Type,
    padding: Size2Type,
    type: str,
) -> SpikeType:
    xcin, xh, xw = x_chw.shape
    kh, kw = ksize
    oh, ow = out_shape
    cout = xcin

    assert (xh + padding[0] * 2 - kh) // stride[0] + 1 == oh
    assert (xw + padding[1] * 2 - kw) // stride[1] + 1 == ow

    out = np.zeros((cout, oh, ow), dtype=np.int32)
    x_padded = np.pad(
        x_chw,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    for c in range(cout):
        for i in range(oh):
            for j in range(ow):
                if type == "avg":
                    out[c, i, j] = np.sum(
                        x_padded[
                            c,
                            stride[0] * i : stride[0] * i + kh,
                            stride[1] * j : stride[1] * j + kw,
                        ]
                    )
                else:
                    out[c, i, j] = np.max(
                        x_padded[
                            c,
                            stride[0] * i : stride[0] * i + kh,
                            stride[1] * j : stride[1] * j + kw,
                        ]
                    )

    if type == "avg":
        thres = kh * kw // 2 + 1
        return out >= thres
    else:
        return out.astype(np.bool_)


_func_avgpool2d = partial(_func_pool2d, type="avg")
_func_maxpool2d = partial(_func_pool2d, type="max")


def _func_conv1d_faster(
    x_cl: np.ndarray,
    out_shape: Size1Type,
    kernel: WeightType,
    stride: Size1Type,
    padding: Size1Type,
) -> SynOutType:
    """Faster 1d convolution.

    XXX: The case where the input feature map is in 'LC' order is not considered for the time being.
    """
    xc, xl = x_cl.shape

    # (O, I, L)
    cout, cin, kl = kernel.shape
    assert xc == cin

    x_padded = np.pad(x_cl, ((0, 0), (padding[0], padding[0])), mode="constant")

    assert (xl + padding[0] * 2 - kl) // stride[0] + 1 == out_shape[0]

    # kernel: (cout, cin, kl) -> (cout, cin*kl)
    col_kernel = kernel.reshape(cout, -1)

    # padded: (cin, xl+2*p[0]-kl) -> (ol, cin*kl)
    col_fm = _1d_im2col(x_padded, out_shape[0], kl, stride)

    # out = np.zeros((cout,) + out_shape, dtype=np.int64)
    # (ol, cin*kl) * (cout, cin*kl)^T = (ol, cout)
    out = col_fm @ col_kernel.T  # + self.bias

    # (ol, cout) -> (cout, ol)
    return out.astype(np.int32).T


def _func_conv2d_faster(
    x_chw: np.ndarray,
    out_shape: Size2Type,
    kernel: WeightType,
    stride: Size2Type,
    padding: Size2Type,
    # fm_order: str,
) -> SynOutType:
    """Faster 2d convolution.

    XXX: The case where the input feature map is in 'HWC' order is not considered for the time being.
    """
    xc, xh, xw = x_chw.shape

    # (O, I, H, W)
    cout, cin, kh, kw = kernel.shape
    assert xc == cin

    x_padded = np.pad(
        x_chw,
        ((0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
        mode="constant",
    )

    assert (xh + padding[0] * 2 - kh) // stride[0] + 1 == out_shape[0]
    assert (xw + padding[1] * 2 - kw) // stride[1] + 1 == out_shape[1]

    # kernel: (cout, cin, kh, kw) -> (cout, cin*kh*kw)
    col_kernel = kernel.reshape(cout, -1)

    # padded: (cin, xh+2*p[0]-kh, xw+2*p[1]-kw) -> (oh*ow, cin*kh*kw)
    col_fm = _2d_im2col(x_padded, out_shape[0], out_shape[1], kh, kw, stride)

    # out = np.zeros((cout,) + out_shape, dtype=np.int64)
    # (oh*ow, cin*kh*kw) * (cout, cin*kh*kw)^T = (oh*ow, cout)
    out = col_fm @ col_kernel.T  # + self.bias

    # (oh*ow, cout) -> (cout, oh*ow) -> (cout, oh, ow)
    out = out.astype(np.int32).T.reshape((cout,) + out_shape)

    return out


def _1d_im2col(
    x_padded: np.ndarray, ol: int, kl: int, stride: Size1Type
) -> NDArray[np.int64]:
    cols = np.zeros((ol, x_padded.shape[0] * kl), dtype=np.int64)

    _, pl = x_padded.shape

    idx = 0
    for i in range(0, pl - kl + 1, stride[0]):
        cols[idx] = x_padded[:, i : i + kl].ravel()
        idx += 1

    return cols


def _2d_im2col(
    x_padded: np.ndarray, oh: int, ow: int, kh: int, kw: int, stride: Size2Type
) -> NDArray[np.int64]:
    cols = np.zeros((oh * ow, x_padded.shape[0] * kh * kw), dtype=np.int64)

    _, ph, pw = x_padded.shape

    idx = 0
    for i in range(0, ph - kh + 1, stride[0]):
        for j in range(0, pw - kw + 1, stride[1]):
            cols[idx] = x_padded[:, i : i + kh, j : j + kw].ravel()
            idx += 1

    return cols
