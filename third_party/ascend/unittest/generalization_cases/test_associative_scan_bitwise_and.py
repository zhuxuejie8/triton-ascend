import triton
import pytest
import torch
import triton.language as tl
import test_common
from test_common import TestUtils, check_ub_mem_overflow
import sys


@triton.jit
def tl_combine_fn(a, b):
    return a & b


def torch_combine_fn(a, b):
    return a & b


def torch_func_scan(x: torch.Tensor, dim: int, combine_fn='max', reverse=False):
    dim = dim % x.ndim  # 0 到 x.ndim-1

    if reverse:
        x = x.flip(dim)

    N = x.size(dim)
    tensors = torch.unbind(x, dim=dim)

    outputs = []
    carry = tensors[0]
    outputs.append(carry)

    for i in range(1, N):
        carry = combine_fn(tensors[i], carry)
        outputs.append(carry)

    output = torch.stack(outputs, dim=dim)

    if reverse:
        output = output.flip(dim)

    return output


@triton.jit
def associative_scan_1d(X, Z, N: tl.constexpr, reverse: tl.constexpr, dim: tl.constexpr):
    off = tl.arange(0, N)
    x = tl.load(X + off)
    out = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=tl_combine_fn)
    tl.store(Z + off, out)


@triton.jit
def associative_scan_2d(X, Z, N: tl.constexpr, M: tl.constexpr, reverse: tl.constexpr, dim: tl.constexpr):
    offx = tl.arange(0, M)
    offy = tl.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = tl.load(X + off2d)
    out = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=tl_combine_fn)
    tl.store(Z + off2d, out)


@triton.jit
def associative_scan_3d(X, Z, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, reverse: tl.constexpr,
                        dim: tl.constexpr):
    off2 = tl.arange(0, D2)
    off1 = tl.arange(0, D1) * D2
    off0 = tl.arange(0, D0) * D1 * D2
    off = off2[None, None, :] + off1[None, :, None] + off0[:, None, None]
    x = tl.load(X + off)
    out = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=tl_combine_fn)
    tl.store(Z + off, out)


@triton.jit
def associative_scan_4d(X, Z, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr,
                        reverse: tl.constexpr, dim: tl.constexpr):
    off3 = tl.arange(0, D3)
    off2 = tl.arange(0, D2) * D3
    off1 = tl.arange(0, D1) * D2 * D3
    off0 = tl.arange(0, D0) * D1 * D2 * D3
    off = off3[None, None, None, :] + off2[None, None, :, None] + off1[None, :, None, None] + off0[:, None, None, None]
    x = tl.load(X + off)
    out = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=tl_combine_fn)
    tl.store(Z + off, out)


@triton.jit
def associative_scan_5d(X, Z, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr,
                        reverse: tl.constexpr, dim: tl.constexpr):
    off4 = tl.arange(0, D4)
    off3 = tl.arange(0, D3) * D4
    off2 = tl.arange(0, D2) * D3 * D4
    off1 = tl.arange(0, D1) * D2 * D3 * D4
    off0 = tl.arange(0, D0) * D1 * D2 * D3 * D4
    off = (off4[None, None, None, None, :] + off3[None, None, None, :, None] + off2[None, None, :, None, None] +
           off1[None, :, None, None, None] + off0[:, None, None, None, None])
    x = tl.load(X + off)
    out = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=tl_combine_fn)
    tl.store(Z + off, out)


# 6D函数
@triton.jit
def associative_scan_6d(X, Z, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr,
                        D5: tl.constexpr,
                        reverse: tl.constexpr, dim: tl.constexpr):
    off5 = tl.arange(0, D5)
    off4 = tl.arange(0, D4) * D5
    off3 = tl.arange(0, D3) * D4 * D5
    off2 = tl.arange(0, D2) * D3 * D4 * D5
    off1 = tl.arange(0, D1) * D2 * D3 * D4 * D5
    off0 = tl.arange(0, D0) * D1 * D2 * D3 * D4 * D5
    off = (off5[None, None, None, None, None, :] +
           off4[None, None, None, None, :, None] +
           off3[None, None, None, :, None, None] +
           off2[None, None, :, None, None, None] +
           off1[None, :, None, None, None, None] +
           off0[:, None, None, None, None, None])
    x = tl.load(X + off)
    out = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=tl_combine_fn)
    tl.store(Z + off, out)


### 7维
@triton.jit
def associative_scan_7d(X, Z, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr,
                        D5: tl.constexpr, D6: tl.constexpr,
                        reverse: tl.constexpr, dim: tl.constexpr):
    off6 = tl.arange(0, D6)
    off5 = tl.arange(0, D5) * D6
    off4 = tl.arange(0, D4) * D5 * D6
    off3 = tl.arange(0, D3) * D4 * D5 * D6
    off2 = tl.arange(0, D2) * D3 * D4 * D5 * D6
    off1 = tl.arange(0, D1) * D2 * D3 * D4 * D5 * D6
    off0 = tl.arange(0, D0) * D1 * D2 * D3 * D4 * D5 * D6
    off = (off6[None, None, None, None, None, None, :] +
           off5[None, None, None, None, None, :, None] +
           off4[None, None, None, None, :, None, None] +
           off3[None, None, None, :, None, None, None] +
           off2[None, None, :, None, None, None, None] +
           off1[None, :, None, None, None, None, None] +
           off0[:, None, None, None, None, None, None])
    x = tl.load(X + off)
    out = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=tl_combine_fn)
    tl.store(Z + off, out)


### 8维
@triton.jit
def associative_scan_8d(X, Z, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, D3: tl.constexpr, D4: tl.constexpr,
                        D5: tl.constexpr, D6: tl.constexpr, D7: tl.constexpr,
                        reverse: tl.constexpr, dim: tl.constexpr):
    off7 = tl.arange(0, D7)
    off6 = tl.arange(0, D6) * D7
    off5 = tl.arange(0, D5) * D6 * D7
    off4 = tl.arange(0, D4) * D5 * D6 * D7
    off3 = tl.arange(0, D3) * D4 * D5 * D6 * D7
    off2 = tl.arange(0, D2) * D3 * D4 * D5 * D6 * D7
    off1 = tl.arange(0, D1) * D2 * D3 * D4 * D5 * D6 * D7
    off0 = tl.arange(0, D0) * D1 * D2 * D3 * D4 * D5 * D6 * D7
    off = (off7[None, None, None, None, None, None, None, :] +
           off6[None, None, None, None, None, None, :, None] +
           off5[None, None, None, None, None, :, None, None] +
           off4[None, None, None, None, :, None, None, None] +
           off3[None, None, None, :, None, None, None, None] +
           off2[None, None, :, None, None, None, None, None] +
           off1[None, :, None, None, None, None, None, None] +
           off0[:, None, None, None, None, None, None, None])
    x = tl.load(X + off)
    out = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=tl_combine_fn)
    tl.store(Z + off, out)


testlist = [
    # shape, scan_dim
    ((1,), (-1,)),
    ((2,), (-1,)),
    ((7,), (-1,)),
    ((8,), (-1,)),
    ((15,), (-1,)),
    ((16,), (-1,)),
    ((37,), (0,)),
    ((741,), (0,)),
    ((256,), (0,)),
    ((333,), (0,)),
    ((2014,), (0,)),
    ((7411,), (0,)),
    ((11411,), (0,)),

    ((1, 1), (-1,)),
    ((5, 1), (-1,)),
    ((1, 15), (-1,)),
    ((37, 16), (-1,)),
    ((37, 31), (0,)),
    ((2, 32), (0,)),
    ((5, 37), (0,)),
    ((3, 741), (0,)),
    ((13, 19), (1,)),
    ((32, 39), (1,)),
    ((64, 16), (1,)),
    ((39, 137), (1,)),
    ((1, 111), (1,)),
    ((12, 1014), (0,)),
    ((1, 1333), (0,)),
    ((3, 1345), (0,)),
    ((1007, 11), (0,)),

    ((1, 1, 1), (-1,)),
    ((1, 1, 23), (-1,)),
    ((23, 1, 1), (-1,)),
    ((1, 23, 1), (-1,)),
    ((37, 5, 3), (-1,)),
    ((2, 29, 4), (0,)),
    ((7, 31, 7), (0,)),
    ((3, 5, 8), (0,)),
    ((7, 17, 15), (0,)),
    ((25, 5, 16), (0,)),
    ((23, 5, 31), (0,)),
    ((19, 11, 32), (0,)),
    ((7, 11, 33), (0,)),
    ((2, 3, 255), (1,)),
    ((3, 3, 256), (1,)),
    ((3, 2, 257), (1,)),
    ((1, 93, 23), (2,)),
    ((1, 22, 39), (2,)),
    ((27, 1, 39), (2,)),
    ((27, 22, 13), (2,)),
    ((8, 1, 25), (2,)),
    ((33, 3, 3), (2,)),
    ((5, 13, 111), (2,)),
    ((1137, 5, 1), (2,)),

    ((1, 1, 1, 1), (-1,)),
    ((2, 29, 4, 1), (-1,)),
    ((3, 3, 4, 8), (1,)),
    ((1, 1, 4, 9), (1,)),
    ((2, 27, 4, 16), (2,)),
    ((3, 23, 4, 17), (2,)),
    ((1, 5, 4, 32), (3,)),
    ((1, 1, 4, 33), (3,)),
    ((1, 3, 2, 741), (0,)),
    ((1, 1, 2, 741), (0,)),
    ((2, 2, 2, 2), (-1,)),
    ((4, 3, 2, 1), (-1,)),
    ((8, 4, 2, 3), (2,)),
    ((16, 2, 3, 2), (2,)),
    ((13, 2, 1, 101), (3,)),
    ((6, 7, 19, 2), (3,)),
    ((3, 2, 1, 128), (0,)),
    ((133, 1, 7, 1), (0,)),
    ((3, 2, 432, 2), (-1,)),
    ((4, 4, 4, 102), (-1,)),
    ((2, 4, 9, 13), (2,)),
    ((9, 278, 1, 1), (2,)),
    ((2, 2, 8, 4), (3,)),
    ((4, 4, 2, 4), (3,)),
    ((1, 1, 1, 2111), (0,)),
    ((1, 1999, 1, 1), (1,)),
    ((168, 1, 2, 4), (2,)),
    ((3688, 1, 1, 1), (3,)),

    ((2, 1, 29, 4, 1), (-1,)),
    ((3, 2, 3, 4, 8), (1,)),
    ((1, 1, 1, 4, 9), (2,)),
    ((2, 1, 27, 4, 16), (3,)),
    ((3, 1, 23, 4, 17), (4,)),
    ((1, 1, 5, 4, 32), (0,)),
    ((1, 9, 1, 4, 33), (-1,)),
    ((1, 1, 2, 2, 741), (1,)),
    ((1, 2, 1, 2, 741), (2,)),
    ((1, 1, 1, 1, 1), (3,)),
    ((2, 6, 2, 54, 2), (4,)),
    ((3, 2, 4, 2, 9), (0,)),
    ((8, 2, 1, 2, 3), (-1,)),
    ((16, 1, 19, 1, 7), (1,)),
    ((899, 1, 1, 1, 1), (2,)),
    ((34, 1, 5, 8, 5), (3,)),
    ((2, 128, 1, 1, 11), (4,)),
    ((256, 1, 2, 3, 4), (0,)),
    ((199, 1, 1, 7, 1), (-1,)),
    ((1, 3, 1, 1, 1999), (4,)),
    ((1, 2, 7, 3, 64), (3,)),
    ((2, 3, 3, 4, 23), (2,)),
    ((1134, 1, 2, 1, 1), (0,)),

    # 切分
    ((12111,), (-1,)),
    ((1, 10099), (-1,)),
    ((10000, 1), (-1,)),
    ((11123, 1, 1), (0,)),
    ((1, 1, 10230), (2,)),
    ((1, 1, 1, 12000), (3,)),
    ((1, 1, 10098, 1), (2,)),
    ((11011, 1, 1, 1), (1,)),
    ((11899, 1, 1, 1, 1), (0,)),
    ((1, 1, 1, 1, 11000), (4,)),
    ((1, 1, 1, 1, 11089), (-1,)),
    ((1, 1, 10301, 1, 1), (0,)),

    # 6D
    ((1, 1, 1, 1, 1, 11000), (5,)),
    ((1, 1, 1, 1, 10311, 1), (4,)),
    ((11673, 1, 1, 1, 1, 1), (3,)),
    ((11, 2, 3, 1, 2, 4), (2,)),
    ((3, 2, 3, 1, 2, 9), (1,)),
    ((13, 2, 3, 1, 2, 2), (0,)),
    ((31, 2, 3, 1, 2, 2), (-1,)),

    # 7D
    ((1, 1, 1, 1, 1, 1, 11057), (6,)),
    ((1, 1, 1, 1, 1, 10311, 1), (5,)),
    ((11673, 1, 1, 1, 1, 1, 1), (4,)),
    ((1, 2, 3, 1, 2, 4, 8, 9), (3,)),
    ((3, 2, 3, 1, 2, 19, 1), (2,)),
    ((13, 2, 3, 1, 2, 2, 1), (1,)),
    ((179, 2, 1, 1, 2, 2, 1), (0,)),
    ((1, 2, 1, 1, 2, 2, 199), (-1,)),

    # 8D
    ((1, 1, 1, 1, 1, 1, 1, 10059), (7,)),
    ((1, 1, 1, 1, 1, 1, 11057, 1), (6,)),
    ((1, 1, 1, 1, 1, 10311, 1, 1), (5,)),
    ((10673, 1, 1, 1, 1, 1, 1, 1), (4,)),
    ((11, 2, 1, 1, 1, 1, 1, 1), (3,)),
    ((3, 2, 3, 1, 2, 19, 1, 3), (2,)),
    ((13, 2, 3, 1, 2, 2, 1, 1), (1,)),
    ((179, 2, 1, 1, 2, 2, 1, 3), (0,)),
    ((1, 1, 1, 1, 1, 1, 1, 10059), (-1,)),
]

typelist = ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']  # and 操作支持的数据类型

ids = ["{}-{}".format("-".join(map(str, ("shape",) + shape)), "-".join(map(str, ("scan_dim",) + scan_dim))) for
       shape, scan_dim in testlist]


@pytest.mark.parametrize("shape,scan_dim", testlist, ids=ids)
@pytest.mark.parametrize("reverse", [False, ])  # 暂不支持reverse=True
@pytest.mark.parametrize("dtype", typelist)
def test_associative_scan(shape, reverse, dtype, scan_dim):
    if check_ub_mem_overflow(dtype, shape):
        pytest.skip(f"dtype:{dtype} shape:{shape} mem overflow")
    x = test_common.generate_tensor_new(shape, dtype).npu()

    if isinstance(scan_dim, int):
        # 单个维度
        scan_dims = [scan_dim]
    else:
        # 维度序列
        scan_dims = scan_dim

    for i, scan_dim in enumerate(scan_dims):
        if scan_dim > len(shape):
            pytest.skip(f"scan_dim:{scan_dim} >= shape dims:{len(shape)}, skipping.")

        triton_res = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
        dim = len(shape)
        func_name = f"associative_scan_{dim}d"
        func = getattr(sys.modules[__name__], func_name)
        func[(1,)](x, triton_res, *x.shape, reverse, scan_dim)
    torch_res = torch_func_scan(x.cpu(), scan_dim, combine_fn=torch_combine_fn, reverse=reverse)
    test_common.validate_cmp(dtype, triton_res.cpu(), torch_res)
