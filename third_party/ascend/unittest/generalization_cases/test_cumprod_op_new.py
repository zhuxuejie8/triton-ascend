import math
import pytest
import torch
import triton
import triton.language as tl
import test_common


@triton.jit
def triton_cumprod_1D(in_ptr0, out_ptr0, dim: tl.constexpr, reverse: tl.constexpr, L: tl.constexpr):
    idx = tl.arange(0, L)
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_cumprod_2D(in_ptr0, out_ptr0, dim: tl.constexpr, reverse: tl.constexpr, L: tl.constexpr, M: tl.constexpr):
    Lidx = tl.arange(0, L)
    Midx = tl.arange(0, M)
    idx = Lidx[:, None] * M + Midx[None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_cumprod_3D(
        in_ptr0, out_ptr0, dim: tl.constexpr, reverse: tl.constexpr,
        L: tl.constexpr, M: tl.constexpr, N: tl.constexpr
):
    Lidx = tl.arange(0, L)
    Midx = tl.arange(0, M)
    Nidx = tl.arange(0, N)
    idx = Lidx[:, None, None] * M * N + Midx[None, :, None] * N + Nidx[None, None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_cumprod_4D(
        in_ptr0, out_ptr0, dim: tl.constexpr, reverse: tl.constexpr,
        L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    Lidx = tl.arange(0, L)
    Midx = tl.arange(0, M)
    Nidx = tl.arange(0, N)
    Kidx = tl.arange(0, K)
    idx = (Lidx[:, None, None, None] * M * N * K + Midx[None, :, None, None] * N * K +
           Nidx[None, None, :, None] * K + Kidx[None, None, None, :])
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_cumprod_5D(
        in_ptr0, out_ptr0, dim: tl.constexpr, reverse: tl.constexpr,
        L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, J: tl.constexpr, K: tl.constexpr
):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    jblk_idx = tl.arange(0, J)
    kblk_idx = tl.arange(0, K)
    idx = lblk_idx[:, None, None, None, None] * K * J * N * M + \
          mblk_idx[None, :, None, None, None] * K * J * N + \
          nblk_idx[None, None, :, None, None] * K * J + \
          jblk_idx[None, None, None, :, None] * K + \
          kblk_idx[None, None, None, None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_cumprod_6D(
        in_ptr0, out_ptr0, dim: tl.constexpr, reverse: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
        N: tl.constexpr, J: tl.constexpr, K: tl.constexpr, X: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    jblk_idx = tl.arange(0, J)
    kblk_idx = tl.arange(0, K)
    xblk_idx = tl.arange(0, X)
    idx = lblk_idx[:, None, None, None, None, None] * X * K * J * N * M + \
          mblk_idx[None, :, None, None, None, None] * X * K * J * N + \
          nblk_idx[None, None, :, None, None, None] * X * K * J + \
          jblk_idx[None, None, None, :, None, None] * X * K + \
          kblk_idx[None, None, None, None, :, None] * X + \
          xblk_idx[None, None, None, None, None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_cumprod_7D(
        in_ptr0, out_ptr0, dim: tl.constexpr, reverse: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
        N: tl.constexpr, J: tl.constexpr, K: tl.constexpr, X: tl.constexpr, Y: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    jblk_idx = tl.arange(0, J)
    kblk_idx = tl.arange(0, K)
    xblk_idx = tl.arange(0, X)
    yblk_idx = tl.arange(0, Y)
    idx = lblk_idx[:, None, None, None, None, None, None] * Y * X * K * J * N * M + \
          mblk_idx[None, :, None, None, None, None, None] * Y * X * K * J * N + \
          nblk_idx[None, None, :, None, None, None, None] * Y * X * K * J + \
          jblk_idx[None, None, None, :, None, None, None] * Y * X * K + \
          kblk_idx[None, None, None, None, :, None, None] * Y * X + \
          xblk_idx[None, None, None, None, None, :, None] * Y + \
          yblk_idx[None, None, None, None, None, None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


@triton.jit
def triton_cumprod_8D(
        in_ptr0, out_ptr0, dim: tl.constexpr, reverse: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
        N: tl.constexpr, J: tl.constexpr, K: tl.constexpr, X: tl.constexpr, Y: tl.constexpr, Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    jblk_idx = tl.arange(0, J)
    kblk_idx = tl.arange(0, K)
    xblk_idx = tl.arange(0, X)
    yblk_idx = tl.arange(0, Y)
    zblk_idx = tl.arange(0, Z)
    idx = lblk_idx[:, None, None, None, None, None, None, None] * Z * Y * X * K * J * N * M + \
          mblk_idx[None, :, None, None, None, None, None, None] * Z * Y * X * K * J * N + \
          nblk_idx[None, None, :, None, None, None, None, None] * Z * Y * X * K * J + \
          jblk_idx[None, None, None, :, None, None, None, None] * Z * Y * X * K + \
          kblk_idx[None, None, None, None, :, None, None, None] * Z * Y * X + \
          xblk_idx[None, None, None, None, None, :, None, None] * Z * Y + \
          yblk_idx[None, None, None, None, None, None, :, None] * Z + \
          zblk_idx[None, None, None, None, None, None, None, :]
    x = tl.load(in_ptr0 + idx)
    ret = tl.cumprod(x, axis=dim, reverse=reverse)
    tl.store(out_ptr0 + idx, ret)


testlist = [
    # 1D
    (triton_cumprod_1D, (1,)),
    (triton_cumprod_1D, (4,)),
    (triton_cumprod_1D, (3,)),
    (triton_cumprod_1D, (40,)),
    (triton_cumprod_1D, (24,)),
    (triton_cumprod_1D, (16,)),
    (triton_cumprod_1D, (32,)),
    (triton_cumprod_1D, (2,)),
    (triton_cumprod_1D, (5,)),
    (triton_cumprod_1D, (17,)),
    (triton_cumprod_1D, (99,)),
    (triton_cumprod_1D, (251,)),
    # 2D
    (triton_cumprod_2D, (1, 1)),
    (triton_cumprod_2D, (2, 4)),
    (triton_cumprod_2D, (128, 4)),
    (triton_cumprod_2D, (267, 32)),
    (triton_cumprod_2D, (9, 7)),
    (triton_cumprod_2D, (13, 15)),
    (triton_cumprod_2D, (15, 22)),
    (triton_cumprod_2D, (15, 1)),
    (triton_cumprod_2D, (1, 155)),
    (triton_cumprod_2D, (85, 27)),
    (triton_cumprod_2D, (32, 32)),
    (triton_cumprod_2D, (2, 2)),
    (triton_cumprod_2D, (9, 81)),
    (triton_cumprod_2D, (102, 21)),
    (triton_cumprod_2D, (1, 1024)),
    (triton_cumprod_2D, (2, 1031)),
    (triton_cumprod_2D, (3, 630)),
    # 3D
    (triton_cumprod_3D, (1, 1, 1)),
    (triton_cumprod_3D, (2, 4, 8)),
    (triton_cumprod_3D, (128, 4, 1)),
    (triton_cumprod_3D, (67, 32, 2)),
    (triton_cumprod_3D, (9, 7, 15)),
    (triton_cumprod_3D, (13, 15, 6)),
    (triton_cumprod_3D, (77, 22, 2)),
    (triton_cumprod_3D, (15, 22, 8)),
    (triton_cumprod_3D, (15, 1, 9)),
    (triton_cumprod_3D, (1, 155, 10)),
    (triton_cumprod_3D, (45, 27, 6)),
    (triton_cumprod_3D, (1, 32, 32)),
    (triton_cumprod_3D, (2, 2, 32)),
    (triton_cumprod_3D, (9, 81, 4)),
    (triton_cumprod_3D, (12, 21, 15)),
    (triton_cumprod_3D, (1, 1, 15)),
    (triton_cumprod_3D, (22, 1, 1)),
    (triton_cumprod_3D, (2, 3, 255)),
    (triton_cumprod_3D, (3, 3, 256)),
    (triton_cumprod_3D, (3, 2, 257)),
    (triton_cumprod_3D, (24, 1, 51)),
    (triton_cumprod_3D, (3, 2, 400)),
    # 4D
    (triton_cumprod_4D, (1, 1, 1, 1)),
    (triton_cumprod_4D, (2, 4, 8, 4)),
    (triton_cumprod_4D, (1, 6, 5, 7)),
    (triton_cumprod_4D, (9, 1, 11, 2)),
    (triton_cumprod_4D, (8, 12, 1, 7)),
    (triton_cumprod_4D, (3, 9, 8, 1)),
    (triton_cumprod_4D, (1, 1, 25, 1)),
    (triton_cumprod_4D, (1, 38, 25, 1)),
    (triton_cumprod_4D, (72, 1, 19, 1)),
    (triton_cumprod_4D, (7, 5, 1, 1)),
    (triton_cumprod_4D, (18, 7, 2, 15)),
    (triton_cumprod_4D, (8, 4, 2, 4)),
    (triton_cumprod_4D, (2, 8, 2, 2)),
    (triton_cumprod_4D, (240, 1, 3, 3)),
    (triton_cumprod_4D, (256, 1, 1, 1)),
    (triton_cumprod_4D, (512, 1, 1, 1)),
    (triton_cumprod_4D, (11, 4, 22, 2)),
    (triton_cumprod_4D, (1, 1, 1, 256)),
    (triton_cumprod_4D, (1, 25, 2, 15)),
    # 5D
    (triton_cumprod_5D, (1, 1, 1, 1, 1)),
    (triton_cumprod_5D, (2, 4, 8, 4, 2)),
    (triton_cumprod_5D, (8, 4, 2, 4, 4)),
    (triton_cumprod_5D, (2, 8, 2, 2, 2)),
    (triton_cumprod_5D, (5, 3, 1, 2, 9)),
    (triton_cumprod_5D, (1, 8, 2, 2, 7)),
    (triton_cumprod_5D, (5, 1, 2, 9, 6)),
    (triton_cumprod_5D, (10, 2, 5, 1, 4)),
    (triton_cumprod_5D, (9, 7, 1, 5, 1)),
    (triton_cumprod_5D, (15, 1, 1, 21, 1)),
    (triton_cumprod_5D, (7, 1, 1, 1, 19)),
    (triton_cumprod_5D, (7, 4, 2, 1, 19)),
    (triton_cumprod_5D, (128, 1, 1, 1, 1)),
    (triton_cumprod_5D, (32, 1, 5, 1, 8)),
    (triton_cumprod_5D, (256, 1, 1, 1, 1)),
    (triton_cumprod_5D, (50, 1, 2, 2, 7)),
    # 6D
    (triton_cumprod_6D, (2, 2, 2, 2, 2, 2)),
    (triton_cumprod_6D, (2, 4, 8, 4, 2, 2)),
    # 7D
    (triton_cumprod_7D, (2, 2, 2, 2, 2, 2, 2)),
    (triton_cumprod_7D, (2, 4, 2, 4, 2, 8, 2)),
    # 8D
    (triton_cumprod_8D, (2, 2, 2, 2, 2, 2, 2, 2)),
    (triton_cumprod_8D, (2, 4, 2, 4, 2, 8, 2, 2)),
]
black_list = [
    (1, 1024),
    (2, 1031),
    (3, 630),
    (128, 4, 1),
    (67, 32, 2),
    (77, 22, 2),
    (45, 27, 6),
    (2, 3, 255),
    (3, 3, 256),
    (3, 2, 257),
    (24, 1, 51),
    (3, 2, 400),
    (8, 12, 1, 7),
    (72, 1, 19, 1),
    (18, 7, 2, 15),
    (240, 1, 3, 3),
    (256, 1, 1, 1),
    (512, 1, 1, 1),
    (11, 4, 22, 2),
    (2, 4, 8, 4, 2),
    (8, 4, 2, 4, 4),
    (10, 2, 5, 1, 4),
    (9, 7, 1, 5, 1),
    (7, 4, 2, 1, 19),
    (128, 1, 1, 1, 1),
    (32, 1, 5, 1, 8),
    (256, 1, 1, 1, 1),
    (50, 1, 2, 2, 7),
    (2, 4, 8, 4, 2, 2),
    (2, 4, 2, 4, 2, 8, 2),
    (2, 2, 2, 2, 2, 2, 2, 2),
    (2, 4, 2, 4, 2, 8, 2, 2),
]

typelist = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'bool', 'uint8', 'uint16', 'uint32',
            'uint64']
# typelist = ['bool']

ids = ["{}-{}".format(
    testfunc.__name__, "-".join(map(str, shape))
) for testfunc, shape in testlist
]


def torch_func(x, dim, reverse, dtype):
    if (dtype == torch.bfloat16) or (dtype == torch.float16) or (dtype == torch.float32):
        x = x.to(torch.float32)
    else:
        x = x.to(torch.int64)
    if reverse:
        x = torch.flip(x, [dim])
    res = torch.cumprod(x, dim=dim)
    res = res.to(dtype)
    return res


@pytest.mark.parametrize('testfunc, shape', testlist, ids=ids)
@pytest.mark.parametrize("dim", [-1, 0, 1, 2, 3, 7])
@pytest.mark.parametrize('sigtype', typelist)
@pytest.mark.parametrize("reverse", [False])
def test_cumprod(testfunc, sigtype, shape, dim, reverse):
    if sigtype in ['int8', 'bool'] and shape in black_list:
        pytest.skip("ub overflow, skipping.")
    if dim >= len(shape):
        pytest.skip("dim >= len(shape), skipping.")
    dtype = test_common.get_torch_typename(sigtype)
    # 生成数据
    x = test_common.generate_tensor(shape, sigtype, seed=20).npu()
    output = torch.zeros(shape, dtype=dtype).npu()
    # 生成标杆
    ans = torch_func(x, dim, reverse, dtype)
    # 生成结果
    testfunc[(1,)](x, output, dim, reverse, *shape)
    # 数据对比
    test_common.validate_cmp(sigtype, ans, output)
