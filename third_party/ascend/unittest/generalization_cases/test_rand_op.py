import triton
import torch
import triton.language as tl
import math
import numpy as np
import scipy
import pytest
import test_common


@triton.jit
def triton_rand1d(out_ptr, seed, L: tl.constexpr):
    idx = tl.arange(0, L)
    rnd = tl.rand(seed, idx)
    tl.store(out_ptr + idx, rnd)


@triton.jit
def triton_rand2d(out_ptr, seed, L: tl.constexpr, M: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    idx = lblk_idx[:, None] * M + mblk_idx[None, :]
    rnd = tl.rand(seed, idx)
    odx = lblk_idx[:, None] * M + mblk_idx[None, :]
    tl.store(out_ptr + odx, rnd)


@triton.jit
def triton_rand3d(out_ptr, seed, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lidx = tl.arange(0, L)
    midx = tl.arange(0, M)
    nidx = tl.arange(0, N)
    idx = lidx[:, None, None] * M * N + midx[None, :, None] * N + nidx[None, None, :]
    rnd = tl.rand(seed, idx)
    odx = lidx[:, None, None] * M * N + midx[None, :, None] * N + nidx[None, None, :]
    tl.store(out_ptr + odx, rnd)


@triton.jit
def triton_rand4d(out_ptr, seed, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, O: tl.constexpr):
    lidx = tl.arange(0, L)
    midx = tl.arange(0, M)
    nidx = tl.arange(0, N)
    oidx = tl.arange(0, O)
    idx = (lidx[:, None, None, None] * M * N * O +
           midx[None, :, None, None] * N * O +
           nidx[None, None, :, None] * O +
           oidx[None, None, None, :])
    rnd = tl.rand(seed, idx)
    odx = idx
    tl.store(out_ptr + odx, rnd)


@triton.jit
def triton_rand5d(out_ptr, seed, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, O: tl.constexpr, P: tl.constexpr):
    lidx = tl.arange(0, L)
    midx = tl.arange(0, M)
    nidx = tl.arange(0, N)
    oidx = tl.arange(0, O)
    pidx = tl.arange(0, P)
    idx = (lidx[:, None, None, None, None] * M * N * O * P +
           midx[None, :, None, None, None] * N * O * P +
           nidx[None, None, :, None, None] * O * P +
           oidx[None, None, None, :, None] * P +
           pidx[None, None, None, None, :])
    rnd = tl.rand(seed, idx)
    odx = idx
    tl.store(out_ptr + odx, rnd)


@triton.jit
def triton_rand6d(
    out_ptr, seed,
    A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
    D: tl.constexpr, E: tl.constexpr, F: tl.constexpr
):
    a_idx = tl.arange(0, A)
    b_idx = tl.arange(0, B)
    c_idx = tl.arange(0, C)
    d_idx = tl.arange(0, D)
    e_idx = tl.arange(0, E)
    f_idx = tl.arange(0, F)

    idx = (
        a_idx[:, None, None, None, None, None] * (B * C * D * E * F)
        + b_idx[None, :, None, None, None, None] * (C * D * E * F)
        + c_idx[None, None, :, None, None, None] * (D * E * F)
        + d_idx[None, None, None, :, None, None] * (E * F)
        + e_idx[None, None, None, None, :, None] * F
        + f_idx[None, None, None, None, None, :]
    )

    rnd = tl.rand(seed, idx)
    odx = idx
    tl.store(out_ptr + odx, rnd)


@triton.jit
def triton_rand7d(
    out_ptr, seed,
    A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
    D: tl.constexpr, E: tl.constexpr, F: tl.constexpr, G: tl.constexpr
):
    a_idx = tl.arange(0, A)
    b_idx = tl.arange(0, B)
    c_idx = tl.arange(0, C)
    d_idx = tl.arange(0, D)
    e_idx = tl.arange(0, E)
    f_idx = tl.arange(0, F)
    g_idx = tl.arange(0, G)

    idx = (
        a_idx[:, None, None, None, None, None, None] * (B * C * D * E * F * G)
        + b_idx[None, :, None, None, None, None, None] * (C * D * E * F * G)
        + c_idx[None, None, :, None, None, None, None] * (D * E * F * G)
        + d_idx[None, None, None, :, None, None, None] * (E * F * G)
        + e_idx[None, None, None, None, :, None, None] * (F * G)
        + f_idx[None, None, None, None, None, :, None] * G
        + g_idx[None, None, None, None, None, None, :]
    )

    rnd = tl.rand(seed, idx)
    odx = idx
    tl.store(out_ptr + odx, rnd)


@triton.jit
def triton_rand8d(
    out_ptr, seed,
    A: tl.constexpr, B: tl.constexpr, C: tl.constexpr,
    D: tl.constexpr, E: tl.constexpr, F: tl.constexpr,
    G: tl.constexpr, H: tl.constexpr
):
    a_idx = tl.arange(0, A)
    b_idx = tl.arange(0, B)
    c_idx = tl.arange(0, C)
    d_idx = tl.arange(0, D)
    e_idx = tl.arange(0, E)
    f_idx = tl.arange(0, F)
    g_idx = tl.arange(0, G)
    h_idx = tl.arange(0, H)

    idx = (
        a_idx[:, None, None, None, None, None, None, None] * (B * C * D * E * F * G * H)
        + b_idx[None, :, None, None, None, None, None, None] * (C * D * E * F * G * H)
        + c_idx[None, None, :, None, None, None, None, None] * (D * E * F * G * H)
        + d_idx[None, None, None, :, None, None, None, None] * (E * F * G * H)
        + e_idx[None, None, None, None, :, None, None, None] * (F * G * H)
        + f_idx[None, None, None, None, None, :, None, None] * (G * H)
        + g_idx[None, None, None, None, None, None, :, None] * H
        + h_idx[None, None, None, None, None, None, None, :]
    )

    rnd = tl.rand(seed, idx)
    odx = idx
    tl.store(out_ptr + odx, rnd)


seed_list = ['int8','uint8','int16','uint16','int32','uint32','int64','uint64','bool']
# seed_list = [22, 854, 1234, 15464]
testlist = [
    (1,),
    (12,),
    (6,),
    (2,),
    (4,),
    (3,),

    (1, 1),
    (2, 4),
    (3, 6),
    (6, 6),
    (4, 3),
    (3, 5),

    (1, 1, 1),
    (1, 1, 6),
    (2, 6, 3),
    (5, 4, 5),

    (1, 1, 1, 1),
    (3, 5, 5, 2),
    (2, 5, 1, 2),
    (5, 5, 1, 3),
    (4, 3, 5, 3),
    (4, 3, 5, 2),

    (1, 1, 1, 1, 1),
    (3, 1, 2, 4, 5),
    (1, 2, 5, 3, 2),

    (1, 4, 3, 4, 3, 2),
    (2, 4, 3, 1, 3, 2),
    (3, 3, 2, 1, 1, 1),
    (3, 2, 1, 4, 1, 2),
    (1, 4, 3, 2, 4, 2),
    (2, 2, 2, 2, 2, 4),

    (4, 2, 2, 1, 1, 1, 3),
    (2, 1, 3, 1, 2, 1, 1),
    (1, 2, 1, 3, 1, 2, 2),

    (3, 1, 3, 1, 2, 1, 2, 2),
    (1, 1, 1, 2, 4, 2, 4, 4),
    (2, 2, 3, 1, 1, 1, 3, 3),
    (2, 1, 2, 1, 1, 2, 2, 2),
    (3, 3, 1, 1, 1, 2, 1, 3),
]
typelist = ['float32']


@pytest.mark.parametrize('shape', testlist, ids=["-".join(map(str, arg)) for arg in testlist])
@pytest.mark.parametrize('sigtype', typelist)
@pytest.mark.parametrize('seed_type', seed_list)
def test_rand(sigtype, shape, seed_type):
    dtype = test_common.get_torch_typename(sigtype)
    test_num = 100
    threshold_num = math.floor(test_num * 0.95927)

    seed_tensor = test_common.generate_tensor_new((1, ), seed_type)
    #seed = seed_tensor[0].item()
    seed = 10000
    if len(shape) == 1:
        L, = shape

        # 验证相同seed两次生成结果一致
        x1 = torch.empty((L,), dtype=dtype, device='npu')
        x2 = torch.empty((L,), dtype=dtype, device='npu')
        triton_rand1d[(1,)](x1, seed, L)
        triton_rand1d[(1,)](x2, seed, L)

        # ktest测试
        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)

            if sigtype == "bool":
                ref = (torch.rand(size=(L,)) > 0.5).flatten()
            else:
                ref = torch.rand(size=(L,), dtype=dtype).flatten()

            x = torch.empty((L,), dtype=dtype, device='npu')
            triton_rand1d[(1,)](x, seed, L)

            if sigtype == "bfloat16":
                ref = ref.to(torch.float32)
                x = x.to(torch.float32)
            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 2:
        L, M = shape

        x1 = torch.empty((L * M,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M,), dtype=dtype, device='npu')
        triton_rand2d[(1,)](x1, seed, L, M)
        triton_rand2d[(1,)](x2, seed, L, M)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)

            if sigtype == "bool":
                ref = (torch.rand(size=(L, M)) > 0.5).flatten()
            else:
                ref = torch.rand(size=(L, M), dtype=dtype).flatten()

            x = torch.empty((L * M,), dtype=dtype, device='npu')
            triton_rand2d[(1,)](x, seed, L, M)

            if sigtype == "bfloat16":
                ref = ref.to(torch.float32)
                x = x.to(torch.float32)
            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 3:
        L, M, N = shape

        x1 = torch.empty((L * M * N,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M * N,), dtype=dtype, device='npu')
        triton_rand3d[(1,)](x1, seed, L, M, N)
        triton_rand3d[(1,)](x2, seed, L, M, N)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)

            if sigtype == "bool":
                ref = (torch.rand(size=(L, M, N)) > 0.5).flatten()
            else:
                ref = torch.rand(size=(L, M, N), dtype=dtype).flatten()

            x = torch.empty((L * M * N,), dtype=dtype, device='npu')
            triton_rand3d[(1,)](x, seed, L, M, N)

            if sigtype == "bfloat16":
                ref = ref.to(torch.float32)
                x = x.to(torch.float32)
            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 4:
        L, M, N, K = shape

        x1 = torch.empty((L * M * N * K,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M * N * K,), dtype=dtype, device='npu')
        triton_rand4d[(1,)](x1, seed, L, M, N, K)
        triton_rand4d[(1,)](x2, seed, L, M, N, K)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)

            if sigtype == "bool":
                ref = (torch.rand(size=(L, M, N, K)) > 0.5).flatten()
            else:
                ref = torch.rand(size=(L, M, N, K), dtype=dtype).flatten()

            x = torch.empty((L * M * N * K,), dtype=dtype, device='npu')
            triton_rand4d[(1,)](x, seed, L, M, N, K)

            if sigtype == "bfloat16":
                ref = ref.to(torch.float32)
                x = x.to(torch.float32)
            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 5:
        L, M, N, K, Z = shape

        x1 = torch.empty((L * M * N * Z,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M * N * Z,), dtype=dtype, device='npu')
        triton_rand5d[(1,)](x1, seed, L, M, N, K, Z)
        triton_rand5d[(1,)](x2, seed, L, M, N, K, Z)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)

            if sigtype == "bool":
                ref = (torch.rand(size=(L, M, N, K, Z)) > 0.5).flatten()
            else:
                ref = torch.rand(size=(L, M, N, K, Z), dtype=dtype).flatten()

            x = torch.empty((L * M * N * Z,), dtype=dtype, device='npu')
            triton_rand5d[(1,)](x, seed, L, M, N, K, Z)

            if sigtype == "bfloat16":
                ref = ref.to(torch.float32)
                x = x.to(torch.float32)
            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 6:
        A, B, C, D, E, F = shape

        x1 = torch.empty((A * B * C * D * E * F,), dtype=dtype, device='npu')
        x2 = torch.empty((A * B * C * D * E * F,), dtype=dtype, device='npu')
        triton_rand6d[(1,)](x1, seed, A, B, C, D, E, F)
        triton_rand6d[(1,)](x2, seed, A, B, C, D, E, F)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)

            if sigtype == "bool":
                ref = (torch.rand(size=(A, B, C, D, E, F)) > 0.5).flatten()
            else:
                ref = torch.rand(size=(A, B, C, D, E, F), dtype=dtype).flatten()

            x = torch.empty((A * B * C * D * E * F,), dtype=dtype, device='npu')
            triton_rand6d[(1,)](x, seed, A, B, C, D, E, F)

            if sigtype == "bfloat16":
                ref = ref.to(torch.float32)
                x = x.to(torch.float32)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1

        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 7:
        A, B, C, D, E, F, G = shape

        x1 = torch.empty((A * B * C * D * E * F * G,), dtype=dtype, device='npu')
        x2 = torch.empty((A * B * C * D * E * F * G,), dtype=dtype, device='npu')
        triton_rand7d[(1,)](x1, seed, A, B, C, D, E, F, G)
        triton_rand7d[(1,)](x2, seed, A, B, C, D, E, F, G)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)

            if sigtype == "bool":
                ref = (torch.rand(size=(A, B, C, D, E, F, G)) > 0.5).flatten()
            else:
                ref = torch.rand(size=(A, B, C, D, E, F, G), dtype=dtype).flatten()

            x = torch.empty((A * B * C * D * E * F * G,), dtype=dtype, device='npu')
            triton_rand7d[(1,)](x, seed, A, B, C, D, E, F, G)

            if sigtype == "bfloat16":
                ref = ref.to(torch.float32)
                x = x.to(torch.float32)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1

        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 8:
        A, B, C, D, E, F, G, H = shape

        x1 = torch.empty((A * B * C * D * E * F * G * H,), dtype=dtype, device='npu')
        x2 = torch.empty((A * B * C * D * E * F * G * H,), dtype=dtype, device='npu')
        triton_rand8d[(1,)](x1, seed, A, B, C, D, E, F, G, H)
        triton_rand8d[(1,)](x2, seed, A, B, C, D, E, F, G, H)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)

            if sigtype == "bool":
                ref = (torch.rand(size=(A, B, C, D, E, F, G, H)) > 0.5).flatten()
            else:
                ref = torch.rand(size=(A, B, C, D, E, F, G, H), dtype=dtype).flatten()

            x = torch.empty((A * B * C * D * E * F * G * H,), dtype=dtype, device='npu')
            triton_rand8d[(1,)](x, seed, A, B, C, D, E, F, G, H)

            if sigtype == "bfloat16":
                ref = ref.to(torch.float32)
                x = x.to(torch.float32)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1

        assert torch.equal(x1, x2) and correctness > threshold_num
    else:
        raise ValueError("shape not supported")
