import triton
import torch
import triton.language as tl
import math
import numpy as np
import scipy
import pytest
import test_common


@triton.jit
def triton_randint4x1d(out_ptr, seed, L: tl.constexpr):
    idx = tl.arange(0, L)
    rnd, _ = tl.randint4x(seed, idx)
    tl.store(out_ptr + idx, rnd)


@triton.jit
def triton_randint4x2d(out_ptr, seed, L: tl.constexpr, M: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    idx = lblk_idx[:, None] * M + mblk_idx[None, :]
    rnd, _ = tl.randint4x(seed, idx)
    tl.store(out_ptr + idx, rnd)


@triton.jit
def triton_randint4x3d(out_ptr, seed, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    idx = (
        lblk_idx[:, None, None] * (M * N)
        + mblk_idx[None, :, None] * N
        + nblk_idx[None, None, :]
    )
    rnd, _ = tl.randint4x(seed, idx)
    tl.store(out_ptr + idx, rnd)


@triton.jit
def triton_randint4x4d(
    out_ptr, seed, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, O: tl.constexpr
):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    oblk_idx = tl.arange(0, O)
    idx = (
        lblk_idx[:, None, None, None] * (M * N * O)
        + mblk_idx[None, :, None, None] * (N * O)
        + nblk_idx[None, None, :, None] * O
        + oblk_idx[None, None, None, :]
    )
    rnd, _ = tl.randint4x(seed, idx)
    tl.store(out_ptr + idx, rnd)


@triton.jit
def triton_randint4x5d(
    out_ptr,
    seed,
    L: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    O: tl.constexpr,
    P: tl.constexpr,
):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    oblk_idx = tl.arange(0, O)
    pblk_idx = tl.arange(0, P)
    idx = (
        lblk_idx[:, None, None, None, None] * (M * N * O * P)
        + mblk_idx[None, :, None, None, None] * (N * O * P)
        + nblk_idx[None, None, :, None, None] * (O * P)
        + oblk_idx[None, None, None, :, None] * P
        + pblk_idx[None, None, None, None, :]
    )
    rnd, _ = tl.randint4x(seed, idx)
    tl.store(out_ptr + idx, rnd)


@triton.jit
def triton_randint4x6d(
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

    rnd, _ = tl.randint4x(seed, idx)
    tl.store(out_ptr + idx, rnd)


@triton.jit
def triton_randint4x7d(
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

    rnd, _ = tl.randint4x(seed, idx)
    tl.store(out_ptr + idx, rnd)


@triton.jit
def triton_randint4x8d(
    out_ptr, seed,
    A: tl.constexpr, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr,
    E: tl.constexpr, F: tl.constexpr, G: tl.constexpr, H: tl.constexpr
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

    rnd, _ = tl.randint4x(seed, idx)
    tl.store(out_ptr + idx, rnd)


ii8 = np.iinfo(np.int8)
ii16 = np.iinfo(np.int16)
ii32 = np.iinfo(np.int32)
ii64 = np.iinfo(np.int64)

extrm_val_dict = {
    "int8": (ii8.min, ii8.max),
    "int16": (ii16.min, ii16.max),
    "int32": (ii32.min, ii32.max),
    "int64": (ii64.min, ii64.max),
    "bool": (0, 1),
}

# seed_list = [22, ]
seed_list = ['int8','uint8','int16','uint16','int32','uint32','int64','uint64','bool']
testlist = [
    # (1,),
    # (6,),
    # (3,),
    # (6,),
    # (9,),

    # (1, 1,),
    (3, 3),
    (2, 1),
    (1, 5),

    # (1, 1, 1,),
    # (7, 5, 1),
    # (1, 7, 3),
    # (3, 7, 5),
    (4, 5, 2),

    # (1, 1, 1, 1,),
    (7, 6, 4, 5),
    (7, 1, 2, 2),

    (1, 1, 1, 1, 1,),
    (6, 7, 5, 2, 1),
    # (4, 1, 3, 3, 4),
    (5, 2, 5, 6, 1),

    (1, 4, 3, 4, 3, 2),
    # (2, 4, 3, 1, 3, 2),
    (3, 3, 2, 1, 1, 1),
    # (3, 2, 1, 4, 1, 2),
    (1, 4, 3, 2, 4, 2),
    (2, 2, 2, 2, 2, 4),

    (4, 2, 2, 1, 1, 1, 3),
    (2, 1, 3, 1, 2, 1, 1),
    (1, 2, 1, 3, 1, 2, 2),

    (3, 1, 3, 1, 2, 1, 2, 2),
    (1, 1, 1, 2, 4, 2, 4, 4),
    # (2, 2, 3, 1, 1, 1, 3, 3),
    # (2, 1, 2, 1, 1, 2, 2, 2),
    (3, 3, 1, 1, 1, 2, 1, 3),
]
typelist = ['int32',]


@pytest.mark.parametrize('shape', testlist, ids=["-".join(map(str, arg)) for arg in testlist])
@pytest.mark.parametrize('sigtype', typelist)
@pytest.mark.parametrize('seed_type', seed_list)
def test_randint4x(sigtype, shape, seed_type):
    dtype = test_common.get_torch_typename(sigtype)
    low, high = extrm_val_dict[sigtype]
    test_num = 100
    threshold_num = math.floor(test_num * 0.95927)
    # seed_tensor = test_common.generate_tensor_new((1, ), seed_type)
    # seed = seed_tensor[0].item()
    seed = 1000
    if len(shape) == 1:
        L, = shape

        x1 = torch.empty((L,), dtype=dtype, device='npu')
        x2 = torch.empty((L,), dtype=dtype, device='npu')
        triton_randint4x1d[(1,)](x1, seed, L)
        triton_randint4x1d[(1,)](x2, seed, L)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)
            ref = torch.randint(low=low, high=high, size=(L,), dtype=dtype).flatten()
            x = torch.empty((L,), dtype=dtype, device='npu')
            triton_randint4x1d[(1,)](x, seed, L)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 2:
        L, M = shape

        x1 = torch.empty((L * M,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M,), dtype=dtype, device='npu')
        triton_randint4x2d[(1,)](x1, seed, L, M)
        triton_randint4x2d[(1,)](x2, seed, L, M)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)
            ref = torch.randint(low=low, high=high, size=(L, M,), dtype=dtype).flatten()
            x = torch.empty((L * M,), dtype=dtype, device='npu')
            triton_randint4x2d[(1,)](x, seed, L, M)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 3:
        L, M, N = shape

        x1 = torch.empty((L * M * N,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M * N,), dtype=dtype, device='npu')
        triton_randint4x3d[(1,)](x1, seed, L, M, N)
        triton_randint4x3d[(1,)](x2, seed, L, M, N)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)
            ref = torch.randint(low=low, high=high, size=(L, M, N,), dtype=dtype).flatten()
            x = torch.empty((L * M * N,), dtype=dtype, device='npu')
            triton_randint4x3d[(1,)](x, seed, L, M, N)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 4:
        L, M, N, K = shape

        x1 = torch.empty((L * M * N * K,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M * N * K,), dtype=dtype, device='npu')
        triton_randint4x4d[(1,)](x1, seed, L, M, N, K)
        triton_randint4x4d[(1,)](x2, seed, L, M, N, K)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)
            ref = torch.randint(low=low, high=high, size=(L, M, N, K,), dtype=dtype).flatten()
            x = torch.empty((L * M * N * K,), dtype=dtype, device='npu')
            triton_randint4x4d[(1,)](x, seed, L, M, N, K)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 5:
        L, M, N, K, Z = shape

        x1 = torch.empty((L * M * N * Z,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M * N * Z,), dtype=dtype, device='npu')
        triton_randint4x5d[(1,)](x1, seed, L, M, N, K, Z)
        triton_randint4x5d[(1,)](x2, seed, L, M, N, K, Z)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)
            ref = torch.randint(low=low, high=high, size=(L, M, N, K, Z,), dtype=dtype).flatten()
            x = torch.empty((L * M * N * Z,), dtype=dtype, device='npu')
            triton_randint4x5d[(1,)](x, seed, L, M, N, K, Z)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 6:
        L, M, N, O, K, Z = shape

        x1 = torch.empty((L * M * N * O * K * Z,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M * N * O * K * Z,), dtype=dtype, device='npu')
        triton_randint4x6d[(1,)](x1, seed, L, M, N, O, K, Z)
        triton_randint4x6d[(1,)](x2, seed, L, M, N, O, K, Z)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)
            ref = torch.randint(low=low, high=high, size=(L, M, N, O, K, Z,), dtype=dtype).flatten()
            x = torch.empty((L * M * N * O * Z,), dtype=dtype, device='npu')
            triton_randint4x6d[(1,)](x, seed, L, M, N, O, K, Z)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 7:
        L, M, N, O, P, K, Z = shape

        x1 = torch.empty((L * M * N * O * P * K * Z,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M * N * O * P * K * Z,), dtype=dtype, device='npu')
        triton_randint4x7d[(1,)](x1, seed, L, M, N, O, P, K, Z)
        triton_randint4x7d[(1,)](x2, seed, L, M, N, O, P, K, Z)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)
            ref = torch.randint(low=low, high=high, size=(L, M, N, O, P, K, Z,), dtype=dtype).flatten()
            x = torch.empty((L * M * N * O * P * K * Z,), dtype=dtype, device='npu')
            triton_randint4x7d[(1,)](x, seed, L, M, N, O, P, K, Z)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    elif len(shape) == 8:
        L, M, N, O, P, Q, K, Z = shape

        x1 = torch.empty((L * M * N * O * P * Q * K * Z,), dtype=dtype, device='npu')
        x2 = torch.empty((L * M * N * O * P * Q * K * Z,), dtype=dtype, device='npu')
        triton_randint4x8d[(1,)](x1, seed, L, M, N, O, P, Q, K, Z)
        triton_randint4x8d[(1,)](x2, seed, L, M, N, O, P, Q, K, Z)

        correctness = 0
        for _ in range(test_num):
            torch.manual_seed(seed)
            ref = torch.randint(low=low, high=high, size=(L, M, N, O, P, Q, K, Z,), dtype=dtype).flatten()
            x = torch.empty((L * M * N * O * P * Q * K * Z,), dtype=dtype, device='npu')
            triton_randint4x8d[(1,)](x, seed, L, M, N, O, P, Q, K, Z)

            pvalue = scipy.stats.kstest(ref, x.cpu().numpy().flatten()).pvalue
            if pvalue > 0.01:
                correctness += 1
        assert torch.equal(x1, x2) and correctness > threshold_num
    else:
        raise ValueError("shape not supported")
