import triton
import triton.language as tl
import torch
import pytest
import test_common
from test_common import TestUtils
import math
import numpy as np
import scipy


@triton.jit
def kernel_rand(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.rand(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals)  # 存储随机数


@triton.jit
def triton_rand_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.rand(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def kernel_randn(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randn(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals)  # 存储随机数


@triton.jit
def triton_randn_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.randn(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def kernel_randint(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(block_size):
        global_offset = block_offset + inner_idx
        rand_vals = tl.randint(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        tl.store(x_ptr + global_offset, rand_vals)  # 存储随机数


@triton.jit
def triton_randint_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret = tl.randint(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def kernel_randint4x(x_ptr, n_rounds: tl.constexpr, N: tl.constexpr, XBLOCK: tl.constexpr):
    block_offset = tl.program_id(0) * XBLOCK
    indices = tl.arange(0, 4)
    block_size = XBLOCK if block_offset + XBLOCK <= N else N - block_offset
    for inner_idx in range(0, block_size + 4, step=4):
        global_offset = block_offset + inner_idx
        rand_vals, _ = tl.randint4x(5, 10 + global_offset, n_rounds)  # 对每个索引生成一个随机数
        mask = (global_offset + indices) < (block_offset + block_size)
        tl.store(x_ptr + global_offset + indices, rand_vals, mask)  # 存储随机数


@triton.jit
def triton_randint4x_4d_5d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr,
        BLOCK_3: tl.constexpr, BLOCK_4: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr,
        SHAPE_3: tl.constexpr, SHAPE_4: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr,
        STRIDE_3: tl.constexpr, STRIDE_4: tl.constexpr
):
    # 1D program_id for flatten multi-d offset
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0
    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if BLOCK_4 > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)

    ret, _ = tl.randint4x(5, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


# With alpha=0.01, z=-3.0902, N=100, we have (1-0.01)+(-3.0902)*sqrt(0.01*(1-0.01)/100)=0.9593,
# so there must be 96 cases for each shape to have pvalue larger than 0.01.
# There is higher possibility to fail with small shapes, so we will use large shape.
@pytest.mark.parametrize('shape', [
    (256, 256),
    (512, 512),
    (1024, 1024),
])
def test_rand_case(shape):
    y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()

    numel = y_calf.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    correctness = 0
    for _ in range(100):
        ref = np.random.random_sample(shape).flatten()
        kernel_rand[ncore, 1, 1](y_calf, 10, numel, xblock)

        pvalue = scipy.stats.kstest(ref, y_calf.cpu().numpy().flatten()).pvalue
        if pvalue > 0.01:
            correctness += 1

    assert correctness > 95


@pytest.mark.parametrize('shape', [
    (256, 256),
    (512, 512),
    (1024, 1024),
])
def test_randn_case(shape):
    y_calf = torch.zeros(shape, dtype=eval('torch.float32')).npu()

    numel = y_calf.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    correctness = 0
    for _ in range(100):
        ref = np.random.standard_normal(shape).flatten()
        kernel_randn[ncore, 1, 1](y_calf, 10, numel, xblock)

        pvalue = scipy.stats.kstest(ref, y_calf.cpu().numpy().flatten()).pvalue
        if pvalue > 0.01:
            correctness += 1

    assert correctness > 95


@pytest.mark.parametrize('shape', [
    (256, 256),
    (512, 512),
    (1024, 1024),
])
def test_randint_case(shape):
    y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    numel = y_cali.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    correctness = 0
    ii32 = np.iinfo(np.int32)
    for _ in range(100):
        ref = np.random.randint(low=ii32.min, high=ii32.max, size=shape).flatten()
        kernel_randint[ncore, 1, 1](y_cali, 10, numel, xblock)

        pvalue = scipy.stats.kstest(ref, y_cali.cpu().numpy().flatten()).pvalue
        if pvalue > 0.01:
            correctness += 1

    assert correctness > 95


@pytest.mark.parametrize('shape', [
    (256, 256),
    (512, 512),
    (1024, 1024),
])
def test_randint4x_case(shape):
    y_cali = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    numel = y_cali.numel()
    ncore = 1 if numel < 32 else 32
    xblock = math.ceil(numel / ncore)

    correctness = 0
    ii32 = np.iinfo(np.int32)
    for _ in range(100):
        ref = np.random.randint(low=ii32.min, high=ii32.max, size=shape).flatten()
        kernel_randint4x[ncore, 1, 1](y_cali, 10, numel, xblock)

        pvalue = scipy.stats.kstest(ref, y_cali.cpu().numpy().flatten()).pvalue
        if pvalue > 0.01:
            correctness += 1

    assert correctness > 95


@pytest.mark.parametrize('shape', TestUtils.test_shape4d + TestUtils.test_shape5d)
def test_rand_4d_5d(shape):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 5:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_rand_4d_5d[grid](x, *blocks, *blocks, *strides)
    triton_randn_4d_5d[grid](x, *blocks, *blocks, *strides)
    triton_randint_4d_5d[grid](y, *blocks, *blocks, *strides)
    triton_randint4x_4d_5d[grid](y, *blocks, *blocks, *strides)


@triton.jit
def triton_randint4x_8d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr, BLOCK_5: tl.constexpr, BLOCK_6: tl.constexpr, BLOCK_7: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr, SHAPE_5: tl.constexpr, SHAPE_6: tl.constexpr, SHAPE_7: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr, STRIDE_5: tl.constexpr, STRIDE_6: tl.constexpr, STRIDE_7: tl.constexpr
):
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0

    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if (BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)
    if (BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, None, :] * STRIDE_5
        mask = mask[:, :, :, :, :, None] & (tl.arange(0, BLOCK_5)[None, None, None, None, None, :] < SHAPE_5)
    if (BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, :, None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, None,
                                                    :] * STRIDE_6
        mask = mask[:, :, :, :, :, :, None] & (tl.arange(0, BLOCK_6)[None, None, None, None, None, None, :] < SHAPE_6)
    if BLOCK_7 > 1:
        offsets = offsets[:, :, :, :, :, :, :, None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None,
                                                       :] * STRIDE_7
        mask = mask[:, :, :, :, :, :, :, None] & (
                    tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None, :] < SHAPE_7)

    ret, _ = tl.randint4x(8, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def triton_randint_8d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr, BLOCK_5: tl.constexpr, BLOCK_6: tl.constexpr, BLOCK_7: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr, SHAPE_5: tl.constexpr, SHAPE_6: tl.constexpr, SHAPE_7: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr, STRIDE_5: tl.constexpr, STRIDE_6: tl.constexpr, STRIDE_7: tl.constexpr
):
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0

    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if (BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)
    if (BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, None, :] * STRIDE_5
        mask = mask[:, :, :, :, :, None] & (tl.arange(0, BLOCK_5)[None, None, None, None, None, :] < SHAPE_5)
    if (BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, :, None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, None,
                                                    :] * STRIDE_6
        mask = mask[:, :, :, :, :, :, None] & (tl.arange(0, BLOCK_6)[None, None, None, None, None, None, :] < SHAPE_6)
    if BLOCK_7 > 1:
        offsets = offsets[:, :, :, :, :, :, :, None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None,
                                                       :] * STRIDE_7
        mask = mask[:, :, :, :, :, :, :, None] & (
                    tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None, :] < SHAPE_7)

    ret = tl.randint(8, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def triton_randn_8d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr, BLOCK_5: tl.constexpr, BLOCK_6: tl.constexpr, BLOCK_7: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr, SHAPE_5: tl.constexpr, SHAPE_6: tl.constexpr, SHAPE_7: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr, STRIDE_5: tl.constexpr, STRIDE_6: tl.constexpr, STRIDE_7: tl.constexpr
):
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0

    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if (BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)
    if (BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, None, :] * STRIDE_5
        mask = mask[:, :, :, :, :, None] & (tl.arange(0, BLOCK_5)[None, None, None, None, None, :] < SHAPE_5)
    if (BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, :, None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, None,
                                                    :] * STRIDE_6
        mask = mask[:, :, :, :, :, :, None] & (tl.arange(0, BLOCK_6)[None, None, None, None, None, None, :] < SHAPE_6)
    if BLOCK_7 > 1:
        offsets = offsets[:, :, :, :, :, :, :, None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None,
                                                       :] * STRIDE_7
        mask = mask[:, :, :, :, :, :, :, None] & (
                    tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None, :] < SHAPE_7)

    ret = tl.randn(8, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def triton_rand_8d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr, BLOCK_5: tl.constexpr, BLOCK_6: tl.constexpr, BLOCK_7: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr, SHAPE_5: tl.constexpr, SHAPE_6: tl.constexpr, SHAPE_7: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr, STRIDE_5: tl.constexpr, STRIDE_6: tl.constexpr, STRIDE_7: tl.constexpr
):
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0

    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if (BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)
    if (BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, None, :] * STRIDE_5
        mask = mask[:, :, :, :, :, None] & (tl.arange(0, BLOCK_5)[None, None, None, None, None, :] < SHAPE_5)
    if (BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, :, None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, None,
                                                    :] * STRIDE_6
        mask = mask[:, :, :, :, :, :, None] & (tl.arange(0, BLOCK_6)[None, None, None, None, None, None, :] < SHAPE_6)
    if BLOCK_7 > 1:
        offsets = offsets[:, :, :, :, :, :, :, None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None,
                                                       :] * STRIDE_7
        mask = mask[:, :, :, :, :, :, :, None] & (
                    tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None, :] < SHAPE_7)

    ret = tl.rand(8, offsets, 10)
    tl.store(output_ptr + offsets, ret, mask=mask)


@pytest.mark.parametrize("shape", TestUtils.test_shape6d + TestUtils.test_shape7d + TestUtils.test_shape8d)
def test_rand_8d(shape):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 8:
        blocks.append(1)
        strides.append(1)

    grid = (32, 1, 1)
    triton_rand_8d[grid](x, *blocks, *blocks, *strides)


@pytest.mark.parametrize("shape", TestUtils.test_shape6d + TestUtils.test_shape7d + TestUtils.test_shape8d)
def test_randn_8d(shape):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 8:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_randn_8d[grid](x, *blocks, *blocks, *strides)


@pytest.mark.parametrize("shape", TestUtils.test_shape6d + TestUtils.test_shape7d + TestUtils.test_shape8d)
def test_randint_8d(shape):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 8:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_randint_8d[grid](y, *blocks, *blocks, *strides)


@pytest.mark.parametrize("shape", TestUtils.test_shape6d + TestUtils.test_shape7d + TestUtils.test_shape8d)
def test_randint4x_8d(shape):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 8:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_randint4x_8d[grid](y, *blocks, *blocks, *strides)

#===================================================================
#参数泛化

@triton.jit
def triton_randint4x_8d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr, BLOCK_5: tl.constexpr, BLOCK_6: tl.constexpr, BLOCK_7: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr, SHAPE_5: tl.constexpr, SHAPE_6: tl.constexpr, SHAPE_7: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr, STRIDE_5: tl.constexpr, STRIDE_6: tl.constexpr, STRIDE_7: tl.constexpr,
        seed: tl.constexpr, n_rounds: tl.constexpr
):
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0

    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if (BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)
    if (BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, None, :] * STRIDE_5
        mask = mask[:, :, :, :, :, None] & (tl.arange(0, BLOCK_5)[None, None, None, None, None, :] < SHAPE_5)
    if (BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, :, None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, None,
                                                    :] * STRIDE_6
        mask = mask[:, :, :, :, :, :, None] & (tl.arange(0, BLOCK_6)[None, None, None, None, None, None, :] < SHAPE_6)
    if BLOCK_7 > 1:
        offsets = offsets[:, :, :, :, :, :, :, None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None,
                                                       :] * STRIDE_7
        mask = mask[:, :, :, :, :, :, :, None] & (
                    tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None, :] < SHAPE_7)

    ret, _ = tl.randint4x(seed, offsets, n_rounds)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def triton_randint_8d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr, BLOCK_5: tl.constexpr, BLOCK_6: tl.constexpr, BLOCK_7: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr, SHAPE_5: tl.constexpr, SHAPE_6: tl.constexpr, SHAPE_7: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr, STRIDE_5: tl.constexpr, STRIDE_6: tl.constexpr, STRIDE_7: tl.constexpr,
        seed: tl.constexpr, n_rounds: tl.constexpr
):
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0

    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if (BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)
    if (BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, None, :] * STRIDE_5
        mask = mask[:, :, :, :, :, None] & (tl.arange(0, BLOCK_5)[None, None, None, None, None, :] < SHAPE_5)
    if (BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, :, None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, None,
                                                    :] * STRIDE_6
        mask = mask[:, :, :, :, :, :, None] & (tl.arange(0, BLOCK_6)[None, None, None, None, None, None, :] < SHAPE_6)
    if BLOCK_7 > 1:
        offsets = offsets[:, :, :, :, :, :, :, None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None,
                                                       :] * STRIDE_7
        mask = mask[:, :, :, :, :, :, :, None] & (
                    tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None, :] < SHAPE_7)

    ret = tl.randint(seed, offsets, n_rounds)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def triton_randn_8d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr, BLOCK_5: tl.constexpr, BLOCK_6: tl.constexpr, BLOCK_7: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr, SHAPE_5: tl.constexpr, SHAPE_6: tl.constexpr, SHAPE_7: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr, STRIDE_5: tl.constexpr, STRIDE_6: tl.constexpr, STRIDE_7: tl.constexpr,
        seed: tl.constexpr, n_rounds: tl.constexpr
):
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0

    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if (BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)
    if (BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, None, :] * STRIDE_5
        mask = mask[:, :, :, :, :, None] & (tl.arange(0, BLOCK_5)[None, None, None, None, None, :] < SHAPE_5)
    if (BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, :, None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, None,
                                                    :] * STRIDE_6
        mask = mask[:, :, :, :, :, :, None] & (tl.arange(0, BLOCK_6)[None, None, None, None, None, None, :] < SHAPE_6)
    if BLOCK_7 > 1:
        offsets = offsets[:, :, :, :, :, :, :, None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None,
                                                       :] * STRIDE_7
        mask = mask[:, :, :, :, :, :, :, None] & (
                    tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None, :] < SHAPE_7)

    ret = tl.randn(seed, offsets, n_rounds)
    tl.store(output_ptr + offsets, ret, mask=mask)


@triton.jit
def triton_rand_8d(
        output_ptr,
        BLOCK_0: tl.constexpr, BLOCK_1: tl.constexpr, BLOCK_2: tl.constexpr, BLOCK_3: tl.constexpr,
        BLOCK_4: tl.constexpr, BLOCK_5: tl.constexpr, BLOCK_6: tl.constexpr, BLOCK_7: tl.constexpr,
        SHAPE_0: tl.constexpr, SHAPE_1: tl.constexpr, SHAPE_2: tl.constexpr, SHAPE_3: tl.constexpr,
        SHAPE_4: tl.constexpr, SHAPE_5: tl.constexpr, SHAPE_6: tl.constexpr, SHAPE_7: tl.constexpr,
        STRIDE_0: tl.constexpr, STRIDE_1: tl.constexpr, STRIDE_2: tl.constexpr, STRIDE_3: tl.constexpr,
        STRIDE_4: tl.constexpr, STRIDE_5: tl.constexpr, STRIDE_6: tl.constexpr, STRIDE_7: tl.constexpr,
        seed: tl.constexpr, n_rounds: tl.constexpr
):
    pid = tl.program_id(0)
    # base offset for dimension 0
    offsets = pid + tl.arange(0, BLOCK_0) * STRIDE_0
    mask = tl.arange(0, BLOCK_0) < SHAPE_0

    # nested offset expansion
    if (BLOCK_1 * BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, None] + tl.arange(0, BLOCK_1)[None, :] * STRIDE_1
        mask = mask[:, None] & (tl.arange(0, BLOCK_1)[None, :] < SHAPE_1)
    if (BLOCK_2 * BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, None] + tl.arange(0, BLOCK_2)[None, None, :] * STRIDE_2
        mask = mask[:, :, None] & (tl.arange(0, BLOCK_2)[None, None, :] < SHAPE_2)
    if (BLOCK_3 * BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, None] + tl.arange(0, BLOCK_3)[None, None, None, :] * STRIDE_3
        mask = mask[:, :, :, None] & (tl.arange(0, BLOCK_3)[None, None, None, :] < SHAPE_3)
    if (BLOCK_4 * BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, None] + tl.arange(0, BLOCK_4)[None, None, None, None, :] * STRIDE_4
        mask = mask[:, :, :, :, None] & (tl.arange(0, BLOCK_4)[None, None, None, None, :] < SHAPE_4)
    if (BLOCK_5 * BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, None] + tl.arange(0, BLOCK_5)[None, None, None, None, None, :] * STRIDE_5
        mask = mask[:, :, :, :, :, None] & (tl.arange(0, BLOCK_5)[None, None, None, None, None, :] < SHAPE_5)
    if (BLOCK_6 * BLOCK_7) > 1:
        offsets = offsets[:, :, :, :, :, :, None] + tl.arange(0, BLOCK_6)[None, None, None, None, None, None,
                                                    :] * STRIDE_6
        mask = mask[:, :, :, :, :, :, None] & (tl.arange(0, BLOCK_6)[None, None, None, None, None, None, :] < SHAPE_6)
    if BLOCK_7 > 1:
        offsets = offsets[:, :, :, :, :, :, :, None] + tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None,
                                                       :] * STRIDE_7
        mask = mask[:, :, :, :, :, :, :, None] & (
                    tl.arange(0, BLOCK_7)[None, None, None, None, None, None, None, :] < SHAPE_7)

    ret = tl.rand(seed, offsets, n_rounds)
    tl.store(output_ptr + offsets, ret, mask=mask)


seed_list = [0, 1, 22, 854, 1234, 15464, 4294967295, -4294967294, -3]
n_rounds = [4, 10, 15, 20]
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
@pytest.mark.parametrize("shape", TestUtils.test_shape6d + TestUtils.test_shape7d + TestUtils.test_shape8d + testlist)
@pytest.mark.parametrize("seed", seed_list)
@pytest.mark.parametrize("n_rounds", n_rounds)
def test_rand_8d(shape, seed, n_rounds):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    print('11111', strides)
    while len(blocks) < 8:
        blocks.append(1)
        strides.append(1)

    grid = (32, 1, 1)
    triton_rand_8d[grid](x, *blocks, *blocks, *strides, seed, n_rounds)


@pytest.mark.parametrize("shape", TestUtils.test_shape6d + TestUtils.test_shape7d + TestUtils.test_shape8d + testlist)
@pytest.mark.parametrize("seed", seed_list)
@pytest.mark.parametrize("n_rounds", n_rounds)
def test_randn_8d(shape, seed, n_rounds):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    print('111111 ', strides)
    while len(blocks) < 8:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_randn_8d[grid](x, *blocks, *blocks, *strides, seed, n_rounds)


@pytest.mark.parametrize("shape", TestUtils.test_shape6d + TestUtils.test_shape7d + TestUtils.test_shape8d + testlist)
@pytest.mark.parametrize("seed", seed_list)
@pytest.mark.parametrize("n_rounds", n_rounds)
def test_randint_8d(shape, seed, n_rounds):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 8:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_randint_8d[grid](y, *blocks, *blocks, *strides, seed, n_rounds)


@pytest.mark.parametrize("shape", TestUtils.test_shape6d + TestUtils.test_shape7d + TestUtils.test_shape8d + testlist)
@pytest.mark.parametrize("seed", seed_list)
@pytest.mark.parametrize("n_rounds", n_rounds)
def test_randint4x_8d(shape, seed, n_rounds):
    x = torch.zeros(shape, dtype=eval('torch.float32')).npu()
    y = torch.zeros(shape, dtype=eval('torch.int32')).npu()

    blocks = list(x.size())
    strides = list(x.stride())
    while len(blocks) < 8:
        blocks.append(1)
        strides.append(1)

    grid = (1,)
    triton_randint4x_8d[grid](y, *blocks, *blocks, *strides, seed, n_rounds)