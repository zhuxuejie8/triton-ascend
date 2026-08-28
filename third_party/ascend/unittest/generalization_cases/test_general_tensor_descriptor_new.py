import triton
import triton.language as tl
import torch
import torch_npu
import pytest
import test_common
from test_common import TestUtils


@triton.jit
def triton_tensor_descriptor_1d(
    out_ptr,
    x_ptr,
    SHAPE_0: tl.constexpr,
    STRIDE_0: tl.constexpr,
    BLOCK_0: tl.constexpr,
):
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[SHAPE_0, ],
        strides=[STRIDE_0, ],
        block_shape=[BLOCK_0, ],
        padding_option="zero",
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[SHAPE_0, ],
        strides=[STRIDE_0, ],
        block_shape=[BLOCK_0, ],
        padding_option="zero",
    )
    o1 = tl.program_id(0) * BLOCK_0
    value = in_desc.load([o1, ])
    out_desc.store([o1, ], value)


@triton.jit
def triton_tensor_descriptor_2d(
    out_ptr,
    x_ptr,
    SHAPE_0: tl.constexpr,
    SHAPE_1: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    BLOCK_0: tl.constexpr,
    BLOCK_1: tl.constexpr,
):
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[SHAPE_0, SHAPE_1],
        strides=[STRIDE_0, STRIDE_1],
        block_shape=[BLOCK_0, BLOCK_1],
        padding_option="zero",
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[SHAPE_0, SHAPE_1],
        strides=[STRIDE_0, STRIDE_1],
        block_shape=[BLOCK_0, BLOCK_1],
        padding_option="zero",
    )
    o1 = tl.program_id(0) * BLOCK_0
    o2 = tl.program_id(1) * BLOCK_1
    value = in_desc.load([o1, o2])
    out_desc.store([o1, o2], value)


@triton.jit
def triton_tensor_descriptor_3d(
    out_ptr,
    x_ptr,
    SHAPE_0: tl.constexpr,
    SHAPE_1: tl.constexpr,
    SHAPE_2: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    BLOCK_0: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
):
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[SHAPE_0, SHAPE_1, SHAPE_2],
        strides=[STRIDE_0, STRIDE_1, STRIDE_2],
        block_shape=[BLOCK_0, BLOCK_1, BLOCK_2],
        padding_option="zero",
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[SHAPE_0, SHAPE_1, SHAPE_2],
        strides=[STRIDE_0, STRIDE_1, STRIDE_2],
        block_shape=[BLOCK_0, BLOCK_1, BLOCK_2],
        padding_option="zero",
    )
    o1 = tl.program_id(0) * BLOCK_0
    o2 = tl.program_id(1) * BLOCK_1
    o3 = tl.program_id(2) * BLOCK_2
    value = in_desc.load([o1, o2, o3])
    out_desc.store([o1, o2, o3], value)


@triton.jit
def triton_tensor_descriptor_4d(
    out_ptr,
    x_ptr,
    SHAPE_0: tl.constexpr,
    SHAPE_1: tl.constexpr,
    SHAPE_2: tl.constexpr,
    SHAPE_3: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr,
    BLOCK_0: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
    BLOCK_3: tl.constexpr,
):
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3],
        strides=[STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3],
        block_shape=[BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3],
        padding_option="zero",
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3],
        strides=[STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3],
        block_shape=[BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3],
        padding_option="zero",
    )
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    idx2 = pid2 // BLOCK_3
    idx3 = pid2 % BLOCK_3
    o1 = pid0 * BLOCK_0
    o2 = pid1 * BLOCK_1
    o3 = idx2 * BLOCK_2
    o4 = idx3 * BLOCK_3
    value = in_desc.load([o1, o2, o3, o4])
    out_desc.store([o1, o2, o3, o4], value)


@triton.jit
def triton_tensor_descriptor_5d(
    out_ptr,
    x_ptr,
    SHAPE_0: tl.constexpr,
    SHAPE_1: tl.constexpr,
    SHAPE_2: tl.constexpr,
    SHAPE_3: tl.constexpr,
    SHAPE_4: tl.constexpr,
    STRIDE_0: tl.constexpr,
    STRIDE_1: tl.constexpr,
    STRIDE_2: tl.constexpr,
    STRIDE_3: tl.constexpr,
    STRIDE_4: tl.constexpr,
    BLOCK_0: tl.constexpr,
    BLOCK_1: tl.constexpr,
    BLOCK_2: tl.constexpr,
    BLOCK_3: tl.constexpr,
    BLOCK_4: tl.constexpr,
):
    in_desc = tl.make_tensor_descriptor(
        x_ptr,
        shape=[SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3, SHAPE_4],
        strides=[STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3, STRIDE_4],
        block_shape=[BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3, BLOCK_4],
        padding_option="zero",
    )
    out_desc = tl.make_tensor_descriptor(
        out_ptr,
        shape=[SHAPE_0, SHAPE_1, SHAPE_2, SHAPE_3, SHAPE_4],
        strides=[STRIDE_0, STRIDE_1, STRIDE_2, STRIDE_3, STRIDE_4],
        block_shape=[BLOCK_0, BLOCK_1, BLOCK_2, BLOCK_3, BLOCK_4],
        padding_option="zero",
    )
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    pid2 = tl.program_id(2)
    idx3 = pid2 // (BLOCK_3 * BLOCK_4)
    idx4 = (pid2 // BLOCK_4) % BLOCK_3
    idx5 = pid2 % BLOCK_4
    o1 = pid0 * BLOCK_0
    o2 = pid1 * BLOCK_1
    o3 = idx3 * BLOCK_2
    o4 = idx4 * BLOCK_3
    o5 = idx5 * BLOCK_4
    value = in_desc.load([o1, o2, o3, o4, o5])
    out_desc.store([o1, o2, o3, o4, o5], value)


shape_list = [
    # 1D
    (1,), (2,), (17,), (32,), (64,), (77,), (111,), (128,), (256,), (321,), (512,), (1024,), (2000,), (2159,),
    # 2D
    (1, 1), (2, 2), (2, 3), (5, 3), (27, 3), (3, 5), (6, 9), (13, 151), (16, 16), (16, 17), (32, 32), (57, 3), (64, 16), (128, 8), (256, 4), (512, 2), (1024, 1), (1, 1024), (2, 1031), (3, 630), (1000, 2),
    # 3D
    (1, 1, 1), (1, 1, 23), (23, 1, 1), (1, 23, 1), (5, 8, 7), (27, 8, 2), (3, 2, 5), (6, 9, 2), (1, 22, 39), (27, 1, 39), (27, 22, 1), (1, 1, 39), (1, 22, 1), (27, 1, 1), (37, 5, 3), (2, 29, 4), (7, 31, 7), (3, 5, 8), (7, 17, 15), (25, 5, 16), (15, 5, 31), (4, 11, 32), (7, 5, 33), (2, 3, 255), (3, 3, 256), (3, 2, 257), (24, 1, 51), (2, 524, 2), (3, 2, 400),
    # 4D
    (1, 1, 1, 1), (1, 4, 4, 4), (2, 2, 2, 2), (2, 2, 8, 4), (4, 4, 2, 4), (4, 4, 4, 1), (4, 2, 18, 11), (1, 4, 5, 7), (8, 4, 2, 2), (11, 1, 2, 5), (16, 2, 2, 2), (32, 2, 1, 1), (1, 1, 64, 1), (65, 1, 1, 3), (3, 1, 111, 3), (1, 1, 128, 1), (240, 1, 3, 3), (256, 1, 1, 1), (512, 1, 1, 1), (11, 4, 22, 2), (1, 1, 1, 256), (1, 1, 1078, 1), (1, 25, 2, 15),
    # 5D
    (1, 1, 1, 1, 1), (2, 2, 2, 2, 2), (4, 2, 2, 2, 2), (8, 2, 1, 1, 1), (5, 3, 2, 4, 17), (16, 1, 1, 1, 1), (32, 1, 1, 1, 1), (3, 5, 1, 3, 42), (64, 1, 1, 1, 1), (2, 13, 3, 13, 1), (128, 1, 1, 1, 1), (32, 1, 5, 1, 8), (256, 1, 1, 1, 1), (50, 1, 2, 2, 7), (512, 1, 1, 1, 1), (3, 1, 1, 2, 250), (1, 1, 1, 1, 256), (1, 2, 2, 2, 64), (15, 2, 2, 2, 13), (2, 2, 2, 4, 4), (2, 2, 4, 2, 4), (2, 11, 4, 2, 4), (1, 1, 3, 2, 125), (5, 2, 4, 1, 26), (2, 2, 2, 4, 8)
]

dtype_list_A3 = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'uint8']
dtype_list_A5 = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'bfloat16', 'uint8', 'uint16', 'uint32', 'uint64', 'fp8e5m2', 'fp8e4m3']

ub_list_A3 = [
    ('int8', (1, 2, 2, 2, 64)),
    #('int8', (8, 8, 8, 2, 8)),
    # ('int8', (8, 8, 8, 12, 32)),
    # ('int16', (4, 8, 8, 12, 32)),
    # ('int32', (4, 4, 8, 12, 32)),
    # ('int64', (4, 4, 4, 12, 32)),
    # ('float16', (4, 8, 8, 12, 32)),
    # ('bfloat16', (4, 8, 8, 12, 32)),
    # ('float32', (4, 4, 8, 12, 32)),
    # ('uint8', (8, 8, 8, 12, 32)),
]

ub_list_A5 = [
    ('int8', (1, 2, 2, 2, 64)),
    # ('int8', (7, 11, 103, 32)),
    # ('int16', (7, 11, 103, 16)),
    # ('int32', (7, 11, 103, 8)),
    # ('int64', (7, 11, 103, 4)),
    # ('float16', (7, 11, 103, 16)),
    # ('bfloat16', (7, 11, 103, 16)),
    # ('float32', (7, 11, 103, 8)),
    # ('uint8', (7, 11, 103, 32)),
    # ('uint16', (7, 11, 103, 16)),
    # ('uint32', (7, 11, 103, 8)),
    # ('uint64', (7, 11, 103, 4)),
]

if "Ascend910_95" in test_common.get_version():
    ub_list = ub_list_A5
    dtype_list = dtype_list_A5
else:
    ub_list = ub_list_A3
    dtype_list = dtype_list_A3


@pytest.mark.parametrize("shape", shape_list)
@pytest.mark.parametrize("dtype", dtype_list)
def test_tensor_descriptor(dtype, shape):
    if shape[-1] * test_common.get_dtype_size(dtype) >= 16:
        _tensor_descriptor_valid(dtype, shape)
    else:
        _tensor_descriptor_invalid(dtype, shape)


def _tensor_descriptor_valid(dtype, shape):
    x = test_common.generate_tensor(shape, dtype).npu()
    y_ref = x
    blocks = list(x.size())
    strides = list(x.stride())
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    if len(shape) == 1:
        triton_tensor_descriptor_1d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 2:
        triton_tensor_descriptor_2d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 3:
        triton_tensor_descriptor_3d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 4:
        triton_tensor_descriptor_4d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 5:
        triton_tensor_descriptor_5d[(1,)](y_cal, x, *shape, *strides, *blocks)
    if dtype in ['fp8e4m3', 'fp8e5m2']:
        y_cal = y_cal.to(torch.float32)
        y_ref = y_ref.to(torch.float32)
    test_common.validate_cmp(dtype, y_cal, y_ref)


@test_common.raises_with_match(triton.compiler.errors.CompilationError, "Descriptor block shape must have at least 16 bytes in the last dimension, but got")
def _tensor_descriptor_invalid(dtype, shape):
    x = test_common.generate_tensor(shape, dtype).npu()
    y_ref = x
    blocks = list(x.size())
    strides = list(x.stride())
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    if len(shape) == 1:
        triton_tensor_descriptor_1d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 2:
        triton_tensor_descriptor_2d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 3:
        triton_tensor_descriptor_3d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 4:
        triton_tensor_descriptor_4d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 5:
        triton_tensor_descriptor_5d[(1,)](y_cal, x, *shape, *strides, *blocks)


@pytest.mark.parametrize('dtype, shape', ub_list)
def test_tensor_descriptor_ub(dtype, shape):
    x = test_common.generate_tensor(shape, dtype).npu()
    y_ref = x
    blocks = list(x.size())
    strides = list(x.stride())
    y_cal = test_common.generate_tensor(shape, dtype).npu()
    if len(shape) == 1:
        triton_tensor_descriptor_1d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 2:
        triton_tensor_descriptor_2d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 3:
        triton_tensor_descriptor_3d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 4:
        triton_tensor_descriptor_4d[(1,)](y_cal, x, *shape, *strides, *blocks)
    elif len(shape) == 5:
        triton_tensor_descriptor_5d[(1,)](y_cal, x, *shape, *strides, *blocks)

    if dtype in ['fp8e4m3', 'fp8e5m2']:
        y_cal = y_cal.to(torch.float32)
        y_ref = y_ref.to(torch.float32)

    test_common.validate_cmp(dtype, y_cal, y_ref)