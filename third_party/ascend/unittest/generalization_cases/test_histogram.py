import triton
import triton.language as tl
import torch
import pytest
import math
import test_common
from test_common import TestUtils


@triton.jit
def histogram_kernel(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, N)
    tl.store(z_ptr + offset2, z)


@triton.jit
def histogram_kernel_func(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = x.histogram(N)
    tl.store(z_ptr + offset2, z)


@triton.jit
def histogram_kernel_invalid(x_ptr, z_ptr, M: tl.constexpr, N: tl.constexpr):
    offset1 = tl.arange(0, M)
    offset2 = tl.arange(0, N)
    x = tl.load(x_ptr + offset1)
    z = tl.histogram(x, 2.1)
    tl.store(z_ptr + offset2, z)


@pytest.mark.parametrize("N", [2000])
@pytest.mark.parametrize("ncore", [1])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"])
@pytest.mark.parametrize('shape', TestUtils.test_shape1d)
def test_histogram(shape, N, ncore, dtype):
    torch.manual_seed(17)

    x = test_common.generate_tensor(shape, dtype).npu()
    # torch结果
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    y_cal = y_cal.to(eval(f'torch.{dtype}'))
    # triton结果
    y_ref = torch.zeros(N, dtype=eval(f'torch.int32'), device="npu")
    histogram_kernel[(ncore,)](x, y_ref, M=x.numel(), N=N)


@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("ncore", [1])
@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", ])
@pytest.mark.parametrize('M', [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 37, ])
def test_histogram_func(M, N, ncore, dtype):
    torch.manual_seed(17)
    x = torch.randint(low=0, high=N, size=(M,), dtype=eval(f'torch.{dtype}')).npu()

    # torch纾S彞~\
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    y_cal = y_cal.to(eval(f'torch.int32'))
    # triton纾S彞~\
    y_ref = torch.zeros(N, dtype=eval(f'torch.int32'), device="npu")
    histogram_kernel_func[(ncore,)](x, y_ref, M=M, N=N)
    test_common.validate_cmp(dtype, y_cal, y_ref)
    print("y_cal", y_cal)


@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("ncore", [1])
@pytest.mark.parametrize("dtype", ["int32"])
@pytest.mark.parametrize('M', [20, 90000])
def test_histogram_invalid(M, N, ncore, dtype):
    torch.manual_seed(17)
    x = torch.randint(low=0, high=N, size=(M,), dtype=eval(f'torch.{dtype}')).npu()

    # torch纾S弾^~\
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    # triton纾S弾^~\
    y_ref = torch.zeros(N, dtype=eval(f'torch.int32'), device="npu")
    if (M == 20):
        with pytest.raises(triton.compiler.errors.CompilationError,
                           match=r"create_histogram\(\)|incompatible function arguments|SupportsInt"):
            histogram_kernel_invalid[(ncore,)](x, y_ref, M=M, N=N)
    if (M == 90000):
        with pytest.raises(triton.compiler.errors.MLIRCompilationError, match="ub overflow"):
            histogram_kernel[(ncore,)](x, y_ref, M=M, N=N)


@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("ncore", [1])
@pytest.mark.parametrize("dtype", ["float32"])
@pytest.mark.parametrize('M', [(20,)])
def test_histogram_float(M, N, ncore, dtype):
    torch.manual_seed(17)
    x = torch.randint(low=0, high=N, size=M, dtype=eval(f'torch.{dtype}')).npu()

    # torch纾S弾^~\
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    y_ref = torch.zeros(N, dtype=eval(f'torch.int32'), device="npu")
    # triton纾S弾^
    if len(M) == 1:
        with pytest.raises(triton.compiler.errors.CompilationError, match="histogram only supports integer input"):
            histogram_kernel[(ncore,)](x, y_ref, M=M[0], N=N)


@pytest.mark.parametrize("N", [2])
@pytest.mark.parametrize("ncore", [1])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "uint64", ])
@pytest.mark.parametrize('M', [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 37, ])
def test_histogram_uint(M, N, ncore, dtype):
    torch.manual_seed(17)
    x = torch.randint(low=0, high=N, size=(M,), dtype=eval(f'torch.{dtype}'), device="cpu")
    x = x.to("npu")
    # torch纾S彞~\
    y_cal = torch.histc(x.float(), bins=N, min=0, max=N - 1)
    y_cal = y_cal.to(eval(f'torch.int32'))
    # triton纾S彞~\
    y_ref = torch.zeros(N, dtype=eval(f'torch.int32'), device="npu")
    histogram_kernel_func[(ncore,)](x, y_ref, M=M, N=N)
    test_common.validate_cmp(dtype, y_cal, y_ref)