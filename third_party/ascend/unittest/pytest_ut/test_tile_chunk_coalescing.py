# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest
import torch
import triton
import triton.language as tl

import test_common
<<<<<<< HEAD:third_party/ascend/unittest/generalization_cases/test_general_arange.py
from test_common import TestUtils


def torch_pointwise(length):
    res = (torch.arange(0, length) / 2.7) * torch.arange(0, length)
    return res


def torch_arange(start, end):
    TRITON_MAX_TENSOR_NUMEL = 1048576
    if end < start:
        raise ValueError("arange's end argument must be greater than the start argument")
    if end - start > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(
            f"end - start must be less than or equal to TRITON_MAX_TENSOR_NUMEL = {TRITON_MAX_TENSOR_NUMEL}")
    return torch.arange(start, end)
=======
>>>>>>> release-3.2.2-0625-b79d137:third_party/ascend/unittest/pytest_ut/test_tile_chunk_coalescing.py


@triton.jit
def kernel_tile_chunk_coalescing_axis1(src, dst, N: tl.constexpr,
                                       BLOCK: tl.constexpr):
    batch = tl.program_id(0)
    tile = tl.program_id(1)
    offsets = tile * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    base = batch * N
    values = tl.load(src + base + offsets, mask=mask, other=0.0)
    tl.store(dst + base + offsets, values, mask=mask)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_tile_chunk_coalescing_grid_axis1_e2e(dtype):
    batch = 2
    block = 16
    num_tiles = 32
    n = block * num_tiles
    src = test_common.generate_tensor((batch, n), dtype).npu()
    dst = torch.empty_like(src)

    kernel_tile_chunk_coalescing_axis1[(batch, num_tiles)](
        src, dst, n, BLOCK=block,
    )

<<<<<<< HEAD:third_party/ascend/unittest/generalization_cases/test_general_arange.py
    triton_arange[(1, )](y_cal, START=start, END=end, BLOCK=block)

    assert torch.equal(y_cal.cpu(), y_ref.cpu())
=======
    assert torch.equal(dst.cpu(), src.cpu())
>>>>>>> release-3.2.2-0625-b79d137:third_party/ascend/unittest/pytest_ut/test_tile_chunk_coalescing.py
