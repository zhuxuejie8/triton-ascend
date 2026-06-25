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

<<<<<<< HEAD:third_party/ascend/unittest/generalization_cases/test_general_cat.py
=======

import pytest
>>>>>>> release-3.2.2-0625-b79d137:third_party/ascend/unittest/pytest_ut/test_neg_index.py
import triton
import triton.language as tl
import time
import torch
import torch_npu
import test_common


@triton.jit
def triton_neg_index_load_kernel(in_ptr, out_ptr, BLOCK_SIZE: tl.constexpr, NEG_INDEX: tl.constexpr):
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset >= NEG_INDEX
    tmp = tl.load(in_ptr + ((-NEG_INDEX) + offset), mask, other=0.0)
    tl.store(out_ptr + offset, tmp)


def triton_neg_index_load(in_tensor, index):
    out_tensor = torch.zeros(in_tensor.shape, device=in_tensor.device, dtype=in_tensor.dtype)
    triton_neg_index_load_kernel[(1,)](in_tensor, out_tensor, in_tensor.numel(), index)
    return out_tensor


def torch_neg_index_load(in_tensor, index):
    out = torch.zeros(in_tensor.shape, device=in_tensor.device, dtype=in_tensor.dtype)
    out[index:] = in_tensor[:index]
    return out

<<<<<<< HEAD:third_party/ascend/unittest/generalization_cases/test_general_cat.py
    tl.store(output_ptr + oidx, ret)


# The CAT operator in the Triton community also does not support boolean types.
@pytest.mark.parametrize('shape', TestUtils.test_shape1d)  #triton only support 1D cat
@pytest.mark.parametrize('dtype', ['float32', 'float16', 'bfloat16', 'int32', 'int16', 'int8', 'int64'])
def test_cat(shape, dtype):
    m = shape[0]
    x = torch.full((m, ), 100, dtype=eval("torch." + dtype)).npu()
    y = torch.full((m, ), 30, dtype=eval("torch." + dtype)).npu()

    output = torch.randint(1, (m * 2, ), dtype=eval("torch." + dtype)).npu()

    ans = torch.cat((x, y), dim=0)

    fn_npu_[1, 1, 1](output, x, y, m)

    test_common.validate_cmp(dtype, ans, output)
=======

def test_neg_index_load():
    input_data = torch.arange(12, device="npu", dtype=torch.float32)
    triton_out = triton_neg_index_load(input_data, 6)
    torch_out = torch_neg_index_load(input_data, 6)
>>>>>>> release-3.2.2-0625-b79d137:third_party/ascend/unittest/pytest_ut/test_neg_index.py
