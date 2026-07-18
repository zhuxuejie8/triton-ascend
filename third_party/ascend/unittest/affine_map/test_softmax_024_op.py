# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import pytest
import triton
import triton.language as tl
import time
import torch
import torch_npu
import test_common

from test_common import should_skip_due_to_mem


@triton.jit
def triton_softmax_dim024_5d(in_ptr0, out_ptr0, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           Z: tl.constexpr):
    lblk_idx = tl.arange(0, L)
    mblk_idx = tl.arange(0, M)
    nblk_idx = tl.arange(0, N)
    kblk_idx = tl.arange(0, K)
    zblk_idx = tl.arange(0, Z)
    idx = lblk_idx[:, None, None, None, None] * Z * K * N * M + mblk_idx[None, :, None, None, None] * Z * K * N + \
          nblk_idx[None, None, :, None, None] * Z * K + kblk_idx[None, None, None, :, None] * Z \
          + zblk_idx[None, None, None, None, :]
    x0 = tl.load(in_ptr0 + idx)
    ret = tl.softmax(x0, dim=4)
    ret = tl.softmax(ret, dim=2)
    ret = tl.softmax(ret, dim=0)
    tl.store(out_ptr0 + idx, ret)


testlist = [

    (1, 2, 1, 1, 23),
    (1, 1, 1, 1, 23),
    (1, 1, 1, 1, 1),
    (7, 5, 3, 4, 1),
    (7, 2, 3, 5, 2),
    (15, 8, 1, 2, 1),
    (3, 5, 1, 17, 2),
    (3, 5, 7, 2, 2),
    (2, 9, 4, 2, 3),
    (7, 8, 7, 1, 5),
    (3, 5, 8, 2, 1),
    (7, 7, 5, 1, 4),
    (7, 3, 5, 4, 2),
    (1, 200, 1, 1, 2),
    (1, 2, 100, 1, 3),
    (1, 1, 1, 1000, 1),
    (5, 5, 2, 1, 3),
    (5, 25, 2, 1, 2),
    (20, 1, 2, 1, 21),
    (19, 3, 2, 1, 5),
    (2, 2, 2, 1, 6144),
    (2, 1, 2, 1, 6144)
]

typelist = ['float16', 'bfloat16', 'float32']
A2A3_max_ub_specific_direct_nd = {
    "float16":19656,
    "float32":24560,
    "bfloat16":19656,
}

A5_max_ub_specific_direct_nd = {
    "float16":12696,
    "float32":15870,
    "bfloat16":12696,
}

@pytest.mark.parametrize('shape', testlist, ids=["-".join(map(str, arg)) for arg in testlist])
@pytest.mark.parametrize('sigtype', typelist)
def test_softmax_dim024(sigtype, shape):
    dtype = test_common.get_torch_typename(sigtype)

    if test_common.is_A5_soc_version():
        max_sizes = A5_max_ub_specific_direct_nd
    else:
        max_sizes = A2A3_max_ub_specific_direct_nd
    input_tensor = [{'dtype': sigtype, 'shape': shape}]
    test_common.should_skip_due_to_mem2(input_tensor, max_sizes)

    x0 = test_common.generate_tensor_new(shape=shape, dtype=sigtype).npu()
    softmax_layer = torch.nn.Softmax(dim=4)
    y_ref = softmax_layer(x0)
    softmax_layer = torch.nn.Softmax(dim=2)
    y_ref1 = softmax_layer(y_ref)
    softmax_layer = torch.nn.Softmax(dim=0)
    y_ref2 = softmax_layer(y_ref1)
    output = torch.zeros(shape, dtype=dtype).npu()
    #breakpoint()
    triton_softmax_dim024_5d[(1,)](x0, output, *shape)
    test_common.validate_cmp(sigtype, output, y_ref2)
