# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import unittest.mock as mock

<<<<<<< HEAD
=======
from test_common import check_axes_parse_res, mock_autotuner

>>>>>>> release-3.2.2-0625-b79d137
import triton
import triton.language as tl


def _mock_single_reduction_axis(self):
    self.vector_axes.ensure_axis("x").length_expr = "n_elements"
    self.vector_axes.apply_semantic_fields(reduction_axes=["x"])
    return ["x"]


def _mock_apply_vv_with_low_dim_axes(orig_apply, low_dim_axes):
    def _mock(self):
        applied = orig_apply(self)
        self.low_dim_axes = list(low_dim_axes)
        self._refresh_vector_axes()
        return applied

    return _mock


def _mock_single_reduction_axis(self):
    if "x" in self.keys and "rx" not in self.keys:
        self.keys["rx"] = self.keys.pop("x")
    return ["rx"]


def test_low_dim_axis_parse_base_case1(mock_autotuner):
    import triton.backends.ascend.runtime

<<<<<<< HEAD
    @triton.autotune(configs=[], key=["n_elements"])
=======
    @triton.autotune(
        configs=[],
        key=["n_elements"],
    )
>>>>>>> release-3.2.2-0625-b79d137
    @triton.jit
    def triton_low_dim_axis_parse_base_case1(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE  # <- Separate assignment

        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y

        tl.store(output_ptr + offsets, output, mask=mask)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {},
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"], )
    act_res = triton_low_dim_axis_parse_base_case1[grid]()

    check_axes_parse_res(act_res, ref_res)


def test_low_dim_axis_parse_empty_is_non_fatal(mock_autotuner):
    import triton.backends.ascend.runtime

<<<<<<< HEAD
    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_low_dim_axis_parse_empty_case(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
=======
    @triton.autotune(
        configs=[],
        key=["n_elements"],
    )
    @triton.jit
    def triton_low_dim_axis_parse_empty_case(
        x_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
>>>>>>> release-3.2.2-0625-b79d137
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        mask = block_start < n_elements

        _ = tl.load(x_ptr + block_start, mask=mask, other=0)

    ref_res = {
        "keys": {"x": "n_elements"},
        "split_params": {"x": "BLOCK_SIZE"},
        "tiling_params": {},
<<<<<<< HEAD
        "low_dim_axes": [],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"], )
=======
        "low_dim_axes": ["x"],
        "reduction_axes": [],
    }
    grid = lambda meta: (meta["BLOCK_SIZE"],)
>>>>>>> release-3.2.2-0625-b79d137
    act_res = triton_low_dim_axis_parse_empty_case[grid]()

    check_axes_parse_res(act_res, ref_res)


def test_low_dim_axis_parse_empty_no_persistent_reduction_index_error(mock_autotuner):
    import triton.backends.ascend.runtime
    from triton.backends.ascend.runtime.autotuner import AutoTilingTuner

<<<<<<< HEAD
    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_low_dim_axis_parse_guard_case(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
=======
    @triton.autotune(
        configs=[],
        key=["n_elements"],
    )
    @triton.jit
    def triton_low_dim_axis_parse_guard_case(
        x_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
>>>>>>> release-3.2.2-0625-b79d137
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        mask = block_start < n_elements

        _ = tl.load(x_ptr + block_start, mask=mask, other=0)

<<<<<<< HEAD
    with mock.patch.object(AutoTilingTuner, "_autoparse_reduction_axes", return_value=["rx"]), \
         mock.patch.object(AutoTilingTuner, "_autoparse_low_dim_axes", return_value=[]), \
         mock.patch.object(AutoTilingTuner, "_autoparse_split_params", return_value={}), \
         mock.patch.object(AutoTilingTuner, "_autoparse_tiling_params", return_value={}):
        act_res = triton_low_dim_axis_parse_guard_case[(1, )]()

    assert act_res["low_dim_axes"] == []
    assert act_res["reduction_axes"] == ["rx"]
=======
    with mock.patch.object(
        AutoTilingTuner, "_autoparse_reduction_axes", new=_mock_single_reduction_axis
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_low_dim_axes", return_value=[]
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_split_params", return_value={}
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_tiling_params", return_value={}
    ):
        act_res = triton_low_dim_axis_parse_guard_case[(1,)]()

    assert act_res["low_dim_axes"] == ["x"]
    assert act_res["reduction_axes"] == ["x"]
>>>>>>> release-3.2.2-0625-b79d137


def test_persistent_reduction_inner_axis_threshold(mock_autotuner):
    import triton.backends.ascend.runtime
    from triton.backends.ascend.runtime.autotuner import AutoTilingTuner

<<<<<<< HEAD
    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_persistent_reduction_inner_axis_case(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
=======
    @triton.autotune(
        configs=[],
        key=["n_elements"],
    )
    @triton.jit
    def triton_persistent_reduction_inner_axis_case(
        x_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
>>>>>>> release-3.2.2-0625-b79d137
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        mask = block_start < n_elements
        _ = tl.load(x_ptr + block_start, mask=mask, other=0)

<<<<<<< HEAD
    with mock.patch.object(AutoTilingTuner, "_autoparse_reduction_axes", new=_mock_single_reduction_axis), \
         mock.patch.object(AutoTilingTuner, "_autoparse_low_dim_axes", return_value=["rx"]), \
         mock.patch.object(AutoTilingTuner, "_autoparse_split_params", return_value={}), \
         mock.patch.object(AutoTilingTuner, "_autoparse_tiling_params", return_value={}):
        act_res = triton_persistent_reduction_inner_axis_case[(1, )](None, 1024)

=======
    orig_apply_vv = AutoTilingTuner._apply_vv_axis_semantic_result
    with mock.patch.object(
        AutoTilingTuner,
        "_apply_vv_axis_semantic_result",
        new=_mock_apply_vv_with_low_dim_axes(orig_apply_vv, ["x"]),
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_reduction_axes", new=_mock_single_reduction_axis
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_low_dim_axes", return_value=["x"]
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_split_params", return_value={}
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_tiling_params", return_value={}
    ):
        act_res = triton_persistent_reduction_inner_axis_case[(1,)](None, 1024)
>>>>>>> release-3.2.2-0625-b79d137
    assert act_res["persistent_reduction"] is True


def test_persistent_reduction_outer_axis_threshold(mock_autotuner):
    import triton.backends.ascend.runtime
    from triton.backends.ascend.runtime.autotuner import AutoTilingTuner

<<<<<<< HEAD
    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_persistent_reduction_outer_axis_case(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
=======
    @triton.autotune(
        configs=[],
        key=["n_elements"],
    )
    @triton.jit
    def triton_persistent_reduction_outer_axis_case(
        x_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
>>>>>>> release-3.2.2-0625-b79d137
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        mask = block_start < n_elements
        _ = tl.load(x_ptr + block_start, mask=mask, other=0)

<<<<<<< HEAD
    with mock.patch.object(AutoTilingTuner, "_autoparse_reduction_axes", new=_mock_single_reduction_axis), \
         mock.patch.object(AutoTilingTuner, "_autoparse_low_dim_axes", return_value=[]), \
         mock.patch.object(AutoTilingTuner, "_autoparse_split_params", return_value={}), \
         mock.patch.object(AutoTilingTuner, "_autoparse_tiling_params", return_value={}):
        act_res = triton_persistent_reduction_outer_axis_case[(1, )](None, 64)

=======
    orig_apply_vv = AutoTilingTuner._apply_vv_axis_semantic_result
    with mock.patch.object(
        AutoTilingTuner,
        "_apply_vv_axis_semantic_result",
        new=_mock_apply_vv_with_low_dim_axes(orig_apply_vv, []),
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_reduction_axes", new=_mock_single_reduction_axis
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_low_dim_axes", return_value=[]
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_split_params", return_value={}
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_tiling_params", return_value={}
    ):
        act_res = triton_persistent_reduction_outer_axis_case[(1,)](None, 64)
>>>>>>> release-3.2.2-0625-b79d137
    assert act_res["persistent_reduction"] is True


def test_persistent_reduction_outer_axis_over_threshold(mock_autotuner):
    import triton.backends.ascend.runtime
    from triton.backends.ascend.runtime.autotuner import AutoTilingTuner

<<<<<<< HEAD
    @triton.autotune(configs=[], key=["n_elements"])
    @triton.jit
    def triton_persistent_reduction_outer_axis_over_threshold_case(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
=======
    @triton.autotune(
        configs=[],
        key=["n_elements"],
    )
    @triton.jit
    def triton_persistent_reduction_outer_axis_over_threshold_case(
        x_ptr, n_elements, BLOCK_SIZE: tl.constexpr
    ):
>>>>>>> release-3.2.2-0625-b79d137
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        mask = block_start < n_elements
        _ = tl.load(x_ptr + block_start, mask=mask, other=0)

<<<<<<< HEAD
    with mock.patch.object(AutoTilingTuner, "_autoparse_reduction_axes", new=_mock_single_reduction_axis), \
         mock.patch.object(AutoTilingTuner, "_autoparse_low_dim_axes", return_value=[]), \
         mock.patch.object(AutoTilingTuner, "_autoparse_split_params", return_value={}), \
         mock.patch.object(AutoTilingTuner, "_autoparse_tiling_params", return_value={}):
        act_res = triton_persistent_reduction_outer_axis_over_threshold_case[(1, )](None, 65)

=======
    orig_apply_vv = AutoTilingTuner._apply_vv_axis_semantic_result
    with mock.patch.object(
        AutoTilingTuner,
        "_apply_vv_axis_semantic_result",
        new=_mock_apply_vv_with_low_dim_axes(orig_apply_vv, []),
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_reduction_axes", new=_mock_single_reduction_axis
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_low_dim_axes", return_value=[]
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_split_params", return_value={}
    ), mock.patch.object(
        AutoTilingTuner, "_autoparse_tiling_params", return_value={}
    ):
        act_res = triton_persistent_reduction_outer_axis_over_threshold_case[(1,)](
            None, 65
        )
>>>>>>> release-3.2.2-0625-b79d137
    assert act_res["persistent_reduction"] is False
