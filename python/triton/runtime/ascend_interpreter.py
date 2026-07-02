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

import warnings
import contextlib
import re as _re
import numpy as np
import triton.language as tl
from .interpreter import InterpreterBuilder, TensorHandle, ReduceOps, _get_np_dtype
from .._C.libtriton import interpreter as _interpreter
from . import interpreter as _interp_module
# Several NumPy aliases used in this file (np.exp10, np.rcbrt, np.recip_hypot) are no longer
# present in modern NumPy. Provide fallbacks so the interpreter can still
# resolve the libdevice-numpy dispatch table.
if not hasattr(np, "exp10"):
    np.exp10 = lambda x: np.power(10.0, np.asarray(x))
if not hasattr(np, "rcbrt"):
    np.rcbrt = lambda x: 1.0 / np.cbrt(np.asarray(x))
if not hasattr(np, "recip_hypot"):
    np.recip_hypot = lambda x, y: 1.0 / np.hypot(x, y)
if not hasattr(np, "fdim"):
    np.fdim = lambda x, y: np.maximum(np.subtract(x, y), 0.0)
if not hasattr(np, "scalbn"):    
    np.scalbn = np.ldexp
# Special-function ufuncs that were removed in newer NumPy: fall back to
# scipy.special when available, otherwise provide NaN-emitting stubs.
try:
    import scipy.special as _scipy_special
    _special_map = {
        "erf": _scipy_special.erf, "erfc": _scipy_special.erfc,
        "erfcx": _scipy_special.erfcx, "erfinv": _scipy_special.erfinv,
        "erfcinv": _scipy_special.erfcinv, "gammaln": _scipy_special.gammaln,
        "gamma": _scipy_special.gamma,
        "i0": _scipy_special.i0, "i1": _scipy_special.i1,
        "j0": _scipy_special.j0, "j1": _scipy_special.j1,
        "y0": _scipy_special.y0, "y1": _scipy_special.y1,
        "jv": _scipy_special.jv, "yv": _scipy_special.yv,
        "ndtr": _scipy_special.ndtr, "ndtri": _scipy_special.ndtri,
    }
    for _name, _fn in _special_map.items():
        if not hasattr(np, _name):
            np.__dict__[_name] = _fn
except ImportError:
    pass

try:
    import scipy
    import scipy.special  
    _HAS_SCIPY = True
except ImportError:
    scipy = None
    _HAS_SCIPY = False

try:
    import ml_dtypes
    _HAS_ML_DTYPES = True
except ImportError:
    ml_dtypes = None
    _HAS_ML_DTYPES = False


def _strip_extern_prefix(symbol):
    """Recover the logical libdevice op name from a ``__hmf_<op><dtype>`` symbol.

    The CANN libdevice frontend emits symbols such as ``__hmf_atan_fp32`` or
    ``__hmf_atanDh`` (the ``Dh``/``Db`` suffix means fp16/bf16). Stripping
    those suffixes yields the bare op name (``atan``) that the
    ``_LIBDEVICE_NUMPY`` table is keyed by.
    """
    base = symbol
    for prefix in ("__hmf_", "__hmf", "hmf_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    base = _re.sub(r"_fp(16|32|64|128)$", "", base)
    base = _re.sub(r"_(bf16|fp16|fp32|fp64)$", "", base, flags=_re.IGNORECASE)
    if base.endswith("f") and len(base) > 1 and base[-2].isalpha():
        base = base[:-1]
    base = base.replace("Dh", "").replace("Db", "")
    return base


def _patched_get_np_dtype(tt_dtype):
    if isinstance(tt_dtype, tl.pointer_type):
        return np.dtype(np.uint64)
    np_types = {
        tl.int1: np.dtype(bool),
        tl.float16: np.dtype(np.float16),
        tl.float32: np.dtype(np.float32),
        tl.float64: np.dtype(np.float64),
        tl.int8: np.dtype(np.int8),
        tl.uint8: np.dtype(np.uint8),
        tl.int16: np.dtype(np.int16),
        tl.uint16: np.dtype(np.uint16),
        tl.int32: np.dtype(np.int32),
        tl.uint32: np.dtype(np.uint32),
        tl.int64: np.dtype(np.int64),
        tl.uint64: np.dtype(np.uint64),
        # ASCEND_AI_TRITON_FIX: use ml_dtypes.bfloat16 for IEEE-compliant bfloat16 arithmetic
        # (uint16 storage has same item size, but integer-overflow semantics differed)
        tl.bfloat16: np.dtype(ml_dtypes.bfloat16) if _HAS_ML_DTYPES else np.dtype(np.uint16),
        # float8 types are stored as uint8
        tl.float8e5: np.dtype(np.uint8),
        tl.float8e5b16: np.dtype(np.uint8),
        tl.float8e4nv: np.dtype(np.uint8),
        tl.float8e4b8: np.dtype(np.uint8),
        tl.float8e4b15: np.dtype(np.uint8),
    }
    if isinstance(tt_dtype, tl.block_type):
        if isinstance(tt_dtype.element_ty, tl.pointer_type):
            return np.dtype(np.uint64)
        return np_types[tt_dtype.element_ty]
    return np_types[tt_dtype]
_interp_module._get_np_dtype = _patched_get_np_dtype

class AscendReduceOps(ReduceOps):
    """
    Ascend reduce operations that override only the apply_impl logic.
    All other methods (sum, min_max, generic_reduce, etc.) are inherited from ReduceOps.
    """
    def apply_impl(self, input_param):
        if self.combine_fn == tl.standard._argmin_combine_tie_break_left:
            return self.min_max(input_param[0], val_reduce_op=np.min, idx_reduce_op=np.argmin)
        elif self.combine_fn == tl.standard._argmax_combine_tie_break_left:
            return self.min_max(input_param[0], val_reduce_op=np.max, idx_reduce_op=np.argmax)
        # Ta has modified the implemention of tl.max
        elif self.combine_fn == tl.standard._elementwise_max_default:
            return self.min_max(input_param[0], val_reduce_op=np.nanmax, idx_reduce_op=None)    
        elif self.combine_fn == tl.standard._elementwise_max_propagate_nan:
            return self.min_max(input_param[0], val_reduce_op=np.max, idx_reduce_op=None)
        elif self.combine_fn == tl.standard._elementwise_min:
            return self.min_max(input_param[0], val_reduce_op=np.nanmin, idx_reduce_op=None)
        elif self.combine_fn == tl.standard._sum_combine:
            return self.sum(input_param[0])
        else:
            # Fall back to the slow mode
            return self.generic_reduce(input_param)


def _compute_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


class AscendInterpreterBuilder(InterpreterBuilder):
    """
    Extended InterpreterBuilder with Ascend-specific extension operations.
    
    This class inherits from InterpreterBuilder and adds support for:
    - get_element (extract_scalar): Extract scalar from tensor using indices
    - insert_slice: Insert sub-tensor into full tensor
    - extract_slice: Extract slice from tensor
    - index_select_simd: SIMD gather operation
    - get_sub_vec_id: Get vector core ID for 1:2 ratio emulation
    - Synchronization operations: sync_block_set/wait/all
    
    All extension operations handle both TensorHandle and Python int types
    for interpreter mode compatibility.
    """
  
    _LIBDEVICE_NUMPY = globals().get("_LIBDEVICE_NUMPY", {})

    def __init__(self) -> None:
        super().__init__()
        # Sub-vector core ID for simulating 1:2 hardware ratio
        self.sub_vec_id = 0
        # Flag to track if sub_vec_id simulation is needed
        self._sub_vec_simulation_enabled = False

    def to_int_val(self, val):
        """
        Convert a value (int or TensorHandle) to Python int.
        
        :param val: Value to convert (int, TensorHandle, or other)
        :return: Python integer
        """
        if isinstance(val, TensorHandle):
            return int(val.data.item())
        return int(val)

    def _patch_lang_ascend(self, fn):

        def _new_range(arg1, arg2=None, step=None, **kwargs):
            if step is None:
                step = 1
            if arg2 is None:
                start, end = 0, arg1
            else:
                start, end = arg1, arg2
            return range(start, end, step)

        def _new_reduce(input_param, axis, combine_fn, keep_dims=False, **kwargs):
            return AscendReduceOps(axis, combine_fn, keep_dims).apply(input_param)

        @contextlib.contextmanager
        def _dummpy_scope(*args, **kwargs):
            yield

        def _np_atan2(y, x):
            y_np = y.handle.data.astype(np.float32)
            x_np = x.handle.data.astype(np.float32)
            res = np.arctan2(y_np, x_np).astype(_get_np_dtype(x.type))
            return tl.tensor(TensorHandle(res, x.type.scalar), x.type)

        def _np_isfinited(x):
            # numpy.isfinite handles nan and inf simultaneously.
            res = np.isfinite(x.handle.data.astype(np.float32)).astype(np.bool_)
            return tl.tensor(TensorHandle(res, tl.int1.scalar), tl.int1)

        def _np_finitef(x):
            # finitef is defined as float32-only; same semantics as isfinited.
            res = np.isfinite(x.handle.data.astype(np.float32)).astype(np.bool_)
            return tl.tensor(TensorHandle(res, tl.int1.scalar), tl.int1)

        tl.extra.cann.extension.scope = _dummpy_scope
        tl.extra.cann.extension.parallel = _new_range
        tl.reduce = _new_reduce
        tl.core.reduce = _new_reduce

        try:
            import triton.language.extra.cann.libdevice as libdevice          
            for _name, _np_fn in (("atan2", _np_atan2),
                                   ("isfinited", _np_isfinited),
                                   ("finitef", _np_finitef)):
                _orig = getattr(libdevice, _name, None)
                if _orig is None:
                    continue
                setattr(libdevice, _name, _np_fn)
                for g_name, g_val in list(fn.__globals__.items()):
                    if g_val is _orig:
                        fn.__globals__[g_name] = _np_fn
        except (ImportError, AttributeError):
            pass
    
    def get_additional_reserved_keywords(self):
        """
        Return additional reserved keywords specific to Ascend backend.
        
        These keywords will be filtered out from kernel call arguments
        and are not supported by the interpreter.
        
        :return: List of additional reserved keyword strings
        """
        return [
            "multibuffer",      # Ascend-specific memory buffering
            "debug",
            "optimize_dynamic_offset",
            "enable_mixed_cv",
            "enable_auto_bind_sub_block",
            "sync_solver",
            # Add more Ascend-specific keywords here as needed
            # "ascend_option1",
            # "ascend_option2",
        ]
    
    def patch_extensions(self, fn):
        """
        Patch Ascend extension modules for the given function.
        
        This method handles all Ascend-specific extension module patching,
        including CANN extensions and any other extension modules found in
        the function's global namespace.
        
        :param fn: The kernel function to patch extensions for
        """
        # Import _patch_builtin from parent module
        from .interpreter import _patch_builtin
        self._patch_lang_ascend(fn)
        
        # Patch all modules in fn's globals that might be extension modules
        for name, value in list(fn.__globals__.items()):
            if value is None:
                continue
            try:
                # Check if it looks like an extension module (has builtin functions)
                if hasattr(value, '__name__') and 'extension' in str(value.__name__):
                    _patch_builtin(value, self)
                # Also try patching any module-like object that might have builtin functions
                elif hasattr(value, '__dict__') and not isinstance(value, type):
                    # Try to patch it and ignore if it fails
                    try:
                        _patch_builtin(value, self)
                    except Exception:
                        pass
            except Exception:
                pass
        
        # Also try importing extension directly as fallback
        try:
            import triton.language.extra.cann.extension as extension
            _patch_builtin(extension, self)
        except (ImportError, AttributeError):
            # Extension module not available (e.g., non-Ascend backend)
            pass

    def execute_with_sub_vec_simulation(self, fn, args, grid):
        """
        Execute function with optional 1:2 sub-vector core simulation.
        
        Sub-vector simulation is only activated when create_get_sub_vec_id() is
        actually called during execution. This avoids unnecessary double execution
        for code that doesn't use sub_vec_id functionality.
        
        :param fn: The kernel function to execute
        :param args: Function arguments
        :param grid: Grid dimensions (nx, ny, nz)
        """
        # Reset simulation flag at the beginning of each execution
        self._sub_vec_simulation_enabled = False
        self.sub_vec_id = 0
        
        # First, try a single execution to see if sub_vec_id is used
        for x in range(grid[0]):
            for y in range(grid[1]):
                for z in range(grid[2]):
                    self.set_grid_idx(x, y, z)
                    fn(**args)
        
        # If sub_vec_id was accessed during execution, run again with sub_vec_id=1
        if self._sub_vec_simulation_enabled:
            self.sub_vec_id = 1
            for x in range(grid[0]):
                for y in range(grid[1]):
                    for z in range(grid[2]):
                        self.set_grid_idx(x, y, z)
                        fn(**args)

    # ========================================================================
    # Extension ops for Ascend
    # ========================================================================

    def create_extract_scalar(self, tensor_handle, indices):
        """
        Extract a scalar from a tensor using indices (equivalent to get_element).
        
        Handles mixed types: Python int (from loops) and TensorHandle (from other ops).
        
        :param tensor_handle: The tensor to extract from (TensorHandle)
        :param indices: List of scalar indices (can be TensorHandle or Python int)
        :return: Scalar value as TensorHandle
        """
        # Convert indices from TensorHandle or Python int to integers
        index_values = []
        for idx in indices:
            if isinstance(idx, int):
                # Python int passed directly (e.g., from loop counter)
                index_values.append(idx)
            elif isinstance(idx, TensorHandle):
                # Interpreter TensorHandle
                index_values.append(int(idx.data.item()) if hasattr(idx.data, 'item') else int(idx.data))
            else:
                # Fallback: try to extract data
                index_values.append(int(idx.data.item()) if hasattr(idx, 'data') and hasattr(idx.data, 'item') 
                                  else int(idx.data) if hasattr(idx, 'data') else int(idx))
        
        # Extract the scalar value
        scalar_data = tensor_handle.data[tuple(index_values)]
        return TensorHandle(np.array([scalar_data]), tensor_handle.dtype.scalar)

    def create_insert_slice(self, full_tensor, sub_tensor, offsets, sizes, strides):
        """
        Insert a sub-tensor into a full tensor at specified offsets.
        
        Handles mixed types: Python int and TensorHandle for offsets.
        
        :param full_tensor: The full tensor (destination, TensorHandle)
        :param sub_tensor: The sub-tensor to insert (TensorHandle)
        :param offsets: List of offset TensorHandle objects or Python ints
        :param sizes: List of size integers
        :param strides: List of stride integers
        :return: Modified tensor with sub_tensor inserted (TensorHandle)
        """
        result = full_tensor.data.copy()
        
        # Convert offsets from TensorHandle or Python int to integers
        offset_values = []
        for off in offsets:
            if isinstance(off, int):
                # Python int passed directly
                offset_values.append(off)
            elif isinstance(off, TensorHandle):
                # Interpreter TensorHandle
                offset_values.append(int(off.data.item()) if hasattr(off.data, 'item') else int(off.data))
            else:
                # Fallback
                offset_values.append(int(off.data.item()) if hasattr(off, 'data') and hasattr(off.data, 'item')
                                   else int(off.data) if hasattr(off, 'data') else int(off))
        
        # Build slices for insertion
        slices = []
        for i, (offset, size, stride) in enumerate(zip(offset_values, sizes, strides)):
            end = offset + size * stride
            if stride == 1:
                slices.append(slice(offset, end))
            else:
                slices.append(slice(offset, end, stride))
        
        # Insert the sub-tensor
        result[tuple(slices)] = sub_tensor.data
        
        return TensorHandle(result, full_tensor.dtype.scalar)

    def create_extract_slice(self, full_tensor, offsets, sizes, strides):
        """
        Extract a slice from a full tensor.
        
        Handles mixed types: Python int and TensorHandle for offsets.
        
        :param full_tensor: The full tensor (TensorHandle)
        :param offsets: List of offset TensorHandle objects or Python ints
        :param sizes: List of size integers
        :param strides: List of stride integers
        :return: Extracted sub-tensor (TensorHandle)
        """
        # Convert offsets from TensorHandle or Python int to integers
        offset_values = []
        for off in offsets:
            if isinstance(off, int):
                # Python int passed directly
                offset_values.append(off)
            elif isinstance(off, TensorHandle):
                # Interpreter TensorHandle
                offset_values.append(int(off.data.item()) if hasattr(off.data, 'item') else int(off.data))
            else:
                # Fallback
                offset_values.append(int(off.data.item()) if hasattr(off, 'data') and hasattr(off.data, 'item')
                                   else int(off.data) if hasattr(off, 'data') else int(off))
        
        # Build slices for extraction
        slices = []
        for i, (offset, size, stride) in enumerate(zip(offset_values, sizes, strides)):
            end = offset + size * stride
            if stride == 1:
                slices.append(slice(offset, end))
            else:
                slices.append(slice(offset, end, stride))
        
        # Extract the slice
        extracted = full_tensor.data[tuple(slices)]
        
        return TensorHandle(extracted, full_tensor.dtype.scalar)

    def create_index_select_simd(self, src_ptr, index_tensor, dim, src_shape, src_offset, read_shape, result_shape):
        """
        SIMD index_select operation (gather with indices along a dimension).
        
        This is a hardware-accelerated gather operation that selects elements
        from a tensor using a set of indices along a specified dimension.
        
        :param src_ptr: Source tensor pointer (TensorHandle), just ptr address, not value
        :param index_tensor: 1D tensor of indices (TensorHandle or array)
        :param dim: Dimension to select from (int)
        :param src_shape: List of source shape (int or TensorHandle)
        :param src_offset: List of source offset (int or TensorHandle)
        :param read_shape: List of read shape (int or TensorHandle)
        :param result_shape: List of result shape (int or TensorHandle)
        :return: Result tensor with selected indices (TensorHandle)
        """
        # Convert src_shape, src_offset, read_shape to integers
        src_shape_vals = [self.to_int_val(s) for s in src_shape]
        src_offset_vals = [self.to_int_val(s) if s != -1 else -1 for s in src_offset]
        read_shape_vals = [self.to_int_val(r) if r != -1 else -1 for r in read_shape]
        result_shape_vals = [self.to_int_val(r) for r in result_shape]
        
        # Get index values - handle both array and TensorHandle
        if isinstance(index_tensor, TensorHandle):
            indices = index_tensor.data.flatten()
        else:
            indices = np.asarray(index_tensor).flatten()
        
        # Ensure indices are integers
        if indices.dtype not in [np.int32, np.int64]:
            indices = indices.astype(np.int32)

        # Element type
        dtype_tt = src_ptr.get_element_ty()
        dtype_np = _get_np_dtype(dtype_tt)
        src_strides = _compute_strides(src_shape_vals)
        base_addr = int(src_ptr.data.item())
        
        # Create result tensor
        result = np.empty(result_shape_vals, dtype=dtype_np)
        
        # Perform index_select: for each index, read the specified data
        for out_idx, in_idx in enumerate(indices):
            in_idx = int(in_idx)
            # Generate all coordinates in the tile
            ranges = []
            for d in range(len(src_shape_vals)):
                if d == dim:
                    ranges.append([in_idx])
                else:
                    offset = src_offset_vals[d] if src_offset_vals[d] != -1 else 0
                    read_size = read_shape_vals[d] if read_shape_vals[d] != -1 else src_shape_vals[d]
                    # Clamp to valid range
                    offset = max(0, min(offset, src_shape_vals[d] - 1))
                    read_size = min(read_size, src_shape_vals[d] - offset)
                    ranges.append(list(range(offset, offset + read_size)))
            from itertools import product
            coords = list(product(*ranges))

            # Compute address for each element in the tile
            addresses = []
            for coord in coords:
                offset = sum(coord[i] * src_strides[i] for i in range(len(coord)))
                addr = base_addr + offset * np.dtype(dtype_np).itemsize
                addresses.append(addr)
            # load data
            addr_array = np.array(addresses, dtype=np.uint64)
            mask_array = np.ones_like(addr_array, dtype=bool)
            other_array = np.zeros_like(addr_array, dtype=dtype_np)
            tile_data = _interpreter.load(addr_array, mask_array, other_array, dtype_np)
            # Reshape tile_data to match read_shape with dim=1 at dim
            tile_shape = []
            for d in range(len(src_shape_vals)):
                if d == dim:
                    tile_shape.append(1)
                else:
                    offset = src_offset_vals[d]
                    read_size = read_shape_vals[d]
                    offset = max(0, min(offset, src_shape_vals[d] - 1))
                    read_size = min(read_size, src_shape_vals[d] - offset)
                    tile_shape.append(read_size)
            tile_data = tile_data.reshape(tile_shape)

            # Build result slice
            result_slices = []
            for d in range(len(result_shape_vals)):
                if d == dim:
                    result_slices.append(slice(out_idx, out_idx + 1))
                else:
                    result_slices.append(slice(None))
            result[tuple(result_slices)] = tile_data
 
        return TensorHandle(result, dtype_tt)

    def create_get_sub_vec_id(self):
        """
        Get the Vector Core index on the AI Core.
        
        In Interpreter mode, simulate multiple vector cores by maintaining
        a sub_vec_id counter. This is used for 1:2 hardware ratio emulation
        where different vector cores process different partitions of the data.
        
        The first call to this method enables sub_vec_simulation, causing
        the kernel to be executed twice (once for each sub_vec_id value).
        
        :return: Vector Core ID as TensorHandle (int64, scalar)
        """
        # Enable sub_vec_id simulation when this method is called
        self._sub_vec_simulation_enabled = True
        
        # Return the current sub_vec_id
        vec_id = np.int64(self.sub_vec_id)
        return TensorHandle(np.array([vec_id], dtype=np.int64), tl.int64)

    def sync_block_set(self, sender, receiver, event_id, sender_pipe_value, receiver_pipe_value):
        """
        Set synchronization event between compute and vector units.
        
        In Interpreter mode, this is a no-op since we execute single-threaded.
        Synchronization is not needed in CPU emulation.
        
        :param sender: Source unit ("cube" or "vector")
        :param receiver: Destination unit ("cube" or "vector")
        :param event_id: Event ID (TensorHandle)
        :param sender_pipe_value: Sender pipe value
        :param receiver_pipe_value: Receiver pipe value
        """
        # No-op in interpreter mode: single-threaded execution doesn't need sync
        pass

    def sync_block_wait(self, sender, receiver, event_id, sender_pipe_value, receiver_pipe_value):
        """
        Wait for synchronization event between compute and vector units.
        
        In Interpreter mode, this is a no-op since we execute single-threaded.
        Synchronization is not needed in CPU emulation.
        
        :param sender: Source unit ("cube" or "vector")
        :param receiver: Destination unit ("cube" or "vector")
        :param event_id: Event ID (TensorHandle)
        :param sender_pipe_value: Sender pipe value
        :param receiver_pipe_value: Receiver pipe value
        """
        # No-op in interpreter mode: single-threaded execution doesn't need sync
        pass

    def sync_block_all(self, mode, event_id):
        """
        Synchronize all compute or vector units globally.
        
        In Interpreter mode, this is a no-op since we execute single-threaded.
        Synchronization is not needed in CPU emulation.
        
        :param mode: Sync mode ("all_cube", "all_vector", "all", "all_sub_vector")
        :param event_id: Event ID (int, constexpr, or TensorHandle)
        """
        # No-op in interpreter mode: single-threaded execution doesn't need sync
        pass

    def get_int1_ty(self):
        return tl.int1

    def get_all_ones_value(self, tt_type):
        np_type = _get_np_dtype(tt_type)
        if "int" in np_type.name:
            return TensorHandle(np.full(1, -1, dtype=np_type), tt_type.scalar)
        elif np_type == np.bool_:
            return TensorHandle(np.full(1, True, dtype=np_type), tt_type.scalar)
        else:
            raise TypeError(f"unsupported type {tt_type}")

    def is_simt_mode(self):
        return False

    def create_sort(self, ptr_data, dim: int, descending: bool):
        ndim = ptr_data.data.ndim
        norm_dim = dim if dim >= 0 else dim + ndim
        if not (0 <= norm_dim < ndim):
            raise IndexError(
                f"Dimension out of range(expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})")

        if descending:
            sorted_asc = np.sort(ptr_data.data, axis=norm_dim)
            sorted_desc = np.flip(sorted_asc, axis=norm_dim)
            return TensorHandle(sorted_desc, ptr_data.dtype.scalar)
        else:
            return TensorHandle(np.sort(ptr_data.data, axis=norm_dim), ptr_data.dtype.scalar)

    def create_flip(self, ptr_data, dim):
        ndim = ptr_data.data.ndim
        norm_dim = dim if dim >= 0 else dim + ndim
        if not (0 <= norm_dim < ndim):
            raise IndexError(
                f"Dimension out of range(expected to be in range of [{-ndim}, {ndim - 1}], but got {dim})")
        return TensorHandle(np.flip(ptr_data.data, axis=norm_dim), ptr_data.dtype.scalar)

    def create_gather_out_to_ub(self, src_ptr, index_tensor, index_boundary, dim, src_stride, end_offset, start_offset, other=None):
        # Convert src_stride, start_offset, end_offset to integers
        src_stride_vals = [self.to_int_val(s) for s in src_stride]
        start_offset_vals = [self.to_int_val(s) for s in start_offset]
        end_offset_vals = [self.to_int_val(s) for s in end_offset]

        # Element type
        dtype_tt = src_ptr.get_element_ty()
        dtype_np = _get_np_dtype(dtype_tt)
        element_size = np.dtype(dtype_np).itemsize
        base_addr = int(src_ptr.data.item())
        index_shape = index_tensor.data.shape
        index_rank = len(index_shape)
        total_elements = np.prod(index_shape)

        # Generate  coordinates
        all_coords = []
        for idx in range(total_elements):
            coord = np.unravel_index(idx, index_shape)
            all_coords.append(coord)
        
        # Compute the source tensor coordinates for each position in all_coords
        src_coords = []
        for coord in all_coords:
            src_coord = []
            for d in range(index_rank):
                if d == dim:
                    index_value = index_tensor.data[coord]
                    if index_value >= index_boundary:
                        src_coord.append(-1)
                    else:
                        src_coord.append(start_offset_vals[d] + index_value)
                else:
                    src_coord.append(start_offset_vals[d] + coord[d])
            src_coords.append(src_coord)

        # Compute address and mask
        addresses = []
        valid_mask = []
        for _, src_coord in enumerate(src_coords):
            if -1 in src_coord:
                addresses.append(0)
                valid_mask.append(False)
            else:
                offset = 0
                for d in range(index_rank):
                    offset += src_coord[d] * src_stride_vals[d]
                address = base_addr + offset * element_size
                addresses.append(address)
                valid_mask.append(True)
        
        addr_array = np.array(addresses, dtype=np.uint64)
        mask_array = np.array(valid_mask, dtype=bool)

        # Create other value array
        if other is not None:
            if isinstance(other, TensorHandle):
                other_value = other.data.item()
            else:
                other_value = other
            other_array = np.full(addr_array.shape, other_value, dtype=dtype_np)
        else:
            other_array = np.zeros(addr_array.shape, dtype=dtype_np)

        # Load data
        flat_result = _interpreter.load(addr_array, mask_array, other_array, dtype_np)
        result = flat_result.reshape(index_shape)
        return TensorHandle(result, dtype_tt)

    def create_scatter_ub_to_out(self, dst_ptr, value_tensor, index_tensor, index_boundary, dim, dst_stride, end_offset, start_offset):
        # Convert dst_stride, start_offset, end_offset to integers
        dst_stride_vals = [self.to_int_val(s) for s in dst_stride]
        start_offset_vals = [self.to_int_val(s) for s in start_offset]
        end_offset_vals = [self.to_int_val(s) for s in end_offset]        

        # Element type
        dtype_tt = dst_ptr.get_element_ty()
        dtype_np = _get_np_dtype(dtype_tt)
        element_size = np.dtype(dtype_np).itemsize        
        base_addr = int(dst_ptr.data.item())

        index_shape = index_tensor.data.shape
        index_rank = len(index_shape)
        total_elements = np.prod(index_shape)
        flat_values = value_tensor.data.flatten()
        flat_indices = index_tensor.data.flatten()

        # Generate  coordinates
        all_coords = []
        for idx in range(total_elements):
            coord = np.unravel_index(idx, index_shape)
            all_coords.append(coord)

        # Compute address and mask
        addresses = []
        valid_mask = []
        for _, coord in enumerate(all_coords):
            index_value = index_tensor.data[coord]
            if index_value >= index_boundary:
                addresses.append(0)
                valid_mask.append(False)
            else:
                dst_coord = []
                for d in range(index_rank):
                    if d == dim:
                        dst_coord.append(start_offset_vals[d] + index_value)                                            
                    else:
                        dst_coord.append(start_offset_vals[d] + coord[d])
                offset = 0
                for d in range(index_rank):
                    offset += dst_coord[d] * dst_stride_vals[d]
                address = base_addr + offset * element_size
                addresses.append(address)
                valid_mask.append(True)
        
        addr_array = np.array(addresses, dtype=np.uint64)
        mask_array = np.array(valid_mask, dtype=bool)

        _interpreter.store(addr_array, flat_values, mask_array)

    def create_index_put(self, dst_ptr, index_tensor, value_tensor, dim, index_boundary, end_offset, start_offset, dst_stride):
        # Convert dst_stride, start_offset, end_offset_ to integers
        dst_stride_vals = [self.to_int_val(s) for s in dst_stride]
        start_offset_vals = [self.to_int_val(s) for s in start_offset]
        end_offset_vals = [self.to_int_val(s) for s in end_offset]        

        # Element type
        dtype_tt = dst_ptr.get_element_ty()
        dtype_np = _get_np_dtype(dtype_tt)
        element_size = np.dtype(dtype_np).itemsize        
        base_addr = int(dst_ptr.data.item())

        value_shape = value_tensor.data.shape
        value_rank = len(value_shape)
        
        flat_values = value_tensor.data.flatten()      
        total_elements = flat_values.size

        # Generate  coordinates
        all_coords = []
        for idx in range(total_elements):
            coord = np.unravel_index(idx, value_shape)
            all_coords.append(coord)

        read_ranges = []
        for d in range(value_rank):
            start = start_offset_vals[d]
            end = end_offset_vals[d]
            read_ranges.append((start, end))

        #Compute address
        addresses = []
        valid_mask = []
        values_to_store = []
        for i, coord in enumerate(all_coords):
            index_pos = coord[dim]
            index_value = index_tensor.data[index_pos]
            if index_value >= index_boundary:
                addresses.append(0)
                valid_mask.append(False)
            else:
                dst_coord = []
                for d in range(value_rank):
                    if d == dim:
                        dst_coord.append(index_value)                                            
                    else:
                        dst_coord.append(start_offset_vals[d] + coord[d])
                offset = 0
                for d in range(value_rank):
                    offset += dst_coord[d] * dst_stride_vals[d]
                address = base_addr + offset * element_size
                addresses.append(address)
                values_to_store.append(flat_values[i])
                valid_mask.append(True)

        addr_array = np.array(addresses, dtype=np.uint64)
        mask_array = np.array(valid_mask, dtype=bool)
        values_array = np.array(values_to_store, dtype=dtype_np)

        _interpreter.store(addr_array, values_array, mask_array)

    def get_bool_attr(self, val):
        return bool(val) 

    def get_unit_attr(self):
        return None  # None valule in compile_hint return uint

    def get_int32_attr(self, val):        
        return int(val)

    def get_str_attr(self, val):
        return str(val)

    def get_i64_array_attr(self, val):      
        return [int(x) for x in val]

    def create_annotation_mark(self, ptr_data, hint_name: str, hint_val):
        if hint_name == "overflow_mode":
            raise ValueError(f"overflow_mode is not supported in interpreter mode, may have accuracy issues")
        else:
            warnings.warn(
            f"compile_hint '{hint_name}' is not supported in interpreter mode, just pass it",
            UserWarning,
            stacklevel=2 
        )    

    def _resolve_extern_symbol(self, symbol, arg_dtypes):
        """Look up the numpy callable for ``symbol`` and the actual input dtypes.

        Resolution order:
        1. Exact symbol match in ``_LIBDEVICE_NUMPY`` (covers the full
           149-op libdevice set in ``cann/libdevice.py``).
        2. Name-only fallback: strip the ``__hmf_`` / dtype suffix and
           consult the same table again. This is what the libdevice
           frontend emits at the IR level (``__hmf_atan_fp32`` -> ``atan``).
        3. Fall back to ``_SYMBOL_TO_NUMPY`` (kept for compatibility with
           any symbols registered at runtime through
           ``_register_extern_symbol``).
        4. Last-ditch name-based dispatch happens inside
           ``create_extern_elementwise`` for op aliases that do not have a
           direct numpy equivalent (e.g. ``atan`` -> ``np.arctan``).

        Returns ``(np_callable, key_dtype)`` where ``key_dtype`` is the
        scalar dtype used for the dispatch lookup.
        """
        # 1) Exact symbol match in the global libdevice table.
        np_types = getattr(AscendInterpreterBuilder, "_LIBDEVICE_NUMPY", {})
        entry = np_types.get(symbol)
        if entry is None:
            # 2) Strip the prefix and dtype suffix to recover the op name.
            entry = np_types.get(_strip_extern_prefix(symbol))
        
        # Choose the dispatch key from the first floating input.
        key_dtype = None
        for d in arg_dtypes:
            if isinstance(d, tl.dtype) and d.is_floating():
                key_dtype = d.scalar if hasattr(d, "scalar") else d
                break
        if key_dtype is None and arg_dtypes:
            key_dtype = arg_dtypes[0].scalar if hasattr(arg_dtypes[0], "scalar") else arg_dtypes[0]
        if isinstance(entry, dict):
            np_fn = entry.get(key_dtype, entry.get(None) or entry.get(tl.float32))
        elif callable(entry):
            np_fn = entry
        else:
            np_fn = None
        return np_fn, key_dtype

    def create_extern_elementwise(self, libName, libPath, symbol, argList, retType, isPure):
        """Numpy fallback for ``core.extern_elementwise`` in interpreter mode.

        The real Ascend backend lowers these calls to CANN libdevice symbols
        (e.g. ``__hmf_atanf``). The interpreter has no dynamic loader for
        them, so we route the call through a numpy ufunc selected by symbol
        name. This covers the test-suite cases (atan, tanh, exp, log, sin,
        cos, sqrt, pow, ...) and any other libdevice ops the test may pull
        in via ``tl.extra.cann.libdevice``.
        """
        # Extract numpy arrays and source dtypes from TensorHandle wrappers.
        np_args = []
        src_dtypes = []
        for arg in argList:
            if isinstance(arg, TensorHandle):
                np_args.append(arg.data)
                src_dtypes.append(arg.dtype)
            else:
                np_args.append(np.asarray(arg))
                src_dtypes.append(getattr(arg, "dtype", tl.float32))

        # Allow positional scalars (e.g. ldexp's exponent) to be passed in.
        # They show up in argList as plain python ints/floats in IR dumps.
        np_fn, key_dtype = self._resolve_extern_symbol(symbol, src_dtypes)   

        if np_fn is None:
            raise NotImplementedError(
                f"extern_elementwise symbol '{symbol}' has no numpy fallback in interpreter mode"
            )

        try:
            result_data = np_fn(*np_args)
        except Exception as e:
            raise NotImplementedError(
                f"extern_elementwise symbol '{symbol}' failed in interpreter fallback: {e}"
            )

        # Determine the result numpy dtype from retType.
        if isinstance(retType, tl.dtype):
            ret_scalar = retType.scalar if hasattr(retType, "scalar") else retType
            ret_np_dtype = _get_np_dtype(retType)
        else:
            ret_scalar = retType
            ret_np_dtype = _get_np_dtype(retType)

        result_data = np.asarray(result_data).astype(ret_np_dtype)
        return TensorHandle(result_data, ret_scalar)

    def get_bf16(self, value):
        return TensorHandle(np.array([value], dtype=np.dtype(ml_dtypes.bfloat16) if _HAS_ML_DTYPES else np.dtype(np.uint16)), tl.bfloat16)


def _gamma_fallback(x):
    """Cheap Stirling-style fallback used when scipy is unavailable."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    mask = x > 0
    out[~mask] = np.nan
    xv = x[mask]
    # Stirling approximation: lgamma ~ (x-0.5)ln(x) - x + 0.5 ln(2 pi)
    out[mask] = np.exp((xv - 0.5) * np.log(np.maximum(xv, 1e-30)) - xv + 0.5 * np.log(2.0 * np.pi))
    return out

_LIBDEVICE_NUMPY = { 
    # ---- transcendental ---------------------------------------------------
    "cos": np.cos, "cosh": np.cosh, "sin": np.sin, "sinh": np.sinh,
    "tan": np.tan, "tanh": np.tanh, "asin": np.arcsin, "asinh": np.arcsinh,
    "acos": np.arccos, "acosh": np.arccosh, "atan": np.arctan, "atanh": np.arctanh,
    "atan2": np.arctan2, "exp": np.exp, "expm1": np.expm1, "exp2": np.exp2,
    "exp10": np.exp10, "log": np.log, "log1p": np.log1p, "log2": np.log2,
    "log10": np.log10, "logb": lambda x: np.log2(np.abs(np.where(np.asarray(x) == 0, 1.0, np.asarray(x)))),
    "ilogb": lambda x: np.floor(np.log2(np.abs(np.where(np.asarray(x) == 0, 1.0, np.asarray(x))))).astype(np.int32),
    "sqrt": np.sqrt, "cbrt": np.cbrt, "rcbrt": np.rcbrt,
    "rsqrt_rn": lambda x: 1.0 / np.sqrt(np.asarray(x)),
    "rcp_rn": lambda x: 1.0 / np.asarray(x),
    "reciprocal": lambda x: 1.0 / np.asarray(x),
    "pow": np.power, "hypot": np.hypot, "rhypot": getattr(np, "recip_hypot", lambda x, y: 1.0 / np.hypot(x, y)),
    "fmod": np.fmod, "remainder": np.remainder, "copysign": np.copysign,
    "fdim": np.fdim, "fmax": np.fmax, "fmin": np.fmin,
    "nextafter": np.nextafter, "ldexp": np.ldexp, "scalbn": np.scalbn,
    # ---- rounding / fp-class ---------------------------------------------
    "fabs": np.abs, "abs": np.abs, "neg": np.negative, "relu": lambda x: np.fmax(np.asarray(x), 0),
    "rint": np.rint, "nearbyint": np.rint, "round": np.round, "trunc": np.trunc,
    "floor": np.floor, "ceil": np.ceil,
    "signbit": lambda x: np.signbit(np.asarray(x)).astype(np.int8),
    "isnan": lambda x: np.isnan(np.asarray(x)), "isinf": lambda x: np.isinf(np.asarray(x)),
    "isfinite": lambda x: np.isfinite(np.asarray(x)),
    "isin": lambda x: np.isinf(np.asarray(x)),  # alias: _strip_extern_prefix strips trailing 'f' from isinf -> isin
    # ---- special functions -----------------------------------------------
    "erf": np.erf, "erfc": np.erfc, "erfcx": np.erfcx,
    "erfinv": (lambda x: scipy.special.erfinv(x)) if _HAS_SCIPY else (lambda x: np.full(np.shape(x), np.nan)),
    "erfcinv": (lambda x: scipy.special.erfcinv(x)) if _HAS_SCIPY else (lambda x: np.full(np.shape(x), np.nan)),
    "lgamma": (lambda x: scipy.special.gammaln(x)) if _HAS_SCIPY else (lambda x: np.log(np.maximum(_gamma_fallback(x), 1e-30))),
    "tgamma": (lambda x: scipy.special.gamma(x)) if _HAS_SCIPY else _gamma_fallback,
    "normcdf": (lambda x: scipy.special.ndtr(x)) if _HAS_SCIPY else (lambda x: 0.5 * (1.0 + np.erf(np.asarray(x) / np.sqrt(2.0)))),
    "normcdfinv": (lambda x: scipy.special.ndtri(x)) if _HAS_SCIPY else (lambda x: np.sqrt(2.0) * scipy.special.erfinv(2 * np.asarray(x) - 1)) if _HAS_SCIPY else (lambda x: np.full(np.shape(x), np.nan)),
    "j0": (lambda x: scipy.special.j0(x)) if _HAS_SCIPY else (lambda x: np.sin(x) / np.where(x == 0, 1.0, x)),
    "j1": (lambda x: scipy.special.j1(x)) if _HAS_SCIPY else (lambda x: np.sin(x) / np.where(x == 0, 1.0, x) - np.cos(x) / np.where(x == 0, 1.0, x)),
    "jn": (lambda n, x: scipy.special.jv(n, x)) if _HAS_SCIPY else (lambda n, x: np.sin(x) / np.where(x == 0, 1.0, x)),
    "y0": (lambda x: scipy.special.y0(x)) if _HAS_SCIPY else (lambda x: -np.cos(x) / np.where(x == 0, 1.0, x)),
    "y1": (lambda x: scipy.special.y1(x)) if _HAS_SCIPY else (lambda x: -np.cos(x) / np.where(x == 0, 1.0, x) - np.sin(x) / np.where(x == 0, 1.0, x)),
    "yn": (lambda n, x: scipy.special.yv(n, x)) if _HAS_SCIPY else (lambda n, x: -np.cos(x) / np.where(x == 0, 1.0, x)),
    "sinpi": lambda x: np.sin(np.asarray(x) * np.pi),
    "cospi": lambda x: np.cos(np.asarray(x) * np.pi),
    # ---- rounding-mode intrinsics (cannot be emulated on a numpy fp core) -
    "add_rd": None, "add_rn": None, "add_ru": None, "add_rz": None,
    "sub_rd": None, "sub_rn": None, "sub_ru": None, "sub_rz": None,
    "mul_rd": None, "mul_rn": None, "mul_ru": None, "mul_rz": None,
    "div_rd": None, "div_rn": None, "div_ru": None, "div_rz": None,
    "sqrt_rd": None, "sqrt_rn": None, "sqrt_ru": None, "sqrt_rz": None,
    "rcp_rd": None, "rcp_ru": None, "rcp_rz": None,
    "fma_rd": None, "fma_rn": None, "fma_ru": None, "fma_rz": None,
    # ---- cast ops with explicit rounding modes ----------------------------
    "float2int_rd": None, "float2int_rn": None, "float2int_ru": None, "float2int_rz": None,
    "float2uint_rd": None, "float2uint_rn": None, "float2uint_ru": None, "float2uint_rz": None,
    "float2ll_rd": None, "float2ll_rn": None, "float2ll_ru": None, "float2ll_rz": None,
    "float2ull_rd": None, "float2ull_rn": None, "float2ull_ru": None, "float2ull_rz": None,
    "int2float_rd": None, "int2float_rn": None, "int2float_ru": None, "int2float_rz": None,
    "uint2float_rd": None, "uint2float_rn": None, "uint2float_ru": None, "uint2float_rz": None,
    "ll2float_rd": None, "ll2float_rn": None, "ll2float_ru": None, "ll2float_rz": None,
    "ull2float_rd": None, "ull2float_rn": None, "ull2float_ru": None, "ull2float_rz": None,
    "float_as_int": None, "float_as_uint": None,
    "int_as_float": None, "uint_as_float": None,
    "llrint": None, "llround": None,
    # ---- integer / bit-level intrinsics ----------------------------------
    "brev": None, "byte_perm": None, "clz": None, "popc": None, "ffs": None,
    "hadd": None, "rhadd": None, "sad": None, "mulhi": None, "mul24": None,
    # ---- fast / approximate (delegate to np equivalent) -------------------
    "fast_cos": np.cos, "fast_sin": np.sin, "fast_tan": np.tan,
    "fast_exp": np.exp, "fast_exp10": np.exp10,
    "fast_log": np.log, "fast_log2": np.log2, "fast_log10": np.log10,
    "fast_pow": np.power, "fast_divide": np.true_divide,
    "fast_cosf": np.cos, "fast_sinf": np.sin, "fast_tanf": np.tan,
    "fast_expf": np.exp, "fast_exp10f": np.exp10,
    "fast_logf": np.log, "fast_log2f": np.log2, "fast_log10f": np.log10,
    "fast_powf": np.power, "fast_dividef": np.true_divide,
    "finitef": np.isfinite, "finite": np.isfinite, "finitelf": np.isfinite,
    "saturatef": None, "saturate": None, "gamma": None,
    # ---- vector norms / bessel --------------------------------------------
    "norm3d": None, "norm4d": None, "rnorm3d": None, "rnorm4d": None,
    "cyl_bessel_i0": (lambda x: scipy.special.i0(x)) if _HAS_SCIPY else (lambda x: np.full(np.shape(x), np.nan)),
    "cyl_bessel_i1": (lambda x: scipy.special.i1(x)) if _HAS_SCIPY else (lambda x: np.full(np.shape(x), np.nan)),
}
AscendInterpreterBuilder._LIBDEVICE_NUMPY = _LIBDEVICE_NUMPY
