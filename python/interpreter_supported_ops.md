# Triton 解释器模式（TRITON_INTERPRET=1）支持的 Op 列表

## 1. 概述

Triton 解释器模式通过 `TRITON_INTERPRET=1` 环境变量启用，允许在不依赖 GPU 的情况下使用 numpy 作为后端模拟执行 Triton kernel，主要用于调试和单元测试。

### 1.1 调用链路

```
tl.xxx        (public API, python/triton/language/__init__.py)
  └─ core.xxx   (@builtin decorator, python/triton/language/core.py)
       └─ _semantic.xxx   (python/triton/language/semantic.py)
            └─ builder.create_xxx   (python/triton/runtime/interpreter.py: InterpreterBuilder)
                 └─ numpy 实现
```

特殊路径：`reduce` / `associative_scan` 不走 builder，而是通过 `_patch_reduce_scan` 直接 patch `tl.core`，使用 `ReduceOps` / `ScanOps` 类用 numpy 实现。

### 1.2 核心实现文件

| 文件 | 作用 |
|------|------|
| [interpreter.py](file:///c:/work/code/triton/triton-lang36/triton/python/triton/runtime/interpreter.py) | `InterpreterBuilder` 提供 `create_*` 方法，基于 numpy 模拟 GPU 行为 |
| [semantic.py](file:///c:/work/code/triton/triton-lang36/triton/python/triton/language/semantic.py) | `TritonSemantic` 连接 public API 和 builder |
| [core.py](file:///c:/work/code/triton/triton-lang36/triton/python/triton/language/core.py) | 定义 `@builtin` 装饰的 public API |
| [math.py](file:///c:/work/code/triton/triton-lang36/triton/python/triton/language/math.py) | 数学函数（exp/log/sqrt 等） |
| [standard.py](file:///c:/work/code/triton/triton-lang36/triton/python/triton/language/standard.py) | reduce/scan 衍生函数（sum/max/sort 等） |
| [random.py](file:///c:/work/code/triton/triton-lang36/triton/python/triton/language/random.py) | 随机数生成 |

---

## 2. 支持的 Op 总表

下表按 `python/triton/language/__init__.py` 导出的 public API 顺序列出解释器模式的支持情况。

### 2.1 编程模型

| Public API | core 函数 | semantic 方法 | InterpreterBuilder 方法 | 备注 |
|-----------|-----------|--------------|------------------------|------|
| `program_id` | `program_id` | `program_id` | `create_get_program_id` | 返回当前 grid 索引 |
| `num_programs` | `num_programs` | `num_programs` | `create_get_num_programs` | 返回 grid 维度大小 |

### 2.2 算术二元运算

| Public API | core/semantic | InterpreterBuilder 方法 | 备注 |
|-----------|---------------|------------------------|------|
| `add` / `__add__` | `add` | `create_add` / `create_fadd` | 整数用 add，浮点用 fadd |
| `sub` / `__sub__` | `sub` | `create_sub` / `create_fsub` | 整数用 sub，浮点用 fsub |
| `mul` / `__mul__` | `mul` | `create_mul` / `create_fmul` | 整数用 mul，浮点用 fmul |
| `__truediv__` | `truediv` | `create_fdiv` / `create_sdiv` / `create_udiv` | 浮点用 fdiv，有符号整数用 sdiv，无符号用 udiv |
| `__floordiv__` | `floordiv` | `create_idiv` | IEEE 语义的整除 |
| `__mod__` | `mod` | `create_frem` / `create_srem` / `create_urem` | 浮点用 frem，整数用 srem/urem |
| `__neg__` | `minus` | `create_sub`（0-x） | 通过 0-x 实现 |
| `math.fdiv` | `fdiv` | `create_fdiv` | 快速浮点除法 |
| `math.div_rn` | - | `create_precise_divf` | IEEE 舍入的精确除法 |
| `math.fma` | - | `create_fma` | 融合乘加 x*y+z |
| `math.umulhi` | - | `create_umulhi` | 高位乘法 |

### 2.3 位运算

| Public API | semantic 方法 | InterpreterBuilder 方法 | 备注 |
|-----------|---------------|------------------------|------|
| `__and__` | `and_` | `create_and` | 按位与 |
| `__or__` | `or_` | `create_or` | 按位或 |
| `__xor__` | `xor_` | `create_xor` | 按位异或 |
| `__invert__` | `invert` | `create_xor`（x ^ 全1） | 通过 xor 实现 |
| `__lshift__` | `shl` | `create_shl` | 左移 |
| `__rshift__`（无符号） | `lshr` | `create_lshr` | 逻辑右移 |
| `__rshift__`（有符号） | `ashr` | `create_ashr` | 算术右移 |

### 2.4 比较运算

| Public API | semantic 方法 | InterpreterBuilder 方法（浮点） | InterpreterBuilder 方法（有符号整数） | InterpreterBuilder 方法（无符号整数） |
|-----------|---------------|--------------------------------|-------------------------------------|-------------------------------------|
| `__gt__` | `greater_than` | `create_fcmpOGT` | `create_icmpSGT` | `create_icmpUGT` |
| `__ge__` | `greater_equal` | `create_fcmpOGE` | `create_icmpSGE` | `create_icmpUGE` |
| `__lt__` | `less_than` | `create_fcmpOLT` | `create_icmpSLT` | `create_icmpULT` |
| `__le__` | `less_equal` | `create_fcmpOLE` | `create_icmpSLE` | `create_icmpULE` |
| `__eq__` | `equal` | `create_fcmpOEQ` | `create_icmpEQ` | `create_icmpEQ` |
| `__ne__` | `not_equal` | `create_fcmpUNE` | `create_icmpNE` | `create_icmpNE` |

### 2.5 Min / Max / Clamp / Select

| Public API | core 函数 | semantic 方法 | InterpreterBuilder 方法 | 备注 |
|-----------|-----------|---------------|------------------------|------|
| `minimum` | `minimum` | `minimum` | `create_minsi` / `create_minui` / `create_minimumf` / `create_minnumf` | 根据 signedness 和 propagate_nan 选择 |
| `maximum` | `maximum` | `maximum` | `create_maxsi` / `create_maxui` / `create_maximumf` / `create_maxnumf` | 根据 signedness 和 propagate_nan 选择 |
| `clamp` | `clamp` | `clamp` | `create_clampf` | 截断到 [min, max] |
| `where` | `where` | `where` | `create_select` | 条件选择 |

### 2.6 类型转换

| Public API | core 函数 | semantic 方法 | InterpreterBuilder 方法 | 备注 |
|-----------|-----------|---------------|------------------------|------|
| `cast` | `cast` | `cast` | `create_int_cast` / `create_fp_to_si` / `create_fp_to_ui` / `create_si_to_fp` / `create_ui_to_fp` / `create_fp_ext` / `create_fp_trunc` / `create_fp_to_fp` / `create_bitcast` / `create_int_to_ptr` / `create_ptr_to_int` | 根据 src/dst 类型组合选择 |
| `view`（bitcast=True） | `cast`（bitcast） | `cast`（bitcast） | `create_bitcast` | 位重新解释 |
| `view`（reshape 语义） | `reshape` | `reshape` | `create_reshape` | 形状变换 |

### 2.7 内存读写

| Public API | core 函数 | semantic 方法 | InterpreterBuilder 方法 | 备注 |
|-----------|-----------|---------------|------------------------|------|
| `load`（普通指针） | `load` | `load` | `create_load` / `create_masked_load` | 支持 mask、other、cache_modifier、eviction_policy、is_volatile |
| `store`（普通指针） | `store` | `store` | `create_store` / `create_masked_store` | 支持 mask、cache_modifier、eviction_policy |
| `make_block_ptr` | `make_block_ptr` | `make_block_ptr` | `create_make_block_ptr` | 创建块指针 |
| `advance` | `advance` | `advance` | `create_advance` | 推进块指针偏移 |
| `load`（块指针 + boundary_check） | `load` | `load` | `create_tensor_pointer_load` | 支持 boundary_check 和 padding PAD_ZERO/PAD_NAN |
| `store`（块指针 + boundary_check） | `store` | `store` | `create_tensor_pointer_store` | 支持 boundary_check |

### 2.8 原子操作

| Public API | core 函数 | semantic 方法 | InterpreterBuilder 方法 | 支持的 RMW op |
|-----------|-----------|---------------|------------------------|---------------|
| `atomic_cas` | `atomic_cas` | `atomic_cas` | `create_atomic_cas` | CAS |
| `atomic_add` | `atomic_add` | `atomic_add` | `create_atomic_rmw` | ADD（整数）/ FADD（浮点） |
| `atomic_max` | `atomic_max` | `atomic_max` | `create_atomic_rmw` | MAX（有符号）/ UMAX（无符号） |
| `atomic_min` | `atomic_min` | `atomic_min` | `create_atomic_rmw` | MIN（有符号）/ UMIN（无符号） |
| `atomic_and` | `atomic_and` | `atomic_and` | `create_atomic_rmw` | AND |
| `atomic_or` | `atomic_or` | `atomic_or` | `create_atomic_rmw` | OR |
| `atomic_xor` | `atomic_xor` | `atomic_xor` | `create_atomic_rmw` | XOR |
| `atomic_xchg` | `atomic_xchg` | `atomic_xchg` | `create_atomic_rmw` | XCHG |

### 2.9 Tensor Descriptor 操作

| Public API | core 函数 | semantic 方法 | InterpreterBuilder 方法 | 备注 |
|-----------|-----------|---------------|------------------------|------|
| `make_tensor_descriptor` | `make_tensor_descriptor` | `make_tensor_descriptor` | `create_make_tensor_descriptor` | 创建描述符（校验 16 字节对齐） |
| `load_tensor_descriptor` | `load_tensor_descriptor` | `descriptor_load` | `create_descriptor_load` | 支持 padding |
| `store_tensor_descriptor` | `store_tensor_descriptor` | `descriptor_store` | `create_descriptor_store` | - |
| （内部）`descriptor_gather` | - | `descriptor_gather` | `create_descriptor_gather` | 2D 描述符 gather |
| （内部）`descriptor_scatter` | - | `descriptor_scatter` | `create_descriptor_scatter` | 2D 描述符 scatter |

### 2.10 张量形状操作

| Public API | core 函数 | semantic 方法 | InterpreterBuilder 方法 | 备注 |
|-----------|-----------|---------------|------------------------|------|
| `arange` | `arange` | `arange` | `create_make_range` | 生成整数范围 |
| `full` | `full` | `full` | `create_splat` | 标量广播到 shape |
| `reshape` | `reshape` | `reshape` | `create_reshape` | 形状变换，支持 can_reorder |
| `trans` | `trans` | `trans` | `create_trans` | 转置（指定 perm） |
| `permute` | `permute` | `permute` | `create_trans` | 维度排列（复用 create_trans） |
| `broadcast` | `broadcast` | `broadcast` | `create_broadcast` | 广播到指定 shape |
| `broadcast_to` | `broadcast_to` | `broadcast` | `create_broadcast` | 广播到指定 shape |
| `expand_dims` | `expand_dims` | `expand_dims` | `create_expand_dims` | 在指定轴扩展维度 |
| `cat` | `cat` | `cat` | `create_cat` | 沿第一维拼接 |
| `join` | `join` | `join` | `create_join` | 沿最后一维 stack 两个张量 |
| `split` | `split` | `split` | `create_split` | 沿最后一维拆分为两个张量 |
| `gather` | `gather` | `gather` | `create_gather` | 沿 axis 用 index 收集 |
| `histogram` | `histogram` | `histogram` | `create_histogram` | 直方图统计 |
| `slice` | - | - | 编译期处理 | 切片在 AST 层处理 |

### 2.11 矩阵乘

| Public API | core 函数 | semantic 方法 | InterpreterBuilder 方法 | 备注 |
|-----------|-----------|---------------|------------------------|------|
| `dot` | `dot` | `dot` | `create_dot` | 支持 fp8 输入（内部转 fp16 计算），支持 input_precision（tf32/tf32x3/ieee） |

### 2.12 Reduce / Scan（通过 `_patch_reduce_scan` 特殊 patch）

| Public API | 实现类 | 实现方法 | 支持情况 |
|-----------|--------|----------|---------|
| `reduce` | `ReduceOps` | `apply_impl` | 支持内置 combine_fn：`_sum_combine` / `_elementwise_max` / `_elementwise_min` / `_argmax_combine_tie_break_left` / `_argmin_combine_tie_break_left`；其他走 `generic_reduce`（逐元素模拟，较慢） |
| `associative_scan` | `ScanOps` | `apply_impl` | 支持内置 combine_fn：`_sum_combine`（cumsum）/ `_prod_combine`（cumprod）；其他走 `generic_scan`（逐元素模拟，较慢） |

### 2.13 标准库 reduce 衍生（standard.py，基于 `reduce`/`associative_scan` 组合）

| Public API | 底层 combine_fn / 依赖 | 备注 |
|-----------|------------------------|------|
| `sum` | `_sum_combine` | 求和归约 |
| `max` | `_elementwise_max` | 最大值归约 |
| `min` | `_elementwise_min` | 最小值归约 |
| `argmax` | `_argmax_combine_tie_break_left` | 最大值索引（左优先） |
| `argmin` | `_argmin_combine_tie_break_left` | 最小值索引（左优先） |
| `xor_sum` | `_xor_combine` | 异或归约 |
| `reduce_or` | `_or_combine` | 按位或归约 |
| `cumsum` | `_sum_combine` | 累积和（基于 associative_scan） |
| `cumprod` | `_prod_combine` | 累积积（基于 associative_scan） |
| `softmax` | `max` + `exp` + `sum` + `fdiv` | softmax 实现 |
| `sigmoid` | `exp` | 1 / (1 + exp(-x)) |
| `ravel` | `reshape` | 展平为 1D |
| `zeros` | `full` | 全零张量 |
| `zeros_like` | `zeros` | 同形状全零张量 |
| `sort` | `reshape` + `where` + `xor_sum` + `reduce` | bitonic 排序 |
| `topk` | `sort_impl`（descending=True） | top-k 选择 |
| `bitonic_merge` | `_bitonic_merge` | bitonic 归并 |
| `flip` | `reshape` + `xor_sum` | 翻转 |
| `interleave` | `reshape` + `cat` | 交错拼接 |
| `swizzle2d` | `minimum` + 算术 | 索引重排 |
| `cdiv` | constexpr | 编译期整除向上取整 |

### 2.14 一元数学函数（math.py）

| Public API | InterpreterBuilder 方法 | 备注 |
|-----------|------------------------|------|
| `math.exp` | `create_exp` | 自然指数 |
| `math.exp2` | `create_exp2` | 2 为底指数 |
| `math.log` | `create_log` | 自然对数 |
| `math.log2` | `create_log2` | 2 为底对数 |
| `math.cos` | `create_cos` | 余弦 |
| `math.sin` | `create_sin` | 正弦 |
| `math.sqrt` | `create_sqrt` | 快速平方根 |
| `math.sqrt_rn` | `create_precise_sqrt` | IEEE 舍入平方根 |
| `math.rsqrt` | `create_rsqrt` | 平方根倒数 |
| `math.abs`（浮点） | `create_fabs` | 浮点绝对值（mask 符号位） |
| `math.abs`（有符号整数） | `create_iabs` | 整数绝对值 |
| `math.abs`（fp8e4b15） | `create_and` | 通过 mask 0x7F 实现 |
| `math.abs`（无符号整数） | no-op | 直接返回 |
| `math.erf` | `create_erf` | 误差函数（用 math.erf 向量化） |
| `math.floor` | `create_floor` | 向下取整 |
| `math.ceil` | `create_ceil` | 向上取整 |

### 2.15 随机数（random.py，基于 philox 算法组合）

| Public API | 底层依赖 | 支持情况 |
|-----------|----------|---------|
| `rand` | `philox` + `uint_to_uniform_float` | 支持 |
| `rand4x` | `philox` + `uint_to_uniform_float` | 支持 |
| `randn` | `philox` + `pair_uniform_to_normal` | 支持 |
| `randn4x` | `philox` + `pair_uniform_to_normal` | 支持 |
| `randint` | `philox` | 支持 |
| `randint4x` | `philox` | 支持 |
| `philox` | `philox_impl` | 支持 |
| `philox_impl` | `mul` / `add` / `umulhi` / `cast` | 支持 |
| `pair_uniform_to_normal` | `sqrt` / `log` / `cos` | 支持 |
| `uint_to_uniform_float` | `cast` / 位运算 | 支持 |

### 2.16 控制流与调试

| Public API | 实现方式 | 备注 |
|-----------|---------|------|
| `range` | patch 为 Python `range` | 通过 `_patch_lang_core` |
| `static_range` | patch 为 Python `range` | 通过 `_patch_lang_core` |
| `static_assert` | patch 为 Python `assert` | 通过 `_patch_lang_core` |
| `static_print` | patch 为 Python `print` | 通过 `_patch_lang_core` |
| `device_print` | `create_print` | 输出 grid_idx + 前缀 + 数据，支持 hex 格式 |
| `device_assert` | `create_assert` | 断言失败抛出 AssertionError |
| `assume` | `create_assume` | 断言（同 device_assert） |
| `debug_barrier` | `create_barrier` | no-op（解释器中程序串行执行） |
| `condition` | 纯包装类 | 无 builder 调用，while 循环条件包装 |

### 2.17 张量属性标注

| Public API | 实现方式 | 备注 |
|-----------|---------|------|
| `multiple_of` | patch 设置 `tt.divisibility` 属性 | 标注张量是某值的倍数 |
| `max_contiguous` | patch 设置 `tt.contiguity` 属性 | 标注最大连续性 |
| `max_constancy` | patch 设置 `tt.constancy` 属性 | 标注最大常量性 |

---

## 3. 不支持的 Op

以下 op 在解释器模式下会报错：

| Public API | 报错类型 | 原因 |
|-----------|---------|------|
| `dot_scaled` | `AttributeError` | `InterpreterBuilder` 未实现 `create_dot_scaled` |
| `map_elementwise` | `AttributeError` | `InterpreterBuilder` 未实现 `create_map_elementwise` |
| `inline_asm_elementwise` | `NotImplementedError` | `create_inline_asm` 显式抛出异常 |
| 外部库函数（`extern_elementwise`） | `NotImplementedError` | `create_extern_elementwise` 显式抛出异常 |
| Tensor Descriptor 原子操作（`descriptor_atomic_add/min/max/and/or/xor`） | `AttributeError` | `InterpreterBuilder` 未实现 `create_descriptor_reduce` |

---

## 4. 支持的数据类型

解释器模式通过 `_get_np_dtype` 支持以下 Triton dtype 到 numpy dtype 的映射：

| Triton dtype | numpy dtype | 备注 |
|--------------|-------------|------|
| `int1` | `bool` | 布尔 |
| `float16` | `float16` | 半精度浮点 |
| `float32` | `float32` | 单精度浮点 |
| `float64` | `float64` | 双精度浮点 |
| `int8` / `uint8` | `int8` / `uint8` | 8 位整数 |
| `int16` / `uint16` | `int16` / `uint16` | 16 位整数 |
| `int32` / `uint32` | `int32` / `uint32` | 32 位整数 |
| `int64` / `uint64` | `int64` / `uint64` | 64 位整数 |
| `bfloat16` | `uint16` | 以 uint16 存储，转换时用 `_convert_float` |
| `float8e5` | `uint8` | 以 uint8 存储 |
| `float8e5b16` | `uint8` | 以 uint8 存储 |
| `float8e4nv` | `uint8` | 以 uint8 存储 |
| `float8e4b8` | `uint8` | 以 uint8 存储 |
| `float8e4b15` | `uint8` | 以 uint8 存储 |
| `pointer_type` | `uint64` | 指针以 64 位无符号整数存储 |

---

## 5. 特殊实现说明

### 5.1 `_patch_lang` 机制

解释器通过 `_patch_lang` 在执行 kernel 前临时替换 `triton.language` 模块中的 builtin 函数，使其调用 `InterpreterBuilder` 而非真实编译器。执行完成后通过 `_LangPatchScope.restore()` 恢复原始属性。

### 5.2 `_patch_reduce_scan` 机制

`reduce` 和 `associative_scan` 由于涉及 region_builder_fn（闭包），无法简单通过 builder 实现，因此通过 `_patch_reduce_scan` 直接替换 `tl.reduce` 和 `tl.associative_scan`，使用 `ReduceOps` 和 `ScanOps` 类用 numpy 实现。

- 内置 combine_fn（sum/max/min/argmin/argmax/cumsum/cumprod）有快速路径
- 其他 combine_fn 走 `generic_reduce` / `generic_scan`（逐元素模拟，性能较慢）

### 5.3 `_patch_lang_core` 机制

以下函数通过 patch `tl.core` 实现，不走 builder：
- `range` / `static_range` → Python `range`
- `static_assert` → Python `assert`
- `static_print` → Python `print`
- `multiple_of` / `max_contiguous` / `max_constancy` → 设置 TensorHandle 的 attr 字典
- `dtype.to_ir` → 返回 tl.dtype 本身（而非 IR 类型）

### 5.4 fp8 支持

`create_dot` 对 fp8 输入有特殊处理：先将 fp8 转换为 fp16，再用 numpy.matmul 计算。fp8 转换通过 `_convert_float` 实现位级模拟。

### 5.5 bf16 支持

bf16 与 fp32 互转通过 `_convert_float` 实现位级转换（调整指数 bias 和尾数宽度），其他类型转换用 `numpy.astype`。

---

## 6. 总结

Triton 解释器模式通过 `InterpreterBuilder` 和多个 patch 机制，基于 numpy 模拟了 Triton IR 的绝大多数 op。基于 `python/triton/language/__init__.py` 的 `__all__` 列表，排除类型定义（`float16`/`int32` 等）、常量（`TRITON_MAX_TENSOR_NUMEL`）、模块（`math`/`extra`）、辅助类（`condition`/`constexpr`）后：

| 分类 | 数量 | 支持的 op 名称 |
|------|------|----------------|
| 编程模型 | 2 | `program_id`, `num_programs` |
| 算术二元运算 | 6 | `add`, `sub`, `mul`, `math.fdiv`, `math.div_rn`, `math.fma` |
| 位运算 | 5 | `__and__`, `__or__`, `__xor__`, `__invert__`, `__lshift__`, `__rshift__`（合并为位运算类，共 5 个核心 op） |
| 比较运算 | 6 | `__gt__`, `__ge__`, `__lt__`, `__le__`, `__eq__`, `__ne__` |
| Min/Max/Clamp/Select | 4 | `minimum`, `maximum`, `clamp`, `where` |
| 类型转换 | 2 | `cast`, `view` |
| 内存读写 | 4 | `load`, `store`, `make_block_ptr`, `advance` |
| 原子操作 | 8 | `atomic_cas`, `atomic_add`, `atomic_max`, `atomic_min`, `atomic_and`, `atomic_or`, `atomic_xor`, `atomic_xchg` |
| Tensor Descriptor 操作 | 3 | `make_tensor_descriptor`, `load_tensor_descriptor`, `store_tensor_descriptor` |
| 张量形状操作 | 12 | `arange`, `full`, `reshape`, `trans`, `permute`, `broadcast`, `broadcast_to`, `expand_dims`, `cat`, `join`, `split`, `gather`, `histogram` |
| 矩阵乘 | 1 | `dot` |
| Reduce / Scan | 2 | `reduce`, `associative_scan` |
| 标准库 reduce 衍生 | 11 | `sum`, `max`, `min`, `argmax`, `argmin`, `xor_sum`, `reduce_or`, `cumsum`, `cumprod`, `softmax`, `sigmoid` |
| 标准库其他衍生 | 7 | `ravel`, `zeros`, `zeros_like`, `sort`, `topk`, `bitonic_merge`, `flip`, `interleave`, `swizzle2d`, `cdiv` |
| 一元数学函数 | 14 | `math.exp`, `math.exp2`, `math.log`, `math.log2`, `math.cos`, `math.sin`, `math.sqrt`, `math.sqrt_rn`, `math.rsqrt`, `math.abs`, `math.erf`, `math.floor`, `math.ceil`, `math.umulhi` |
| 随机数生成 | 10 | `rand`, `rand4x`, `randn`, `randn4x`, `randint`, `randint4x`, `philox`, `philox_impl`, `pair_uniform_to_normal`, `uint_to_uniform_float` |
| 控制流与调试 | 8 | `range`, `static_range`, `static_assert`, `static_print`, `device_print`, `device_assert`, `assume`, `debug_barrier` |
| 张量属性标注 | 3 | `multiple_of`, `max_contiguous`, `max_constancy` |
| **合计支持** | **~102** | — |
| **不支持** | 3 | `dot_scaled`, `inline_asm_elementwise`, `map_elementwise` |
| **总 op 数** | **~105** | `__all__` 中的可执行函数 |

> 注：运算符重载（`__add__`/`__sub__`/`__mul__` 等 19 个）底层复用上述 op，不重复计数；Tensor Descriptor 原子操作（`descriptor_atomic_add` 等 6 个）不在 `__all__` 中导出，且解释器不支持。

Triton 解释器模式覆盖范围：
- 编程模型（program_id / num_programs）
- 算术 / 位 / 比较运算（全套运算符重载）
- 类型转换（含 fp8 / bf16 特殊处理）
- 内存读写（普通指针 / 块指针 / Tensor Descriptor）
- 原子操作（CAS + 7 种 RMW）
- 张量形状操作（reshape / trans / broadcast / cat / split / gather 等）
- 矩阵乘（dot）
- Reduce / Scan（含 11 种标准库衍生）
- 一元数学函数（14 种）
- 随机数生成（philox 系列）
- 控制流与调试（print / assert / barrier）

**仅 3 个 op 不支持**（在 `__all__` 中导出的）：`dot_scaled`、`inline_asm_elementwise`、`map_elementwise`。这些 op 在解释器模式下会抛出 `AttributeError` 或 `NotImplementedError`。

此外，未在 `__all__` 中导出的 Tensor Descriptor 原子操作（`descriptor_atomic_add/min/max/and/or/xor`）和外部库函数（`extern_elementwise`）也不支持。
