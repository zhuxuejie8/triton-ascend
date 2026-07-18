import os
import re
import torch
import torch_npu
import triton
import triton.language as tl
import numpy as np
import math
import logging
from typing import AnyStr
import pytest
import functools


def set_stack():
    from triton.runtime.driver import driver
    driver.active.utils.set_device_limit(0, "WARP_STACK_SIZE", 0x100000)
    driver.active.set_current_device(0)


log_level = os.getenv("LOG_LEVEL", "WARN").upper()
level_mapping = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

logging.basicConfig(
    level=level_mapping.get(log_level, logging.WARNING),
    format="[%(asctime)s][%(levelname)s] %(message)s"
)

bisheng_not_support_dtypes = {
    'abs': [],
    'eq': [],
    'ne': [],
    'flip': ['int64', 'bfloat16'],
    'load_store': ['int64'],
    'permute2d': ['int64'],
    'permute3d': ['int64'],
    'trans2d': ['int64'],
    'trans3d': ['int64'],
    'matmul': ['int16', 'int32', 'uint32', 'int64', 'bool']
}

tritonascend_not_support_dtypes = {
    'abs': ['bool'],
    'eq': ['bool'],
    'ne': ['bool'],
    'flip': ['bool'],
    'load_store': ['bool'],
    'permute2d': ['bool'],
    'permute3d': ['bool'],
    'trans2d': ['bool'],
    'trans3d': ['bool'],
}

extreme_range_dict = {
    'int8': [-128, 127],
    'int16': [-32768, 32767],
    'int32': [-2147483648, 2147483647],
    'int64': [-9223372036854775808, 9223372036854775807],
    'uint8': [0, 255],
    'uint16': [0, 65535],
    'uint32': [0, 4294967295],
    'uint64': [0, 18446744073709551615],
    'float16': [-65504, 65504],
    'float32': [-3.4e+38, 3.4e+38],
    'bfloat16': [-3.38953e+38, 3.38953e+38],
    'bool': [-1, 1],  # 不生效，实际只在True和False中取值
    'fp8e4m3': [-448, 448],
    'fp8e5m2': [-57344, 57344],
    'fp8e5b16': [-57344, 57344],
    'fp4': [-6, 6],
}


def avoid_not_support(op: AnyStr):
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(shape, dtype, *args, **kwargs):
            if dtype in bisheng_not_support_dtypes.get(op, []):
                logging.warn(f'skiped bisheng not support dtype:{dtype}')
                return
            if dtype in tritonascend_not_support_dtypes.get(op, []):
                logging.warn(f'skiped triton ascend not support dtype:{dtype}')
                return
            return test_func(shape, dtype, *args, **kwargs)

        return wrapper

    return decorator


def get_shape1d(in_shape1d):
    result = []
    for i in in_shape1d:
        v = tuple((i,))
        result.append(v)
    return result


def get_shape2d(in_shape1d, custom_shape):
    result = []
    for a in in_shape1d:
        for b in custom_shape:
            t1 = tuple((a, b))
            t2 = tuple((b, a))
            if t1 not in result:
                result.append(t1)
            if t2 not in result:
                result.append(t2)
    return result


def get_shape3d():
    return [(1, 22, 39), (27, 1, 39), (27, 22, 1), (23, 1, 1), (1, 23, 1), (1, 1, 23), (37, 5, 3), (2, 29, 4),
            (7, 31, 7), (3, 5, 8), (7, 17, 15), (23, 5, 16), (23, 5, 31), (7, 11, 32), (7, 11, 33), (2, 3, 255),
            (3, 3, 256), (3, 2, 257)]


def get_shape1_2_3d(in_shape1d, custom_shape):
    return get_shape1d(in_shape1d) + get_shape2d(in_shape1d, custom_shape) + get_shape3d()


class TestUtils:
    in_shape1d = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 37, 741]
    custom_shape = [3, 13, 32, 256]
    batch = [1, 2, 3, 4, 5, 8]
    test_shape1d = get_shape1d(in_shape1d)
    test_shape2d = get_shape2d(in_shape1d, custom_shape)
    test_shape3d = [(1, 22, 39), (27, 1, 39), (27, 22, 1), (1, 1, 23), (23, 1, 1), (1, 23, 1),
                    (37, 5, 3), (2, 29, 4), (7, 31, 7), (3, 5, 8), (7, 17, 15), (25, 5, 16),
                    (23, 5, 31), (7, 11, 32), (7, 11, 33), (2, 3, 255), (3, 3, 256), (3, 2, 257), ]
    test_shape4d = [(8, 4, 8, 8), (1, 11, 16, 2)]
    test_shape5d = [(2, 3, 4, 5, 6), (1, 3, 4, 5, 6), (3, 6, 2, 4, 4)]
    test_shape6d = [(2, 3, 5, 6, 3, 2)]
    test_shape7d = [(1, 2, 3, 4, 3, 2, 2)]
    test_shape8d = [(1, 2, 3, 2, 5, 3, 7, 2), (1, 3, 2, 5, 6, 7, 2, 1), (2, 3, 7, 3, 2, 3, 2, 3)]
    full_shape_4_8d = test_shape4d + test_shape5d + test_shape6d + test_shape7d + test_shape8d

    full_shape = test_shape1d + test_shape2d + test_shape3d
    test_shape1_2_3d = full_shape
    full_dtype = ['int8', 'int16', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']
    ub_size = 98304 * 2
    dtype_list = full_dtype


def get_dtype_size(dtype):
    torch_dtype = eval('torch.' + dtype)
    bits = 0
    if torch_dtype == torch.bool:
        bits = 8
    elif torch.is_floating_point(torch.tensor(0, dtype=torch_dtype)):
        bits = torch.finfo(torch_dtype).bits
    else:
        bits = torch.iinfo(torch_dtype).bits
    return bits // 8


def check_ub_mem_overflow(dtype, shape):
    bytes = get_dtype_size(dtype)
    if bytes * math.prod(shape) > TestUtils.ub_size:
        logging.warning(f'dtype:{dtype} shape:{shape} mem overflow')
        return True
    return False


# 生成泛化用例数值tensor函数
# 若op需要控制数据区间，可将各ratio置0，避免出现datarange之外的值
def generate_tensor_new(shape,  # 生成tensor的shape（必填）
                        dtype,  # 生成tensor的数据类型（必填）
                        seed=None,  # 种子，用于复现数据生成结果
                        data_range=None,  # 主要数据的数据区间
                        special_ratio=0.05,  # 插入 inf/-inf/nan的比例（仅当dtype为float类时生效）
                        precision_ratio=0.35,  # 精度小浮点数的比例（仅当dtype为float类时生效）
                        extreme_ratio=0.1,  # 极值区域占比（其中20%为 min~min+10, max-10~max，80%为min和max本身）（dtype为bool时不生效）
                        zero_ratio=0,  # 额外加入 0 的比例 （dtype为bool时不生效）
                        bool_true_ratio=0.5,  # True 的占比 （仅当dtype为bool时生效）
                        ):
    # 各数据类型的数据极值字典

    # 各数据类型主要生成数据区间字典
    middle_range_dict = {
        'int8': [-20, 20],
        'int16': [-200, 200],
        'int32': [-500, 500],
        'int64': [-1000, 1000],
        'uint8': [0, 40],
        'uint16': [0, 400],
        'uint32': [0, 1000],
        'uint64': [0, 2000],
        'float16': [-500, 500],
        'float32': [-1000, 1000],
        'bfloat16': [-1000, 1000],
        'bool': [-1, 1],  # 不生效，实际只在True和False中取值
        'fp8e4m3': [-50, 50],
        'fp8e5m2': [-500, 500],
        'fp8e5b16': [-500, 500],
        'fp4': [-6, 6],
    }

    # 浮点数据类型生成精度小浮点数的区间字典
    float_precision_dict = {
        'float16': [-1, 1],
        'float32': [-1, 1],
        'bfloat16': [-1, 1],
        'fp8e4m3': [-1, 1],
        'fp8e5m2': [-1, 1],
        'fp8e5b16': [-1, 1],
        'fp4': [-2, 2],
    }

    # sigtype to dype字典
    ddtype_dict = {
        'int8': torch.int8,
        'int16': torch.int16,
        'int32': torch.int32,
        'int64': torch.int64,
        'uint8': torch.uint8,
        'uint16': torch.uint16,
        'uint32': torch.uint32,
        'uint64': torch.uint64,
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'bool': torch.bool,
        'fp8e4m3': torch.float8_e4m3fn,
        'fp8e5m2': torch.float8_e5m2,
        'fp8e5b16': None,
        'fp4': None,
    }

    # 输入种子时手动设置种子值
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 读取字典生成数据区间并进行一次数据类型支持过滤
    if dtype not in ddtype_dict:
        raise ValueError(f"Unsupported dtype: {dtype}")

    min_val, max_val = extreme_range_dict[dtype]
    total = np.prod(shape)

    # 鉴于10对于int64粒度过小会引发np.random.randint的value Error：low>=high，此处对int64特殊处理
    if dtype == 'int64':
        left_range = (min_val, min_val + 1000)
        right_range = (max_val - 1000, max_val)
    elif dtype == 'fp4':
        left_range = (min_val, min_val + 1)
        right_range = (max_val - 1, max_val)
    else:
        left_range = (min_val, min_val + 10)
        right_range = (max_val - 10, max_val)

    # 设置主要数据范围
    if data_range == None:
        mid_range = middle_range_dict[dtype]
    else:
        mid_range = data_range

    # int类处理
    if dtype in ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'):

        # 指定分布概率和超过100%异常处理
        if (extreme_ratio + zero_ratio > 1):
            raise ValueError(f"Total ratio over 100%, unable to allocate sampling:{extreme_ratio + zero_ratio}")

        # 生成原始一维全0数据，据类型使用int64或uint64防溢出
        data = np.zeros(total, dtype=np.int64)
        if dtype in ['uint8', 'uint16', 'uint32', 'uint64']:
            data = np.zeros(total, dtype=np.uint64)

        # 计算各区间元素个数（其中极值区域为累计计数：按极小值附近，极小值本身，极大值附近，极大值本身顺序）
        left_count = int(total * extreme_ratio / 10)
        left_accumulate_count = int(total * extreme_ratio / 2)
        left_right_count = int(total * extreme_ratio * 3 / 5)
        left_right_accmulate_count = int(total * extreme_ratio)
        zero_count = int(total * zero_ratio)

        # 生成打乱的索引映射数组
        indices = np.random.choice(total, size=total, replace=False)

        # 按区间元素个数划分区间索引
        left_idx = indices[:left_count]
        left_min_idx = indices[left_count:left_accumulate_count]
        right_idx = indices[left_accumulate_count:left_right_count]
        right_max_idx = indices[left_right_count:left_right_accmulate_count]
        zero_idx = indices[left_right_accmulate_count:left_right_accmulate_count + zero_count]
        mid_idx = indices[left_right_accmulate_count + zero_count:]

        # 按区间索引生成数据
        data[left_idx] = np.random.randint(left_range[0], left_range[1], size=len(left_idx), dtype=eval('np.' + dtype))
        data[left_min_idx] = min_val
        data[right_idx] = np.random.randint(right_range[0], right_range[1], size=len(right_idx),
                                            dtype=eval('np.' + dtype))
        data[right_max_idx] = max_val
        data[mid_idx] = np.random.randint(mid_range[0], mid_range[1], size=len(mid_idx), dtype=eval('np.' + dtype))
        data[zero_idx] = 0

        tensor = torch.from_numpy(data.reshape(shape)).to(ddtype_dict[dtype])

    # float类处理
    elif dtype in ('float16', 'float32', 'bfloat16', 'fp8e4m3', 'fp8e5m2'):

        # 指定分布概率和超过100%异常处理
        if (extreme_ratio + precision_ratio + zero_ratio + special_ratio > 1):
            raise ValueError(f"Total ratio over 100%, unable to allocate sampling:{extreme_ratio + precision_ratio}")

        # 读字典取得精度小浮点数数据范围
        precision_range = float_precision_dict[dtype]

        # 生成原始一维全0数据，float32防溢出
        data = np.zeros(total, dtype=np.float32)

        # 计算各区间元素个数（其中极值区域为累计计数：按极小值附近，极小值本身，极大值附近，极大值本身顺序）
        left_count = int(total * extreme_ratio / 10)
        left_accumulate_count = int(total * extreme_ratio / 2)
        left_right_count = int(total * extreme_ratio * 3 / 5)
        left_right_accmulate_count = int(total * extreme_ratio)
        precision_count = int(total * precision_ratio)
        zero_count = int(total * zero_ratio)
        special_count = int(total * special_ratio)

        # 生成打乱的索引映射数组
        indices = np.random.choice(total, size=total, replace=False)

        # 按区间元素个数划分区间索引
        left_idx = indices[:left_count]
        left_min_idx = indices[left_count:left_accumulate_count]
        right_idx = indices[left_accumulate_count:left_right_count]
        right_max_idx = indices[left_right_count:left_right_accmulate_count]
        precision_idx = indices[left_right_accmulate_count:left_right_accmulate_count + precision_count]
        zero_idx = indices[left_right_accmulate_count + precision_count:
                           left_right_accmulate_count + precision_count + zero_count]
        special_idx = indices[left_right_accmulate_count + precision_count + zero_count:
                              left_right_accmulate_count + precision_count + zero_count + special_count]
        mid_idx = indices[left_right_accmulate_count + precision_count + zero_count + special_count:]

        # 按区间索引生成数据
        data[left_idx] = np.random.uniform(left_range[0], left_range[1], size=len(left_idx))
        data[left_min_idx] = min_val
        data[right_idx] = np.random.uniform(right_range[0], right_range[1], size=len(right_idx))
        data[right_max_idx] = max_val
        data[precision_idx] = np.random.uniform(precision_range[0], precision_range[1], size=len(precision_idx))
        data[zero_idx] = 0
        for idx in special_idx:
            r = np.random.rand()
            if r < 1 / 3:
                data[idx] = float('inf')
            elif r < 2 / 3:
                data[idx] = float('-inf')
            else:
                data[idx] = float('nan')
        data[mid_idx] = np.random.uniform(mid_range[0], mid_range[1], size=len(mid_idx))

        tensor = torch.from_numpy(data.reshape(shape)).to(ddtype_dict[dtype])

    # bool类型处理：直接按照参数比例生成true和false值
    elif dtype == 'bool':
        if (bool_true_ratio > 1):
            raise ValueError(f"Total ratio over 100%, unable to allocate sampling:{bool_true_ratio}")
        data = np.random.choice([True, False], size=total, p=[bool_true_ratio, 1 - bool_true_ratio])
        tensor = torch.from_numpy(data.reshape(shape)).to(ddtype_dict[dtype])

    # 第二次异常数据类型处理
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    return tensor


def generate_tensor(shape, dtype, data_range=None, insertinf=False):
    if data_range == None:
        if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
            output_tensor = torch.randn(size=shape, dtype=eval('torch.' + dtype))
        elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'uint32':
            output_tensor = torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
        elif dtype == 'int8':
            output_tensor = torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
        elif dtype == 'bool':
            output_tensor = torch.randint(low=0, high=2, size=shape).bool()
        elif dtype == 'uint8':
            output_tensor = torch.randint(low=0, high=255, size=shape, dtype=torch.uint8)
        elif dtype == 'uint16':
            output_tensor = torch.randint(low=0, high=65535, size=shape, dtype=torch.uint16)
        elif dtype == 'uint64':
            output_tensor = torch.randint(low=0, high=2 * 65535, size=shape, dtype=torch.uint64)
        else:
            raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    else:
        [min, max] = data_range
        if dtype == 'float32' or dtype == 'float16':
            output_tensor = torch.from_numpy(np.random.uniform(low=min, high=max, size=shape).astype(dtype))
        elif dtype == 'bfloat16':
            output_tensor = torch.from_numpy(np.random.uniform(low=min, high=max, size=shape).astype(np.float32)).to(
                torch.bfloat16)
        elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
            output_tensor = torch.randint(low=min, high=max, size=shape, dtype=eval('torch.' + dtype))
        elif dtype == 'bool':
            output_tensor = torch.randint(low=min, high=max, size=shape).bool()
        elif dtype == 'uint8':
            return torch.randint(low=0, high=255, size=shape, dtype=torch.uint8)
        else:
            raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

    if insertinf and dtype in ['float32', 'float16', 'bfloat16']:
        mask = torch.rand(shape) < 0.000001
        special_indices = torch.where(mask)
        if len(special_indices[0]) > 0:
            # Randomly choose which special value to insert
            special_values = [float('inf'), float('-inf'), float('nan')]

            for i in range(len(special_indices[0])):
                # Get random indices for this position
                idx = tuple(special_indices[j][i] for j in range(len(special_indices)))

                # Randomly select a special value
                special_val = np.random.choice(special_values)
                output_tensor[idx] = special_val

    return output_tensor


def generate_tensor_int_withSigns(shape, dtype):
    if dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=-32768, high=32767, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=-128, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


# 该常量定义了在generate_multi_index函数中，用来表示某一个维度未使用时的值
SHAPEDIMUNUSE: tl.constexpr = None


# 生成 1-8 维的索引数组
@triton.jit
def generate_multi_index(
        L: tl.constexpr = SHAPEDIMUNUSE,
        M: tl.constexpr = SHAPEDIMUNUSE,
        N: tl.constexpr = SHAPEDIMUNUSE,
        K: tl.constexpr = SHAPEDIMUNUSE,
        P: tl.constexpr = SHAPEDIMUNUSE,
        Q: tl.constexpr = SHAPEDIMUNUSE,
        R: tl.constexpr = SHAPEDIMUNUSE,
        S: tl.constexpr = SHAPEDIMUNUSE
):
    if L == SHAPEDIMUNUSE:
        raise ValueError("维度不能为空")
    elif M == SHAPEDIMUNUSE:
        lblk_idx = tl.arange(0, L)
        return lblk_idx[:]
    elif N == SHAPEDIMUNUSE:
        lblk_idx = tl.arange(0, L)
        mblk_idx = tl.arange(0, M)
        return lblk_idx[:, None] * M + mblk_idx[None, :]
    elif K == SHAPEDIMUNUSE:
        lblk_idx = tl.arange(0, L)
        mblk_idx = tl.arange(0, M)
        nblk_idx = tl.arange(0, N)
        return lblk_idx[:, None, None] * N * M + mblk_idx[None, :, None] * N + nblk_idx[None, None, :]
    elif P == SHAPEDIMUNUSE:
        lblk_idx = tl.arange(0, L)
        mblk_idx = tl.arange(0, M)
        nblk_idx = tl.arange(0, N)
        kblk_idx = tl.arange(0, K)
        return (lblk_idx[:, None, None, None] * K * N * M + mblk_idx[None, :, None, None] * K * N + \
                nblk_idx[None, None, :, None] * K + kblk_idx[None, None, None, :])
    elif Q == SHAPEDIMUNUSE:
        lblk_idx = tl.arange(0, L)
        mblk_idx = tl.arange(0, M)
        nblk_idx = tl.arange(0, N)
        kblk_idx = tl.arange(0, K)
        pblk_idx = tl.arange(0, P)
        return (lblk_idx[:, None, None, None, None] * P * K * N * M + mblk_idx[None, :, None, None, None] * P * K * N + \
                nblk_idx[None, None, :, None, None] * P * K + kblk_idx[None, None, None, :, None] * P \
                + pblk_idx[None, None, None, None, :])
    elif R == SHAPEDIMUNUSE:
        lblk_idx = tl.arange(0, L)
        mblk_idx = tl.arange(0, M)
        nblk_idx = tl.arange(0, N)
        kblk_idx = tl.arange(0, K)
        pblk_idx = tl.arange(0, P)
        qblk_idx = tl.arange(0, Q)
        return (lblk_idx[:, None, None, None, None, None] * P * K * N * M * Q + \
                mblk_idx[None, :, None, None, None, None] * P * K * N * Q + \
                nblk_idx[None, None, :, None, None, None] * P * K * Q + \
                kblk_idx[None, None, None, :, None, None] * P * Q + \
                pblk_idx[None, None, None, None, :, None] * Q + \
                qblk_idx[None, None, None, None, None, :])
    elif S == SHAPEDIMUNUSE:
        lblk_idx = tl.arange(0, L)
        mblk_idx = tl.arange(0, M)
        nblk_idx = tl.arange(0, N)
        kblk_idx = tl.arange(0, K)
        pblk_idx = tl.arange(0, P)
        qblk_idx = tl.arange(0, Q)
        rblk_idx = tl.arange(0, R)
        return (lblk_idx[:, None, None, None, None, None, None] * Q * P * K * N * M * R + \
                mblk_idx[None, :, None, None, None, None, None] * Q * P * K * N * R + \
                nblk_idx[None, None, :, None, None, None, None] * Q * P * K * R + \
                kblk_idx[None, None, None, :, None, None, None] * Q * P * R + \
                pblk_idx[None, None, None, None, :, None, None] * Q * R + \
                qblk_idx[None, None, None, None, None, :, None] * R + \
                rblk_idx[None, None, None, None, None, None, :])
    else:
        lblk_idx = tl.arange(0, L)
        mblk_idx = tl.arange(0, M)
        nblk_idx = tl.arange(0, N)
        kblk_idx = tl.arange(0, K)
        pblk_idx = tl.arange(0, P)
        qblk_idx = tl.arange(0, Q)
        rblk_idx = tl.arange(0, R)
        sblk_idx = tl.arange(0, S)
        return (lblk_idx[:, None, None, None, None, None, None, None] * R * Q * P * K * N * M * S + \
                mblk_idx[None, :, None, None, None, None, None, None] * R * Q * P * K * N * S + \
                nblk_idx[None, None, :, None, None, None, None, None] * R * Q * P * K * S + \
                kblk_idx[None, None, None, :, None, None, None, None] * R * Q * P * S + \
                pblk_idx[None, None, None, None, :, None, None, None] * R * Q * S + \
                qblk_idx[None, None, None, None, None, :, None, None] * R * S + \
                rblk_idx[None, None, None, None, None, None, :, None] * S + \
                sblk_idx[None, None, None, None, None, None, None, :])


def get_torch_typename(dtype):
    if dtype == 'float32':
        tyname = torch.float32
    elif dtype == 'int32':
        tyname = torch.int32
    elif dtype == 'int64':
        tyname = torch.int64
    elif dtype == 'float16':
        tyname = torch.float16
    elif dtype == 'int16':
        tyname = torch.int16
    elif dtype == 'int8':
        tyname = torch.int8
    elif dtype == 'bool':
        tyname = torch.bool
    elif dtype == 'bfloat16':
        tyname = torch.bfloat16
    elif dtype == 'uint8':
        tyname = torch.uint8
    elif dtype == 'uint16':
        tyname = torch.uint16
    elif dtype == 'uint32':
        tyname = torch.uint32
    elif dtype == 'uint64':
        tyname = torch.uint64
    elif dtype == 'fp8e4m3':
        tyname = torch.float8_e4m3fn
    elif dtype == 'fp8e5m2':
        tyname = torch.float8_e5m2
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    return tyname


def get_triton_typename(dtype):
    if dtype == 'float32':
        tyname = tl.float32
    elif dtype == 'int32':
        tyname = tl.int32
    elif dtype == 'int64':
        tyname = tl.int64
    elif dtype == 'float16':
        tyname = tl.float16
    elif dtype == 'int16':
        tyname = tl.int16
    elif dtype == 'int8':
        tyname = tl.int8
    elif dtype == 'bool':
        tyname = tl.int1
    elif dtype == 'bfloat16':
        tyname = tl.bfloat16
    elif dtype == 'uint8':
        tyname = tl.uint8
    elif dtype == 'uint16':
        tyname = tl.uint16
    elif dtype == 'uint32':
        tyname = tl.uint32
    elif dtype == 'uint64':
        tyname = tl.uint64
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    return tyname


def get_triton_sig_typename(dtype):
    if dtype == 'float32':
        tyname = "*fp32"
    elif dtype == 'int32':
        tyname = "*i32"
    elif dtype == 'int64':
        tyname = "*i64"
    elif dtype == 'float16':
        tyname = "*fp16"
    elif dtype == 'int16':
        tyname = "*i16"
    elif dtype == 'int8':
        tyname = "*i8"
    elif dtype == 'bool':
        tyname = "*i1"
    elif dtype == 'uint8':
        tyname = "*ui8"
    elif dtype == 'uint16':
        tyname = "*ui16"
    elif dtype == 'uint32':
        tyname = "*ui32"
    elif dtype == 'uint64':
        tyname = "*ui64"
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    return tyname


# Relative error: abs(x_ref - x_cal) / abs(x_ref)
# Absolute error: abs(x_ref - x_cal)

# calculation type operators require different error range
# It is a stricter verification and not satisfied now, save it here
def validate_cal(dtype, y_cal, y_ref):
    if dtype == 'float16' or dtype == 'fp8e4m3' or dtype == 'fp8e5m2':
        if torch.mean(y_ref) < 0.001:
            assert torch.abs(y_cal - y_ref) < 0.001, "|y_cal - y_ref| < 0.001 is required !"
        else:
            diff = torch.div(torch.abs(y_cal - y_ref), torch.abs(y_cal)) < 0.001
            # all true
            assert diff.all(), "Relative error is less than 0.001 !"
    if dtype == 'float32':
        if torch.mean(y_ref) < 0.0001:
            assert torch.abs(y_cal - y_ref) < 0.0001, "|y_cal - y_ref| < 0.0001 is required !"
        else:
            diff = torch.div(torch.abs(y_cal - y_ref), torch.abs(y_cal)) < 0.0001
            assert diff.all(), "Relative error is less than 0.001 !"
    elif dtype == 'bfloat16':
        diff = torch.div(torch.abs(y_cal - y_ref), torch.abs(y_cal)) < 0.001
        assert diff.all(), "Relative error is less than 0.001 !"
    elif dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


# moving and comparison ops require no precision error
def validate_cmp(dtype, y_cal, y_ref, custom_rtol=None, custom_atol=None):
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()
    default_tolerances = {
        'float16': (1e-03, 1e-03),
        'bfloat16': (5e-03, 5e-03),
        'float32': (1e-04, 1e-04),
        'fp8e4m3': (125e-03, 125e-03),
        'fp8e5m2': (25e-02, 25e-02),
    }
    if dtype in default_tolerances:
        default_rtol, default_atol = default_tolerances[dtype]
        rtol = custom_rtol if custom_rtol is not None else default_rtol
        atol = custom_atol if custom_atol is not None else default_atol
        # bfloat16特殊处理
        if dtype == 'bfloat16':
            torch.testing.assert_close(
                y_ref.to(torch.float32), y_cal.to(torch.float32),
                rtol=rtol, atol=atol, equal_nan=True
            )
        else:
            torch.testing.assert_close(y_ref, y_cal, rtol=rtol, atol=atol, equal_nan=True)
    elif dtype in ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal.cpu(), y_ref.cpu())
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def validate_cmp_with_expection(dtype, y_cal, y_ref, expect):
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        if expect:
            assert torch.allclose(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
        else:
            assert not torch.allclose(y_ref, y_cal, rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        if expect:
            assert torch.equal(y_cal, y_ref)
        else:
            assert not torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))


def raises_with_match(expected_exception, match_pattern):
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            with pytest.raises(expected_exception, match=match_pattern):
                return test_func(*args, **kwargs)

        return wrapper

    return decorator


def capture_output(expected_output):
    def decorator(test_func):
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            capsys = kwargs.pop('capsys', None)
            if capsys is None:
                try:
                    capsys = pytest.fixture(capsys)()
                except:
                    raise RuntimeError("This decorator requires pytest's capsys fixture")
            test_func(capsys, *args, **kwargs)
            captured = capsys.readouterr()
            # pybind11::scoped_ostream_redirect captures std::cout with \x00 inserted
            # for now, no idea how to eliminate \x00 from C++ side.
            cleaned = re.sub(r"\x00", "", captured.out)
            assert expected_output in cleaned

        return wrapper

    return decorator


def get_version():
    import ctypes
    rtsdll = ctypes.CDLL("libruntime.so")
    c_char_t = ctypes.create_string_buffer(b'\xff' * 256, 256)
    rtsdll.rtGetSocVersion(c_char_t, ctypes.c_uint32(256))
    soc_version = c_char_t.value.decode("utf-8")
    return soc_version


def is_A5_soc_version():
    soc_version = get_version()
    return "Ascend910_95" in soc_version or "Ascend950" in soc_version


# 因子分解函数
def prime_factorization(num):
    factors = []
    d = 2
    while d * d <= num:
        while num % d == 0:
            factors.append(d)
            num //= d
        d += 1
    if num > 1:
        factors.append(num)
    return factors


def generate_random_shape(sum_val, n):
    """
    生成一个随机 shape，使得各轴的乘积等于 sum_val。
    n: 新 shape 的维度数量。
    """

    import random

    # 排除非法以及全1情况
    if sum_val <= 0:
        raise ValueError("sum_val must be a positive integer.")
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if sum_val == 1:
        return [1] * n
    # 因子分解
    prime_factors = prime_factorization(sum_val)
    total_prime_count = len(prime_factors)
    # 因子不足补1
    if total_prime_count < n:
        factors = prime_factors[:]
        for i in range(n - total_prime_count):
            factors.append(1)
        random.shuffle(factors)
    # 取前n个因子后继续随机组合
    else:
        random.shuffle(prime_factors)
        factors = prime_factors[:n]
        for p in prime_factors[n:]:
            # 随机选择一个因子位置，乘上这个质因子
            idx = random.randint(0, n - 1)
            factors[idx] *= p
        random.shuffle(factors)
    return tuple(factors)


def generate_random_shape_nth_root(max_mul_shape, dim):
    """
    给定 max_mul_shape，随机生成dim维shape，要求每个维度不超过max_mul_shape的dim次方根
    """
    import random

    root = max_mul_shape ** (1 / dim)
    max_value = int(root)

    # 若max_value小于等于2，则使用generate_random_shape方式生成shape
    if max_value <= 2:
        shape = generate_random_shape(max_mul_shape, dim)
        return shape

    # 生成维度为 dim 的列表，每个元素在 2 到 max_mul_shape 之间
    shape = [random.randint(2, max_value) for _ in range(dim)]

    # 计算乘积
    product = math.prod(shape)

    # 确保乘积不超过 max_mul_shape
    while product > max_mul_shape:
        # 如果乘积超过 max_mul_shape，重新生成列表
        shape = [random.randint(2, max_value) for _ in range(dim)]
        product = math.prod(shape)

    return tuple(shape)


# 根据输入计算单个Tensor内存占用
def tensor_mem(dtype, shape):
    dtype_size = get_dtype_size(dtype)
    return dtype_size * math.prod(shape)


# 计算多个Tensor的总内存（输入和输出）
def total_mem_for_tensors(tensors):
    """
    tensors: Tensor描述列表，每个元素为dict，包含'dtype'和'shape'
    示例: [{'dtype': 'int32', 'shape': [3,6,3,4]}, ...]
    """
    total = 0
    for tensor in tensors:
        total += tensor_mem(tensor['dtype'], tensor['shape'])
    return total


# 系统UB规格阈值（根据已知信息中的Ascend型号）
UB_threshold_A5 = 248 * 1024
UB_threshold_A2A3 = 192 * 1024


# 根据版本号，确认系统UB阈值
def UB_threshold_value():
    system_version = get_version()
    if "Ascend910_95" in system_version or "Ascend950" in system_version:
        UB_threshold = UB_threshold_A5
    else:
        UB_threshold = UB_threshold_A2A3
    return UB_threshold


# 根据数据类型*（shape*张量数量）与系统UB规格判定，排除异常随机值导致的UB超限误报
def should_skip_due_to_mem(tensors, UB_threshold=None):
    total_mem = total_mem_for_tensors(tensors)
    if UB_threshold is None:
        UB_threshold = UB_threshold_value()
    # 调用需要确认：输入输出数据类型/shape不一致的特殊情况
    if total_mem > UB_threshold:
        tensor_info = "; ".join([f"dtype:{t['dtype']} shape:{t['shape']}" for t in tensors])
        pytest.skip(f"内存超限：总占用{total_mem}字节，阈值{UB_threshold}字节。Tensor详情: {tensor_info}")


# 根据单个shape的mul最大值来判断排除UB超限误报
def should_skip_due_to_mem2(tensors, mul_shape_threshold_custom=None):
    default_direct = mul_shape_threshold(len(tensors))
    if mul_shape_threshold_custom is None:
        mul_shape_threshold_custom = default_direct

    for tensor in tensors:
        if tensor['dtype'] not in mul_shape_threshold_custom:
            mul_shape_threshold_custom[tensor['dtype']] = default_direct[tensor['dtype']]
        if math.prod(tensor['shape']) > mul_shape_threshold_custom[tensor['dtype']]:
            pytest.skip(
                f"内存超限：当前类型：{tensor['dtype']}，单个shape最大乘积为{mul_shape_threshold_custom[tensor['dtype']]}，实际为{math.prod(tensor['shape'])}。")


# 根据单个shape的mul最大值来判断排除UB超限误报
def should_skip_due_to_mem2(tensors, mul_shape_threshold_custom=None):
    default_direct = mul_shape_threshold(len(tensors))
    if mul_shape_threshold_custom is None:
        mul_shape_threshold_custom = default_direct

    for tensor in tensors:
        if tensor['dtype'] not in mul_shape_threshold_custom:
            mul_shape_threshold_custom[tensor['dtype']] = default_direct[tensor['dtype']]
        if math.prod(tensor['shape']) > mul_shape_threshold_custom[tensor['dtype']]:
            pytest.skip(
                f"内存超限：当前类型：{tensor['dtype']}，单个shape最大乘积为{mul_shape_threshold_custom[tensor['dtype']]}，实际为{math.prod(tensor['shape'])}。")


dtype_list = ['uint8', 'int8', 'uint16', 'int16', 'uint32', 'uint64', 'int32', 'int64', 'float16', 'bfloat16', 'float32', 'bool']


# 给定同shape的tensor个数num，计算不同dtype下不超UB的最大shape乘积值
def mul_shape_threshold(num):
    UB_threshold = UB_threshold_value()

    max_mul_shape_dict = {}
    for dtype in dtype_list:
        max_mul_shape_dict[dtype] = UB_threshold // (get_dtype_size(dtype) * num)
    return max_mul_shape_dict


def get_reduced_shape(reduce_list, to_be_reduced_shape):
    result_shape = tuple()
    for index, value in enumerate(to_be_reduced_shape):
        if index not in reduce_list and index - len(to_be_reduced_shape) not in reduce_list:
            result_shape += (value,)
    return result_shape