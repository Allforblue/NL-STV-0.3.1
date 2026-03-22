# backend/core/sdk/operators.py
import pandas as pd
import numpy as np
from typing import List, Union


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: Union[str, List[str]] = None, how: str = 'left', **kwargs):
    """
    强制类型对齐的安全合并 (终极版)
    AI 在处理多表级联时调用此方法。自动处理 left_on/right_on 的类型对齐，
    并自动剥离因为空值导致的 .0 浮点数后缀，保证 Lookup 字典表关联 100% 成功。
    """
    left = left.copy()
    right = right.copy()

    left_cols = []
    right_cols = []

    # 提取所有关联键
    if on:
        cols = [on] if isinstance(on, str) else on
        left_cols.extend(cols)
        right_cols.extend(cols)

    if 'left_on' in kwargs:
        l_on = kwargs['left_on']
        left_cols.extend([l_on] if isinstance(l_on, str) else l_on)
    if 'right_on' in kwargs:
        r_on = kwargs['right_on']
        right_cols.extend([r_on] if isinstance(r_on, str) else r_on)

    # 【修复核心】：强制转为字符串，并剥离 '.0' 后缀，消灭 int32/float64 vs object 报错
    for c in set(left_cols):
        if c in left.columns:
            left[c] = left[c].astype(str).str.replace(r'\.0$', '', regex=True)
    for c in set(right_cols):
        if c in right.columns:
            right[c] = right[c].astype(str).str.replace(r'\.0$', '', regex=True)

    if on:
        return pd.merge(left, right, on=on, how=how, **kwargs)
    else:
        return pd.merge(left, right, how=how, **kwargs)


def prepare_animation_grid(df: pd.DataFrame, ids: List[str], time_col: str, id_col: str):
    """构建时空笛卡尔积网格，防止 Plotly 动画闪烁"""
    if time_col not in df.columns:
        return pd.DataFrame(columns=[id_col, time_col])
    all_times = sorted(df[time_col].unique())
    grid = pd.MultiIndex.from_product([ids, all_times], names=[id_col, time_col]).to_frame(index=False)

    # 同样剥离动画网格中的 .0
    grid[id_col] = grid[id_col].astype(str).str.replace(r'\.0$', '', regex=True)
    return grid


def safe_resample(
    df: pd.DataFrame,
    time_col: str,
    val_col: str = None,
    freq: str = '1h',
    agg_func: str = 'sum'
):
    """
    安全且准确的时间重采样（Accuracy-First）

    设计原则：
    1. 不采样，不做近似，保证统计准确性。
    2. 不偷偷改变用户指定的时间粒度 freq。
    3. 仅裁剪与聚合无关的列，以降低内存占用，但不影响结果。
    4. 显式支持:
       - sum
       - mean
       - size
       - count
    5. 保证输出时间列为 datetime，便于 Plotly 正确渲染时间轴。

    参数:
        df: 输入 DataFrame
        time_col: 时间列名
        val_col: 数值列名
            - 对 sum / mean: 必须提供且必须存在
            - 对 size / count: 可为空；为空时默认输出列名为 'count'
        freq: 重采样频率，如 '1h', '1d'
        agg_func: 聚合方式，支持 'sum' / 'mean' / 'size' / 'count'

    返回:
        DataFrame，包含:
        - [time_col, output_col]
    """

    if time_col not in df.columns:
        raise ValueError(
            f"[SDK Error] safe_resample: 时间列 '{time_col}' 不存在！可用列: {df.columns.tolist()}"
        )

    agg_func = str(agg_func or 'sum').lower().strip()
    freq = str(freq or '1h').replace('H', 'h').replace('D', 'd')

    supported_aggs = {'sum', 'mean', 'size', 'count'}
    if agg_func not in supported_aggs:
        raise ValueError(
            f"[SDK Error] safe_resample: 不支持的 agg_func='{agg_func}'。"
            f"仅支持 {sorted(supported_aggs)}"
        )

    # --------------------------------------------------
    # 1. 只保留必要列（减少内存，但不影响准确性）
    # --------------------------------------------------
    required_cols = [time_col]

    if agg_func in {'sum', 'mean'}:
        if val_col is None:
            raise ValueError(
                f"[SDK Error] safe_resample: agg_func='{agg_func}' 时必须提供 val_col。"
            )
        if val_col not in df.columns:
            raise ValueError(
                f"[SDK Error] safe_resample: 数值列 '{val_col}' 不存在！可用列: {df.columns.tolist()}"
            )
        required_cols.append(val_col)

    elif agg_func in {'size', 'count'}:
        # count/size 允许不传 val_col
        if val_col is not None and val_col in df.columns:
            required_cols.append(val_col)

    df_small = df[required_cols].copy()

    # --------------------------------------------------
    # 2. 时间列强制转 datetime
    # --------------------------------------------------
    df_small[time_col] = pd.to_datetime(df_small[time_col], errors='coerce')
    df_small = df_small.dropna(subset=[time_col])

    if df_small.empty:
        output_col = val_col if val_col else 'count'
        return pd.DataFrame(columns=[time_col, output_col])

    # --------------------------------------------------
    # 3. 建立 resampler
    # --------------------------------------------------
    resampler = df_small.set_index(time_col).resample(freq)

    # --------------------------------------------------
    # 4. 聚合逻辑
    # --------------------------------------------------
    if agg_func == 'sum':
        # sum 必须使用 val_col
        output_col = val_col
        df_resampled = resampler[val_col].sum().reset_index()

    elif agg_func == 'mean':
        output_col = val_col
        df_resampled = resampler[val_col].mean().reset_index()

    elif agg_func == 'size':
        # size = 每个时间桶的行数（最适合订单数/记录数）
        output_col = val_col if val_col else 'count'
        df_resampled = resampler.size().reset_index(name=output_col)

    elif agg_func == 'count':
        # count = 非空值数量
        # 若 val_col 存在，则统计 val_col 非空数；否则退化为 size
        output_col = val_col if val_col else 'count'
        if val_col is not None and val_col in df_small.columns:
            df_resampled = resampler[val_col].count().reset_index(name=output_col)
        else:
            df_resampled = resampler.size().reset_index(name=output_col)

    # --------------------------------------------------
    # 5. 缺失值处理
    # --------------------------------------------------
    # sum / size / count 的空时间桶填 0
    # mean 的空桶保留 NaN 后删除，避免误导
    if agg_func in {'sum', 'size', 'count'}:
        df_resampled[output_col] = df_resampled[output_col].fillna(0)
    else:  # mean
        df_resampled = df_resampled.dropna(subset=[output_col])

    # --------------------------------------------------
    # 6. 再次保证时间列为 datetime（防止后续显示成科学计数法）
    # --------------------------------------------------
    df_resampled[time_col] = pd.to_datetime(df_resampled[time_col], errors='coerce')

    return df_resampled