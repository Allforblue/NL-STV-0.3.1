# backend/core/sdk/data_utils.py
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
# 导入同级目录下的常量
from .constraints import SKEWNESS_THRESHOLD


def get_log_transform_params(series: pd.Series) -> Dict[str, Any]:
    """
    长尾数据分析与对数刻度生成
    【终极修复】：在对数空间内进行均匀等分，确保前端渲染的刻度视觉上绝对均匀分布！
    """
    if series.empty:
        return {"use_log": False}

    max_val = series.max()
    median_val = series.median()
    min_val = series.min()

    if max_val > median_val * SKEWNESS_THRESHOLD and max_val > 0:
        # 1. 将极大值和极小值映射到对数空间
        log_min = np.log1p(min_val)
        log_max = np.log1p(max_val)

        # 2. 在对数空间里均匀地切 5 刀 (这才是视觉上均匀的关键)
        log_ticks = np.linspace(log_min, log_max, 5)

        # 3. 将对数值还原回现实世界的真实业务数值 (用于给用户看)
        real_ticks = np.expm1(log_ticks)

        return {
            "use_log": True,
            "tickvals": log_ticks.tolist(),
            "ticktext": [f"{int(v):,}" for v in real_ticks]
        }

    return {"use_log": False}


def sanitize_for_json(obj: Any) -> Any:
    """
    递归清洗对象以确保其兼容 JSON/GeoJSON 序列化。
    主要处理 Numpy 类型和 Pandas 时间戳。
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, np.datetime64):
        return str(pd.to_datetime(obj))
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    else:
        return obj


def filter_positive_values(df: pd.DataFrame, cols: List[str]):
    """
    过滤非正值 (针对需要对数刻度的柱状图/饼图，对应 Rule 4)
    """
    for col in cols:
        if col in df.columns:
            df = df[df[col] > 0]
    return df


def infer_spatial_cols(df: pd.DataFrame, lat_col: str = None, lon_col: str = None) -> Tuple[str, str]:
    """自动推断或校验经纬度列"""
    cols_lower = {c.lower(): c for c in df.columns}

    # 1. 如果 AI 传了列名，严格校验它是否存在
    if lat_col and lon_col:
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(
                f"[SDK Error] 传入的经纬度列 '{lat_col}' 或 '{lon_col}' 在表中不存在！可用列: {df.columns.tolist()}")
        return lat_col, lon_col

    # 2. 如果 AI 没传（或传了 None），启动自动推断
    lat_candidates = ['lat', 'latitude', 'pickup_latitude', 'dropoff_latitude', 'y', '纬度']
    lon_candidates = ['lon', 'lng', 'longitude', 'pickup_longitude', 'dropoff_longitude', 'x', '经度']

    found_lat = next((cols_lower[k] for k in lat_candidates if k in cols_lower), None)
    found_lon = next((cols_lower[k] for k in lon_candidates if k in cols_lower), None)

    if found_lat and found_lon:
        return found_lat, found_lon

    raise ValueError(
        f"[SDK Error] 无法自动推断出经纬度列，请显式传入 lat_col 和 lon_col。当前可用列: {df.columns.tolist()}")


def infer_time_col(df: pd.DataFrame, time_col: str = None) -> str:
    """自动推断或校验时间列"""
    if time_col:
        if time_col not in df.columns:
            raise ValueError(f"[SDK Error] 传入的时间列 '{time_col}' 在表中不存在！可用列: {df.columns.tolist()}")
        return time_col

    # 扫描 Datetime 类型的列
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # 退而求其次，根据列名猜测并尝试转换
    time_keywords = ['time', 'date', 'timestamp', '日期', '时间']
    for col in df.columns:
        if any(kw in str(col).lower() for kw in time_keywords):
            return col

    raise ValueError(
        f"[SDK Error] 无法推断出时间列 (没有 datetime 类型或包含 time 的列名)。可用列: {df.columns.tolist()}")


def infer_numeric_col(df: pd.DataFrame, val_col: str = None) -> str:
    """校验或推断度量列 (数值型)"""
    if val_col:
        if val_col not in df.columns:
            raise ValueError(f"[SDK Error] 传入的度量列 '{val_col}' 在表中不存在！可用列: {df.columns.tolist()}")
        return val_col

    # 寻找第一个数值型列（排除经纬度和 ID）
    exclude_kws = ['id', 'lat', 'lon', 'lng']
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not any(kw in str(col).lower() for kw in exclude_kws):
            return col

    # 如果全都是 ID/坐标，无奈返回第一个数值列
    if len(numeric_cols) > 0:
        return numeric_cols[0]

    raise ValueError(f"[SDK Error] 数据表中没有可用的数值型列作为度量值。")