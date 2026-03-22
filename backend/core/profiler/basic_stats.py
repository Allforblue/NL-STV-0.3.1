import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, Any, List


def get_column_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    提取每一列的基础统计信息，用于构建 LLM Prompt。
    优化点：增加了对时间、分类基数、地理边界的深度提取，确保 JSON 序列化安全。
    """
    stats = {}

    # 1. 确定需要处理的列（处理 GeoDataFrame 的特殊情况）
    columns_to_process = df.columns.tolist()
    is_gdf = isinstance(df, gpd.GeoDataFrame)

    if is_gdf and df.geometry.name not in columns_to_process:
        columns_to_process.append(df.geometry.name)

    for col in columns_to_process:
        series = df[col]
        # 基础信息
        col_type = str(series.dtype)

        # 优化点 1：修改采样策略，均匀获取大约 10 个样本，更好地向大模型体现数据的真实跨度
        valid_series = series.dropna()
        n_samples = 3
        if len(valid_series) > n_samples:
            # 使用 linspace 生成均匀分布的索引，确保覆盖首、中、尾数据
            indices = np.linspace(0, len(valid_series) - 1, num=n_samples, dtype=int)
            samples = valid_series.iloc[indices].tolist()
        else:
            samples = valid_series.tolist()
        samples = [str(s) for s in samples]

        col_info = {
            "dtype": col_type,
            "samples": samples,
            "missing_count": int(series.isna().sum()),
            "missing_rate": round(float(series.isna().mean()), 4)
        }

        # 优化点 2：增强对伪装成 Object/String 的时间列的检测 (基于列名探测和样本解析)
        is_dt = pd.api.types.is_datetime64_any_dtype(series) or "datetime" in col_type
        if not is_dt and pd.api.types.is_object_dtype(series):
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in ['date', 'time', 'timestamp']):
                try:
                    if not valid_series.empty:
                        pd.to_datetime(valid_series.iloc[0])
                        is_dt = True
                except Exception:
                    pass

        # 2. 针对数值类型的统计 (int, float)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            try:
                if not valid_series.empty:
                    col_info.update({
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": float(series.mean())
                    })
            except Exception:
                pass

        # 3. 针对时间类型的统计 (Datetime 或伪装的时间列)
        elif is_dt:
            try:
                col_info["dtype"] = "datetime"
                if not valid_series.empty:
                    # 优化点 3：如果是字符串形式的时间，强制转换为 datetime 后再求正确的业务 min/max
                    if pd.api.types.is_object_dtype(series):
                        dt_series = pd.to_datetime(valid_series, errors='coerce').dropna()
                        if not dt_series.empty:
                            col_info["min"] = str(dt_series.min())
                            col_info["max"] = str(dt_series.max())
                    else:
                        col_info["min"] = str(valid_series.min())
                        col_info["max"] = str(valid_series.max())
            except Exception:
                pass

        # 4. 针对类别/字符串类型的统计 (Object, Category, String)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
            try:
                nunique = int(series.nunique())
                col_info["unique_count"] = nunique
                # 如果类别较少，直接给出类别名称，方便 LLM 做颜色映射或过滤器
                if nunique <= 10:
                    col_info["categories"] = [str(x) for x in series.unique() if pd.notna(x)]
            except Exception:
                pass

        # 5. 针对几何类型的统计 (GeoPandas Geometry)
        if is_gdf and col == df.geometry.name:
            col_info["dtype"] = "geometry"
            try:
                if not df.empty:
                    # 获取主流几何类型 (Point, LineString, Polygon)
                    col_info["geom_type"] = str(df.geom_type.mode()[0])
                    # 获取地理边界框 [minx, miny, maxx, maxy]
                    bounds = df.total_bounds
                    col_info["bounds"] = [float(x) for x in bounds] if len(bounds) == 4 else []
            except Exception:
                col_info["geom_type"] = "unknown"

        stats[col] = col_info

    return stats


def get_dataset_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    """
    获取数据集层面的指纹信息。
    优化点：增加地理特征检测提示（即使不是 GeoDataFrame）。
    """
    # 基础结构
    fingerprint = {
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "column_names": df.columns.tolist(),
        "column_stats": get_column_stats(df)
    }

    # 1. 显式地理检测 (GeoDataFrame)
    if isinstance(df, gpd.GeoDataFrame):
        fingerprint["is_geospatial"] = True
        fingerprint["crs"] = str(df.crs) if df.crs else "EPSG:4326 (assumed)"
        fingerprint["primary_geometry"] = str(df.geometry.name)
    else:
        # 2. 隐式地理检测 (针对普通 CSV 中的经纬度列)
        # 这种提示能帮助 LLM 决定是否需要先执行 pd.to_numeric() 或 gpd.points_from_xy()
        lat_candidates = [c for c in df.columns if str(c).lower() in ['lat', 'latitude', 'y', '纬度']]
        lon_candidates = [c for c in df.columns if str(c).lower() in ['lon', 'lng', 'longitude', 'x', '经度']]

        if lat_candidates and lon_candidates:
            fingerprint["is_geospatial"] = True
            fingerprint["potential_lat_lon"] = {"lat": lat_candidates[0], "lon": lon_candidates[0]}
            fingerprint["is_geodataframe"] = False
        else:
            fingerprint["is_geospatial"] = False

    return fingerprint