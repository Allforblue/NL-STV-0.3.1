# backend/core/sdk/visualizer.py

import json
import pandas as pd
import geopandas as gpd
import plotly.express as px
import numpy as np
from typing import Optional, List, Dict, Any
import warnings

from .constraints import THEME_CONFIG, COLORBAR_STYLE, SKEWNESS_THRESHOLD, MAX_SCATTER_POINTS
from .operators import safe_merge, prepare_animation_grid
from .data_utils import get_log_transform_params, infer_spatial_cols, infer_time_col, infer_numeric_col

warnings.filterwarnings(
    "ignore",
    message="Geometry is in a geographic CRS. Results from 'centroid' are likely incorrect."
)


class STVisualizer:
    """
    Spatio-Temporal Visualization SDK (Protocol Unified Map Edition)

    当前版本核心原则：
    1. 所有“中央地图”相关方法统一返回 STV Map Protocol，而不是 Plotly 地图 Figure。
    2. 所有“右侧统计图”方法继续返回 Plotly Figure。
    3. 统一中央地图渲染路径，彻底绕开 Plotly mapbox / choropleth_mapbox 的兼容性问题。
    """

    def __init__(self):
        self.theme = THEME_CONFIG
        self._global_id_map = {}

    def _apply_pro_layout(self, fig, colorbar_title: str = ""):
        fig.update_layout(
            template=self.theme["template"],
            margin=dict(t=20, b=20, l=20, r=80),
            coloraxis_colorbar={**COLORBAR_STYLE, "title": {"text": colorbar_title, "side": "top"}}
        )
        fig.update_traces(
            marker_line_width=self.theme["marker_line_width"],
            marker_line_color=self.theme["marker_line_color"]
        )
        return fig

    def _safe_hover_cols(self, df: pd.DataFrame, hover_cols: Optional[List[str]]) -> List[str]:
        if not hover_cols:
            return []
        return [col for col in hover_cols if col in df.columns]

    def _semantic_cols_from_df(self, df: pd.DataFrame) -> List[str]:
        semantic_kws = ['zone', 'borough', 'name', 'neighborhood', 'district', 'city', 'region']
        return [c for c in df.columns if str(c).lower() in semantic_kws]

    def _build_protocol_legend(self, values: pd.Series, display_title: str) -> Optional[Dict[str, Any]]:
        log_info = get_log_transform_params(values)
        if log_info.get("use_log"):
            return {
                "title": display_title,
                "tickvals": log_info["tickvals"],
                "ticktext": log_info["ticktext"]
            }
        return None

    # ==========================================
    # 面状地图渲染（静态）→ 协议输出
    # ==========================================
    def choropleth(
        self,
        df: pd.DataFrame,
        gdf: gpd.GeoDataFrame,
        data_key: str,
        geo_key: str,
        val_col: str = None,
        hover_cols: Optional[List[str]] = None,
        label_map: Optional[Dict[str, str]] = None
    ):
        if data_key not in df.columns:
            raise ValueError(f"[SDK Error] 业务表 df 中缺失 data_key '{data_key}'！")
        if geo_key not in gdf.columns:
            raise ValueError(f"[SDK Error] 地理表 gdf 中缺失 geo_key '{geo_key}'！")

        val_col = infer_numeric_col(df, val_col)

        df = df.copy().dropna(subset=[data_key])
        gdf_wgs = gdf.copy().to_crs(epsg=4326)

        df[data_key] = df[data_key].astype(str).str.replace(r'\.0$', '', regex=True)
        gdf_wgs[geo_key] = gdf_wgs[geo_key].astype(str).str.replace(r'\.0$', '', regex=True)

        merged = safe_merge(gdf_wgs, df, left_on=geo_key, right_on=data_key, how='left')
        merged[val_col] = merged[val_col].fillna(0)

        log_info = get_log_transform_params(merged[val_col])
        color_col = val_col
        if log_info["use_log"]:
            merged[f"{val_col}_log"] = np.log1p(np.clip(merged[val_col], 0, None))
            color_col = f"{val_col}_log"

        auto_semantic_cols = self._semantic_cols_from_df(gdf_wgs)
        safe_hover = self._safe_hover_cols(merged, hover_cols)

        geojson_keep_cols = list(set([geo_key, 'geometry'] + auto_semantic_cols))
        data_keep_cols = list(set([data_key, val_col, color_col] + safe_hover + auto_semantic_cols))

        payload = {
            "is_stv_map_protocol": True,
            "layer_type": "choropleth",
            "geojson": json.loads(gdf_wgs[geojson_keep_cols].to_json()),
            "data": merged[data_keep_cols].to_dict(orient="records"),
            "mapping": {
                "id": data_key,
                "geo_id": geo_key,
                "color_value": color_col,
                "display_value": val_col
            }
        }

        if log_info["use_log"]:
            payload["legend"] = {
                "title": val_col,
                "tickvals": log_info["tickvals"],
                "ticktext": log_info["ticktext"]
            }
        else:
            payload["legend"] = {
                "title": val_col,
                "tickvals": [0, float(merged[val_col].max()) if len(merged) > 0 else 0],
                "ticktext": ["0", f"{float(merged[val_col].max()):,.0f}" if len(merged) > 0 else "0"]
            }

        if label_map:
            payload["label_map"] = label_map

        return payload

    # ==========================================
    # 面状地图渲染（动画）→ 协议输出
    # ==========================================
    def animated_map(
        self,
        df: pd.DataFrame,
        gdf: gpd.GeoDataFrame,
        data_key: str,
        geo_key: str,
        time_col: str = None,
        val_col: str = None,
        hover_cols: Optional[List[str]] = None
    ):
        time_col = infer_time_col(df, time_col)
        val_col = infer_numeric_col(df, val_col)

        if pd.api.types.is_numeric_dtype(df[time_col]):
            if df[time_col].max() <= 366:
                raise ValueError(
                    f"[SDK Error] 动画时间列 '{time_col}' 不能是离散整数(如 dt.hour 或 dt.day)! "
                    f"你必须保留真实的绝对时间戳！请使用 `dt.floor('h')`，"
                    f"绝对禁止使用 `dt.hour` 或 `dt.day`！"
                )

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        if df[time_col].isna().mean() > 0.5:
            raise ValueError(
                f"[SDK Error] 动画时间列 '{time_col}' 无法解析为连续的 DateTime 对象，请检查你的 Pandas 数据清洗逻辑！"
            )

        if data_key not in df.columns:
            raise ValueError(f"[SDK Error] 动画业务表 df 中缺失 data_key '{data_key}'")
        if geo_key not in gdf.columns:
            raise ValueError(f"[SDK Error] 动画地理表 gdf 中缺失 geo_key '{geo_key}'")

        gdf_wgs = gdf.copy().to_crs(epsg=4326)

        df = df.dropna(subset=[data_key, time_col])
        df[data_key] = df[data_key].astype(str).str.replace(r'\.0$', '', regex=True)
        gdf_wgs[geo_key] = gdf_wgs[geo_key].astype(str).str.replace(r'\.0$', '', regex=True)

        geo_ids = gdf_wgs[geo_key].unique()

        grid = prepare_animation_grid(df, list(geo_ids), time_col, data_key)
        df_agg = df.groupby([data_key, time_col])[val_col].sum().reset_index()
        df_agg[data_key] = df_agg[data_key].astype(str)

        df_anim = pd.merge(grid, df_agg, on=[data_key, time_col], how='left').fillna(0)

        auto_semantic_cols = self._semantic_cols_from_df(gdf_wgs)
        attr_cols = list(set([geo_key] + (hover_cols or []) + auto_semantic_cols))
        attr_cols = [c for c in attr_cols if c in gdf_wgs.columns]

        name_map = gdf_wgs[attr_cols].drop_duplicates()
        if geo_key != data_key:
            name_map = name_map.rename(columns={geo_key: data_key})

        df_anim = pd.merge(df_anim, name_map, on=data_key, how='left')
        df_anim = df_anim.sort_values([time_col, data_key])

        log_info = get_log_transform_params(df_anim[val_col])
        color_col = val_col
        if log_info["use_log"]:
            df_anim[f"{val_col}_log"] = np.log1p(np.clip(df_anim[val_col], 0, None))
            color_col = f"{val_col}_log"

        df_anim['_timestamp'] = pd.to_datetime(df_anim[time_col]).astype('int64') // 10 ** 9

        keep_cols = list(set(
            [data_key, time_col, '_timestamp', val_col, color_col] + (hover_cols or []) + auto_semantic_cols
        ))
        keep_cols = [c for c in keep_cols if c in df_anim.columns]

        geojson_keep_cols = list(set([geo_key, 'geometry'] + auto_semantic_cols))

        payload = {
            "is_stv_map_protocol": True,
            "layer_type": "animated_choropleth",
            "geojson": json.loads(gdf_wgs[geojson_keep_cols].to_json()),
            "data": df_anim[keep_cols].to_dict(orient="records"),
            "mapping": {
                "id": data_key,
                "geo_id": geo_key,
                "color_value": color_col,
                "display_value": val_col,
                "timestamp": "_timestamp"
            }
        }

        if log_info["use_log"]:
            payload["legend"] = {
                "title": val_col,
                "tickvals": log_info["tickvals"],
                "ticktext": log_info["ticktext"]
            }
        else:
            payload["legend"] = {
                "title": val_col,
                "tickvals": [0, float(df_anim[val_col].max()) if len(df_anim) > 0 else 0],
                "ticktext": ["0", f"{float(df_anim[val_col].max()):,.0f}" if len(df_anim) > 0 else "0"]
            }

        return payload

    # ==========================================
    # 点状地图渲染（静态）→ 协议输出
    # ==========================================
    def scatter_map(
        self,
        df: pd.DataFrame,
        lat_col: str = None,
        lon_col: str = None,
        val_col: Optional[str] = None,
        size_col: Optional[str] = None,
        hover_cols: Optional[List[str]] = None
    ):
        lat_col, lon_col = infer_spatial_cols(df, lat_col, lon_col)
        if val_col:
            val_col = infer_numeric_col(df, val_col)
        if size_col:
            size_col = infer_numeric_col(df, size_col)

        df = df.copy().dropna(subset=[lat_col, lon_col])

        if len(df) > MAX_SCATTER_POINTS:
            df = df.sample(MAX_SCATTER_POINTS, random_state=42)

        auto_semantic_cols = self._semantic_cols_from_df(df)
        safe_hover = self._safe_hover_cols(df, hover_cols)

        keep_cols = list(set([lat_col, lon_col] + safe_hover + auto_semantic_cols))
        if val_col:
            keep_cols.append(val_col)
        if size_col:
            keep_cols.append(size_col)

        payload = {
            "is_stv_map_protocol": True,
            "layer_type": "scatter",
            "data": df[list(set(keep_cols))].to_dict(orient="records"),
            "mapping": {
                "latitude": lat_col,
                "longitude": lon_col,
                "color_value": val_col,
                "size_value": size_col
            }
        }

        if val_col and val_col in df.columns:
            log_info = get_log_transform_params(df[val_col])
            if log_info["use_log"]:
                payload["legend"] = {
                    "title": val_col,
                    "tickvals": log_info["tickvals"],
                    "ticktext": log_info["ticktext"]
                }
            else:
                payload["legend"] = {
                    "title": val_col,
                    "tickvals": [0, float(df[val_col].max()) if len(df) > 0 else 0],
                    "ticktext": ["0", f"{float(df[val_col].max()):,.0f}" if len(df) > 0 else "0"]
                }

        return payload

    # ==========================================
    # 热力图渲染（静态）→ 协议输出
    # ==========================================
    def heatmap(
        self,
        df: pd.DataFrame,
        lat_col: str = None,
        lon_col: str = None,
        val_col: str = None
    ):
        lat_col, lon_col = infer_spatial_cols(df, lat_col, lon_col)
        if val_col:
            val_col = infer_numeric_col(df, val_col)

        df = df.copy().dropna(subset=[lat_col, lon_col])

        keep_cols = [lat_col, lon_col]
        if val_col:
            keep_cols.append(val_col)

        payload = {
            "is_stv_map_protocol": True,
            "layer_type": "heatmap",
            "data": df[list(set(keep_cols))].to_dict(orient="records"),
            "mapping": {
                "latitude": lat_col,
                "longitude": lon_col,
                "color_value": val_col
            }
        }

        if val_col and val_col in df.columns:
            log_info = get_log_transform_params(df[val_col])
            if log_info["use_log"]:
                payload["legend"] = {
                    "title": val_col,
                    "tickvals": log_info["tickvals"],
                    "ticktext": log_info["ticktext"]
                }
            else:
                payload["legend"] = {
                    "title": val_col,
                    "tickvals": [0, float(df[val_col].max()) if len(df) > 0 else 0],
                    "ticktext": ["0", f"{float(df[val_col].max()):,.0f}" if len(df) > 0 else "0"]
                }
        else:
            payload["legend"] = {
                "title": "Density",
                "tickvals": [0, 1],
                "ticktext": ["Low", "High"]
            }

        return payload

    # ==========================================
    # 点状地图渲染（动画）→ 协议输出
    # ==========================================
    def animated_scatter(
        self,
        df: pd.DataFrame,
        lat_col: str = None,
        lon_col: str = None,
        time_col: str = None,
        val_col: str = None,
        hover_cols: Optional[List[str]] = None
    ):
        lat_col, lon_col = infer_spatial_cols(df, lat_col, lon_col)
        time_col = infer_time_col(df, time_col)
        if val_col:
            val_col = infer_numeric_col(df, val_col)

        if pd.api.types.is_numeric_dtype(df[time_col]):
            if df[time_col].max() <= 366:
                raise ValueError(
                    f"[SDK Error] 动画时间列 '{time_col}' 不能是离散整数(如 dt.hour)! "
                    f"你必须保留真实的绝对时间戳！请使用 `dt.floor('h')`，绝对禁止使用 `dt.hour` 或 `dt.day`！"
                )

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        if df[time_col].isna().mean() > 0.5:
            raise ValueError(f"[SDK Error] 动画时间列 '{time_col}' 无法解析为连续的 DateTime 对象。")

        df = df.dropna(subset=[lat_col, lon_col, time_col])
        df = df.sort_values(time_col)

        df['_timestamp'] = pd.to_datetime(df[time_col]).astype('int64') // 10 ** 9
        df[time_col] = pd.to_datetime(df[time_col]).dt.strftime('%Y-%m-%d %H:00')

        keep_cols = [lat_col, lon_col, time_col, '_timestamp']
        if val_col:
            keep_cols.append(val_col)
        if hover_cols:
            keep_cols.extend([c for c in hover_cols if c in df.columns])

        payload = {
            "is_stv_map_protocol": True,
            "layer_type": "animated_scatter",
            "data": df[list(set(keep_cols))].to_dict(orient="records"),
            "mapping": {
                "latitude": lat_col,
                "longitude": lon_col,
                "color_value": val_col,
                "timestamp": "_timestamp"
            }
        }

        if val_col and val_col in df.columns:
            log_info = get_log_transform_params(df[val_col])
            if log_info["use_log"]:
                payload["legend"] = {
                    "title": val_col,
                    "tickvals": log_info["tickvals"],
                    "ticktext": log_info["ticktext"]
                }
            else:
                payload["legend"] = {
                    "title": val_col,
                    "tickvals": [0, float(df[val_col].max()) if len(df) > 0 else 0],
                    "ticktext": ["0", f"{float(df[val_col].max()):,.0f}" if len(df) > 0 else "0"]
                }

        return payload

    # ==========================================
    # 统计图表渲染（右侧图表，保留 Plotly）
    # ==========================================
    def bar(self, df: pd.DataFrame, x_val: str, y_cat: str, top_n: int = 15, color_col: Optional[str] = None):
        if x_val not in df.columns or y_cat not in df.columns:
            raise ValueError(f"[SDK Error] 柱状图所需列 '{x_val}' 或 '{y_cat}' 不存在！")

        df_plot = df.copy()
        is_x_num = pd.api.types.is_numeric_dtype(df_plot[x_val])
        is_y_num = pd.api.types.is_numeric_dtype(df_plot[y_cat])

        if is_y_num and not is_x_num:
            cat_col, val_col = x_val, y_cat
        elif is_x_num and not is_y_num:
            cat_col, val_col = y_cat, x_val
        else:
            cat_col, val_col = (
                (x_val, y_cat)
                if df_plot[x_val].nunique() <= df_plot[y_cat].nunique()
                else (y_cat, x_val)
            )

        if len(df_plot) > top_n * 2 or df_plot[cat_col].duplicated().any():
            df_plot = df_plot.groupby(cat_col, as_index=False)[val_col].sum()

        df_plot = df_plot.sort_values(val_col, ascending=True).tail(top_n)

        df_plot[cat_col] = df_plot[cat_col].astype(str).str.replace(r'\.0$', '', regex=True)
        if self._global_id_map:
            df_plot[cat_col] = df_plot[cat_col].map(lambda x: self._global_id_map.get(str(x), x))

        fig = px.bar(
            df_plot,
            x=val_col,
            y=cat_col,
            orientation='h',
            color=color_col or cat_col,
            color_discrete_sequence=px.colors.qualitative.Prism
        )

        self._apply_pro_layout(fig)
        fig.update_layout(yaxis={'type': 'category'}, showlegend=False, margin=dict(l=150))
        return fig

    def pie(self, df: pd.DataFrame, names: str, values: str, top_n: int = 8):
        if names not in df.columns or values not in df.columns:
            raise ValueError(f"[SDK Error] 饼图所需列 names='{names}' 或 values='{values}' 不存在！")

        df_plot = df.copy()
        if len(df_plot) > top_n * 2 or df_plot[names].duplicated().any():
            df_plot = df_plot.groupby(names, as_index=False)[values].sum()

        df_plot = df_plot.sort_values(values, ascending=False).head(top_n)

        df_plot[names] = df_plot[names].astype(str).str.replace(r'\.0$', '', regex=True)
        if self._global_id_map:
            df_plot[names] = df_plot[names].map(lambda x: self._global_id_map.get(str(x), x))

        fig = px.pie(
            df_plot,
            names=names,
            values=values,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        return self._apply_pro_layout(fig)

    def line(self, df: pd.DataFrame, time_col: str = None, val_col: str = None):
        time_col = infer_time_col(df, time_col)
        val_col = infer_numeric_col(df, val_col)
        df_trend = df.sort_values(time_col)

        fig = px.line(df_trend, x=time_col, y=val_col)
        fig.update_traces(
            mode='lines',
            line=dict(width=2, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(99, 110, 250, 0.1)'
        )
        return self._apply_pro_layout(fig)

    def periodic_bar(self, df: pd.DataFrame, time_col: str = None, val_col: str = None, cycle: str = 'hour'):
        time_col = infer_time_col(df, time_col)
        val_col = infer_numeric_col(df, val_col)

        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        if cycle == 'hour':
            df['period'] = df[time_col].dt.hour
        elif cycle == 'dayofweek':
            df['period'] = df[time_col].dt.day_name()

        df_agg = df.groupby('period')[val_col].count().reset_index(name='count')
        if cycle == 'hour':
            df_agg = df_agg.sort_values('period')

        fig = px.bar(df_agg, x='period', y='count')
        return self._apply_pro_layout(fig)