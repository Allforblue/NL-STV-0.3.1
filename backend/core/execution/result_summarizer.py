# backend/core/execution/result_summarizer.py

import logging
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd

logger = logging.getLogger(__name__)


class ResultSummarizer:
    """
    执行结果摘要器

    目标：
    1. 将执行产物（Plotly Figure / STV Map Protocol / DataFrame）转换为统一的语义摘要
    2. 为 InsightExtractor 提供稳定、低歧义的结构化上下文
    3. 避免直接依赖 trace 顺序 / 图表视觉排列来推断图意
    4. 在不引入 compare 专项 schema 的前提下，提升摘要质量与可解释性
    """

    def __init__(self):
        pass

    # =========================================================
    # 公共工具
    # =========================================================
    def _to_native(self, obj: Any) -> Any:
        """将 numpy / pandas 类型递归转为 JSON-friendly native types"""
        if isinstance(obj, dict):
            return {str(k): self._to_native(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [self._to_native(v) for v in obj]
        if isinstance(obj, (np.int64, np.int32, np.int16, np.int8, np.integer)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32, np.float16, np.floating)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        if isinstance(obj, np.datetime64):
            return str(pd.to_datetime(obj))
        if isinstance(obj, np.ndarray):
            return self._to_native(obj.tolist())
        return obj

    def _safe_translate(self, cat_str: Any, global_id_map: Optional[Dict[str, str]] = None) -> str:
        text = str(cat_str).replace('.0', '')
        if global_id_map and text.isdigit() and text in global_id_map:
            return str(global_id_map[text])
        return str(cat_str)

    def _safe_number_list(self, values: Any) -> List[float]:
        try:
            arr = np.array(values, dtype=object)
            cleaned = []
            for v in arr:
                num = pd.to_numeric(v, errors='coerce')
                if pd.notna(num):
                    cleaned.append(float(num))
            return cleaned
        except Exception:
            return []

    def _detect_metric_label(self, field_name: Optional[str]) -> str:
        name = str(field_name or "").lower()
        if any(k in name for k in ['count', 'trip_count', 'order_count', 'freq', 'volume', 'num']):
            return "订单数量"
        if any(k in name for k in ['amount', 'fare', 'revenue', 'price', 'gmv', 'cost']):
            return "金额"
        if any(k in name for k in ['distance', 'trip_distance', 'mileage']):
            return "距离"
        if any(k in name for k in ['duration', 'travel_time', 'time_spent']):
            return "时长"
        return str(field_name or "指标")

    def _detect_dimension_label(self, field_name: Optional[str]) -> str:
        name = str(field_name or "").lower()
        if any(k in name for k in ['zone', 'location', 'region', 'district', 'borough', 'city', 'area']):
            return "区域"
        if any(k in name for k in ['time', 'date', 'hour', 'day', 'month', 'week']):
            return "时间"
        if any(k in name for k in ['category', 'type', 'class']):
            return "类别"
        return str(field_name or "维度")

    def _make_headline_facts(
        self,
        max_category: Any = None,
        max_value: Any = None,
        min_category: Any = None,
        min_value: Any = None
    ) -> Dict[str, Any]:
        return self._to_native({
            "max_category": max_category,
            "max_value": max_value,
            "min_category": min_category,
            "min_value": min_value
        })

    def _pick_dimension_field(self, chart_config: Dict[str, Any]) -> Optional[str]:
        if not isinstance(chart_config, dict):
            return None

        for key in ["display_dimension", "target_dimension", "primary_dimension", "x_axis", "series_name"]:
            value = chart_config.get(key)
            if value and str(value).lower() != "auto":
                return value

        return None

    def _pick_metric_field(self, chart_config: Dict[str, Any]) -> Optional[str]:
        if not isinstance(chart_config, dict):
            return None

        for key in ["display_metric", "target_metric", "primary_metric"]:
            value = chart_config.get(key)
            if value and str(value).lower() != "auto":
                return value

        y_axis = chart_config.get("y_axis")
        if isinstance(y_axis, list) and y_axis:
            first = y_axis[0]
            if first and str(first).lower() != "auto":
                return first

        return None

    def _safe_range(self, values: List[float]) -> Dict[str, Optional[float]]:
        if not values:
            return {"min": None, "max": None}
        return {
            "min": float(min(values)),
            "max": float(max(values))
        }

    def _safe_mean(self, values: List[float]) -> Optional[float]:
        if not values:
            return None
        try:
            return float(np.mean(values))
        except Exception:
            return None

    # =========================================================
    # 对外统一入口
    # =========================================================
    def summarize_component(
        self,
        component_id: str,
        result_obj: Any,
        component_meta: Optional[Dict[str, Any]] = None,
        global_id_map: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        统一入口：根据结果对象类型派发到不同摘要器
        """
        component_meta = component_meta or {}

        try:
            if isinstance(result_obj, (pd.DataFrame, pd.Series, gpd.GeoDataFrame)):
                return self._summarize_dataframe_like(
                    component_id=component_id,
                    result_obj=result_obj,
                    component_meta=component_meta
                )

            if isinstance(result_obj, dict) and result_obj.get("is_stv_map_protocol"):
                return self._summarize_protocol_map(
                    component_id=component_id,
                    payload=result_obj,
                    component_meta=component_meta,
                    global_id_map=global_id_map or {}
                )

            if hasattr(result_obj, "data") and isinstance(result_obj.data, (list, tuple)):
                return self._summarize_plotly_figure(
                    component_id=component_id,
                    fig=result_obj,
                    component_meta=component_meta,
                    global_id_map=global_id_map or {}
                )

            return self._to_native({
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "unknown"),
                "viz_type": "unknown",
                "semantic_role": "unknown",
                "narrative_hints": ["该组件结果类型未识别，未生成详细摘要。"]
            })

        except Exception as e:
            logger.warning(f"[Summarizer] Failed to summarize component {component_id}: {e}")
            return self._to_native({
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "unknown"),
                "viz_type": "unknown",
                "semantic_role": "unknown",
                "narrative_hints": [f"摘要提取失败: {str(e)}"]
            })

    # =========================================================
    # Plotly Figure 摘要
    # =========================================================
    def _summarize_plotly_figure(
        self,
        component_id: str,
        fig: Any,
        component_meta: Dict[str, Any],
        global_id_map: Dict[str, str]
    ) -> Dict[str, Any]:
        if not fig.data or len(fig.data) == 0:
            return self._to_native({
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "chart"),
                "viz_type": "empty_figure",
                "semantic_role": "empty_chart",
                "narrative_hints": ["图表为空，无法提取摘要。"]
            })

        trace = fig.data[0]
        trace_type = getattr(trace, 'type', 'unknown')

        if trace_type == 'bar':
            return self._summarize_bar_trace(component_id, trace, component_meta, global_id_map)
        elif trace_type == 'pie':
            return self._summarize_pie_trace(component_id, trace, component_meta, global_id_map)
        elif trace_type in ['scatter', 'scattergl']:
            mode = getattr(trace, 'mode', '')
            if mode and 'lines' in str(mode):
                return self._summarize_line_trace(component_id, trace, component_meta)
            return self._summarize_scatter_trace(component_id, trace, component_meta)
        else:
            return self._to_native({
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "chart"),
                "title": component_meta.get("title", ""),
                "viz_type": trace_type,
                "semantic_role": "generic_chart",
                "narrative_hints": [
                    f"未针对 trace_type={trace_type} 做专门摘要，使用通用摘要。",
                    "建议优先结合 headline_facts 与图表标题理解该组件。"
                ]
            })

    def _summarize_bar_trace(
        self,
        component_id: str,
        trace: Any,
        component_meta: Dict[str, Any],
        global_id_map: Dict[str, str]
    ) -> Dict[str, Any]:
        orientation = getattr(trace, 'orientation', 'v') or 'v'

        x_arr = np.array(trace.x if trace.x is not None else [], dtype=object)
        y_arr = np.array(trace.y if trace.y is not None else [], dtype=object)

        if orientation == 'h':
            categories = [self._safe_translate(v, global_id_map) for v in y_arr]
            values = self._safe_number_list(x_arr)
            semantic_role = "ranking_chart"
        else:
            categories = [self._safe_translate(v, global_id_map) for v in x_arr]
            values = self._safe_number_list(y_arr)
            semantic_role = "ranking_chart"

        pair_count = min(len(categories), len(values))
        ranked_items = [
            {"category": categories[i], "value": values[i]}
            for i in range(pair_count)
        ]

        if not ranked_items:
            return self._to_native({
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "chart"),
                "viz_type": "bar",
                "semantic_role": semantic_role,
                "narrative_hints": ["柱状图没有可用数据。"]
            })

        ranked_desc = sorted(ranked_items, key=lambda x: x["value"], reverse=True)
        ranked_asc = sorted(ranked_items, key=lambda x: x["value"], reverse=False)

        max_item = ranked_desc[0]
        min_item = ranked_asc[0]
        top_second = ranked_desc[1] if len(ranked_desc) > 1 else None

        title = component_meta.get("title", "")
        chart_config = component_meta.get("chart_config", {}) or {}
        metric_field_name = self._pick_metric_field(chart_config) or "unknown"
        dimension_field_name = self._pick_dimension_field(chart_config) or "unknown"

        gap_hint = None
        if top_second is not None:
            try:
                gap = max_item["value"] - top_second["value"]
                if abs(gap) > 0:
                    gap_hint = (
                        f"头部第一项与第二项差值约为 {gap:.2f}。"
                    )
            except Exception:
                pass

        summary = {
            "component_id": component_id,
            "component_type": component_meta.get("component_type", "chart"),
            "title": title,
            "viz_type": "bar",
            "semantic_role": semantic_role,
            "data_profile": {
                "row_count": pair_count,
                "dimension_field": dimension_field_name,
                "metric_field": metric_field_name,
                "dimension_label": self._detect_dimension_label(dimension_field_name),
                "metric_label": self._detect_metric_label(metric_field_name)
            },
            "headline_facts": self._make_headline_facts(
                max_category=max_item["category"],
                max_value=max_item["value"],
                min_category=min_item["category"],
                min_value=min_item["value"]
            ),
            "ranking_insights": {
                "orientation": "horizontal" if orientation == 'h' else "vertical",
                "top_items_desc": ranked_desc[:5],
                "bottom_items_asc": ranked_asc[:5],
                "visible_order_note": (
                    "该图为横向条形图，视觉上通常从上到下显示由大到小，顶部条目是最大值。"
                    if orientation == 'h' else
                    "该图为纵向条形图，请以数值排序结果为准，而不是仅凭视觉位置判断。"
                )
            },
            "narrative_hints": [
                "这是一个排名型柱状图。",
                "该图适合用于识别头部类别、尾部类别和排序结构。",
                f"最大类别是 {max_item['category']}，数值为 {max_item['value']:.0f}。",
                f"最小类别是 {min_item['category']}，数值为 {min_item['value']:.0f}。",
                *( [gap_hint] if gap_hint else [] ),
                "洞察生成时应优先使用 ranking_insights.top_items_desc[0] 作为最高项依据。"
            ]
        }
        return self._to_native(summary)

    def _summarize_line_trace(
        self,
        component_id: str,
        trace: Any,
        component_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        x_arr = np.array(trace.x if trace.x is not None else [], dtype=object)
        y_arr = np.array(trace.y if trace.y is not None else [], dtype=object)

        values = self._safe_number_list(y_arr)
        pair_count = min(len(x_arr), len(values))

        points = []
        for i in range(pair_count):
            points.append({
                "time": str(x_arr[i]),
                "value": values[i]
            })

        if not points:
            return self._to_native({
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "chart"),
                "viz_type": "line",
                "semantic_role": "trend_chart",
                "narrative_hints": ["折线图没有可用数据。"]
            })

        peak_point = max(points, key=lambda x: x["value"])
        valley_point = min(points, key=lambda x: x["value"])

        start_point = points[0]
        end_point = points[-1]

        trend_direction = "fluctuating"
        if end_point["value"] > start_point["value"]:
            trend_direction = "upward"
        elif end_point["value"] < start_point["value"]:
            trend_direction = "downward"

        title = component_meta.get("title", "")
        chart_config = component_meta.get("chart_config", {}) or {}
        metric_field_name = self._pick_metric_field(chart_config) or "unknown"
        dimension_field_name = self._pick_dimension_field(chart_config) or "unknown"

        delta_hint = None
        try:
            delta = end_point["value"] - start_point["value"]
            if abs(delta) > 0:
                delta_hint = f"起止点变化约为 {delta:.2f}。"
        except Exception:
            pass

        summary = {
            "component_id": component_id,
            "component_type": component_meta.get("component_type", "chart"),
            "title": title,
            "viz_type": "line",
            "semantic_role": "trend_chart",
            "data_profile": {
                "row_count": len(points),
                "dimension_field": dimension_field_name,
                "metric_field": metric_field_name,
                "dimension_label": self._detect_dimension_label(dimension_field_name),
                "metric_label": self._detect_metric_label(metric_field_name)
            },
            "headline_facts": {
                "peak_time": peak_point["time"],
                "peak_value": peak_point["value"],
                "valley_time": valley_point["time"],
                "valley_value": valley_point["value"],
                "start_time": start_point["time"],
                "start_value": start_point["value"],
                "end_time": end_point["time"],
                "end_value": end_point["value"]
            },
            "temporal_insights": {
                "trend_direction": trend_direction,
                "peak_point": peak_point,
                "valley_point": valley_point,
                "start_point": start_point,
                "end_point": end_point
            },
            "narrative_hints": [
                "这是一个趋势型折线图。",
                "该图适合用于识别峰值时点、谷值时点和整体趋势方向。",
                f"峰值出现在 {peak_point['time']}，数值为 {peak_point['value']:.2f}。",
                f"谷值出现在 {valley_point['time']}，数值为 {valley_point['value']:.2f}。",
                f"整体趋势判断为 {trend_direction}。",
                *( [delta_hint] if delta_hint else [] )
            ]
        }
        return self._to_native(summary)

    def _summarize_pie_trace(
        self,
        component_id: str,
        trace: Any,
        component_meta: Dict[str, Any],
        global_id_map: Dict[str, str]
    ) -> Dict[str, Any]:
        labels = np.array(trace.labels if trace.labels is not None else [], dtype=object)
        values = self._safe_number_list(trace.values if trace.values is not None else [])

        pair_count = min(len(labels), len(values))
        items = [
            {"category": self._safe_translate(labels[i], global_id_map), "value": values[i]}
            for i in range(pair_count)
        ]

        if not items:
            return self._to_native({
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "chart"),
                "viz_type": "pie",
                "semantic_role": "composition_chart",
                "narrative_hints": ["饼图没有可用数据。"]
            })

        total = sum(i["value"] for i in items) or 1
        ranked_desc = sorted(items, key=lambda x: x["value"], reverse=True)

        top_item = ranked_desc[0]
        top_item_share = top_item["value"] / total

        chart_config = component_meta.get("chart_config", {}) or {}
        metric_field_name = self._pick_metric_field(chart_config) or "unknown"
        dimension_field_name = self._pick_dimension_field(chart_config) or "unknown"

        concentration_level = (
            "high" if top_item_share >= 0.5 else
            "medium" if top_item_share >= 0.25 else
            "low"
        )

        concentration_hint = None
        if concentration_level == "high":
            concentration_hint = "头部类别集中度较高，说明构成明显偏向少数类别。"
        elif concentration_level == "medium":
            concentration_hint = "头部类别存在一定集中，但仍保留一定分散性。"
        else:
            concentration_hint = "整体构成相对分散，没有单一类别形成绝对主导。"

        summary = {
            "component_id": component_id,
            "component_type": component_meta.get("component_type", "chart"),
            "title": component_meta.get("title", ""),
            "viz_type": "pie",
            "semantic_role": "composition_chart",
            "data_profile": {
                "row_count": len(items),
                "dimension_field": dimension_field_name,
                "metric_field": metric_field_name,
                "dimension_label": self._detect_dimension_label(dimension_field_name),
                "metric_label": self._detect_metric_label(metric_field_name)
            },
            "headline_facts": {
                "max_category": top_item["category"],
                "max_value": top_item["value"],
                "max_share": top_item_share
            },
            "composition_insights": {
                "top_items_desc": ranked_desc[:5],
                "total_value": total,
                "concentration_level": concentration_level
            },
            "narrative_hints": [
                "这是一个构成型饼图。",
                "该图适合用于识别头部构成项及整体集中程度。",
                f"占比最高的类别是 {top_item['category']}，对应数值 {top_item['value']:.2f}。",
                f"其占整体比例约为 {top_item_share:.1%}。",
                concentration_hint
            ]
        }
        return self._to_native(summary)

    def _summarize_scatter_trace(
        self,
        component_id: str,
        trace: Any,
        component_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        x_arr = np.array(trace.x if trace.x is not None else [], dtype=object)
        y_values = self._safe_number_list(trace.y if trace.y is not None else [])

        x_numeric = self._safe_number_list(x_arr)
        row_count = min(len(x_arr), len(y_values)) if y_values else len(x_arr)

        chart_config = component_meta.get("chart_config", {}) or {}
        metric_field_name = self._pick_metric_field(chart_config) or "unknown"
        dimension_field_name = self._pick_dimension_field(chart_config) or "unknown"

        summary = {
            "component_id": component_id,
            "component_type": component_meta.get("component_type", "chart"),
            "title": component_meta.get("title", ""),
            "viz_type": "scatter",
            "semantic_role": "distribution_chart",
            "data_profile": {
                "row_count": row_count,
                "dimension_field": dimension_field_name,
                "metric_field": metric_field_name,
                "dimension_label": self._detect_dimension_label(dimension_field_name),
                "metric_label": self._detect_metric_label(metric_field_name),
                "x_range": self._safe_range(x_numeric),
                "y_range": self._safe_range(y_values)
            },
            "narrative_hints": [
                "这是一个散点分布图。",
                "该图适合用于观察分布范围、离散程度和异常值位置。",
                f"当前有效点数约为 {row_count}。"
            ]
        }

        if y_values:
            summary["headline_facts"] = {
                "max_y": max(y_values),
                "min_y": min(y_values),
                "mean_y": self._safe_mean(y_values)
            }

        return self._to_native(summary)

    # =========================================================
    # STV Protocol Map 摘要
    # =========================================================
    def _summarize_protocol_map(
        self,
        component_id: str,
        payload: Dict[str, Any],
        component_meta: Dict[str, Any],
        global_id_map: Dict[str, str]
    ) -> Dict[str, Any]:
        layer_type = payload.get("layer_type", "map")
        data_list = payload.get("data", []) or []

        if layer_type in ['choropleth', 'animated_choropleth']:
            return self._summarize_protocol_choropleth(
                component_id, payload, component_meta, global_id_map
            )

        if layer_type in ['scatter', 'animated_scatter']:
            return self._summarize_protocol_scatter(
                component_id, payload, component_meta
            )

        if layer_type == 'heatmap':
            return self._summarize_protocol_heatmap(
                component_id, payload, component_meta
            )

        return self._to_native({
            "component_id": component_id,
            "component_type": component_meta.get("component_type", "map"),
            "title": component_meta.get("title", ""),
            "viz_type": layer_type,
            "semantic_role": "map",
            "data_profile": {
                "row_count": len(data_list)
            },
            "narrative_hints": [
                f"未针对地图类型 {layer_type} 做专门摘要。",
                "建议优先结合地图标题和核心指标字段理解该组件。"
            ]
        })

    def _summarize_protocol_choropleth(
        self,
        component_id: str,
        payload: Dict[str, Any],
        component_meta: Dict[str, Any],
        global_id_map: Dict[str, str]
    ) -> Dict[str, Any]:
        data_list = payload.get("data", []) or []
        mapping = payload.get("mapping", {}) or {}

        value_key = mapping.get("display_value") or mapping.get("color_value")
        id_key = mapping.get("id")

        ranked_items = []
        for row in data_list:
            try:
                value = float(row.get(value_key, 0))
                raw_id = str(row.get(id_key, "unknown"))
                translated = global_id_map.get(
                    raw_id,
                    row.get("zone") or row.get("name") or row.get("borough") or raw_id
                )
                ranked_items.append({
                    "entity_id": raw_id,
                    "entity_name": str(translated),
                    "value": value
                })
            except Exception:
                continue

        if not ranked_items:
            return self._to_native({
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "map"),
                "title": component_meta.get("title", ""),
                "viz_type": payload.get("layer_type", "choropleth"),
                "semantic_role": "spatial_hotspot_map",
                "narrative_hints": ["地图协议中无可用区域数值数据。"]
            })

        ranked_desc = sorted(ranked_items, key=lambda x: x["value"], reverse=True)
        ranked_asc = sorted(ranked_items, key=lambda x: x["value"], reverse=False)

        hottest = ranked_desc[0]
        coldest = ranked_asc[0]

        title = component_meta.get("title", "")
        summary = {
            "component_id": component_id,
            "component_type": component_meta.get("component_type", "map"),
            "title": title,
            "viz_type": payload.get("layer_type", "choropleth"),
            "semantic_role": (
                "animated_spatiotemporal_map"
                if payload.get("layer_type") == "animated_choropleth"
                else "spatial_hotspot_map"
            ),
            "data_profile": {
                "row_count": len(ranked_items),
                "entity_field": id_key,
                "metric_field": value_key,
                "metric_label": self._detect_metric_label(value_key)
            },
            "headline_facts": {
                "max_region": hottest["entity_name"],
                "max_value": hottest["value"],
                "min_region": coldest["entity_name"],
                "min_value": coldest["value"]
            },
            "spatial_insights": {
                "top_regions_desc": ranked_desc[:5],
                "bottom_regions_asc": ranked_asc[:5]
            },
            "narrative_hints": [
                "这是一个区域热点分布地图。",
                "该图适合用于识别热点区域、冷点区域和区域排序结构。",
                f"最高值区域是 {hottest['entity_name']}，数值为 {hottest['value']:.2f}。",
                f"最低值区域是 {coldest['entity_name']}，数值为 {coldest['value']:.2f}。",
                "洞察生成时应优先使用 spatial_insights.top_regions_desc[0] 作为热点区域依据。"
            ]
        }

        if mapping.get("timestamp"):
            summary["temporal_insights"] = {
                "is_time_animated": True,
                "timestamp_field": mapping.get("timestamp")
            }
            summary["narrative_hints"].append("该地图包含时间动画语义，可结合时间变化理解热点迁移。")

        return self._to_native(summary)

    def _summarize_protocol_scatter(
        self,
        component_id: str,
        payload: Dict[str, Any],
        component_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        data_list = payload.get("data", []) or []
        mapping = payload.get("mapping", {}) or {}

        lat_key = mapping.get("latitude")
        lon_key = mapping.get("longitude")
        value_key = mapping.get("color_value")
        size_key = mapping.get("size_value")

        valid_points = []
        for row in data_list:
            try:
                lat = float(row.get(lat_key))
                lon = float(row.get(lon_key))
                if np.isfinite(lat) and np.isfinite(lon):
                    valid_points.append(row)
            except Exception:
                continue

        summary = {
            "component_id": component_id,
            "component_type": component_meta.get("component_type", "map"),
            "title": component_meta.get("title", ""),
            "viz_type": payload.get("layer_type", "scatter"),
            "semantic_role": (
                "animated_point_distribution_map"
                if payload.get("layer_type") == "animated_scatter"
                else "point_distribution_map"
            ),
            "data_profile": {
                "row_count": len(valid_points),
                "latitude_field": lat_key,
                "longitude_field": lon_key,
                "metric_field": value_key,
                "size_field": size_key
            },
            "narrative_hints": [
                "这是一个点分布地图。",
                "该图适合用于识别点位密度、分布范围和热点聚集位置。",
                f"有效点数量为 {len(valid_points)}。"
            ]
        }

        if valid_points:
            if value_key:
                vals = []
                for row in valid_points:
                    try:
                        vals.append(float(row.get(value_key, 0)))
                    except Exception:
                        pass
                if vals:
                    summary["headline_facts"] = {
                        "max_value": max(vals),
                        "min_value": min(vals),
                        "mean_value": self._safe_mean(vals)
                    }

            if lat_key and lon_key:
                try:
                    lats = [float(r[lat_key]) for r in valid_points]
                    lons = [float(r[lon_key]) for r in valid_points]
                    summary["spatial_insights"] = {
                        "bounds": {
                            "min_lat": min(lats),
                            "max_lat": max(lats),
                            "min_lon": min(lons),
                            "max_lon": max(lons)
                        }
                    }
                except Exception:
                    pass

        return self._to_native(summary)

    def _summarize_protocol_heatmap(
        self,
        component_id: str,
        payload: Dict[str, Any],
        component_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        data_list = payload.get("data", []) or []
        mapping = payload.get("mapping", {}) or {}

        lat_key = mapping.get("latitude")
        lon_key = mapping.get("longitude")
        value_key = mapping.get("color_value")

        weights = []
        valid_points = 0

        for row in data_list:
            try:
                lat = float(row.get(lat_key))
                lon = float(row.get(lon_key))
                if np.isfinite(lat) and np.isfinite(lon):
                    valid_points += 1
                    if value_key and row.get(value_key) is not None:
                        weights.append(float(row.get(value_key)))
            except Exception:
                continue

        summary = {
            "component_id": component_id,
            "component_type": component_meta.get("component_type", "map"),
            "title": component_meta.get("title", ""),
            "viz_type": "heatmap",
            "semantic_role": "density_heatmap",
            "data_profile": {
                "row_count": valid_points,
                "latitude_field": lat_key,
                "longitude_field": lon_key,
                "metric_field": value_key
            },
            "narrative_hints": [
                "这是一个热力密度图。",
                "该图适合用于识别空间密度高低与热点聚集区域。",
                f"有效空间点数量为 {valid_points}。"
            ]
        }

        if weights:
            summary["headline_facts"] = {
                "max_weight": max(weights),
                "min_weight": min(weights),
                "mean_weight": float(np.mean(weights))
            }

        return self._to_native(summary)

    # =========================================================
    # DataFrame / Series / GeoDataFrame 摘要
    # =========================================================
    def _summarize_dataframe_like(
        self,
        component_id: str,
        result_obj: Any,
        component_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        if isinstance(result_obj, pd.Series):
            row_count = len(result_obj)
            numeric_values = self._safe_number_list(result_obj.tolist())

            summary = {
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "table"),
                "title": component_meta.get("title", ""),
                "viz_type": "series",
                "semantic_role": "data_series",
                "data_profile": {
                    "row_count": row_count,
                    "dtype": str(result_obj.dtype)
                },
                "narrative_hints": [
                    f"结果为 Series，共 {row_count} 条数据。",
                    "该结果适合用于数值趋势或单列统计校验。"
                ]
            }

            if numeric_values:
                summary["headline_facts"] = {
                    "max_value": max(numeric_values),
                    "min_value": min(numeric_values),
                    "mean_value": self._safe_mean(numeric_values)
                }

            return self._to_native(summary)

        if isinstance(result_obj, gpd.GeoDataFrame):
            row_count = len(result_obj)
            summary = {
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "table"),
                "title": component_meta.get("title", ""),
                "viz_type": "geodataframe",
                "semantic_role": "spatial_table",
                "data_profile": {
                    "row_count": row_count,
                    "columns": list(result_obj.columns),
                    "crs": str(result_obj.crs) if result_obj.crs is not None else None
                },
                "narrative_hints": [
                    f"结果为 GeoDataFrame，共 {row_count} 条空间记录。",
                    "该结果适合用于空间明细校验与区域属性检查。"
                ]
            }
            return self._to_native(summary)

        if isinstance(result_obj, pd.DataFrame):
            row_count = len(result_obj)
            columns = list(result_obj.columns)
            numeric_cols = list(result_obj.select_dtypes(include=[np.number]).columns)
            object_cols = list(result_obj.select_dtypes(include=['object', 'category']).columns)
            time_cols = [c for c in columns if pd.api.types.is_datetime64_any_dtype(result_obj[c])]

            summary = {
                "component_id": component_id,
                "component_type": component_meta.get("component_type", "table"),
                "title": component_meta.get("title", ""),
                "viz_type": "dataframe",
                "semantic_role": "table_result",
                "data_profile": {
                    "row_count": row_count,
                    "columns": columns[:50],
                    "numeric_columns": numeric_cols[:10],
                    "categorical_columns": object_cols[:10],
                    "time_columns": time_cols[:10]
                },
                "narrative_hints": [
                    f"结果为 DataFrame，共 {row_count} 条记录。",
                    "该结果适合用于明细验证、数值对比和结构化分析补充。"
                ]
            }

            if numeric_cols:
                first_num = numeric_cols[0]
                series = pd.to_numeric(result_obj[first_num], errors='coerce').dropna()
                if not series.empty:
                    summary["headline_facts"] = {
                        "primary_numeric_field": first_num,
                        "max_value": float(series.max()),
                        "min_value": float(series.min()),
                        "mean_value": float(series.mean())
                    }

            if time_cols and numeric_cols:
                tc = time_cols[0]
                vc = numeric_cols[0]
                series = pd.to_numeric(result_obj[vc], errors='coerce').dropna()
                if not series.empty:
                    summary["temporal_insights"] = {
                        "time_field": tc,
                        "metric_field": vc,
                        "peak_value": float(series.max()),
                        "valley_value": float(series.min()),
                        "start_time": str(result_obj[tc].min()),
                        "end_time": str(result_obj[tc].max())
                    }
                    summary["narrative_hints"].append("结果同时包含时间列与数值列，可用于时间变化分析。")

            return self._to_native(summary)

        return self._to_native({
            "component_id": component_id,
            "component_type": component_meta.get("component_type", "unknown"),
            "viz_type": "unknown",
            "semantic_role": "unknown",
            "narrative_hints": ["结果对象不是 DataFrame-like。"]
        })