# backend/core/execution/executor.py

import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import random
import json
from shapely.geometry import Point, Polygon, LineString
import traceback
import sys
import io
import textwrap
from typing import Dict, Any, List, Optional
import logging

from pydantic import BaseModel

# 引入 SDK
from core.sdk.visualizer import STVisualizer
from core.sdk import operators as ops

# 执行结果摘要器
from core.execution.result_summarizer import ResultSummarizer

logger = logging.getLogger(__name__)


class ComponentResult(BaseModel):
    """单个组件的执行结果"""
    component_id: str
    data: Any
    summary_stats: Optional[Dict[str, Any]] = None


class DashboardExecutionResult(BaseModel):
    """整个看板的执行结果包"""
    success: bool
    results: Dict[str, ComponentResult] = {}
    global_insight_data: Dict[str, Any] = {}
    error: Optional[str] = None
    code: str = ""


class CodeExecutor:
    """
    代码执行器（Metadata-Aware 版）

    当前职责：
    1. 在安全上下文中执行 LLM 生成代码
    2. 构建全局 ID 语义字典
    3. 接收完整 component metadata，并传给 ResultSummarizer
    4. 返回结构化执行结果
    5. 为 InsightExtractor 注入 interaction / compare 上下文摘要

    不再承担：
    - 图表/地图的手工摘要逻辑
    """

    def __init__(self):
        self.global_context = {
            "pd": pd,
            "gpd": gpd,
            "px": px,
            "go": go,
            "np": np,
            "random": random,
            "json": json,
            "Point": Point,
            "Polygon": Polygon,
            "LineString": LineString,
            "print": print,
            "stv": STVisualizer(),
            "ops": ops
        }
        self.summarizer = ResultSummarizer()

    def _dedent_code(self, code: str) -> str:
        return textwrap.dedent(code).strip()

    def _make_serializable(self, obj: Any) -> Any:
        """
        递归序列化执行结果，兼容：
        - Plotly Figure
        - numpy 标量
        - pandas 时间对象
        - ndarray
        - dict/list
        """
        if hasattr(obj, "to_dict") and hasattr(obj, "layout") and hasattr(obj, "data"):
            return obj.to_dict()

        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)

        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)

        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)

        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)

        elif isinstance(obj, np.datetime64):
            return str(pd.to_datetime(obj))

        elif hasattr(obj, "__geo_interface__"):
            return "GEOMETRY_OBJECT"

        elif isinstance(obj, np.ndarray):
            return self._make_serializable(obj.tolist())

        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}

        elif isinstance(obj, (list, tuple, set)):
            return [self._make_serializable(v) for v in obj]

        else:
            return obj

    def _build_global_id_map(self, safe_data_context: Dict[str, Any]) -> Dict[str, str]:
        """
        从数据上下文中自动嗅探字典表，构建全局 ID -> Name 映射。
        用于摘要解释层和洞察层的可读性增强。
        """
        global_id_map = {}

        for k, df in safe_data_context.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                cols = [str(c).lower() for c in df.columns]
                id_col_idx = next((i for i, c in enumerate(cols) if 'id' in c), None)
                name_col_idx = next(
                    (i for i, c in enumerate(cols) if any(kw in c for kw in ['zone', 'name', 'borough', 'city'])),
                    None
                )

                if id_col_idx is not None and name_col_idx is not None and len(df.columns) <= 6:
                    id_col = df.columns[id_col_idx]
                    name_col = df.columns[name_col_idx]
                    try:
                        temp_dict = dict(zip(
                            df[id_col].astype(str).str.replace(r'\.0$', '', regex=True),
                            df[name_col].astype(str)
                        ))
                        global_id_map.update(temp_dict)
                        logger.info(f">>> [Executor] 成功构建全局语义字典，共提取 {len(temp_dict)} 个映射词条。")
                    except Exception:
                        pass

        return global_id_map

    def _extract_component_meta(self, comp: Any) -> Dict[str, Any]:
        """
        从 component definition（dict 或 Pydantic 对象）中提取摘要器所需的 metadata。
        """
        is_dict = isinstance(comp, dict)

        component_id = comp.get('id') if is_dict else getattr(comp, 'id', 'unknown')
        component_type = comp.get('type') if is_dict else getattr(comp, 'type', 'unknown')
        title = comp.get('title') if is_dict else getattr(comp, 'title', component_id)

        chart_config = comp.get('chart_config', {}) if is_dict else getattr(comp, 'chart_config', {})
        map_config = comp.get('map_config', []) if is_dict else getattr(comp, 'map_config', [])
        layout = comp.get('layout', {}) if is_dict else getattr(comp, 'layout', {})

        component_type_str = str(component_type).split('.')[-1].lower()

        if hasattr(chart_config, "model_dump"):
            chart_config = chart_config.model_dump()
        elif chart_config is None:
            chart_config = {}

        if hasattr(layout, "model_dump"):
            layout = layout.model_dump()

        if isinstance(map_config, list):
            normalized_map_config = []
            for item in map_config:
                if hasattr(item, "model_dump"):
                    normalized_map_config.append(item.model_dump())
                else:
                    normalized_map_config.append(item)
            map_config = normalized_map_config
        elif hasattr(map_config, "model_dump"):
            map_config = map_config.model_dump()

        return {
            "component_id": component_id,
            "component_type": component_type_str,
            "title": title,
            "chart_config": chart_config or {},
            "map_config": map_config or [],
            "layout": layout or {}
        }

    def _build_component_meta_map(
        self,
        component_ids: List[str],
        component_defs: Optional[List[Any]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        优先使用 workflow 传入的完整 component definitions 构建 metadata map。
        如果缺失，则回退到基于 component_id 的弱推断。
        """
        meta_map: Dict[str, Dict[str, Any]] = {}

        if component_defs:
            for comp in component_defs:
                try:
                    meta = self._extract_component_meta(comp)
                    cid = meta["component_id"]
                    meta_map[cid] = meta
                except Exception as e:
                    logger.warning(f"[Executor] Failed to parse component meta: {e}")

        for cid in component_ids:
            if cid not in meta_map:
                inferred_type = "map" if "map" in cid else ("insight" if "insight" in cid else "chart")
                meta_map[cid] = {
                    "component_id": cid,
                    "component_type": inferred_type,
                    "title": cid,
                    "chart_config": {},
                    "map_config": [],
                    "layout": {}
                }

        return meta_map

    def _build_interaction_context(
        self,
        interaction_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        构建顶层 interaction/compare 上下文，供 InsightExtractor 使用。
        当前策略：
        - 不改变组件级摘要结构
        - 在 global_insight_data 顶层增加 _context
        """
        if not interaction_state or not isinstance(interaction_state, dict):
            return {
                "selection_enabled": False,
                "active_selection_count": 0,
                "active_selection_labels": [],
                "compare_enabled": False,
                "baseline_mode": "global",
                "left_selection_labels": [],
                "right_selection_labels": [],
                "active_component_id": None
            }

        selections = interaction_state.get("selections", []) or []
        active_selection_ids = interaction_state.get("active_selection_ids", []) or []
        active_id_set = set(active_selection_ids)

        comparison_state = interaction_state.get("comparison_state", {}) or {}
        left_ids = comparison_state.get("left_selection_ids", []) or []
        right_ids = comparison_state.get("right_selection_ids", []) or []

        selection_map = {}
        for item in selections:
            if not isinstance(item, dict):
                continue
            sid = item.get("selection_id")
            if sid:
                selection_map[sid] = item

        def _label_for_selection_id(sid: str) -> str:
            item = selection_map.get(sid, {}) or {}
            return (
                item.get("label")
                or item.get("selection_context", {}).get("display_value")
                or item.get("selection_id")
                or str(sid)
            )

        active_labels = [
            _label_for_selection_id(sid)
            for sid in active_selection_ids
            if sid in selection_map
        ]

        left_labels = [
            _label_for_selection_id(sid)
            for sid in left_ids
            if sid in selection_map
        ]

        right_labels = [
            _label_for_selection_id(sid)
            for sid in right_ids
            if sid in selection_map
        ]

        active_items = [
            selection_map[sid]
            for sid in active_selection_ids
            if sid in selection_map
        ]

        active_selection_types = []
        for item in active_items:
            sel_type = item.get("selection_type")
            if sel_type and sel_type not in active_selection_types:
                active_selection_types.append(sel_type)

        return {
            "selection_enabled": len(active_id_set) > 0,
            "active_selection_count": len(active_selection_ids),
            "active_selection_labels": active_labels,
            "active_selection_types": active_selection_types,
            "compare_enabled": bool(comparison_state.get("enabled")),
            "baseline_mode": comparison_state.get("baseline_mode", "global"),
            "left_selection_labels": left_labels,
            "right_selection_labels": right_labels,
            "left_selection_count": len(left_ids),
            "right_selection_count": len(right_ids),
            "active_component_id": interaction_state.get("active_component_id"),
            "compare_question": comparison_state.get("compare_question"),
            "compare_metric": comparison_state.get("compare_metric"),
            "compare_dimension": comparison_state.get("compare_dimension")
        }

    def execute_dashboard_logic(
        self,
        code_str: str,
        data_context: Dict[str, Any],
        component_ids: List[str],
        summaries: List[Dict[str, Any]] = None,
        component_defs: Optional[List[Any]] = None,
        interaction_state: Optional[Dict[str, Any]] = None
    ) -> DashboardExecutionResult:

        clean_code = self._dedent_code(code_str)
        local_scope = {}

        print("运行的代码:\n", clean_code)

        old_stdout = sys.stdout
        redirected_output = io.StringIO()
        sys.stdout = redirected_output

        try:
            logger.info(">>> [Executor] 启动沙箱执行环境...")

            # 1. 构建安全数据上下文
            safe_data_context = {}
            for k, v in data_context.items():
                if hasattr(v, 'copy'):
                    safe_data_context[k] = v.copy()
                else:
                    safe_data_context[k] = v

            if summaries is not None:
                safe_data_context['_metadata'] = summaries

            # 2. 构建全局 ID 字典
            global_id_map = self._build_global_id_map(safe_data_context)

            # 3. 将全局 ID 字典注入 stv
            try:
                self.global_context["stv"]._global_id_map = global_id_map
            except Exception:
                pass

            # 4. 执行生成代码
            exec(clean_code, self.global_context, local_scope)

            if "get_dashboard_data" not in local_scope:
                raise ValueError("Generated code missing 'get_dashboard_data' function.")

            all_results = local_scope["get_dashboard_data"](safe_data_context)

            final_results: Dict[str, ComponentResult] = {}
            insight_payload: Dict[str, Any] = {}

            # 5. 顶层 interaction / compare context
            insight_payload["_context"] = self._build_interaction_context(interaction_state)

            # 6. 使用完整 component metadata 构建 meta_map
            component_meta_map = self._build_component_meta_map(
                component_ids=component_ids,
                component_defs=component_defs
            )

            # 7. 对每个组件结果做统一摘要
            for cid in component_ids:
                if cid not in all_results:
                    continue

                res_obj = all_results[cid]
                component_meta = component_meta_map.get(cid, {
                    "component_id": cid,
                    "component_type": "unknown",
                    "title": cid,
                    "chart_config": {},
                    "map_config": [],
                    "layout": {}
                })

                try:
                    summary = self.summarizer.summarize_component(
                        component_id=cid,
                        result_obj=res_obj,
                        component_meta=component_meta,
                        global_id_map=global_id_map
                    )
                except Exception as e:
                    logger.warning(f"[Executor] Result summarization failed for {cid}: {e}")
                    summary = {
                        "component_id": cid,
                        "component_type": component_meta.get("component_type", "unknown"),
                        "title": component_meta.get("title", cid),
                        "viz_type": "unknown",
                        "semantic_role": "unknown",
                        "narrative_hints": [f"摘要提取失败: {str(e)}"]
                    }

                if summary:
                    insight_payload[cid] = summary

                final_results[cid] = ComponentResult(
                    component_id=cid,
                    data=res_obj,
                    summary_stats=summary
                )

            sys.stdout = old_stdout

            clean_results = self._make_serializable(final_results)
            clean_insight = self._make_serializable(insight_payload)

            return DashboardExecutionResult(
                success=True,
                results=clean_results,
                global_insight_data=clean_insight,
                code=clean_code
            )

        except Exception as e:
            sys.stdout = old_stdout
            error_trace = traceback.format_exc()

            if "[SDK Error]" in str(e):
                logger.warning(f"⚠️ [SDK AI Interaction Error]: {str(e)}")
            else:
                logger.error(f"Sandbox Execution Failed:\n{error_trace}")

            return DashboardExecutionResult(
                success=False,
                error=error_trace,
                code=clean_code
            )

        finally:
            sys.stdout = old_stdout
            captured = redirected_output.getvalue()
            if captured.strip():
                logger.info(f"Sandbox Log Output:\n{captured.strip()}")