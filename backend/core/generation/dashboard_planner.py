# backend/core/generation/dashboard_planner.py

import logging
import uuid
import json
import re
from typing import Dict, Any, List

from core.llm.AI_client import AIClient
from core.schemas.dashboard import (
    DashboardSchema, DashboardComponent, ComponentType,
    LayoutZone, LayoutConfig
)
from core.generation.templates import LayoutTemplates

logger = logging.getLogger(__name__)


class DashboardPlanner:
    """
    看板编排器 (V1.2.0 Primary-Intent Locked Edition)

    升级点：
    1. 引入 primary_metric / primary_dimension 显式输出，锁定用户主分析目标。
    2. 强化“图表必须围绕主指标组织”的约束，避免 planner 擅自引入无关指标（如金额）。
    3. 主地图标题与右侧图表标题统一围绕 primary_metric 命名。
    4. 提升 fallback 规划的一致性与可解释性。
    """

    def __init__(self, llm_client: AIClient):
        self.llm = llm_client

    # ==========================================
    # 工具函数
    # ==========================================
    def _normalize_chart_type(self, chart_type: str) -> str:
        valid_types = {'bar', 'line', 'pie', 'periodic_bar'}
        ct = str(chart_type or 'bar').lower()
        return ct if ct in valid_types else 'bar'

    def _safe_str(self, value: Any, default: str = "auto") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text if text else default

    def _infer_metric_label_from_name(self, metric_name: str) -> str:
        metric = str(metric_name or "").lower()

        if any(k in metric for k in ['count', 'trip_count', 'order_count', 'freq', 'volume', 'num']):
            return "订单数量"
        if any(k in metric for k in ['amount', 'fare', 'revenue', 'price', 'gmv', 'cost']):
            return "金额"
        if any(k in metric for k in ['distance', 'mileage', 'trip_distance']):
            return "距离"
        if any(k in metric for k in ['duration', 'time_spent', 'travel_time']):
            return "时长"
        return "核心指标"

    def _infer_dimension_label_from_name(self, dim_name: str) -> str:
        dim = str(dim_name or "").lower()

        if any(k in dim for k in ['zone', 'location', 'region', 'district', 'borough', 'city', 'area']):
            return "区域"
        if any(k in dim for k in ['time', 'date', 'hour', 'day', 'month', 'week']):
            return "时间"
        if any(k in dim for k in ['category', 'type', 'class']):
            return "类别"
        return "维度"

    def _build_map_title(self, is_animated: bool, primary_metric: str, primary_dimension: str) -> str:
        metric_label = self._infer_metric_label_from_name(primary_metric)
        dimension_label = self._infer_dimension_label_from_name(primary_dimension)

        if is_animated:
            return f"{metric_label}{dimension_label}时空演变"
        return f"{metric_label}{dimension_label}空间分布"

    def _build_default_chart_title(self, chart_type: str, primary_metric: str, primary_dimension: str) -> str:
        metric_label = self._infer_metric_label_from_name(primary_metric)
        dimension_label = self._infer_dimension_label_from_name(primary_dimension)

        ct = self._normalize_chart_type(chart_type)
        if ct == 'line':
            return f"{metric_label}随{dimension_label}变化趋势"
        if ct == 'pie':
            return f"{metric_label}按{dimension_label}构成分布"
        if ct == 'periodic_bar':
            return f"{metric_label}按周期时间分布"
        return f"{metric_label}按{dimension_label}排名"

    def _extract_primary_fields_from_ai_plan(self, ai_plan: Dict[str, Any]) -> Dict[str, str]:
        primary_metric = self._safe_str(ai_plan.get("primary_metric"), "auto")
        primary_dimension = self._safe_str(ai_plan.get("primary_dimension"), "auto")

        return {
            "primary_metric": primary_metric,
            "primary_dimension": primary_dimension
        }

    # ==========================================
    # 主流程
    # ==========================================
    async def plan_dashboard(self, query: str, summaries: List[Dict[str, Any]]) -> DashboardSchema:
        logger.info(f">>> [Planner] 启动 AI 深度意图解析模式...")

        main_summary = next((s for s in summaries if s.get('is_geospatial')), summaries[0])
        main_var = main_summary['variable_name']

        context_str = ""
        found_time_col = None
        time_source_summary = None

        for s in summaries:
            var_name = s['variable_name']
            cols = list(s.get('column_stats', {}).keys())
            time_ctx = s.get('semantic_analysis', {}).get('temporal_context', {})
            p_time = time_ctx.get('primary_time_col')

            sem_tags = s.get('semantic_analysis', {}).get('column_metadata', {})
            tag_hints = {}
            for k, v in sem_tags.items():
                if isinstance(v, dict) and v.get('semantic_tag'):
                    concept = v.get('concept_name', k)
                    tag = v.get('semantic_tag')
                    tag_hints[k] = f"{concept} ({tag})"

            time_hint = (
                f"(主时间列: {p_time}, 建议粒度: {time_ctx.get('suggested_resampling')})"
                if p_time else "无时间维度"
            )
            context_str += (
                f"- 数据集 `{var_name}`: {time_hint}\n"
                f"  语义字段与含义: {json.dumps(tag_hints, ensure_ascii=False)}\n"
                f"  全部列: {cols[:20]}\n"
            )

            if p_time and not found_time_col:
                found_time_col = p_time
                time_source_summary = s

        system_prompt = f"""
你是一位资深时空数据分析专家。请根据用户的【分析意图】和【数据特征】，规划一套仪表板方案。

=== 数据上下文 ===
{context_str}

=== 任务 0：锁定主分析目标 (PRIMARY ANALYSIS LOCK) ===
你必须先从用户问题中识别两个核心字段：
- `primary_metric`: 用户真正关心的核心度量（例如 订单量 / 金额 / 距离 / 时长 / 次数）
- `primary_dimension`: 用户真正关心的分析维度（例如 区域 / 时间 / 类型 / 类别）

🚨 PRIMARY LOCK RULES 🚨
1. 所有组件必须优先围绕 `primary_metric` 组织。
2. 除非用户明确要求，不要引入与 `primary_metric` 无关的新度量。
   - 例如：如果用户问“订单数量最多”，不要擅自规划“总金额”图。
3. 如果用户问“哪个区域最多/最少/排名/分布”，则 `primary_dimension` 通常应为区域类字段。
4. 如果用户问趋势/变化/波动，则至少一个图表应围绕时间维度。
5. 你的规划目标不是“让看板看起来丰富”，而是“最大程度回答用户问题”。

=== 任务 1：识别时空动态意图与空间类型 ===
- `is_animated`: 是否观察时间演变（true/false）。
- `time_granularity`: 聚合步长（如 '1H', '1D'）。
- `map_spatial_type`: 判断地图类型。如果上下文中包含带 `ST_LOC_ID` 标签的 GeoDataFrame，选 `polygon`；如果只有 `ST_LAT/ST_LON`，选 `point`。

=== 任务 2：提取时间约束 ===
- 提取特定的日期/时段 (ISO 格式 start/end)。未提及设为 null。

=== 任务 3：规划统计图表 (CRITICAL CONTRACT) ===
- 在右侧边栏规划 1-2 个统计图表 (类型从['bar', 'line', 'pie', 'periodic_bar'] 选)。
- 必须根据上下文中的【语义字段】，显式指定图表的维度和度量。
- `target_dimension`: X轴/分类列名 (参考 `BIZ_CAT` 或 `ST_LOC_ID` 或 `ST_TIME`)
- `target_metric`: Y轴/度量列名 (参考 `BIZ_METRIC`)
- 每一个 `planned_chart.target_metric` 默认必须与 `primary_metric` 保持一致。
- 只有在用户明确要求对比其他指标时，才允许第二张图使用不同 metric。
- 如果用户问题是“最多/最少/排名”，优先使用 `bar`。
- 如果用户问题是“构成/占比”，优先使用 `pie`。
- 如果用户问题是“趋势/变化”，优先使用 `line` 或 `periodic_bar`。

🚨 标题通用防歧义法则 (UNIVERSAL NAMING RULES) 🚨
1. 必须使用上下文【语义字段与含义】中提供的【中文业务名词】来生成图表 `title`。
2. 命名公式：`[度量业务名词] + 随/按 + [维度业务名词] + 的分布/排名/趋势`。
3. 如果规划了多个图表，必须在标题中清晰体现它们维度的差异。
4. 所有标题必须与 `primary_metric` 保持一致，除非用户明确要求比较多个指标。

=== 输出格式 (JSON) ===
{{
    "primary_metric": "trip_count",
    "primary_dimension": "PULocationID",
    "is_animated": false,
    "time_granularity": "1H",
    "requested_time_range": {{
        "start": null,
        "end": null
    }},
    "map_spatial_type": "polygon",
    "planned_charts": [
        {{
            "title": "订单数量按上车区域排名",
            "chart_type": "bar",
            "target_dimension": "PULocationID",
            "target_metric": "trip_count",
            "analysis_intent": "识别订单量最高的区域"
        }}
    ]
}}
"""

        user_prompt = f"""用户原始指令: "{query}"
请分析意图并给出规划方案。
"""

        try:
            ai_plan = await self.llm.query_json_async(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
        except Exception as e:
            logger.error(f"AI 意图解析失败: {e}")
            ai_plan = {
                "primary_metric": "auto",
                "primary_dimension": "auto",
                "is_animated": False,
                "planned_charts": [],
                "map_spatial_type": "polygon" if main_summary.get('is_geospatial') else "point"
            }

        # ==========================================
        # 提取主目标
        # ==========================================
        primary_info = self._extract_primary_fields_from_ai_plan(ai_plan)
        primary_metric = primary_info["primary_metric"]
        primary_dimension = primary_info["primary_dimension"]

        is_anim_requested = ai_plan.get("is_animated", False)
        spatial_type = ai_plan.get("map_spatial_type", "polygon")

        default_resample = '1H'
        if time_source_summary:
            default_resample = (
                time_source_summary
                .get('semantic_analysis', {})
                .get('temporal_context', {})
                .get('suggested_resampling', '1H')
            )

        final_step = str(ai_plan.get("time_granularity", default_resample)).upper()
        chart_list = ai_plan.get("planned_charts", [])

        components = []
        start_t, end_t = None, None

        # ==========================================
        # A. 时间轴组件
        # ==========================================
        if is_anim_requested and found_time_col and time_source_summary:
            time_stats = time_source_summary.get('column_stats', {}).get(found_time_col, {})
            physical_min = time_stats.get('min', "2025-01-01T00:00:00")
            physical_max = time_stats.get('max', "2025-01-01T23:59:59")

            req_range = ai_plan.get("requested_time_range")
            if req_range and req_range.get("start") and req_range.get("end"):
                start_t = max(req_range["start"], physical_min)
                end_t = min(req_range["end"], physical_max)
                if start_t >= end_t:
                    start_t, end_t = physical_min, physical_max
            else:
                start_t, end_t = physical_min, physical_max

            if 'D' in final_step:
                f_format = "%Y-%m-%d"
            elif 'H' in final_step:
                f_format = "%H:00"
            elif 'T' in final_step or 'M' in final_step:
                f_format = "%H:%M"
            else:
                f_format = "%Y-%m-%d %H:%M"

            components.append(
                DashboardComponent(
                    id="global_timeline",
                    title=f"时间轴：{found_time_col}",
                    type=ComponentType.TIMELINE_CONTROLLER,
                    layout=LayoutConfig(zone=LayoutZone.TOP_NAV),
                    timeline_config={
                        "start_time": start_t,
                        "end_time": end_t,
                        "step": final_step,
                        "frame_format": f_format,
                        "column": found_time_col
                    }
                )
            )

        # ==========================================
        # B. 主地图组件
        # ==========================================
        map_title = self._build_map_title(
            is_animated=is_anim_requested,
            primary_metric=primary_metric,
            primary_dimension=primary_dimension
        )

        components.append(
            DashboardComponent(
                id="main_map",
                title=map_title,
                type=ComponentType.MAP,
                layout=LayoutConfig(zone=LayoutZone.CENTER_MAIN),
                map_config=[{
                    "layer_id": "L1",
                    "layer_type": spatial_type,
                    "data_var": main_var,
                    "is_animated": is_anim_requested,
                    "animation_column": found_time_col,
                    "primary_metric": primary_metric,
                    "primary_dimension": primary_dimension
                }]
            )
        )

        # ==========================================
        # C. 统计图表组件
        # ==========================================
        if not chart_list:
            chart_list = [{
                "title": self._build_default_chart_title(
                    chart_type="bar",
                    primary_metric=primary_metric,
                    primary_dimension=primary_dimension
                ),
                "chart_type": "bar",
                "target_dimension": primary_dimension,
                "target_metric": primary_metric,
                "analysis_intent": "围绕主指标的默认分析"
            }]

        normalized_chart_list = []
        for chart_plan in chart_list[:2]:
            chart_type = self._normalize_chart_type(chart_plan.get("chart_type", "bar"))
            target_dimension = self._safe_str(chart_plan.get("target_dimension"), primary_dimension)
            target_metric = self._safe_str(chart_plan.get("target_metric"), primary_metric)
            title = self._safe_str(chart_plan.get("title"), "")

            # 关键约束：如果 AI 没显式要求跨指标，就把 metric 拉回 primary_metric
            if target_metric == "auto" or target_metric.strip().lower() in ["none", "null", "unknown"]:
                target_metric = primary_metric

            if not title or title == "auto":
                title = self._build_default_chart_title(chart_type, target_metric, target_dimension)

            normalized_chart_list.append({
                "title": title,
                "chart_type": chart_type,
                "target_dimension": target_dimension,
                "target_metric": target_metric,
                "analysis_intent": chart_plan.get("analysis_intent", "")
            })

        for i, chart_plan in enumerate(normalized_chart_list):
            components.append(
                DashboardComponent(
                    id=f"chart_dynamic_{i + 1}",
                    title=chart_plan.get("title", f"分析图表 {i + 1}"),
                    type=ComponentType.CHART,
                    layout=LayoutConfig(zone=LayoutZone.RIGHT_SIDEBAR, index=i),
                    chart_config={
                        "chart_type": chart_plan.get("chart_type", "bar"),
                        "target_dimension": chart_plan.get("target_dimension", primary_dimension),
                        "target_metric": chart_plan.get("target_metric", primary_metric),
                        "theme": "plotly_dark",
                        "primary_metric": primary_metric,
                        "primary_dimension": primary_dimension
                    }
                )
            )

        # ==========================================
        # D. 洞察组件
        # ==========================================
        components.append(
            DashboardComponent(
                id="ai_insight",
                title="智能分析结论",
                type=ComponentType.INSIGHT,
                layout=LayoutConfig(zone=LayoutZone.BOTTOM_INSIGHT),
                insight_config={
                    "summary": "分析中...",
                    "detail": "提取业务价值中",
                    "tags": ["AI Reasoning"]
                }
            )
        )

        dashboard = DashboardSchema(
            dashboard_id=f"dash_{uuid.uuid4().hex[:6]}",
            title="时空智能看板",
            global_time_range=[start_t, end_t] if (is_anim_requested and start_t) else None,
            components=components
        )

        LayoutTemplates.apply_layout(dashboard.components)

        logger.info(
            f">>> [Planner] 规划完成 | primary_metric={primary_metric} | "
            f"primary_dimension={primary_dimension} | animated={is_anim_requested}"
        )

        return dashboard