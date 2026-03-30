# backend/core/generation/dashboard_planner.py

import logging
import uuid
import json
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
    看板编排器（V1.3.0 Semantic Title Enhanced Edition）

    升级点：
    1. 引入更细粒度的维度语义命名（如 上车区域 / 下车区域 / 上车时间 / 下车时间）
    2. 优化地图和图表标题生成，使其更贴合业务语义
    3. 保持 primary_metric / primary_dimension 锁定策略不变
    4. 兼容现有 planner / workflow / generator 架构
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
        """
        保留旧接口，兼容现有逻辑。
        """
        dim = str(dim_name or "").lower()

        if any(k in dim for k in ['zone', 'location', 'region', 'district', 'borough', 'city', 'area']):
            return "区域"
        if any(k in dim for k in ['time', 'date', 'hour', 'day', 'month', 'week']):
            return "时间"
        if any(k in dim for k in ['category', 'type', 'class']):
            return "类别"
        return "维度"

    def _infer_dimension_phrase_from_name(self, dim_name: str) -> str:
        """
        新增：更细粒度的业务语义短语推断。
        用于标题生成，不再只返回“区域/时间/类别”，而是返回更贴切的中文短语。
        """
        dim = str(dim_name or "").strip()
        lower_dim = dim.lower()

        # 1. NYC Taxi / 通用 pickup / dropoff 方向性字段
        if any(k in lower_dim for k in ['pulocationid', 'pickup_zone', 'pickup_region', 'pickup_area']):
            return "上车区域"

        if any(k in lower_dim for k in ['dolocationid', 'dropoff_zone', 'dropoff_region', 'dropoff_area']):
            return "下车区域"

        if any(k in lower_dim for k in ['pickup_time', 'pickup_datetime', 'tpep_pickup_datetime']):
            return "上车时间"

        if any(k in lower_dim for k in ['dropoff_time', 'dropoff_datetime', 'tpep_dropoff_datetime']):
            return "下车时间"

        # 2. Borough / Zone / Region 这类更具体的空间层级
        if 'borough' in lower_dim:
            if 'pu' in lower_dim or 'pickup' in lower_dim:
                return "上车行政区"
            if 'do' in lower_dim or 'dropoff' in lower_dim:
                return "下车行政区"
            return "行政区"

        if 'zone' in lower_dim:
            if 'pu' in lower_dim or 'pickup' in lower_dim:
                return "上车区域"
            if 'do' in lower_dim or 'dropoff' in lower_dim:
                return "下车区域"
            return "区域"

        if any(k in lower_dim for k in ['location', 'region', 'district', 'city', 'area']):
            if 'pu' in lower_dim or 'pickup' in lower_dim:
                return "上车区域"
            if 'do' in lower_dim or 'dropoff' in lower_dim:
                return "下车区域"
            return "区域"

        # 3. 时间类
        if any(k in lower_dim for k in ['time', 'date', 'hour', 'day', 'month', 'week']):
            if 'pickup' in lower_dim:
                return "上车时间"
            if 'dropoff' in lower_dim:
                return "下车时间"
            return "时间"

        # 4. 类别类
        if any(k in lower_dim for k in ['category', 'type', 'class']):
            return "类别"

        return "维度"

    def _build_metric_dimension_title(
        self,
        metric_label: str,
        dimension_phrase: str,
        chart_type: str
    ) -> str:
        """
        根据 metric + 更细粒度的 dimension phrase 生成更自然的中文标题。
        """
        ct = self._normalize_chart_type(chart_type)

        if ct == 'line':
            return f"{metric_label}随{dimension_phrase}变化趋势"

        if ct == 'pie':
            return f"{metric_label}按{dimension_phrase}构成分布"

        if ct == 'periodic_bar':
            if "时间" in dimension_phrase:
                return f"{metric_label}按{dimension_phrase}周期分布"
            return f"{metric_label}按周期时间分布"

        # 默认 bar / ranking
        return f"{metric_label}{dimension_phrase}排名"

    def _build_map_title(self, is_animated: bool, primary_metric: str, primary_dimension: str) -> str:
        metric_label = self._infer_metric_label_from_name(primary_metric)
        dimension_phrase = self._infer_dimension_phrase_from_name(primary_dimension)

        if is_animated:
            return f"{metric_label}{dimension_phrase}时空演变"
        return f"{metric_label}{dimension_phrase}空间分布"

    def _build_default_chart_title(self, chart_type: str, primary_metric: str, primary_dimension: str) -> str:
        metric_label = self._infer_metric_label_from_name(primary_metric)
        dimension_phrase = self._infer_dimension_phrase_from_name(primary_dimension)
        return self._build_metric_dimension_title(metric_label, dimension_phrase, chart_type)

    def _build_chart_reason_text(
        self,
        chart_type: str,
        analysis_intent: str,
        metric: str,
        dimension: str
    ) -> str:
        """
        生成给前端展示的简短解释：
        为什么当前组件会被规划成这种图。
        """
        ct = self._normalize_chart_type(chart_type)
        intent = str(analysis_intent or "").strip()

        if ct == "line":
            return (
                f"该图选择折线图，是因为当前问题更关注 `{metric}` 随 `{dimension}` 的连续变化趋势，"
                f"折线图更适合表达时间或有序维度上的变化。"
            )

        if ct == "pie":
            return (
                f"该图选择饼图，是因为当前问题更关注 `{metric}` 在不同 `{dimension}` 上的构成关系，"
                f"饼图更适合展示占比和组成结构。"
            )

        if ct == "periodic_bar":
            return (
                f"该图选择周期分布图，是因为当前问题更关注 `{metric}` 在周期性时间维度上的模式变化，"
                f"这种图更适合展示小时或周内分布特征。"
            )

        # 默认 bar
        if "排名" in intent or "最多" in intent or "最少" in intent:
            return (
                f"该图选择柱状图，是因为当前问题更关注不同 `{dimension}` 之间 `{metric}` 的高低比较与排名，"
                f"柱状图更适合展示类别间的对比关系。"
            )

        return (
            f"该图选择柱状图，是因为当前问题需要对不同 `{dimension}` 上的 `{metric}` 进行直观比较，"
            f"柱状图能够清晰展示各类别之间的差异。"
        )

    def _build_chart_explanation(
        self,
        chart_type: str,
        analysis_intent: str,
        metric: str,
        dimension: str,
        user_query: str
    ) -> Dict[str, Any]:
        """
        生成组件级 explanation，挂到 chart_config.render_meta 中。
        """
        return {
            "source": "planner",
            "analysis_intent": analysis_intent or "",
            "why_this_chart": self._build_chart_reason_text(
                chart_type=chart_type,
                analysis_intent=analysis_intent,
                metric=metric,
                dimension=dimension
            ),
            "why_this_dimension": (
                f"该图使用 `{dimension}` 作为分析维度，因为它与当前问题最相关，"
                f"能够帮助用户从这个维度观察差异或变化。"
            ),
            "why_this_metric": (
                f"该图使用 `{metric}` 作为核心指标，因为它是当前问题所关注的主要度量。"
            ),
            "user_request": user_query
        }
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
- `target_dimension`: X轴/分类列名
- `target_metric`: Y轴/度量列名
- 每一个 `planned_chart.target_metric` 默认必须与 `primary_metric` 保持一致。
- 只有在用户明确要求对比其他指标时，才允许第二张图使用不同 metric。
- 如果用户问题是“最多/最少/排名”，优先使用 `bar`。
- 如果用户问题是“构成/占比”，优先使用 `pie`。
- 如果用户问题是“趋势/变化”，优先使用 `line` 或 `periodic_bar`。

🚨 标题生成规则（TITLE RULES）🚨
1. 标题必须尽量贴近业务语义，不要机械使用“区域/时间/类别”。
2. 若字段体现明显方向性，应优先体现：
   - PULocationID / pickup_* -> 上车区域 / 上车时间
   - DOLocationID / dropoff_* -> 下车区域 / 下车时间
3. 标题应简洁、自然，适合直接展示在看板中。
4. 例如：
   - “订单数量上车区域排名”
   - “订单数量下车区域排名”
   - “订单数量按上车区域构成分布”
   - “订单数量随上车时间变化趋势”
5. 所有标题必须与 `primary_metric` 保持一致，除非用户明确要求比较多个指标。

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
            "title": "订单数量上车区域排名",
            "chart_type": "bar",
            "target_dimension": "PULocationID",
            "target_metric": "trip_count",
            "analysis_intent": "识别订单量最高的上车区域"
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
            analysis_intent = self._safe_str(chart_plan.get("analysis_intent"), "")

            # 如果 AI 没显式要求跨指标，就把 metric 拉回 primary_metric
            if target_metric == "auto" or target_metric.strip().lower() in ["none", "null", "unknown"]:
                target_metric = primary_metric

            if not title or title == "auto":
                title = self._build_default_chart_title(chart_type, target_metric, target_dimension)

            explanation = self._build_chart_explanation(
                chart_type=chart_type,
                analysis_intent=analysis_intent,
                metric=target_metric,
                dimension=target_dimension,
                user_query=query
            )

            normalized_chart_list.append({
                "title": title,
                "chart_type": chart_type,
                "target_dimension": target_dimension,
                "target_metric": target_metric,
                "analysis_intent": analysis_intent,
                "explanation": explanation
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
                        "primary_dimension": primary_dimension,
                        "render_meta": {
                            "analysis_intent": chart_plan.get("analysis_intent", ""),
                            "explanation": chart_plan.get("explanation", {})
                        }
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