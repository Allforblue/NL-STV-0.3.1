# core/generation/view_ir.py

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


AnalysisIntent = Literal[
    "trend",
    "composition",
    "ranking",
    "distribution",
    "comparison",
    "unknown",
]

AggregationType = Literal[
    "sum",
    "mean",
    "avg",
    "count",
    "max",
    "min",
    "none",
]

SortOrderType = Literal["asc", "desc"]


class ViewIR(BaseModel):
    """
    轻量级 Chart View IR

    设计原则：
    1. 只服务当前第一阶段 chart 编辑
    2. 不侵入 dashboard schema 主结构
    3. 可从 component dict 兼容构建
    4. 可回写到 component["chart_config"]["render_meta"]["view_ir"]
    """

    # ---------------------------------------------------------
    # 基础信息
    # ---------------------------------------------------------
    component_id: str = Field(default="unknown")
    component_type: str = Field(default="chart")
    title: Optional[str] = None

    # ---------------------------------------------------------
    # 数据/分析语义层
    # ---------------------------------------------------------
    data_var: Optional[str] = Field(
        default=None,
        description="组件使用的数据变量名。当前阶段可为空。"
    )

    analysis_intent: AnalysisIntent = Field(default="unknown")

    dimension: Optional[str] = Field(
        default=None,
        description="当前图表的主要维度字段"
    )
    metric: Optional[str] = Field(
        default=None,
        description="当前图表的主要指标字段"
    )
    aggregation: Optional[AggregationType] = Field(
        default=None,
        description="主要聚合方式"
    )

    # ---------------------------------------------------------
    # 时间变换层
    # ---------------------------------------------------------
    time_field: Optional[str] = Field(
        default=None,
        description="时间字段名"
    )
    time_bucket: Optional[str] = Field(
        default=None,
        description="时间粒度，如 1H / 1D / 1M"
    )

    filters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="预留给后续选择态/局部过滤的轻量字段，当前阶段可为空"
    )

    sort_by: Optional[str] = None
    sort_order: Optional[SortOrderType] = None
    limit: Optional[int] = None

    # ---------------------------------------------------------
    # 编码层
    # ---------------------------------------------------------
    x: Optional[str] = None
    y: Optional[str] = None
    series: Optional[str] = None
    color: Optional[str] = None
    label: Optional[str] = None

    # ---------------------------------------------------------
    # 视觉层
    # ---------------------------------------------------------
    chart_type: Optional[str] = Field(
        default=None,
        description="bar / line / pie / scatter / heatmap / area / table / timeline_heatmap"
    )
    stack: Optional[bool] = None
    unit: Optional[str] = None
    theme: Optional[str] = None

    # ---------------------------------------------------------
    # 交互/比较预留层
    # 当前阶段不重点使用，只做预留，避免后续再改模型
    # ---------------------------------------------------------
    interaction_field: Optional[str] = None
    interaction_key_field: Optional[str] = None
    fact_link_field: Optional[str] = None
    selection_scope: Optional[str] = None
    compare_mode: Optional[str] = None

    def to_prompt_dict(self) -> Dict[str, Any]:
        """
        输出给 prompt 使用的精简 dict：
        - 去掉 None
        - 去掉空 list / 空 dict
        """
        raw = self.model_dump()
        return {
            k: v for k, v in raw.items()
            if v is not None and v != [] and v != {}
        }


class ViewIRPatch(BaseModel):
    """
    用户一次编辑请求对应的 IR 增量修改。
    第一阶段主要覆盖：
    - 改图类型
    - 改标题
    - 改时间粒度
    - 改成占比图
    - 改成排名图
    """
    title: Optional[str] = None

    analysis_intent: Optional[AnalysisIntent] = None
    chart_type: Optional[str] = None

    time_field: Optional[str] = None
    time_bucket: Optional[str] = None

    dimension: Optional[str] = None
    metric: Optional[str] = None
    aggregation: Optional[AggregationType] = None

    sort_by: Optional[str] = None
    sort_order: Optional[SortOrderType] = None
    limit: Optional[int] = None

    x: Optional[str] = None
    y: Optional[str] = None
    series: Optional[str] = None
    color: Optional[str] = None

    stack: Optional[bool] = None
    unit: Optional[str] = None
    theme: Optional[str] = None

    def non_null_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


# =========================================================
# 内部工具
# =========================================================

def _normalize_component_type(comp_type: Any) -> str:
    if comp_type is None:
        return "unknown"
    text = str(comp_type)
    return text.split(".")[-1].lower()


def _safe_get_chart_config(component: Dict[str, Any]) -> Dict[str, Any]:
    chart_config = component.get("chart_config") or {}
    if hasattr(chart_config, "model_dump"):
        chart_config = chart_config.model_dump()
    return chart_config if isinstance(chart_config, dict) else {}


def _safe_get_render_meta(chart_config: Dict[str, Any]) -> Dict[str, Any]:
    render_meta = chart_config.get("render_meta") or {}
    if hasattr(render_meta, "model_dump"):
        render_meta = render_meta.model_dump()
    return render_meta if isinstance(render_meta, dict) else {}


def _pick_first_non_empty(*values: Any) -> Optional[Any]:
    for val in values:
        if val is None:
            continue
        if isinstance(val, str) and val.strip().lower() in {"", "auto", "unknown", "none", "null"}:
            continue
        return val
    return None


def _extract_first_y(y_axis: Any) -> Optional[str]:
    if isinstance(y_axis, list) and y_axis:
        first = y_axis[0]
        if first is not None:
            return str(first)
    if isinstance(y_axis, str) and y_axis.strip():
        return y_axis.strip()
    return None


# =========================================================
# Component <-> ViewIR 映射
# =========================================================

def build_view_ir_from_component(component: Dict[str, Any]) -> Optional[ViewIR]:
    """
    从当前 dashboard component dict 构建 ViewIR。

    优先级：
    1. chart_config.render_meta.view_ir
    2. 从现有 chart_config 兼容映射构建

    注意：
    - 当前只处理 chart 组件
    - 对非 chart 组件返回 None
    """
    if not isinstance(component, dict):
        return None

    comp_type = _normalize_component_type(component.get("type"))
    if comp_type != "chart":
        return None

    chart_config = _safe_get_chart_config(component)
    render_meta = _safe_get_render_meta(chart_config)

    # ---------------------------------------------------------
    # 1. 优先读取已有 view_ir
    # ---------------------------------------------------------
    stored_view_ir = render_meta.get("view_ir")
    if isinstance(stored_view_ir, dict):
        try:
            merged = dict(stored_view_ir)
            merged.setdefault("component_id", component.get("id", "unknown"))
            merged.setdefault("component_type", "chart")
            merged.setdefault("title", component.get("title"))
            return ViewIR(**merged)
        except Exception:
            # 若历史数据结构不完全兼容，则回退到兼容推断
            pass

    # ---------------------------------------------------------
    # 2. 从现有 chart_config 做兼容映射
    # ---------------------------------------------------------
    chart_type = chart_config.get("chart_type")
    if chart_type is not None:
        chart_type = str(chart_type).split(".")[-1].lower()

    target_dimension = chart_config.get("target_dimension")
    target_metric = chart_config.get("target_metric")
    primary_dimension = chart_config.get("primary_dimension")
    primary_metric = chart_config.get("primary_metric")
    display_dimension = chart_config.get("display_dimension")
    display_metric = chart_config.get("display_metric")

    x_axis = chart_config.get("x_axis")
    y_axis = chart_config.get("y_axis")
    series_name = chart_config.get("series_name")

    dimension = _pick_first_non_empty(
        display_dimension,
        target_dimension,
        primary_dimension,
    )
    metric = _pick_first_non_empty(
        display_metric,
        target_metric,
        primary_metric,
        _extract_first_y(y_axis),
    )

    x = _pick_first_non_empty(x_axis, display_dimension, target_dimension, primary_dimension)
    y = _pick_first_non_empty(_extract_first_y(y_axis), display_metric, target_metric, primary_metric)

    # 尝试从 render_meta 中补充更语义化的字段
    inferred_analysis_intent = _pick_first_non_empty(render_meta.get("analysis_intent"), "unknown")
    inferred_time_field = _pick_first_non_empty(render_meta.get("time_field"))
    inferred_aggregation = _pick_first_non_empty(render_meta.get("aggregation"))
    inferred_sort_by = _pick_first_non_empty(render_meta.get("sort_by"))
    inferred_sort_order = _pick_first_non_empty(render_meta.get("sort_order"))
    inferred_limit = render_meta.get("limit")

    # 若没有显式 analysis_intent，做一层保守推断
    if inferred_analysis_intent in {None, "unknown"}:
        inferred_analysis_intent = infer_analysis_intent_from_fields(
            chart_type=chart_type,
            time_bucket=chart_config.get("time_bucket"),
            sort_order=inferred_sort_order
        )

    return ViewIR(
        component_id=str(component.get("id", "unknown")),
        component_type="chart",
        title=component.get("title"),

        analysis_intent=inferred_analysis_intent or "unknown",
        dimension=dimension,
        metric=metric,
        aggregation=inferred_aggregation,

        time_field=inferred_time_field,
        time_bucket=chart_config.get("time_bucket"),

        sort_by=inferred_sort_by,
        sort_order=inferred_sort_order,
        limit=inferred_limit,

        x=x,
        y=y,
        series=_pick_first_non_empty(series_name),
        color=_pick_first_non_empty(render_meta.get("color")),

        chart_type=chart_type,
        stack=chart_config.get("stack"),
        unit=chart_config.get("unit"),
        theme=chart_config.get("theme"),

        interaction_field=chart_config.get("interaction_field"),
        interaction_key_field=chart_config.get("interaction_key_field"),
        fact_link_field=chart_config.get("fact_link_field"),
        selection_scope=chart_config.get("selection_scope"),
        compare_mode=chart_config.get("compare_mode"),
    )


def apply_view_ir_to_component(component: Dict[str, Any], view_ir: ViewIR) -> Dict[str, Any]:
    """
    将 ViewIR 回写到 component dict。

    当前策略：
    1. 保守更新 title / chart_config 中已有兼容字段
    2. 将完整 view_ir 写入 chart_config.render_meta.view_ir
    3. 不强行覆盖 planner locked 字段（target_dimension / target_metric）
    4. 以 display_* / x_axis / y_axis / series_name 为主要可编辑承载层
    """
    comp = deepcopy(component)

    comp_type = _normalize_component_type(comp.get("type"))
    if comp_type != "chart":
        return comp

    if view_ir.title:
        comp["title"] = view_ir.title

    chart_config = comp.get("chart_config") or {}
    if hasattr(chart_config, "model_dump"):
        chart_config = chart_config.model_dump()
    if not isinstance(chart_config, dict):
        chart_config = {}

    render_meta = chart_config.get("render_meta") or {}
    if hasattr(render_meta, "model_dump"):
        render_meta = render_meta.model_dump()
    if not isinstance(render_meta, dict):
        render_meta = {}

    # ---------------------------------------------------------
    # 显式兼容字段回写
    # ---------------------------------------------------------
    if view_ir.chart_type:
        chart_config["chart_type"] = view_ir.chart_type

    chart_config["time_bucket"] = view_ir.time_bucket

    if view_ir.x:
        chart_config["x_axis"] = view_ir.x

    if view_ir.y:
        existing_y = chart_config.get("y_axis")
        if isinstance(existing_y, list) and existing_y:
            chart_config["y_axis"] = [view_ir.y]
        else:
            chart_config["y_axis"] = [view_ir.y]

    if view_ir.series:
        chart_config["series_name"] = view_ir.series

    if view_ir.stack is not None:
        chart_config["stack"] = view_ir.stack

    if view_ir.unit is not None:
        chart_config["unit"] = view_ir.unit

    if view_ir.theme is not None:
        chart_config["theme"] = view_ir.theme

    # display 层优先作为编辑后的最终呈现语义
    if view_ir.dimension:
        chart_config["display_dimension"] = view_ir.dimension
        # 若 x_axis 为空，则顺手填充
        if not chart_config.get("x_axis"):
            chart_config["x_axis"] = view_ir.dimension

    if view_ir.metric:
        chart_config["display_metric"] = view_ir.metric
        # 若 y_axis 为空，则顺手填充
        if not chart_config.get("y_axis"):
            chart_config["y_axis"] = [view_ir.metric]

    if view_ir.interaction_field is not None:
        chart_config["interaction_field"] = view_ir.interaction_field
    if view_ir.interaction_key_field is not None:
        chart_config["interaction_key_field"] = view_ir.interaction_key_field
    if view_ir.fact_link_field is not None:
        chart_config["fact_link_field"] = view_ir.fact_link_field
    if view_ir.selection_scope is not None:
        chart_config["selection_scope"] = view_ir.selection_scope
    if view_ir.compare_mode is not None:
        chart_config["compare_mode"] = view_ir.compare_mode

    # ---------------------------------------------------------
    # render_meta 写回完整 view_ir + 关键语义镜像
    # ---------------------------------------------------------
    render_meta["view_ir"] = view_ir.model_dump()

    # 这些镜像字段便于旧逻辑 / prompt / 调试共存
    render_meta["analysis_intent"] = view_ir.analysis_intent
    render_meta["aggregation"] = view_ir.aggregation
    render_meta["time_field"] = view_ir.time_field
    render_meta["sort_by"] = view_ir.sort_by
    render_meta["sort_order"] = view_ir.sort_order
    render_meta["limit"] = view_ir.limit
    render_meta["color"] = view_ir.color

    chart_config["render_meta"] = render_meta
    comp["chart_config"] = chart_config

    return comp


# =========================================================
# 补充辅助函数
# =========================================================

def infer_analysis_intent_from_fields(
    chart_type: Optional[str],
    time_bucket: Optional[str],
    sort_order: Optional[str] = None
) -> AnalysisIntent:
    """
    基于少量显式字段做保守意图推断。
    用于没有历史 view_ir 的老组件兜底。
    """
    ctype = (chart_type or "").strip().lower()
    if time_bucket:
        return "trend"

    if ctype == "line" or ctype == "area":
        return "trend"

    if ctype == "pie":
        return "composition"

    if ctype == "bar" and sort_order in {"asc", "desc"}:
        return "ranking"

    return "unknown"


def merge_view_ir(base_ir: ViewIR, patch: ViewIRPatch) -> ViewIR:
    """
    将 patch 非空字段覆盖到 base_ir 上，返回新的 ViewIR。
    """
    merged = base_ir.model_dump()
    for k, v in patch.non_null_dict().items():
        merged[k] = v
    return ViewIR(**merged)


def ensure_view_ir_dict(component: Dict[str, Any]) -> Dict[str, Any]:
    """
    便捷方法：
    从 component 构建 ViewIR，并直接返回 dict 形式。
    若不是 chart 或构建失败，返回空 dict。
    """
    ir = build_view_ir_from_component(component)
    if ir is None:
        return {}
    return ir.model_dump()