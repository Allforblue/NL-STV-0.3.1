import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple

from core.llm.AI_client import AIClient
from core.schemas.interaction import InteractionTriggerType

from core.generation.dashboard_script_utils import (
    safe_func_name,
    parse_dashboard_script,
    assemble_dashboard_script,
)
from core.generation.scaffold import STChartScaffold
from core.generation.view_ir import (
    ViewIR,
    build_view_ir_from_component,
    apply_view_ir_to_component,
)
from core.generation.view_transformer import ViewTransformer

logger = logging.getLogger(__name__)


class VizEditor:
    """
    VizEditor（受控编辑增强版）

    当前定位：
    1. 仅处理组件级自然语言图表编辑
    2. 不再处理 bbox / selected_ids / selected_values / time_range
    3. 不承担 selection runtime patch
    4. 始终返回完整 dashboard script
    5. 与现有 workflow / executor 契约兼容
    6. 接入轻量 View IR
    7. 接入 scaffold + validation + retry，避免 LLM 自由写代码
    8. 增加 planner-locked constraints，减少编辑语义漂移

    当前优先支持：
    - 改图类型（折线图 / 柱状图 / 饼图 / 热力图 / 散点图 / 面积图）
    - 改标题
    - 改时间粒度（按小时 / 按天 / 按月）
    - 趋势 / 构成 / 排名等简单意图重生成
    """

    def __init__(self, llm_client: AIClient):
        self.llm = llm_client
        self.scaffold = STChartScaffold()
        self.view_transformer = ViewTransformer()

    # =========================================================
    # 公共工具
    # =========================================================
    def _clean_markdown(self, text: str) -> str:
        if not text:
            return ""

        pattern = r"```(?:python)?\s*(.*?)```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        else:
            text = text.strip()
            text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        if "def get_dashboard_data" in text and "def get_" in text:
            lines = text.split("\n")
            inner_func_lines = []
            capture = False
            for line in lines:
                if line.strip().startswith("def get_") and not line.strip().startswith("def get_dashboard_data"):
                    capture = True
                if capture:
                    if line.startswith("    "):
                        inner_func_lines.append(line[4:])
                    else:
                        inner_func_lines.append(line)
            if inner_func_lines:
                return "\n".join(inner_func_lines).strip()

        return text.strip()

    def _build_context_str(self, summaries: List[Dict[str, Any]]) -> str:
        context_str = ""
        for s in summaries:
            var_name = s.get("variable_name")
            col_stats = s.get("column_stats") or s.get("basic_stats", {}).get("column_stats", {})
            sem_analysis = s.get("semantic_analysis", {})
            col_meta = sem_analysis.get("column_metadata", {})

            col_desc_list = [
                f"{col}({info.get('dtype', 'unknown')})"
                for col, info in col_stats.items()
            ] if col_stats else ["(无列信息)"]

            semantic_hints = {
                col: meta.get("semantic_tag")
                for col, meta in col_meta.items()
                if isinstance(meta, dict) and meta.get("semantic_tag")
            }

            context_str += f"- 变量 `{var_name}`:\n"
            context_str += f"  - 列信息: {', '.join(col_desc_list[:50])}\n"
            if semantic_hints:
                context_str += f"  - 语义标签 (CRITICAL FOR MAPPING): {json.dumps(semantic_hints, ensure_ascii=False)}\n"
            context_str += "\n"

        return context_str

    # =========================================================
    # Layout / Component 辅助
    # =========================================================
    def _normalize_component_type(self, comp_type: Any) -> str:
        if comp_type is None:
            return "unknown"
        text = str(comp_type)
        return text.split(".")[-1].lower()

    def _get_layout_components(self, last_layout: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not last_layout:
            return []
        return last_layout.get("components", []) or []

    def _get_component_by_id(
        self,
        last_layout: Optional[Dict[str, Any]],
        component_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not component_id or not last_layout:
            return None

        for comp in self._get_layout_components(last_layout):
            if comp.get("id") == component_id:
                return comp

        return None

    def _replace_component_in_layout(
        self,
        last_layout: Optional[Dict[str, Any]],
        updated_component: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not last_layout or not updated_component:
            return last_layout

        new_layout = json.loads(json.dumps(last_layout, ensure_ascii=False))
        components = new_layout.get("components", []) or []
        target_id = updated_component.get("id")

        replaced = False
        for idx, comp in enumerate(components):
            if comp.get("id") == target_id:
                components[idx] = updated_component
                replaced = True
                break

        if not replaced:
            logger.warning(f"[VizEditor] Failed to replace component in layout: {target_id}")

        new_layout["components"] = components
        return new_layout

    # =========================================================
    # 编辑类型识别
    # =========================================================
    def _detect_structural_edit(self, query: str) -> bool:
        if not query:
            return False

        q = query.strip().lower()
        structural_keywords = [
            "新增一个图", "增加一个图", "再加一个图", "加一个图表",
            "删除这个图", "删掉这个图", "移除这个图",
            "调整布局", "重新布局", "交换位置", "左右交换", "放到下面", "放到右边",
            "增加对比", "整体对比", "区域对比", "多个区域", "多选区", "对比分析",
            "add a chart", "add another chart", "remove this chart",
            "change layout", "rearrange", "compare selected area", "compare regions"
        ]
        return any(k in q for k in structural_keywords)

    def _classify_edit_intent(
        self,
        payload: Any,
        query: str,
        active_component: Optional[Any]
    ) -> Dict[str, Any]:
        if self._detect_structural_edit(query or ""):
            return {"kind": "structural_edit"}

        if (
            payload.trigger_type == InteractionTriggerType.NATURAL_LANGUAGE
            and active_component is not None
        ):
            return {"kind": "component_nl_edit"}

        return {"kind": "unknown"}

    # =========================================================
    # Planner Locked Constraints（对齐 generator）
    # =========================================================
    def _extract_component_constraints(self, comp: Any) -> Dict[str, Any]:
        is_dict = isinstance(comp, dict)

        c_id = comp.get("id") if is_dict else getattr(comp, "id", "unknown")
        c_type = comp.get("type") if is_dict else getattr(comp, "type", "unknown")
        c_type_str = str(c_type).split(".")[-1].lower()

        constraints = {
            "component_id": c_id,
            "component_type": c_type_str,
            "locked_metric": None,
            "locked_dimension": None,
            "locked_chart_type": None,
            "locked_map_layer_type": None,
            "locked_is_animated": None
        }

        if c_type_str == "chart":
            conf = comp.get("chart_config", {}) if is_dict else getattr(comp, "chart_config", {})
            if hasattr(conf, "model_dump"):
                conf = conf.model_dump()
            conf = conf or {}

            constraints["locked_metric"] = conf.get("target_metric") or conf.get("primary_metric")
            constraints["locked_dimension"] = conf.get("target_dimension") or conf.get("primary_dimension")
            constraints["locked_chart_type"] = conf.get("chart_type")

        elif c_type_str == "map":
            m_conf = comp.get("map_config", []) if is_dict else getattr(comp, "map_config", [])
            if isinstance(m_conf, list) and len(m_conf) > 0:
                first_layer = m_conf[0]
                if hasattr(first_layer, "model_dump"):
                    first_layer = first_layer.model_dump()

                constraints["locked_metric"] = first_layer.get("primary_metric")
                constraints["locked_dimension"] = first_layer.get("primary_dimension")
                constraints["locked_map_layer_type"] = first_layer.get("layer_type")
                constraints["locked_is_animated"] = first_layer.get("is_animated")

        return constraints

    def _build_locked_constraints_text(self, constraints: Dict[str, Any]) -> str:
        ctype = constraints.get("component_type", "unknown")
        locked_metric = constraints.get("locked_metric")
        locked_dimension = constraints.get("locked_dimension")
        locked_chart_type = constraints.get("locked_chart_type")
        locked_map_layer_type = constraints.get("locked_map_layer_type")
        locked_is_animated = constraints.get("locked_is_animated")

        lines = [
            "=== PLANNER-LOCKED CONSTRAINTS (HIGHEST PRIORITY) ===",
            f"- component_type: {ctype}",
            f"- locked_metric: {locked_metric}",
            f"- locked_dimension: {locked_dimension}",
        ]

        if locked_chart_type:
            lines.append(f"- locked_chart_type: {locked_chart_type}")
        if locked_map_layer_type:
            lines.append(f"- locked_map_layer_type: {locked_map_layer_type}")
        if locked_is_animated is not None:
            lines.append(f"- locked_is_animated: {locked_is_animated}")

        lines.extend([
            "",
            "You MUST obey these locked constraints.",
            "Do NOT replace the locked metric with another metric such as total_amount, fare_amount, revenue, amount, etc.",
            "Do NOT replace the locked dimension with another dimension unless the component config explicitly requires it.",
            "The planner has already decided the analytical focus for this component.",
            "Your job is to EDIT the implementation/view semantics, not to redesign the analytical target.",
        ])

        if locked_metric and str(locked_metric).lower() not in ["auto", "none", "null", "unknown"]:
            lines.append(f"CRITICAL: You MUST preserve `{locked_metric}` as the metric for this component unless the user explicitly changes the metric.")
        if locked_dimension and str(locked_dimension).lower() not in ["auto", "none", "null", "unknown"]:
            lines.append(f"CRITICAL: You MUST preserve `{locked_dimension}` as the main dimension/key for this component unless the user explicitly changes the dimension.")

        return "\n".join(lines)

    # =========================================================
    # 验证与重试（对齐 generator 约束风格）
    # =========================================================
    def _validate_generated_code(self, code: str) -> Tuple[bool, str]:
        if not code or not code.strip():
            return False, "[Validation] 生成代码为空。"

        lower_code = code.lower()

        forbidden_imports = [
            "import plotly",
            "from plotly",
            "import matplotlib",
            "from matplotlib",
            "import seaborn",
            "from seaborn",
            "import bokeh",
            "from bokeh",
            "import altair",
            "from altair",
            "import folium",
            "from folium",
            "import pydeck",
            "from pydeck",
            "import keplergl",
            "from keplergl",
        ]
        for pattern in forbidden_imports:
            if pattern in lower_code:
                return False, f"[Validation] 禁止导入可视化库：{pattern}"

        forbidden_calls = [
            "px.bar(",
            "px.line(",
            "px.scatter(",
            "px.pie(",
            "px.choropleth(",
            "px.choropleth_mapbox(",
            "px.scatter_mapbox(",
            "px.density_mapbox(",
            "go.figure(",
            "go.bar(",
            "go.scatter(",
            "plt.plot(",
            "plt.bar(",
            "plt.show(",
            "fig.update_layout(",
            "fig.update_traces(",
            "fig.add_trace(",
            "fig.show(",
        ]
        for pattern in forbidden_calls:
            if pattern in lower_code:
                return False, f"[Validation] 检测到禁止的原生绘图调用：{pattern}"

        if "def get_" not in code or "data_context" not in code:
            return False, "[Validation] 缺少合法函数签名，必须为 def get_xxx(data_context):"

        if "stv." not in code:
            return False, "[Validation] 未检测到 stv SDK调用，必须使用 stv.xxx(...) 进行渲染。"

        if "stv.animated_map(" in code or "stv.animated_scatter(" in code:
            forbidden_time_patterns = [
                ".dt.hour",
                ".dt.day",
                ".dt.month",
                ".dt.week",
                ".dt.dayofweek",
            ]
            for pattern in forbidden_time_patterns:
                if pattern in code:
                    return False, f"[Validation] 动画时间列禁止使用离散时间片：{pattern}，请改用 dt.floor('h') 等连续时间戳。"

        return True, ""

    async def _retry_on_validation_failure(
        self,
        system_prompt: str,
        original_code: str,
        validation_error: str,
        func_name: str,
        user_prompt: str
    ) -> str:
        retry_prompt = f"""
Your previous edited component code was rejected by the validator.

=== VALIDATION ERROR ===
{validation_error}

=== REJECTED CODE ===
{original_code}

=== ORIGINAL EDIT TASK ===
{user_prompt}

Rewrite the function using ONLY the pre-injected `stv` and `ops` SDK objects.

Rules:
1. Keep the exact function signature: `def {func_name}(data_context):`
2. Do NOT import any visualization library
3. Do NOT use raw Plotly / Matplotlib / Seaborn
4. Use pandas/ops for data prep, then stv for rendering
5. If this is a chart component, return the Figure directly
6. If this is a map component, return the protocol Dict directly
7. Follow the resolved view_ir semantics strictly
8. Respect planner-locked metric and dimension constraints
9. Do NOT wrap code inside get_dashboard_data

Return ONLY the fixed Python code block.
"""
        raw_response = await self.llm.chat_async([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": retry_prompt}
        ], json_mode=False)

        return self._clean_markdown(raw_response)

    # =========================================================
    # View IR 辅助
    # =========================================================
    def _build_view_ir_from_component(self, component: Dict[str, Any]) -> Optional[ViewIR]:
        try:
            return build_view_ir_from_component(component)
        except Exception as e:
            logger.warning(f"[VizEditor] Failed to build view_ir from component: {e}")
            return None

    def _apply_view_ir_to_component(
        self,
        component: Dict[str, Any],
        resolved_view_ir: ViewIR,
        user_query: str = ""
    ) -> Dict[str, Any]:
        try:
            updated = apply_view_ir_to_component(component, resolved_view_ir)

            # 编辑后补 explanation，供前端展示“为什么改成这样”
            chart_config = updated.get("chart_config") or {}
            if hasattr(chart_config, "model_dump"):
                chart_config = chart_config.model_dump()
            if not isinstance(chart_config, dict):
                chart_config = {}

            render_meta = chart_config.get("render_meta") or {}
            if hasattr(render_meta, "model_dump"):
                render_meta = render_meta.model_dump()
            if not isinstance(render_meta, dict):
                render_meta = {}

            render_meta["explanation"] = self._build_editor_explanation(
                view_ir=resolved_view_ir,
                user_query=user_query
            )

            chart_config["render_meta"] = render_meta
            updated["chart_config"] = chart_config
            return updated

        except Exception as e:
            logger.warning(f"[VizEditor] Failed to apply view_ir to component: {e}")
            return component

    def _build_chart_reason_text_from_view_ir(self, view_ir: ViewIR) -> str:
        """
        基于编辑后的 ViewIR，为前端生成简短解释：
        为什么当前组件会被改成这种图。
        """
        chart_type = (view_ir.chart_type or "").lower()
        intent = (view_ir.analysis_intent or "unknown").lower()
        metric = view_ir.metric or "核心指标"
        dimension = view_ir.dimension or view_ir.x or "当前维度"

        if chart_type == "line":
            return (
                f"该图当前选择折线图，是因为编辑后的分析目标更偏向观察 `{metric}` "
                f"随 `{dimension}` 的连续变化趋势，折线图更适合表达时间或有序维度上的变化。"
            )

        if chart_type == "pie":
            return (
                f"该图当前选择饼图，是因为编辑后的分析目标更偏向观察 `{metric}` "
                f"在不同 `{dimension}` 上的构成关系，饼图更适合展示占比与组成结构。"
            )

        if chart_type == "area":
            return (
                f"该图当前选择面积图，是因为编辑后的分析目标希望同时体现 `{metric}` "
                f"随 `{dimension}` 的变化趋势及其整体量感。"
            )

        if chart_type == "scatter":
            return (
                f"该图当前选择散点图，是因为编辑后的分析目标更适合通过点位分布观察 "
                f"`{dimension}` 与 `{metric}` 之间的关系。"
            )

        if chart_type == "heatmap":
            return (
                f"该图当前选择热力图，是因为编辑后的分析目标更适合通过颜色强弱查看 "
                f"`{metric}` 在 `{dimension}` 相关空间或二维分布中的差异。"
            )

        if intent == "ranking":
            return (
                f"该图当前选择柱状图，是因为编辑后的问题更关注不同 `{dimension}` 上 "
                f"`{metric}` 的高低比较与排名，柱状图更适合展示类别间对比。"
            )

        if intent == "composition":
            return (
                f"该图当前选择饼图，是因为编辑后的问题更关注 `{metric}` 在不同 `{dimension}` 上的构成关系。"
            )

        return (
            f"该图当前保持这种图形，是因为它更适合表达编辑后的分析目标，"
            f"帮助用户观察 `{dimension}` 上的 `{metric}`。"
        )

    def _build_editor_explanation(
        self,
        view_ir: ViewIR,
        user_query: str
    ) -> Dict[str, Any]:
        """
        编辑后 explanation，写入 chart_config.render_meta["explanation"]
        """
        dimension = view_ir.dimension or view_ir.x or "当前维度"
        metric = view_ir.metric or view_ir.y or "核心指标"

        return {
            "source": "editor",
            "analysis_intent": view_ir.analysis_intent or "",
            "why_this_chart": self._build_chart_reason_text_from_view_ir(view_ir),
            "why_this_dimension": (
                f"编辑后仍以 `{dimension}` 作为主要分析维度，"
                f"因为它最适合承载当前用户想观察的差异或变化。"
            ),
            "why_this_metric": (
                f"编辑后继续围绕 `{metric}` 作为核心指标进行展示，"
                f"以保持分析目标的一致性。"
            ),
            "edit_request": user_query or ""
        }

    # =========================================================
    # Prompt 构造
    # =========================================================
    def _build_component_edit_user_prompt(
        self,
        query: str,
        component: Dict[str, Any],
        current_view_ir: Optional[ViewIR],
        resolved_view_ir: Optional[ViewIR]
    ) -> str:
        comp_id = component.get("id", "unknown")
        comp_type = self._normalize_component_type(component.get("type"))
        title = component.get("title", comp_id)
        chart_config = component.get("chart_config", {}) or {}
        map_config = component.get("map_config", []) or []
        links = component.get("links", []) or []
        component_state = component.get("component_state", {}) or {}

        func_name = safe_func_name(comp_id)

        current_ir_dict = current_view_ir.to_prompt_dict() if current_view_ir else {}
        resolved_ir_dict = resolved_view_ir.to_prompt_dict() if resolved_view_ir else {}

        constraints = self._extract_component_constraints(component)
        locked_constraints_text = self._build_locked_constraints_text(constraints)

        return f"""
You are editing ONE existing spatio-temporal visualization component.

=== TARGET COMPONENT ===
component_id: {comp_id}
component_type: {comp_type}
title: {title}
chart_config: {json.dumps(chart_config, ensure_ascii=False)}
map_config: {json.dumps(map_config, ensure_ascii=False)}
links: {json.dumps(links, ensure_ascii=False)}
component_state: {json.dumps(component_state, ensure_ascii=False)}

=== USER EDIT REQUEST ===
{query}

=== CURRENT VIEW IR ===
{json.dumps(current_ir_dict, ensure_ascii=False, indent=2)}

=== RESOLVED VIEW IR (MUST FOLLOW) ===
{json.dumps(resolved_ir_dict, ensure_ascii=False, indent=2)}

{locked_constraints_text}

=== EDIT REQUIREMENTS ===
1. Rewrite ONLY the target component function.
2. Keep the exact function signature:
   def {func_name}(data_context)
3. You MUST follow the resolved View IR, not just replace the final chart type call.
4. If analysis_intent is "trend" and time_bucket is set, you MUST do temporal aggregation before rendering.
5. If analysis_intent is "composition", you MUST aggregate category values before calling stv.pie(...).
6. If analysis_intent is "ranking", you MUST aggregate and sort values before rendering.
7. Preserve the original analytical target unless the user explicitly asks to change metric/dimension.
8. Do NOT guess missing columns; use only fields supported by the data metadata from system prompt.
9. Do NOT wrap the function inside get_dashboard_data.
10. Return ONLY Python code.

=== OUTPUT FORMAT ===
```python
def {func_name}(data_context):
    # Step 1: Data prep
    # Step 2: Render
    return result
""".strip()

    # =========================================================
    # 单组件重生成（受控版本）
    # =========================================================
    async def _regenerate_component_function(
        self,
        query: str,
        component: Dict[str, Any],
        summaries: List[Dict[str, Any]],
        current_view_ir: Optional[ViewIR],
        resolved_view_ir: Optional[ViewIR]
    ) -> str:
        context_str = self._build_context_str(summaries)
        system_prompt = self.scaffold.get_system_prompt(context_str, [component])
        user_prompt = self._build_component_edit_user_prompt(
            query=query,
            component=component,
            current_view_ir=current_view_ir,
            resolved_view_ir=resolved_view_ir
        )

        raw = await self.llm.chat_async(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            json_mode=False
        )

        code = self._clean_markdown(raw)

        expected_func_name = safe_func_name(component.get("id", "unknown"))
        if f"def {expected_func_name}(data_context):" not in code:
            logger.warning(f"[VizEditor] Regenerated code missing expected signature for {expected_func_name}")
            return ""

        is_valid, validation_error = self._validate_generated_code(code)
        if not is_valid:
            logger.warning(f"[VizEditor] Validation failed for edited component: {validation_error}")
            code = await self._retry_on_validation_failure(
                system_prompt=system_prompt,
                original_code=code,
                validation_error=validation_error,
                func_name=expected_func_name,
                user_prompt=user_prompt
            )

            if f"def {expected_func_name}(data_context):" not in code:
                logger.warning(f"[VizEditor] Retry code missing expected signature for {expected_func_name}")
                return ""

            is_valid_retry, validation_error_retry = self._validate_generated_code(code)
            if not is_valid_retry:
                logger.warning(f"[VizEditor] Retry validation still failed: {validation_error_retry}")
                return ""

        return code

    # =========================================================
    # 组件级自然语言编辑
    # =========================================================
    async def _handle_component_nl_edit(
        self,
        parsed: Dict[str, Any],
        payload: Any,
        summaries: List[Dict[str, Any]],
        last_layout: Optional[Dict[str, Any]],
        active_component: Optional[Any]
    ) -> Dict[str, Any]:
        result = {
            "code": "",
            "updated_layout": last_layout,
            "updated_component_id": payload.active_component_id,
            "view_ir": {},
            "warnings": []
        }

        if not payload.active_component_id:
            logger.warning("[VizEditor] Missing active_component_id for component edit.")
            return result

        comp = None
        if active_component is not None:
            if hasattr(active_component, "model_dump"):
                comp = active_component.model_dump()
            elif isinstance(active_component, dict):
                comp = active_component

        if comp is None:
            comp = self._get_component_by_id(last_layout, payload.active_component_id)

        if not comp:
            logger.warning("[VizEditor] Target component not found for NL edit.")
            return result

        comp_type = self._normalize_component_type(comp.get("type"))

        # 当前阶段更保守：非 chart 组件不走 View IR 编辑增强，直接返回空 result 由 workflow fallback
        if comp_type != "chart":
            logger.warning(f"[VizEditor] Non-chart component edit is not supported in current controlled editor: {comp_type}")
            return result

        current_view_ir = self._build_view_ir_from_component(comp)

        resolved_view_ir = current_view_ir
        warnings: List[str] = []

        if current_view_ir is not None:
            transform_result = self.view_transformer.transform(
                query=payload.query or "",
                current_ir=current_view_ir
            )
            try:
                resolved_view_ir = ViewIR(**(transform_result.get("view_ir") or {}))
            except Exception:
                resolved_view_ir = current_view_ir

            warnings = transform_result.get("warnings") or []

        edited_comp = comp
        if resolved_view_ir is not None:
            edited_comp = self._apply_view_ir_to_component(
                comp,
                resolved_view_ir,
                user_query=payload.query or ""
            )

        new_func_code = await self._regenerate_component_function(
            query=payload.query or "",
            component=edited_comp,
            summaries=summaries,
            current_view_ir=current_view_ir,
            resolved_view_ir=resolved_view_ir
        )

        if not new_func_code:
            logger.warning("[VizEditor] Failed to regenerate target component function.")
            return result

        component_functions = parsed["component_functions"].copy()
        target_id = payload.active_component_id

        if target_id not in component_functions:
            logger.warning(f"[VizEditor] Target component function not found in dashboard script: {target_id}")
            return result

        updated_functions = {}
        for comp_id, info in component_functions.items():
            if comp_id == target_id:
                updated_functions[comp_id] = {
                    "component_id": comp_id,
                    "func_name": safe_func_name(comp_id),
                    "code": new_func_code
                }
            else:
                updated_functions[comp_id] = info

        new_code = assemble_dashboard_script(parsed, updated_functions)
        updated_layout = self._replace_component_in_layout(last_layout, edited_comp)

        result["code"] = new_code
        result["updated_layout"] = updated_layout
        result["warnings"] = warnings
        if resolved_view_ir is not None:
            result["view_ir"] = resolved_view_ir.model_dump()

        return result

    # =========================================================
    # 主入口
    # =========================================================
    async def edit_dashboard_code(
        self,
        original_code: str,
        payload: Any,
        summaries: List[Dict[str, Any]],
        last_layout: Optional[Dict[str, Any]] = None,
        active_component: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        总入口（受控编辑增强版）

        返回：
        {
            "code": str,                  # 完整 dashboard script
            "updated_layout": dict|None,  # 更新后的 layout（用于 workflow 持久化）
            "updated_component_id": str|None,
            "view_ir": dict,
            "warnings": list[str]
        }

        兼容原则：
        - structural / unknown -> 返回原代码与原 layout
        - component_nl_edit -> 返回更新后的完整 code + layout
        """
        fallback_result = {
            "code": original_code,
            "updated_layout": last_layout,
            "updated_component_id": getattr(payload, "active_component_id", None),
            "view_ir": {},
            "warnings": []
        }

        try:
            parsed = parse_dashboard_script(original_code)
        except Exception as e:
            logger.error(f"[VizEditor] Failed to parse dashboard script: {e}")
            return fallback_result

        try:
            intent = self._classify_edit_intent(
                payload=payload,
                query=payload.query or "",
                active_component=active_component
            )
            kind = intent.get("kind", "unknown")

            logger.info(f"[VizEditor] Edit intent classified as: {kind}")

            if kind == "component_nl_edit":
                result = await self._handle_component_nl_edit(
                    parsed=parsed,
                    payload=payload,
                    summaries=summaries,
                    last_layout=last_layout,
                    active_component=active_component
                )

                if result.get("code"):
                    return result
                return fallback_result

            if kind == "structural_edit":
                logger.warning(
                    "[VizEditor] Structural edit detected but unsupported in current editor. Returning original code."
                )
                return fallback_result

            logger.warning("[VizEditor] Unknown or unsupported edit type. Returning original code.")
            return fallback_result

        except Exception as e:
            logger.error(f"[VizEditor] Edit failed: {e}", exc_info=True)
            return fallback_result