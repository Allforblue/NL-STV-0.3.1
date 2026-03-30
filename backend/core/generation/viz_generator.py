# backend/core/generation/viz_generator.py

import logging
import re
import json
import asyncio
import textwrap
from typing import Dict, Any, List, Tuple, Optional

from core.llm.AI_client import AIClient
from core.generation.scaffold import STChartScaffold

logger = logging.getLogger(__name__)


class CodeGenerator:
    """
    代码生成器（长期 RenderMeta 兼容版）

    核心目标：
    1. 强化 LLM 对 stv / ops SDK 的遵循
    2. 减少原生 plotly / matplotlib 幻觉代码
    3. 在执行前用静态规则拦截明显违规代码
    4. 与“地图统一返回协议对象、图表返回 Figure”的 SDK 约定对齐
    5. 强制遵守 planner 锁定的 primary_metric / primary_dimension
    6. 兼容输出 component_render_meta，为前端点击与 workflow 状态化提供稳定语义
    """

    def __init__(self, llm_client: AIClient):
        self.llm = llm_client
        self.scaffold = STChartScaffold()

    # =========================================================
    # 基础清洗 / 解析工具
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
            text = re.sub(r"^```(python)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)

        if "def get_dashboard_data" in text and "def get_" in text:
            lines = text.split('\n')
            inner_func_lines = []
            capture = False
            for line in lines:
                if line.strip().startswith('def get_') and not line.strip().startswith('def get_dashboard_data'):
                    capture = True
                if capture:
                    if line.startswith('    '):
                        inner_func_lines.append(line[4:])
                    else:
                        inner_func_lines.append(line)
            if inner_func_lines:
                return '\n'.join(inner_func_lines)

        return text.strip()

    def _build_context_str(self, summaries: List[Dict[str, Any]]) -> str:
        context_str = ""
        for s in summaries:
            var_name = s.get('variable_name')
            col_stats = s.get('column_stats') or s.get('basic_stats', {}).get('column_stats', {})
            sem_analysis = s.get('semantic_analysis', {})
            col_meta = sem_analysis.get('column_metadata', {})

            col_desc_list = [
                f"{col}({info.get('dtype', 'unknown')})"
                for col, info in col_stats.items()
            ] if col_stats else ["(无列信息)"]

            semantic_hints = {
                col: meta.get('semantic_tag')
                for col, meta in col_meta.items()
                if isinstance(meta, dict) and meta.get('semantic_tag')
            }

            context_str += f"- 变量 `{var_name}`:\n"
            context_str += f"  - 列信息: {', '.join(col_desc_list[:50])}\n"
            if semantic_hints:
                context_str += f"  - 语义标签 (CRITICAL FOR MAPPING): {json.dumps(semantic_hints, ensure_ascii=False)}\n"
            context_str += "\n"
        return context_str

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

    # =========================================================
    # Planner 约束抽取
    # =========================================================
    def _extract_component_constraints(self, comp: Any) -> Dict[str, Any]:
        is_dict = isinstance(comp, dict)

        c_id = comp.get('id') if is_dict else getattr(comp, 'id', 'unknown')
        c_type = comp.get('type') if is_dict else getattr(comp, 'type', 'unknown')
        c_type_str = str(c_type).split('.')[-1].lower()

        constraints = {
            "component_id": c_id,
            "component_type": c_type_str,
            "locked_metric": None,
            "locked_dimension": None,
            "locked_chart_type": None,
            "locked_map_layer_type": None,
            "locked_is_animated": None
        }

        if c_type_str == 'chart':
            conf = comp.get('chart_config', {}) if is_dict else getattr(comp, 'chart_config', {})
            if hasattr(conf, "model_dump"):
                conf = conf.model_dump()

            constraints["locked_metric"] = conf.get("target_metric")
            constraints["locked_dimension"] = conf.get("target_dimension")
            constraints["locked_chart_type"] = conf.get("chart_type")

        elif c_type_str == 'map':
            m_conf = comp.get('map_config', []) if is_dict else getattr(comp, 'map_config', [])
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
            "Your job is to IMPLEMENT the component, not to redesign the analytical target.",
        ])

        if locked_metric and str(locked_metric).lower() not in ["auto", "none", "null", "unknown"]:
            lines.append(f"CRITICAL: You MUST use `{locked_metric}` as the metric for this component.")
        if locked_dimension and str(locked_dimension).lower() not in ["auto", "none", "null", "unknown"]:
            lines.append(f"CRITICAL: You MUST use `{locked_dimension}` as the main dimension/key for this component where applicable.")

        if ctype == "map":
            lines.extend([
                "For map components, do NOT switch from the locked metric to a different continuous metric just because maps often use revenue/amount.",
                "If the locked metric is trip_count/order_count/count-like, the map MUST still visualize that locked metric."
            ])

        return "\n".join(lines)

    # =========================================================
    # Render Meta 推断（保守版）
    # =========================================================
    def _infer_display_dimension_from_locked_dimension(self, locked_dimension: Optional[str]) -> Optional[str]:
        if not locked_dimension:
            return None

        dim = str(locked_dimension).strip()
        lower_dim = dim.lower()

        # 这是保守推断，不是最终语义解析器
        if lower_dim in ["pulocationid", "dolocationid", "locationid"]:
            return "Zone"
        return dim

    def _infer_fact_link_field(self, locked_dimension: Optional[str], component_type: str) -> Optional[str]:
        if not locked_dimension:
            return None

        dim = str(locked_dimension).strip()
        lower_dim = dim.lower()

        if component_type == "chart":
            if lower_dim in ["pulocationid", "dolocationid", "locationid"]:
                return dim
            return dim

        if component_type == "map":
            if lower_dim in ["pulocationid", "dolocationid", "locationid"]:
                return dim
            return dim

        return dim

    def _infer_render_meta_from_component_plan(self, comp: Any) -> Dict[str, Any]:
        """
        基于 planner component plan 的保守推断。
        当 LLM 暂时还没有返回 render_meta 时，至少保证组件配置不完全失真。
        """
        constraints = self._extract_component_constraints(comp)
        ctype = constraints.get("component_type", "unknown")
        locked_metric = constraints.get("locked_metric")
        locked_dimension = constraints.get("locked_dimension")

        display_dimension = self._infer_display_dimension_from_locked_dimension(locked_dimension)
        fact_link_field = self._infer_fact_link_field(locked_dimension, ctype)

        meta = {
            "display_dimension": display_dimension,
            "display_metric": locked_metric,
            "interaction_field": display_dimension or locked_dimension,
            "interaction_key_field": None,
            "fact_link_field": fact_link_field
        }

        if ctype == "map":
            meta["interaction_field"] = display_dimension or locked_dimension
            meta["fact_link_field"] = fact_link_field

        return meta

    def _extract_render_meta_from_code(self, code: str, fallback_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        轻量级静态提取器（保守增强版）：
        尝试从代码中提取最终展示字段，提取失败时回退到 fallback_meta。
        """
        meta = dict(fallback_meta or {})

        try:
            # stv.bar(df=..., x_val='Zone', y_cat='trip_count')
            bar_match = re.search(
                r"stv\.bar\(\s*.*?x_val\s*=\s*['\"]([^'\"]+)['\"].*?y_cat\s*=\s*['\"]([^'\"]+)['\"]",
                code,
                flags=re.DOTALL
            )
            if bar_match:
                meta["display_dimension"] = bar_match.group(1)
                meta["display_metric"] = bar_match.group(2)
                meta["interaction_field"] = bar_match.group(1)
                return meta

            # stv.pie(df=..., names='Borough', values='trip_count')
            pie_match = re.search(
                r"stv\.pie\(\s*.*?names\s*=\s*['\"]([^'\"]+)['\"].*?values\s*=\s*['\"]([^'\"]+)['\"]",
                code,
                flags=re.DOTALL
            )
            if pie_match:
                meta["display_dimension"] = pie_match.group(1)
                meta["display_metric"] = pie_match.group(2)
                meta["interaction_field"] = pie_match.group(1)
                return meta

            # stv.line(df=..., time_col='pickup_time', val_col='trip_count')
            line_match = re.search(
                r"stv\.line\(\s*.*?time_col\s*=\s*['\"]([^'\"]+)['\"].*?val_col\s*=\s*['\"]([^'\"]+)['\"]",
                code,
                flags=re.DOTALL
            )
            if line_match:
                meta["display_dimension"] = line_match.group(1)
                meta["display_metric"] = line_match.group(2)
                meta["interaction_field"] = line_match.group(1)
                return meta

            # stv.choropleth / animated_map
            map_match = re.search(
                r"stv\.(?:choropleth|animated_map)\(\s*.*?data_key\s*=\s*['\"]([^'\"]+)['\"].*?val_col\s*=\s*['\"]([^'\"]+)['\"]",
                code,
                flags=re.DOTALL
            )
            if map_match:
                meta["interaction_field"] = map_match.group(1)
                meta["fact_link_field"] = map_match.group(1)
                meta["display_metric"] = map_match.group(2)
                return meta

        except Exception:
            pass

        return meta

    # =========================================================
    # 修复重试
    # =========================================================
    async def _retry_on_validation_failure(
        self,
        system_prompt: str,
        original_code: str,
        validation_error: str,
        safe_func_name: str,
        locked_constraints_text: str = ""
    ) -> str:
        retry_prompt = f"""
Your previous code was rejected by the validator.

=== VALIDATION ERROR ===
{validation_error}

=== REJECTED CODE ===
{original_code}

{locked_constraints_text}

Rewrite the function using ONLY the pre-injected `stv` and `ops` SDK objects.

Rules:
1. Keep the exact function signature: `def {safe_func_name}(data_context):`
2. Do NOT import any visualization library
3. Do NOT use raw Plotly / Matplotlib / Seaborn
4. Use Pandas/ops for data prep, then stv for rendering
5. If you use a map method, return the protocol Dict directly
6. If you use a chart method, return the Figure directly
7. You MUST respect the planner-locked metric and dimension

Return ONLY the fixed Python code block.
"""
        raw_response = await self.llm.chat_async([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": retry_prompt}
        ], json_mode=False)

        return self._clean_markdown(raw_response)

    # =========================================================
    # 单组件生成
    # =========================================================
    async def _generate_single_component(
        self,
        comp: Any,
        query: str,
        context_str: str,
        available_vars: List[str],
        time_bounds_hint: str,
        frame_format_hint: str,
        interaction_hint: str
    ) -> Dict[str, Any]:
        is_dict = isinstance(comp, dict)
        c_id = comp.get('id') if is_dict else getattr(comp, 'id', 'unknown')
        safe_func_name = f"get_{c_id.replace('-', '_')}"

        system_prompt = self.scaffold.get_system_prompt(context_str, [comp])

        component_constraints = self._extract_component_constraints(comp)
        locked_constraints_text = self._build_locked_constraints_text(component_constraints)

        user_prompt = f"""
User Query: "{query}"

Available Variables:
{json.dumps(available_vars, ensure_ascii=False)}

Temporal Constraints:
- Range: {time_bounds_hint}
- Frame Format Hint: {frame_format_hint}

Interaction Hint:
{interaction_hint if interaction_hint else "None"}

{locked_constraints_text}

Your job:
Generate ONLY the function `{safe_func_name}(data_context)` for the current component.
Use the Metadata Context to map exact column names.
Do not guess missing fields.
Do not redesign the metric/dimension target chosen by the planner.
If the component is not animated, do not hallucinate animation logic.
If the component is animated, use real timestamps, not dt.hour/dt.day.
If the component is a map, return the stv map protocol Dict directly.
If the component is a chart, return the chart Figure directly.

Special rule for count-based line charts:
- If the chart type is line and the locked metric is count-like (trip_count, order_count, count, event_count),
  you MUST first create a helper numeric count column.
- Example:
  df = data_context.get('df_name')[['time_col']].copy()
  df['trip_count'] = 1
  df_ts = ops.safe_resample(df, time_col='time_col', val_col='trip_count', freq='1h', agg_func='sum')
  return stv.line(df_ts, time_col='time_col', val_col='trip_count')
- NEVER use:
  ops.safe_resample(..., val_col=None, agg_func='sum')

Return ONLY Python code.
"""

        logger.info(f"  -> Concurrently generating SDK-based code for: {c_id}")
        raw_response = await self.llm.chat_async([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], json_mode=False)

        code = self._clean_markdown(raw_response)

        is_valid, validation_error = self._validate_generated_code(code)
        if not is_valid:
            logger.warning(f"  -> Validation failed for {c_id}: {validation_error}")
            code = await self._retry_on_validation_failure(
                system_prompt=system_prompt,
                original_code=code,
                validation_error=validation_error,
                safe_func_name=safe_func_name,
                locked_constraints_text=locked_constraints_text
            )

            is_valid_retry, validation_error_retry = self._validate_generated_code(code)
            if not is_valid_retry:
                logger.warning(f"  -> Retry validation still failed for {c_id}: {validation_error_retry}")

        # Render Meta：先从 component plan 保守推断，再尝试用代码静态增强
        fallback_meta = self._infer_render_meta_from_component_plan(comp)
        render_meta = self._extract_render_meta_from_code(code, fallback_meta)

        return {
            "id": c_id,
            "func_name": safe_func_name,
            "code": code,
            "render_meta": render_meta
        }

    # =========================================================
    # Dashboard 代码生成
    # =========================================================
    async def generate_dashboard_code(
        self,
        query: str,
        summaries: List[Dict[str, Any]],
        component_plans: List[Any],
        interaction_hint: str = ""
    ) -> Dict[str, Any]:
        context_str = self._build_context_str(summaries)
        available_vars = [s.get('variable_name') for s in summaries]

        frame_format_hint = "%H:00"
        time_bounds_hint = "No specific bounds"

        target_components = []
        insight_components = []

        if component_plans:
            for comp in component_plans:
                is_dict = isinstance(comp, dict)
                c_type = comp.get('type') if is_dict else getattr(comp, 'type', 'unknown')
                c_type_str = str(c_type).split('.')[-1].lower()
                c_id = comp.get('id') if is_dict else getattr(comp, 'id', 'unknown')

                if c_type_str == 'timeline_controller':
                    t_conf = comp.get('timeline_config', {}) if is_dict else getattr(comp, 'timeline_config', {})
                    if t_conf:
                        start = t_conf.get('start_time') if isinstance(t_conf, dict) else getattr(t_conf, 'start_time', '')
                        end = t_conf.get('end_time') if isinstance(t_conf, dict) else getattr(t_conf, 'end_time', '')
                        fmt = t_conf.get('frame_format') if isinstance(t_conf, dict) else getattr(t_conf, 'frame_format', '')

                        if start and end:
                            time_bounds_hint = f"FROM {start} TO {end}"
                        if fmt:
                            frame_format_hint = fmt

                elif c_type_str == 'insight':
                    insight_components.append(c_id)
                else:
                    target_components.append(comp)

        logger.info(f"Firing up {len(target_components)} parallel generation tasks...")

        tasks = [
            self._generate_single_component(
                comp=comp,
                query=query,
                context_str=context_str,
                available_vars=available_vars,
                time_bounds_hint=time_bounds_hint,
                frame_format_hint=frame_format_hint,
                interaction_hint=interaction_hint
            )
            for comp in target_components
        ]

        generated_results = await asyncio.gather(*tasks)

        master_script = "import pandas as pd\nimport numpy as np\n\n"
        master_script += "def get_dashboard_data(data_context):\n"
        master_script += "    dashboard_results = {}\n\n"

        component_render_meta: Dict[str, Any] = {}

        for res in generated_results:
            indented_code = textwrap.indent(res['code'], '    ')
            master_script += indented_code + "\n\n"

            component_render_meta[res["id"]] = res.get("render_meta", {}) or {}

        for cid in insight_components:
            safe_name = f"get_{cid.replace('-', '_')}"
            insight_code = f"def {safe_name}(data_context):\n    return None"
            master_script += textwrap.indent(insight_code, '    ') + "\n\n"

        master_script += "    # ================= Execution Block =================\n"

        for res in generated_results:
            comp_id = res['id']
            func_name = res['func_name']
            master_script += f"    try:\n"
            master_script += f"        dashboard_results['{comp_id}'] = {func_name}(data_context)\n"
            master_script += f"    except Exception as e:\n"
            master_script += f"        print(f'Error generating {comp_id}: {{str(e)}}')\n"
            master_script += f"        dashboard_results['{comp_id}'] = None\n\n"

        for cid in insight_components:
            safe_name = f"get_{cid.replace('-', '_')}"
            master_script += f"    try:\n"
            master_script += f"        dashboard_results['{cid}'] = {safe_name}(data_context)\n"
            master_script += f"    except Exception as e:\n"
            master_script += f"        print(f'Error generating {cid}: {{str(e)}}')\n"
            master_script += f"        dashboard_results['{cid}'] = None\n\n"

        master_script += "    return dashboard_results\n"

        logger.info("Parallel assembly complete. Returning unified execution script with sandbox isolation.")

        return {
            "code": master_script,
            "component_render_meta": component_render_meta
        }

    # =========================================================
    # 自愈修复
    # =========================================================
    async def fix_code(
        self,
        original_code: str,
        error_trace: str,
        summaries: List[Dict[str, Any]],
        component_plans: List[Any] = None
    ) -> str:
        context_str = self._build_context_str(summaries)
        available_vars = [s.get('variable_name') for s in summaries]

        component_plans = component_plans or []
        base_prompt = self.scaffold.get_system_prompt(context_str, component_plans)

        constraint_blocks = []
        for comp in component_plans:
            try:
                cons = self._extract_component_constraints(comp)
                constraint_blocks.append(self._build_locked_constraints_text(cons))
            except Exception:
                pass

        all_locked_constraints = "\n\n".join(constraint_blocks) if constraint_blocks else ""

        fix_prompt = f"""
CODE EXECUTION FAILED!

=== ERROR TRACEBACK ===
{error_trace}

=== ORIGINAL CODE ===
{original_code}

=== AVAILABLE VARIABLES ===
{json.dumps(available_vars, ensure_ascii=False)}

{all_locked_constraints}

=== SDK FIXING DIAGNOSTIC ===
1. If you see `[SDK Error]`:
   - You likely passed the WRONG column name or wrong parameter to `stv` or `ops`.
   - Read the traceback carefully and replace the column argument with an exact existing column name.

2. If you see `KeyError`:
   - You misspelled a DataFrame column or used the wrong merge/groupby key.
   - Check the Metadata Context for exact case-sensitive column names.

3. If you see merge/join issues:
   - Use `ops.safe_merge(...)` instead of fragile manual merge logic when possible.

4. If you see datetime/time animation issues:
   - Do NOT use `dt.hour`, `dt.day`, or other discrete extracted parts for animation time.
   - Use real timestamps, e.g. `pd.to_datetime(...).dt.floor('h')`.

5. If the component is a chart:
   - Ensure the DataFrame is aggregated before calling `stv.bar`, `stv.pie`, or `stv.line`.

6. If the component is a map:
   - Continue using `stv` map methods
   - Return the protocol Dict directly
   - Do NOT rewrite the map as raw Plotly code or custom frontend layer code

7. PLANNER LOCK RULE:
   - You MUST preserve the planner-selected metric and dimension.
   - Do NOT silently replace trip_count with total_amount, fare_amount, revenue, or any other metric.

8. DO NOT rewrite SDK internals:
   - Continue using `stv` and `ops`
   - ONLY fix your column names, arguments, aggregation logic, or Pandas prep

=== OUTPUT REQUIREMENT ===
Return the COMPLETE fixed function code only.
Do not explain anything.
Do not use raw Plotly/Matplotlib/Seaborn.
"""

        logger.warning("Attempting SDK-aware self-healing fix...")
        raw_response = await self.llm.chat_async([
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": fix_prompt}
        ], json_mode=False)

        fixed_code = self._clean_markdown(raw_response)

        is_valid, validation_error = self._validate_generated_code(fixed_code)
        if not is_valid:
            logger.warning(f"Fixed code still violates validation: {validation_error}")

        return fixed_code