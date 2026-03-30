import logging
import traceback
import asyncio
import orjson
from typing import List, Dict, Any, Tuple
from datetime import date, datetime

# --- 核心模块导入 ---
from core.llm.AI_client import AIClient
from core.profiler.semantic_analyzer import SemanticAnalyzer
from core.profiler.relation_mapper import RelationMapper
from core.profiler.interaction_mapper import InteractionMapper
from core.generation.dashboard_planner import DashboardPlanner
from core.generation.viz_generator import CodeGenerator
from core.generation.viz_editor import VizEditor
from core.execution.executor import CodeExecutor
from core.execution.insight_extractor import InsightExtractor

from core.interaction_solver.interaction_router import InteractionRouter
from core.interaction_solver.selection_resolver import SelectionResolver
from core.interaction_solver.selection_applier import SelectionApplier

# --- 协议与 Schema 导入 ---
from core.schemas.dashboard import DashboardSchema, ComponentType, InsightCard
from core.schemas.interaction import InteractionPayload, InteractionTriggerType

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AnalysisWorkflow:
    """
    全链路指挥官（Interaction Solver 拆分版）

    主要能力：
    1. 数据画像与语义增强
    2. 交互锚点识别
    3. Generate / Selection Edit / Component Edit 三路分流
    4. 看板规划、代码生成与执行
    5. 将完整组件 metadata 传入 executor，提升执行摘要质量
    6. 洞察生成与状态快照固化
    7. Edit Mode 失败时安全降级为 Generate
    8. 支持 interaction_state / component_render_meta 的读写
    """

    def __init__(self, llm_client: AIClient):
        self.llm = llm_client
        self.analyzer = SemanticAnalyzer(llm_client)
        self.relation_mapper = RelationMapper(llm_client)
        self.interaction_mapper = InteractionMapper(llm_client)
        self.planner = DashboardPlanner(llm_client)
        self.generator = CodeGenerator(llm_client)
        self.editor = VizEditor(llm_client)
        self.executor = CodeExecutor()
        self.insight_extractor = InsightExtractor(llm_client)

        self.interaction_router = InteractionRouter()
        self.selection_resolver = SelectionResolver()
        self.selection_applier = SelectionApplier()

    # =========================================================
    # 模式判定辅助函数
    # =========================================================
    def _looks_like_edit_query(self, query: str) -> bool:
        if not query:
            return False

        q = query.strip().lower()

        edit_keywords = [
            "修改", "改成", "改为", "改一下", "调整", "优化", "放大", "缩小",
            "换成", "切换", "增加", "新增", "删除", "移除", "保留", "联动",
            "筛选", "高亮", "隐藏", "显示", "标题改", "颜色改", "把这个图",
            "把这张图", "当前图", "现有图", "看板里", "右侧图", "地图上",
            "change", "edit", "modify", "update", "switch", "replace", "adjust",
            "filter", "highlight", "rename", "retitle"
        ]

        generate_keywords = [
            "重新生成", "重新规划", "重新分析", "新建", "生成一个", "做一个",
            "帮我分析", "请分析", "哪个区域", "哪一天", "什么时间", "趋势",
            "分布", "最多", "最少", "比较", "找出", "分析一下",
            "regenerate", "generate", "analyze", "show me", "which", "what", "find"
        ]

        edit_score = sum(1 for kw in edit_keywords if kw in q)
        gen_score = sum(1 for kw in generate_keywords if kw in q)

        if gen_score >= 1 and edit_score == 0:
            return False
        if edit_score >= 1 and gen_score == 0:
            return True
        if gen_score >= edit_score:
            return False

        return True

    def _looks_like_structural_edit_query(self, query: str) -> bool:
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

        return any(kw in q for kw in structural_keywords)

    def _has_dashboard_wrapper(self, code: str) -> bool:
        if not code:
            return False
        return "def get_dashboard_data" in code

    async def _decide_workflow_mode(self, payload: InteractionPayload, last_state: dict) -> bool:
        # UI 操作一定优先视为编辑类流转
        if payload.trigger_type == InteractionTriggerType.UI_ACTION:
            logger.info(">>> [Mode Decision] 判定为 Edit: 触发源为 UI 交互")
            return True

        if not last_state or not last_state.get("last_code"):
            logger.info(">>> [Mode Decision] 判定为 Generate: 无历史代码上下文")
            return False

        force_mode = getattr(payload, "force_mode", "auto")
        if force_mode == "edit":
            logger.info(">>> [Mode Decision] 判定为 Edit: 用户强制要求编辑模式")
            return True
        elif force_mode == "generate" or payload.force_new:
            logger.info(">>> [Mode Decision] 判定为 Generate: 用户强制要求生成模式")
            return False

        if payload.trigger_type == InteractionTriggerType.SYSTEM and not payload.query:
            logger.info(">>> [Mode Decision] 判定为 Generate: 系统自动初始化看板")
            return False

        query = payload.query or ""
        rule_edit = self._looks_like_edit_query(query)

        if self._looks_like_structural_edit_query(query):
            logger.info(">>> [Mode Decision] 判定为 Generate: 检测到结构性编辑需求")
            return False

        if not rule_edit:
            logger.info(">>> [Mode Decision] 规则判定为 Generate: 查询更像全新分析问题")
            return False

        logger.info(">>> [Mode Decision] 启动 LLM 意图识别，判定用户 Query 倾向...")
        prompt = f"""
当前系统已经生成了一个数据可视化看板。用户输入了一个新的自然语言需求。
请你判断用户是希望：
1. "edit" (修改/增量): 在当前图表基础上微调（例如：改颜色、换图表、增减筛选、修改标题、在侧边加个图等）。
2. "generate" (重建): 完全推翻当前分析方向，生成一个截然不同的新主题看板。

用户的新需求: "{query}"

请严格输出 JSON 格式，包含字段 "mode"，值为 "edit" 或 "generate"。
"""
        try:
            decision = await self.llm.query_json_async(
                prompt=prompt,
                system_prompt="You are a strict JSON-output Intent Classifier."
            )
            mode = decision.get("mode", "edit").lower()
            logger.info(f">>> [Mode Decision] LLM 意图识别结果: {mode}")
            return mode == "edit"
        except Exception as e:
            logger.warning(f"意图识别失败，降级为规则结果: {e}")
            return rule_edit

    # =========================================================
    # 路由判定
    # =========================================================
    def _fallback_edit_route(self, payload: InteractionPayload) -> str:
        """
        当 interaction_router 不可用或返回异常时的兜底路由：
        - UI_ACTION -> selection_interaction
        - NL + active_component_id -> component_visual_edit
        - structural -> structural_edit
        - 其它 -> unknown
        """
        if payload.trigger_type == InteractionTriggerType.UI_ACTION:
            return "selection_interaction"

        if payload.trigger_type == InteractionTriggerType.NATURAL_LANGUAGE:
            if self._looks_like_structural_edit_query(payload.query or ""):
                return "structural_edit"
            if payload.active_component_id:
                return "component_visual_edit"

        return "unknown"

    # =========================================================
    # Render Meta / Interaction State 辅助
    # =========================================================
    def _normalize_generator_output(self, gen_output: Any) -> Tuple[str, Dict[str, Any]]:
        if isinstance(gen_output, str):
            return gen_output, {}

        if isinstance(gen_output, dict):
            code = gen_output.get("code", "") or ""
            render_meta = gen_output.get("component_render_meta", {}) or {}
            return code, render_meta

        return str(gen_output or ""), {}

    def _merge_component_render_meta(
        self,
        dashboard_plan: DashboardSchema,
        render_meta_map: Dict[str, Any]
    ) -> DashboardSchema:
        if not dashboard_plan or not render_meta_map:
            return dashboard_plan

        for comp in dashboard_plan.components:
            meta = render_meta_map.get(comp.id)
            if not meta:
                continue

            if comp.type == ComponentType.CHART and comp.chart_config is not None:
                if meta.get("display_dimension") is not None:
                    comp.chart_config.display_dimension = meta.get("display_dimension")
                    if not comp.chart_config.x_axis:
                        comp.chart_config.x_axis = meta.get("display_dimension")

                if meta.get("display_metric") is not None:
                    comp.chart_config.display_metric = meta.get("display_metric")

                if meta.get("interaction_field") is not None:
                    comp.chart_config.interaction_field = meta.get("interaction_field")

                if meta.get("interaction_key_field") is not None:
                    comp.chart_config.interaction_key_field = meta.get("interaction_key_field")

                if meta.get("fact_link_field") is not None:
                    comp.chart_config.fact_link_field = meta.get("fact_link_field")

                if getattr(comp.chart_config, "render_meta", None) is not None:
                    comp.chart_config.render_meta.update(meta)

            elif comp.type == ComponentType.MAP and comp.map_config:
                for layer in comp.map_config:
                    if meta.get("display_dimension") is not None:
                        layer.display_dimension = meta.get("display_dimension")
                    if meta.get("display_metric") is not None:
                        layer.display_metric = meta.get("display_metric")
                    if meta.get("interaction_field") is not None:
                        layer.interaction_field = meta.get("interaction_field")
                    if meta.get("interaction_key_field") is not None:
                        layer.interaction_key_field = meta.get("interaction_key_field")
                    if meta.get("fact_link_field") is not None:
                        layer.fact_link_field = meta.get("fact_link_field")
                    if getattr(layer, "render_meta", None) is not None:
                        layer.render_meta.update(meta)

        return dashboard_plan

    def _resolve_selection_mode_for_fallback(self, payload: InteractionPayload) -> str:
        mode = getattr(payload, "selection_mode", None)

        if not mode and payload.extra_params and isinstance(payload.extra_params, dict):
            mode = payload.extra_params.get("selection_mode")

        if not mode and payload.extra_params and isinstance(payload.extra_params, dict):
            selection_context = payload.extra_params.get("selection_context", {}) or {}
            mode = selection_context.get("selection_mode")

        mode = str(mode or "replace").strip().lower()

        if mode not in {"replace", "append", "compare_left", "compare_right"}:
            mode = "replace"

        return mode

    def _set_selection_active_flags(
        self,
        selections: List[Dict[str, Any]],
        active_selection_ids: List[str]
    ) -> None:
        active_id_set = set(active_selection_ids or [])
        for item in selections:
            if not isinstance(item, dict):
                continue
            item["active"] = item.get("selection_id") in active_id_set

    def _build_interaction_state_from_payload(
        self,
        payload: InteractionPayload,
        existing_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        兼容兜底版 interaction_state 更新器。
        若 selection_resolver 已经返回了更完整的 interaction_state，
        本函数只作为 fallback。

        升级点：
        - 支持 selection_mode:
          replace / append / compare_left / compare_right
        - 正确维护 active_selection_ids / comparison_state
        """
        state = dict(existing_state or {})

        state["active_component_id"] = payload.active_component_id
        state["last_interaction_payload"] = (
            payload.model_dump() if hasattr(payload, "model_dump") else {}
        )

        if payload.time_range:
            state["global_time_range"] = payload.time_range

        state.setdefault("selections", [])
        state.setdefault("active_selection_ids", [])
        state.setdefault("comparison_state", {
            "enabled": False,
            "baseline_mode": "global",
            "left_selection_ids": [],
            "right_selection_ids": [],
            "compare_metric": None,
            "compare_dimension": None,
            "compare_question": None
        })
        state.setdefault("notes", {})

        if payload.trigger_type == InteractionTriggerType.UI_ACTION:
            selection_context = {}
            if payload.extra_params and isinstance(payload.extra_params, dict):
                selection_context = payload.extra_params.get("selection_context", {}) or {}

            selection_type = None
            raw_payload = {}
            label = None

            if payload.bbox:
                selection_type = "bbox"
                raw_payload = {"bbox": payload.bbox}
                label = selection_context.get("label", "bbox_selection")

            elif payload.selected_ids:
                selection_type = "selected_ids"
                raw_payload = {"selected_ids": payload.selected_ids}
                label = selection_context.get("label", f"id_selection_{len(payload.selected_ids)}")

            elif payload.selected_values:
                selection_type = "selected_values"
                raw_payload = {"selected_values": payload.selected_values}
                try:
                    first_val = next(iter(payload.selected_values.values()))
                    label = selection_context.get("label", str(first_val))
                except Exception:
                    label = selection_context.get("label", "value_selection")

            elif payload.time_range:
                selection_type = "time"
                raw_payload = {"time_range": payload.time_range}
                label = selection_context.get("label", "time_range")

            if selection_type:
                selection_mode = self._resolve_selection_mode_for_fallback(payload)

                scope = selection_context.get("selection_scope", "global")
                if selection_mode == "compare_left":
                    scope = "compare_left"
                elif selection_mode == "compare_right":
                    scope = "compare_right"

                selection_id = f"sel_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                selection_item = {
                    "selection_id": selection_id,
                    "source_component_id": payload.active_component_id,
                    "selection_type": selection_type,
                    "scope": scope,
                    "label": label,
                    "active": True,
                    "raw_payload": raw_payload,
                    "resolved_filters": [],
                    "selection_context": selection_context
                }

                selections = state.get("selections", [])
                active_selection_ids = list(state.get("active_selection_ids", []) or [])
                comparison_state = dict(state.get("comparison_state", {}) or {})

                comparison_state.setdefault("enabled", False)
                comparison_state.setdefault("baseline_mode", "global")
                comparison_state.setdefault("left_selection_ids", [])
                comparison_state.setdefault("right_selection_ids", [])
                comparison_state.setdefault("compare_metric", None)
                comparison_state.setdefault("compare_dimension", None)
                comparison_state.setdefault("compare_question", None)

                selections.append(selection_item)

                if selection_mode == "append":
                    if selection_id not in active_selection_ids:
                        active_selection_ids.append(selection_id)

                    comparison_state["enabled"] = False
                    comparison_state["left_selection_ids"] = []
                    comparison_state["right_selection_ids"] = []

                elif selection_mode == "compare_left":
                    old_left_ids = comparison_state.get("left_selection_ids", []) or []
                    old_left_id_set = set(old_left_ids)

                    for item in selections:
                        if not isinstance(item, dict):
                            continue
                        if item.get("selection_id") in old_left_id_set:
                            item["active"] = False

                    comparison_state["enabled"] = True
                    comparison_state["left_selection_ids"] = [selection_id]
                    comparison_state["right_selection_ids"] = comparison_state.get("right_selection_ids", []) or []

                    active_selection_ids = []
                    for sid in comparison_state["left_selection_ids"] + comparison_state["right_selection_ids"]:
                        if sid and sid not in active_selection_ids:
                            active_selection_ids.append(sid)

                elif selection_mode == "compare_right":
                    old_right_ids = comparison_state.get("right_selection_ids", []) or []
                    old_right_id_set = set(old_right_ids)

                    for item in selections:
                        if not isinstance(item, dict):
                            continue
                        if item.get("selection_id") in old_right_id_set:
                            item["active"] = False

                    comparison_state["enabled"] = True
                    comparison_state["left_selection_ids"] = comparison_state.get("left_selection_ids", []) or []
                    comparison_state["right_selection_ids"] = [selection_id]

                    active_selection_ids = []
                    for sid in comparison_state["left_selection_ids"] + comparison_state["right_selection_ids"]:
                        if sid and sid not in active_selection_ids:
                            active_selection_ids.append(sid)

                else:
                    # replace（默认）
                    active_selection_ids = [selection_id]
                    comparison_state["enabled"] = False
                    comparison_state["left_selection_ids"] = []
                    comparison_state["right_selection_ids"] = []

                state["selections"] = selections
                state["active_selection_ids"] = active_selection_ids
                state["comparison_state"] = comparison_state

                self._set_selection_active_flags(
                    selections=state["selections"],
                    active_selection_ids=state["active_selection_ids"]
                )

        return state

    # =========================================================
    # Generate 逻辑封装
    # =========================================================
    async def _run_generate_mode(
        self,
        payload: InteractionPayload,
        data_summaries: List[Dict[str, Any]],
        interaction_hint: str
    ) -> Tuple[DashboardSchema, str, Dict[str, Any]]:
        logger.info(">>> [Generate Mode] 正在规划全新时空看板...")
        dashboard_plan = await self.planner.plan_dashboard(
            query=f"{payload.query}\n{interaction_hint}",
            summaries=data_summaries
        )

        gen_output = await self.generator.generate_dashboard_code(
            query=payload.query,
            summaries=data_summaries,
            component_plans=dashboard_plan.components,
            interaction_hint=interaction_hint
        )

        current_code, generated_render_meta = self._normalize_generator_output(gen_output)
        return dashboard_plan, current_code, (generated_render_meta or {})

    # =========================================================
    # 主流程
    # =========================================================
    async def execute_step(
        self,
        payload: InteractionPayload,
        data_summaries: List[Dict[str, Any]],
        data_context: Dict[str, Any],
        session_service: Any
    ) -> DashboardSchema:

        # === 0. 历史回溯 ===
        if payload.trigger_type == InteractionTriggerType.BACKTRACK and payload.target_snapshot_id:
            logger.info(f">>> [Backtrack] 正在还原历史快照: {payload.target_snapshot_id}")
            snapshot = session_service.get_snapshot(payload.session_id, payload.target_snapshot_id)
            if snapshot:
                restored_metadata = {
                    "last_code": snapshot.code_snapshot,
                    "last_layout": snapshot.layout_data.model_dump(),
                    "snapshot_id": snapshot.snapshot_id
                }

                try:
                    if getattr(snapshot.layout_data, "metadata", None):
                        restored_metadata.update({
                            k: v for k, v in snapshot.layout_data.metadata.items()
                            if k not in restored_metadata
                        })
                except Exception:
                    pass

                try:
                    if getattr(snapshot, "interaction_state", None) is not None:
                        session_service.update_interaction_state(
                            payload.session_id,
                            snapshot.interaction_state.model_dump()
                            if hasattr(snapshot.interaction_state, "model_dump")
                            else snapshot.interaction_state
                        )
                except Exception:
                    pass

                try:
                    if getattr(snapshot, "component_render_meta", None) is not None:
                        session_service.update_component_render_meta(
                            payload.session_id,
                            snapshot.component_render_meta or {}
                        )
                except Exception:
                    pass

                session_service.update_session_metadata(payload.session_id, restored_metadata)

                logger.info(f">>> [Backtrack] 已同步恢复会话编辑基线 | snapshot={snapshot.snapshot_id}")
                return snapshot.layout_data
            else:
                logger.error("快照不存在，降级为普通分析")

        # === 1. 数据增强与交互映射 ===
        analysis_tasks = []
        task_indices = []

        for i, summary in enumerate(data_summaries):
            sem_analysis = summary.get("semantic_analysis", {})
            if not sem_analysis.get("column_metadata") and "file_info" in summary:
                logger.info(f">>>[Analysis] 调度异步语义画像: {summary['variable_name']}")
                fingerprint = summary.get("basic_stats", {})
                task = self.analyzer.analyze(summary["file_info"].get("path"), fingerprint)
                analysis_tasks.append(task)
                task_indices.append(i)
            else:
                logger.info(f">>> [Skip] 变量 {summary['variable_name']} 已有完整画像")

        if analysis_tasks:
            results = await asyncio.gather(*analysis_tasks)
            for idx, res in zip(task_indices, results):
                data_summaries[idx]["semantic_analysis"] = res.get("semantic_analysis", {})
                data_summaries[idx]["variable_name"] = res.get(
                    "variable_name",
                    data_summaries[idx]["variable_name"]
                )

        session_info = session_service.get_session(payload.session_id)
        if session_info and not session_info.get("is_full_data"):
            logger.info(">>> [Eager Load] 触发后台全量数据预加载 (Non-Blocking)...")

        # 1.2 交互锚点识别（带缓存）
        session_state = session_service.get_session(payload.session_id)
        cached_anchors = session_state.get("cached_interaction_anchors") if session_state else None

        if not cached_anchors:
            logger.info(">>> [Interaction] 计算并缓存多数据集交互锚点...")
            interaction_anchors = await self.interaction_mapper.identify_interaction_anchors(data_summaries)
            if session_state is not None:
                session_state["cached_interaction_anchors"] = interaction_anchors
        else:
            logger.info(">>> [Cache] 命中交互锚点缓存")
            interaction_anchors = cached_anchors

        interaction_hint = self.interaction_mapper.get_planner_hints(interaction_anchors)

        try:
            # === 1.5 interaction_state 更新：优先走 resolver，失败则 fallback ===
            existing_interaction_state = session_service.get_interaction_state(payload.session_id)
            updated_interaction_state = existing_interaction_state

            try:
                resolved_state = self.selection_resolver.resolve(
                    payload=payload,
                    interaction_state=existing_interaction_state,
                    summaries=data_summaries,
                    last_layout=session_state.get("last_workflow_state", {}).get("last_layout")
                    if session_state and session_state.get("last_workflow_state") else None
                )
                if isinstance(resolved_state, dict) and resolved_state.get("interaction_state"):
                    updated_interaction_state = resolved_state["interaction_state"]
                elif isinstance(resolved_state, dict):
                    updated_interaction_state = resolved_state
            except Exception as e:
                logger.warning(f"[Workflow] selection_resolver.resolve 失败，降级使用旧版 interaction_state builder: {e}")
                updated_interaction_state = self._build_interaction_state_from_payload(
                    payload=payload,
                    existing_state=existing_interaction_state
                )

            session_service.update_interaction_state(payload.session_id, updated_interaction_state)

            # === 2. 模式决策 ===
            last_state = session_state.get("last_workflow_state") if session_state else None

            if payload.trigger_type == InteractionTriggerType.SYSTEM and not payload.query:
                logger.info(">>> [Auto-Dashboard] 检测到初始化请求，自动注入探索指令...")
                payload.query = "请详细分析数据的字段分布、地理特征和业务逻辑，并为我自动生成一个最具洞察力的初始看板。"

            is_edit_mode = await self._decide_workflow_mode(payload, last_state)

            if is_edit_mode and (not last_state or not last_state.get("last_layout")):
                logger.warning(">>> [Edit Fallback] 缺少 last_layout，上下文不足，自动降级为 Generate Mode...")
                is_edit_mode = False

            current_code = ""
            dashboard_plan: DashboardSchema = None
            component_render_meta: Dict[str, Any] = session_service.get_component_render_meta(payload.session_id) or {}

            if is_edit_mode:
                logger.info(">>> [Edit Mode] 响应交互动作，正在进入细粒度路由...")
                dashboard_plan = DashboardSchema(**last_state["last_layout"])

                active_comp = next(
                    (c for c in dashboard_plan.components if c.id == payload.active_component_id),
                    None
                )

                # === 2.1 使用 interaction_router 判定具体编辑路径 ===
                try:
                    route_info = self.interaction_router.classify(
                        payload=payload,
                        interaction_state=updated_interaction_state,
                        last_layout=last_state.get("last_layout")
                    )
                    edit_route = route_info.get("kind", "unknown")
                except Exception as e:
                    logger.warning(f"[Workflow] interaction_router.classify 失败，使用 fallback 路由: {e}")
                    edit_route = self._fallback_edit_route(payload)
                    route_info = {"kind": edit_route}

                logger.info(
                    f">>> [Edit Routing] 当前编辑路由: {edit_route} | "
                    f"selection_mode={getattr(payload, 'selection_mode', 'replace')}"
                )

                # -------------------------
                # 路由 1：Selection Interaction
                # -------------------------
                if edit_route == "selection_interaction":
                    try:
                        current_code = self.selection_applier.apply_to_code(
                            original_code=last_state["last_code"],
                            payload=payload,
                            interaction_state=updated_interaction_state,
                            summaries=data_summaries,
                            last_layout=last_state["last_layout"],
                            active_component=active_comp
                        )
                    except Exception as e:
                        logger.warning(f"[Workflow] selection_applier.apply_to_code 失败，降级为 Generate: {e}")
                        is_edit_mode = False

                # -------------------------
                # 路由 2：Component Visual Edit
                # -------------------------
                elif edit_route == "component_visual_edit":
                    if (
                        payload.trigger_type == InteractionTriggerType.NATURAL_LANGUAGE
                        and not payload.active_component_id
                        and payload.query
                    ):
                        logger.warning(
                            ">>> [Edit Fallback] 自然语言编辑缺少 active_component_id，自动降级为 Generate Mode..."
                        )
                        is_edit_mode = False

                    elif (
                        payload.trigger_type == InteractionTriggerType.NATURAL_LANGUAGE
                        and payload.active_component_id
                        and active_comp is None
                    ):
                        logger.warning(
                            f">>> [Edit Fallback] 未找到目标组件 {payload.active_component_id}，自动降级为 Generate Mode..."
                        )
                        is_edit_mode = False

                    else:
                        edit_result = await self.editor.edit_dashboard_code(
                            original_code=last_state["last_code"],
                            payload=payload,
                            summaries=data_summaries,
                            last_layout=last_state["last_layout"],
                            active_component=active_comp
                        )

                        if isinstance(edit_result, dict):
                            current_code = edit_result.get("code") or ""

                            updated_layout = edit_result.get("updated_layout")

                            if updated_layout:
                                try:
                                    dashboard_plan = DashboardSchema(**updated_layout)
                                    logger.info(
                                        f">>> [Edit Mode] 已应用 editor 返回的 updated_layout | component={edit_result.get('updated_component_id')}"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f">>> [Edit Mode] editor 返回的 updated_layout 无法反序列化为 DashboardSchema，保留旧 layout: {e}"
                                    )
                                try:
                                    updated_comp = next(
                                        (c for c in dashboard_plan.components if
                                         c.id == edit_result.get("updated_component_id")),
                                        None
                                    )
                                    if updated_comp and updated_comp.chart_config:
                                        logger.info(
                                            f">>> [Edit Mode] updated view_ir = {updated_comp.chart_config.render_meta.get('view_ir')}"
                                        )
                                except Exception:
                                    pass

                            edit_warnings = edit_result.get("warnings") or []
                            if edit_warnings:
                                logger.warning(f">>> [Edit Mode] editor warnings: {edit_warnings}")

                        else:
                            current_code = edit_result or ""

                # -------------------------
                # 路由 3：结构性编辑 -> Generate
                # -------------------------
                elif edit_route == "structural_edit":
                    logger.info(">>> [Edit Routing] 检测到结构性编辑，交由 Generate Mode 处理")
                    is_edit_mode = False

                # -------------------------
                # Unknown -> Generate
                # -------------------------
                else:
                    logger.warning(">>> [Edit Routing] 未知编辑路由，自动降级为 Generate Mode...")
                    is_edit_mode = False

                if is_edit_mode and not self._has_dashboard_wrapper(current_code):
                    logger.warning(">>> [Edit Fallback] Edit 路径未返回完整 dashboard wrapper，自动降级为 Generate Mode...")
                    is_edit_mode = False

            # === 2.2 Generate / 补偿生成 ===
            if not is_edit_mode:
                dashboard_plan, current_code, component_render_meta = await self._run_generate_mode(
                    payload=payload,
                    data_summaries=data_summaries,
                    interaction_hint=interaction_hint
                )

            if not current_code or not self._has_dashboard_wrapper(current_code):
                logger.info(">>> [Generate Mode] 进入补偿式重新生成...")
                dashboard_plan, current_code, component_render_meta = await self._run_generate_mode(
                    payload=payload,
                    data_summaries=data_summaries,
                    interaction_hint=interaction_hint
                )
                is_edit_mode = False

            # === 2.3 回填 render meta ===
            if component_render_meta:
                dashboard_plan = self._merge_component_render_meta(dashboard_plan, component_render_meta)
                session_service.update_component_render_meta(payload.session_id, component_render_meta)

            # === 2.5 同步全局时间范围与视图状态 ===
            if payload.time_range:
                dashboard_plan.global_time_range = payload.time_range

            if payload.view_state:
                if not getattr(dashboard_plan, "initial_view_state", None):
                    dashboard_plan.initial_view_state = {}
                dashboard_plan.initial_view_state.update(payload.view_state)

            # === 3. 代码执行 ===
            session_service.ensure_full_data_context(payload.session_id)
            full_session = session_service.get_session(payload.session_id)
            actual_data_context = full_session["data_context"]

            comp_ids = [c.id for c in dashboard_plan.components]

            exec_result = self.executor.execute_dashboard_logic(
                code_str=current_code,
                data_context=actual_data_context,
                component_ids=comp_ids,
                summaries=data_summaries,
                component_defs=dashboard_plan.components
            )

            # === 3.1 自愈 / 降级机制 ===
            if not exec_result.success:
                if is_edit_mode:
                    logger.warning(">>> [Edit Fallback] Edit Mode 执行失败，自动降级为 Generate 重新规划...")
                    dashboard_plan, current_code, component_render_meta = await self._run_generate_mode(
                        payload=payload,
                        data_summaries=data_summaries,
                        interaction_hint=interaction_hint
                    )

                    if component_render_meta:
                        dashboard_plan = self._merge_component_render_meta(dashboard_plan, component_render_meta)
                        session_service.update_component_render_meta(payload.session_id, component_render_meta)

                    comp_ids = [c.id for c in dashboard_plan.components]

                    exec_result = self.executor.execute_dashboard_logic(
                        code_str=current_code,
                        data_context=actual_data_context,
                        component_ids=comp_ids,
                        summaries=data_summaries,
                        component_defs=dashboard_plan.components
                    )

                    if not exec_result.success:
                        raise Exception(f"Generate fallback after Edit also failed: {exec_result.error}")

                    is_edit_mode = False

                else:
                    logger.warning(f"代码执行失败，启动 AI 自动纠错: {exec_result.error[:100]}...")
                    current_code = await self.generator.fix_code(
                        original_code=current_code,
                        error_trace=exec_result.error,
                        summaries=data_summaries,
                        component_plans=dashboard_plan.components
                    )

                    if not self._has_dashboard_wrapper(current_code):
                        raise Exception("代码修复失败：fix_code 未返回完整 get_dashboard_data 包装函数。")

                    exec_result = self.executor.execute_dashboard_logic(
                        code_str=current_code,
                        data_context=actual_data_context,
                        component_ids=comp_ids,
                        summaries=data_summaries,
                        component_defs=dashboard_plan.components
                    )

                    if not exec_result.success:
                        raise Exception(f"代码修复失败: {exec_result.error}")

            # === 4. 结果装配与洞察提取 ===
            insight_card = await self.insight_extractor.generate_insights(
                query=payload.query or "交互更新分析",
                execution_stats=exec_result.global_insight_data,
                summaries=data_summaries
            )

            logger.info(">>> [Serialization] 正在进行数据序列化...")

            for component in dashboard_plan.components:
                if component.type == ComponentType.INSIGHT:
                    clean_insight = self._sanitize_data_fast(insight_card)

                    if isinstance(clean_insight, dict):
                        if "Description" in clean_insight and "detail" not in clean_insight:
                            clean_insight["detail"] = clean_insight.pop("Description")

                    component.insight_config = InsightCard(**clean_insight)
                    component.data_payload = clean_insight
                    continue

                if component.id in exec_result.results:
                    res = exec_result.results[component.id]
                    component.data_payload = self._sanitize_data_fast(res.data)

            # === 5. metadata 组装 ===
            final_interaction_state = session_service.get_interaction_state(payload.session_id)
            final_component_render_meta = session_service.get_component_render_meta(payload.session_id)

            dashboard_plan.metadata = {
                "last_code": current_code,
                "last_layout": dashboard_plan.model_dump(),
                "enriched_summaries": self._sanitize_data_fast(data_summaries),
                "execution_stats": self._sanitize_data_fast(exec_result.global_insight_data),
                "interaction_state": self._sanitize_data_fast(final_interaction_state),
                "component_render_meta": self._sanitize_data_fast(final_component_render_meta)
            }

            # === 6. 状态固化与快照存档 ===
            snapshot_id = session_service.save_snapshot(
                session_id=payload.session_id,
                query=payload.query or f"交互: {payload.active_component_id or '全局筛选'}",
                code=current_code,
                layout_data=dashboard_plan,
                summary=insight_card.summary,
                intent="EDIT" if is_edit_mode else "GENERATE"
            )

            dashboard_plan.metadata["snapshot_id"] = snapshot_id
            dashboard_plan.metadata["last_layout"] = dashboard_plan.model_dump()

            session_service.update_session_metadata(payload.session_id, dashboard_plan.metadata)

            return dashboard_plan

        except Exception as e:
            logger.error(f"Analysis Workflow Failed: {traceback.format_exc()}")
            raise e

    # =========================================================
    # 序列化辅助
    # =========================================================
    def _sanitize_data_fast(self, obj: Any) -> Any:
        def deep_clean(o):
            if isinstance(o, (str, int, float, bool, type(None))):
                return o

            if isinstance(o, dict) and o.get("is_stv_map_protocol") is True:
                return {str(k): deep_clean(v) for k, v in o.items()}

            if isinstance(o, dict):
                return {str(k): deep_clean(v) for k, v in o.items()}

            if isinstance(o, (list, tuple, set)):
                return [deep_clean(i) for i in o]

            if isinstance(o, np.ndarray):
                return o.tolist()

            if isinstance(o, (np.integer, np.floating)):
                return o.item()

            if isinstance(o, (pd.Series, pd.Index)):
                return o.tolist()

            if isinstance(o, pd.DataFrame):
                return o.to_dict(orient="records")

            if hasattr(o, "__geo_interface__"):
                return o.__geo_interface__

            if hasattr(o, "to_plotly_json"):
                return deep_clean(o.to_plotly_json())

            if hasattr(o, "to_dict") and hasattr(o, "layout") and hasattr(o, "data"):
                return deep_clean(o.to_dict())

            if isinstance(o, np.datetime64):
                return pd.to_datetime(o).isoformat()

            if isinstance(o, (date, datetime, pd.Timestamp)):
                return o.isoformat()

            if hasattr(o, "model_dump"):
                return deep_clean(o.model_dump())

            return str(o)

        try:
            clean_obj = deep_clean(obj)
            return orjson.loads(orjson.dumps(clean_obj, option=orjson.OPT_NON_STR_KEYS))
        except Exception as e:
            logger.error(f"Fast sanitization failed: {e}")
            return self._sanitize_data_legacy(obj)

    def _sanitize_data_legacy(self, obj: Any) -> Any:
        if hasattr(obj, "to_dict"):
            try:
                return self._sanitize_data_legacy(obj.to_dict(orient="records"))
            except Exception:
                return self._sanitize_data_legacy(obj.to_dict())

        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()

        elif hasattr(obj, "to_plotly_json"):
            return self._sanitize_data_legacy(obj.to_plotly_json())

        elif isinstance(obj, dict):
            return {str(k): self._sanitize_data_legacy(v) for k, v in obj.items()}

        elif isinstance(obj, list):
            return [self._sanitize_data_legacy(i) for i in obj]

        return str(obj)