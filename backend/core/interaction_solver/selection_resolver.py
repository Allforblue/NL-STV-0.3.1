import logging
import uuid
from typing import Dict, Any, List, Optional

from core.schemas.interaction import InteractionPayload, InteractionTriggerType

logger = logging.getLogger(__name__)


class SelectionResolver:
    """
    选择解析器（第一阶段）

    当前能力：
    1. 将 payload 结构化写回 interaction_state
    2. 为 selection item 填充基础 resolved_filters
    3. 支持 selection_mode:
       - replace
       - append
       - compare_left
       - compare_right
    4. 不做复杂 relation 推理，只做轻量规则解析

    说明：
    - 这是过渡版
    - 重点是让 interaction_state 真正可用
    """

    def __init__(self):
        pass

    def _make_selection_id(self) -> str:
        return f"sel_{uuid.uuid4().hex[:8]}"

    def _ensure_state_shape(self, interaction_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        state = dict(interaction_state or {})
        state.setdefault("active_component_id", None)
        state.setdefault("global_time_range", None)
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
        state.setdefault("last_interaction_payload", {})
        state.setdefault("notes", {})
        return state

    def _build_selection_context(self, payload: InteractionPayload) -> Dict[str, Any]:
        selection_context = {}
        if payload.extra_params and isinstance(payload.extra_params, dict):
            selection_context = payload.extra_params.get("selection_context", {}) or {}
        return selection_context

    def _resolve_selection_mode(self, payload: InteractionPayload) -> str:
        """
        解析 selection mode，优先级：
        1. payload.selection_mode
        2. extra_params.selection_mode
        3. extra_params.selection_context.selection_mode
        4. 默认 replace
        """
        mode = getattr(payload, "selection_mode", None)

        if not mode and payload.extra_params and isinstance(payload.extra_params, dict):
            mode = payload.extra_params.get("selection_mode")

        if not mode:
            selection_context = self._build_selection_context(payload)
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

    def _resolve_bbox_filters(
        self,
        payload: InteractionPayload,
        summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        第一阶段 bbox 仅保留轻量 resolved_filters 描述。
        真正数据裁剪仍在 selection_applier 中完成。
        """
        if not payload.bbox or len(payload.bbox) != 4:
            return []

        return [{
            "dataset_var": None,
            "field": "__geometry__",
            "op": "bbox",
            "value": payload.bbox,
            "meta": {
                "selection_type": "bbox",
                "source_component_id": payload.active_component_id
            }
        }]

    def _resolve_selected_ids_filters(
        self,
        payload: InteractionPayload,
        summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not payload.selected_ids:
            return []

        ids = [str(v) for v in payload.selected_ids]

        return [{
            "dataset_var": None,
            "field": "__id__",
            "op": "in",
            "value": ids,
            "meta": {
                "selection_type": "selected_ids",
                "source_component_id": payload.active_component_id
            }
        }]

    def _resolve_selected_values_filters(
        self,
        payload: InteractionPayload,
        summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not payload.selected_values:
            return []

        filters = []
        for k, v in payload.selected_values.items():
            filters.append({
                "dataset_var": None,
                "field": str(k),
                "op": "eq",
                "value": v,
                "meta": {
                    "selection_type": "selected_values",
                    "source_component_id": payload.active_component_id
                }
            })
        return filters

    def _resolve_time_range_filters(
        self,
        payload: InteractionPayload,
        summaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not payload.time_range or len(payload.time_range) != 2:
            return []

        return [{
            "dataset_var": None,
            "field": "__time__",
            "op": "between",
            "value": payload.time_range,
            "meta": {
                "selection_type": "time_range",
                "source_component_id": payload.active_component_id
            }
        }]

    def _build_selection_item(
        self,
        payload: InteractionPayload,
        resolved_filters: List[Dict[str, Any]],
        selection_mode: str
    ) -> Optional[Dict[str, Any]]:
        selection_context = self._build_selection_context(payload)

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

        if not selection_type:
            return None

        scope = selection_context.get("selection_scope", "global")
        if selection_mode == "compare_left":
            scope = "compare_left"
        elif selection_mode == "compare_right":
            scope = "compare_right"

        return {
            "selection_id": self._make_selection_id(),
            "source_component_id": payload.active_component_id,
            "selection_type": selection_type,
            "scope": scope,
            "label": label,
            "active": True,
            "raw_payload": raw_payload,
            "resolved_filters": resolved_filters,
            "selection_context": selection_context
        }

    def _apply_selection_mode(
        self,
        state: Dict[str, Any],
        selection_item: Dict[str, Any],
        selection_mode: str
    ) -> Dict[str, Any]:
        """
        基于 selection_mode 更新 state。
        第一阶段策略：
        - replace: 单选覆盖，关闭 compare
        - append: 追加 active selections，关闭 compare
        - compare_left: 写入 compare 左侧（单槽位），开启 compare
        - compare_right: 写入 compare 右侧（单槽位），开启 compare
        """
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

        new_selection_id = selection_item["selection_id"]
        selections.append(selection_item)

        if selection_mode == "append":
            if new_selection_id not in active_selection_ids:
                active_selection_ids.append(new_selection_id)

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
            comparison_state["left_selection_ids"] = [new_selection_id]
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
            comparison_state["right_selection_ids"] = [new_selection_id]

            active_selection_ids = []
            for sid in comparison_state["left_selection_ids"] + comparison_state["right_selection_ids"]:
                if sid and sid not in active_selection_ids:
                    active_selection_ids.append(sid)

        else:
            # replace（默认）
            active_selection_ids = [new_selection_id]
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

    def resolve(
        self,
        payload: InteractionPayload,
        interaction_state: Dict[str, Any],
        summaries: List[Dict[str, Any]],
        last_layout: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        返回：
        {
            "interaction_state": updated_state,
            "active_selection": selection_item | None,
            "resolved_filters": [...]
        }
        """
        state = self._ensure_state_shape(interaction_state)

        state["active_component_id"] = payload.active_component_id
        state["last_interaction_payload"] = (
            payload.model_dump() if hasattr(payload, "model_dump") else {}
        )

        if payload.time_range:
            state["global_time_range"] = payload.time_range

        # 非 UI_ACTION 且无 selection payload，则只更新基础状态
        has_selection = bool(
            payload.bbox or payload.selected_ids or payload.selected_values or payload.time_range
        )

        if payload.trigger_type != InteractionTriggerType.UI_ACTION and not has_selection:
            return {
                "interaction_state": state,
                "active_selection": None,
                "resolved_filters": []
            }

        resolved_filters: List[Dict[str, Any]] = []

        if payload.bbox:
            resolved_filters = self._resolve_bbox_filters(payload, summaries)
        elif payload.selected_ids:
            resolved_filters = self._resolve_selected_ids_filters(payload, summaries)
        elif payload.selected_values:
            resolved_filters = self._resolve_selected_values_filters(payload, summaries)
        elif payload.time_range:
            resolved_filters = self._resolve_time_range_filters(payload, summaries)

        selection_mode = self._resolve_selection_mode(payload)
        selection_item = self._build_selection_item(payload, resolved_filters, selection_mode)

        if selection_item:
            state = self._apply_selection_mode(
                state=state,
                selection_item=selection_item,
                selection_mode=selection_mode
            )

        return {
            "interaction_state": state,
            "active_selection": selection_item,
            "resolved_filters": resolved_filters
        }