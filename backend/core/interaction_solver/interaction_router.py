import logging
from typing import Dict, Any, Optional

from core.schemas.interaction import InteractionPayload, InteractionTriggerType

logger = logging.getLogger(__name__)


class InteractionRouter:
    """
    交互路由器（第一阶段）

    目标：
    1. 将交互分成 selection_interaction / component_visual_edit / structural_edit / unknown
    2. 让 workflow 不再把所有 edit 都交给 viz_editor
    """

    def __init__(self):
        pass

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

    def _has_selection_payload(self, payload: InteractionPayload) -> bool:
        return bool(
            payload.bbox
            or payload.selected_ids
            or payload.selected_values
            or payload.time_range
        )

    def classify(
            self,
            payload: InteractionPayload,
            interaction_state: Dict[str, Any],
            last_layout: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        返回示例：
        {
            "kind": "selection_interaction",
            "selection_type": "bbox"
        }
        """
        try:
            # 1. UI 动作优先视为 selection interaction
            if payload.trigger_type == InteractionTriggerType.UI_ACTION:
                if payload.bbox:
                    return {"kind": "selection_interaction", "selection_type": "bbox"}
                if payload.selected_ids:
                    return {"kind": "selection_interaction", "selection_type": "selected_ids"}
                if payload.selected_values:
                    return {"kind": "selection_interaction", "selection_type": "selected_values"}
                if payload.time_range:
                    return {"kind": "selection_interaction", "selection_type": "time_range"}
                return {"kind": "selection_interaction", "selection_type": "noop"}

            # 2. 自然语言结构性编辑
            if payload.trigger_type == InteractionTriggerType.NATURAL_LANGUAGE:
                query = payload.query or ""

                if self._looks_like_structural_edit_query(query):
                    return {"kind": "structural_edit"}

                # 3. 有 active_component_id 的自然语言修改 -> component edit
                if payload.active_component_id:
                    return {"kind": "component_visual_edit"}

                # 4. 无 active_component_id 但带明显 selection payload，也走 selection
                if self._has_selection_payload(payload):
                    return {"kind": "selection_interaction", "selection_type": "mixed"}

            return {"kind": "unknown"}

        except Exception as e:
            logger.warning(f"[InteractionRouter] classify failed: {e}")
            return {"kind": "unknown"}