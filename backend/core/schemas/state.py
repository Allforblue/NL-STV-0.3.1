import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

from .dashboard import DashboardSchema


# =========================================================
# 交互状态模型（长期扩展核心）
# =========================================================

class SelectionFilter(BaseModel):
    """
    解析后的事实表过滤条件

    例子：
    - dataset_var='df_yellow_tripdata_2025_01', field='PULocationID', op='in', value=[161, 162]
    - dataset_var='df_lookup', field='Zone', op='eq', value='Midtown Center'
    """
    dataset_var: Optional[str] = Field(
        None,
        description="该过滤条件作用的数据集变量名；若为空，表示可由后续解析层再决定"
    )
    field: str = Field(..., description="过滤字段名")
    op: str = Field(
        default="eq",
        description="过滤操作符：eq / in / between / intersects / contains 等"
    )
    value: Any = Field(..., description="过滤值")
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="附加说明，如 relation 来源、lookup 结果、置信度等"
    )


class SelectionItem(BaseModel):
    """
    单次选择对象：
    用于表示一次 click / bbox / selected_ids / selected_values / time range 交互
    """
    selection_id: str = Field(
        default_factory=lambda: f"sel_{uuid.uuid4().hex[:8]}",
        description="选择对象唯一 ID"
    )

    source_component_id: Optional[str] = Field(
        None,
        description="触发此次选择的源组件 ID"
    )

    selection_type: str = Field(
        ...,
        description="选择类型：bbox / selected_ids / selected_values / time / compare"
    )

    scope: str = Field(
        default="global",
        description="作用范围：global / local / compare_left / compare_right"
    )

    label: Optional[str] = Field(
        None,
        description="给用户看的标签，例如 'Midtown Center', 'Queens', '框选区域 A'"
    )

    active: bool = Field(
        default=True,
        description="该选择当前是否生效"
    )

    raw_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="保留原始输入负载，便于回溯、调试与后续追问"
    )

    resolved_filters: List[SelectionFilter] = Field(
        default_factory=list,
        description="解析后的结构化过滤条件"
    )

    selection_context: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "结构化交互上下文，如 display_field / display_value / "
            "interaction_field / fact_link_field / selection_scope 等"
        )
    )


class ComparisonState(BaseModel):
    """
    对比分析状态：
    用于支持局部 vs 整体、多个选区之间的比较
    """
    enabled: bool = Field(default=False, description="是否开启对比模式")

    baseline_mode: str = Field(
        default="global",
        description="基准模式：global / selection / snapshot"
    )

    left_selection_ids: List[str] = Field(
        default_factory=list,
        description="对比左侧选择集合"
    )

    right_selection_ids: List[str] = Field(
        default_factory=list,
        description="��比右侧选择集合"
    )

    compare_metric: Optional[str] = Field(
        None,
        description="当前比较的主指标"
    )

    compare_dimension: Optional[str] = Field(
        None,
        description="当前比较的主维度"
    )

    compare_question: Optional[str] = Field(
        None,
        description="由用户显式提出的比较问题"
    )


class InteractionState(BaseModel):
    """
    会话级交互状态：
    用于持久保存多轮交互上下文，而不仅仅是一次性 payload
    """
    active_component_id: Optional[str] = Field(
        None,
        description="当前激活的组件 ID"
    )

    global_time_range: Optional[List[str]] = Field(
        None,
        description="当前全局时间范围"
    )

    selections: List[SelectionItem] = Field(
        default_factory=list,
        description="当前会话中累积的选择列表"
    )

    active_selection_ids: List[str] = Field(
        default_factory=list,
        description="当前正在生效的 selection id 列表"
    )

    comparison_state: ComparisonState = Field(
        default_factory=ComparisonState,
        description="当前比较分析状态"
    )

    last_interaction_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="最近一次交互的原始 payload"
    )

    notes: Dict[str, Any] = Field(
        default_factory=dict,
        description="预留给 workflow / editor / planner 的附加交互上下文"
    )


# =========================================================
# 快照与会话状态存储
# =========================================================

class SessionStateSnapshot(BaseModel):
    """
    会话状态快照模型：
    用于保存分析过程中的每一个“时间点”，支撑历史回溯功能。
    """
    snapshot_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="快照唯一ID"
    )

    parent_snapshot_id: Optional[str] = Field(
        None,
        description="父快照 ID，用于构建分析路径树 (Undo/Redo / Branching)"
    )

    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="快照创建时间"
    )

    # 触发上下文
    user_query: str = Field(..., description="触发该看板生成的用户原始指令")

    intent: Optional[str] = Field(
        None,
        description="意图分类，如 REGENERATE / MODIFY / EXPLORE / COMPARE"
    )

    # 核心逻辑备份
    code_snapshot: str = Field(
        ...,
        description="生成该看板的 Python 代码快照"
    )

    # 看板结构与数据
    layout_data: DashboardSchema = Field(
        ...,
        description="当时的看板完整结构与数据负载"
    )

    # UI 历史展示信息
    summary_text: Optional[str] = Field(
        None,
        description="用于历史列表展示的简短结论"
    )

    execution_time_ms: Optional[float] = Field(
        None,
        description="执行耗时（毫秒）"
    )

    # ------------------------------
    # 新增：交互状态快照
    # ------------------------------
    interaction_state: Optional[InteractionState] = Field(
        default=None,
        description="生成该快照时的交互状态"
    )

    # ------------------------------
    # 新增：组件渲染元信息快照
    # ------------------------------
    component_render_meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="各组件的展示/交互元信息快照"
    )

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class SessionStateStore(BaseModel):
    """
    会话全状态存储：
    管理一个会话中所有的快照序列。
    """
    session_id: str
    user_id: Optional[str] = None

    # 历史快照序列
    snapshots: List[SessionStateSnapshot] = Field(default_factory=list)

    # 当前激活快照
    current_snapshot_id: Optional[str] = None

    # 当前会话关联的文件列表
    active_files: List[str] = Field(default_factory=list)

    # ------------------------------
    # 新增：当前交互状态
    # ------------------------------
    interaction_state: InteractionState = Field(
        default_factory=InteractionState,
        description="当前会话的交互状态"
    )

    # ------------------------------
    # 新增：组件渲染元信息
    # ------------------------------
    component_render_meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="当前会话下各组件的 render meta / interaction meta"
    )

    def get_snapshot(self, snapshot_id: str) -> Optional[SessionStateSnapshot]:
        """快速检索指定快照"""
        for ss in self.snapshots:
            if ss.snapshot_id == snapshot_id:
                return ss
        return None

    def get_latest(self) -> Optional[SessionStateSnapshot]:
        """获取最近一次生成的快照"""
        if not self.snapshots:
            return None
        return self.snapshots[-1]

    def add_snapshot(self, snapshot: SessionStateSnapshot):
        """添加新快照并更新当前指针"""
        self.snapshots.append(snapshot)
        self.current_snapshot_id = snapshot.snapshot_id

    def rollback(self, target_snapshot_id: str) -> bool:
        """
        回滚到指定状态
        注意：这里只更新 current_snapshot_id；
        具体恢复 layout / code / interaction_state 由上层 workflow / session_service 决定。
        """
        snapshot = self.get_snapshot(target_snapshot_id)
        if snapshot:
            self.current_snapshot_id = target_snapshot_id
            return True
        return False