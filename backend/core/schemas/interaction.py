from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal
from enum import Enum


class InteractionTriggerType(str, Enum):
    """交互触发源类型"""
    NATURAL_LANGUAGE = "nl"  # 底部对话框输入的文字指令
    UI_ACTION = "ui"  # 地图上的框选(bbox)、点选(click)、缩放(zoom)等 UI 行为
    BACKTRACK = "backtrack"  # 点击左侧历史对话区域进行回溯
    SYSTEM = "system"  # 系统自动触发（如初始化或任务自动跳转）


class InteractionPayload(BaseModel):
    """
    增强版多模态交互载荷：
    支持“自然语言指令”、“地图/图表 UI 操作”以及“历史回溯”协同驱动。
    """
    session_id: str = Field(..., description="用于维持对话上下文的会话ID")

    # 标识本次交互的性质
    trigger_type: InteractionTriggerType = Field(
        default=InteractionTriggerType.NATURAL_LANGUAGE,
        description="交互触发来源类型"
    )

    # --- 模态 1: 自然语言 (Text Input) ---
    query: Optional[str] = Field(None, description="用户的文字指令，如 '分析这里的拥堵原因'")

    # --- 模态 2: UI 交互负载 (UI State) ---
    active_component_id: Optional[str] = Field(
        None,
        description="触发交互的源组件ID，例如 'main_map' 或 'right_pie_1'"
    )

    # 空间维度
    bbox: Optional[List[float]] = Field(
        None,
        description="地图框选或当前视窗范围[min_lon, min_lat, max_lon, max_lat]"
    )

    # 视图状态：保存当前的经纬度、缩放、仰角等，确保看板更新后视角不重置
    view_state: Optional[Dict[str, Any]] = Field(
        None,
        description="当前的地图视图状态 (longitude, latitude, zoom, pitch, bearing)"
    )

    # 实体与分类选择
    selected_ids: Optional[List[Union[str, int]]] = Field(
        None,
        description="地图上点击选中的特定实体 ID 列表"
    )

    # 通用分类选择值：如点击柱状图选中的 'Taxi' 类别
    selected_values: Optional[Dict[str, Any]] = Field(
        None,
        description="分类筛选值映射，如 {'vehicle_type': 'bus'}"
    )

    # 时间维度
    time_range: Optional[List[str]] = Field(
        None,
        description="时间范围过滤 [开始时间, 结束时间]"
    )

    # [新增] selection 状态更新模式：
    # - replace: 替换当前激活选区（默认，兼容旧行为）
    # - append: 追加到当前激活选区集合
    # - compare_left: 将本次选区写入对比左侧
    # - compare_right: 将本次选区写入对比右侧
    selection_mode: Literal["replace", "append", "compare_left", "compare_right"] = Field(
        default="replace",
        description="选区更新模式：replace / append / compare_left / compare_right"
    )

    # --- 模态 3: 历史回溯 (History/State Management) ---
    target_snapshot_id: Optional[str] = Field(
        None,
        description="回溯的目标快照ID。当点击历史记录时，后端直接返回对应的状态快照"
    )

    # --- 交互控制与上下文 ---
    force_new: bool = Field(
        False,
        description="是否强制重新规划看板（即忽略现有布局，完全推翻重做）"
    )

    # 人机协同的模式决策字段
    force_mode: Literal["auto", "edit", "generate"] = Field(
        default="auto",
        description="看板更新模式：'auto' 由 AI 判定，'edit' 强制修改现有代码，'generate' 强制从零生成。"
    )

    current_dashboard_id: Optional[str] = Field(
        None,
        description="当前页面正在显示的看板ID"
    )

    # 扩展参数：用于存储特定组件的自定义交互参数
    extra_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="扩展参数容器，可承载 selection_context、组件自定义参数等"
    )