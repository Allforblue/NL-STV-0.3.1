from pydantic import BaseModel, Field, AliasChoices, ConfigDict
from typing import List, Optional, Dict, Any, Union
from enum import Enum


# --- 基础枚举 ---

class ComponentType(str, Enum):
    MAP = "map"
    CHART = "chart"
    KPI = "kpi"
    INSIGHT = "insight"
    TABLE = "table"
    TIMELINE_CONTROLLER = "timeline_controller"


class LayoutZone(str, Enum):
    """适配原型图的固定布局区域"""
    CENTER_MAIN = "center_main"
    RIGHT_SIDEBAR = "right_sidebar"
    BOTTOM_INSIGHT = "bottom_insight"
    LEFT_HISTORY = "left_history"
    TOP_NAV = "top_nav"


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    PIE = "pie"
    HEATMAP = "heatmap"
    TABLE = "table"
    TIMELINE_HEATMAP = "timeline_heatmap"
    AREA = "area"


class InteractionType(str, Enum):
    """交互行为类型"""
    BBOX = "bbox"
    CLICK = "click"
    FILTER = "filter"
    TIME = "time"


# --- 联动逻辑定义 ---

class ComponentLink(BaseModel):
    """组件间的联动关系定义"""
    target_id: str = Field(..., description="响应联动的目标组件ID")
    interaction_type: InteractionType
    link_key: str = Field(..., description="关联的字段名，如 'zone_id' 或 'timestamp'")
    description: Optional[str] = None


# --- 细分配置 ---

class LayoutConfig(BaseModel):
    """看板布局配置：支持浮点数以实现精确对齐"""
    zone: LayoutZone = Field(..., description="所属布局区域")
    index: int = Field(0, description="在区域内的排序索引")
    x: float = 0
    y: float = 0
    w: float = 12
    h: float = 6


class TimelineConfig(BaseModel):
    """时间播放器配置契约"""
    column: str = Field(..., description="驱动时间动画的数据字段名")
    start_time: str = Field(..., description="ISO 格式开始时间")
    end_time: str = Field(..., description="ISO 格式结束时间")
    step: str = Field("1H", description="时间步长 (如 '1H', '1D')")
    frame_format: str = Field("%H:00", description="Plotly 动画帧 ID 的格式契约")
    enable_playback: Optional[bool] = True
    auto_play: Optional[bool] = False


class MapLayerConfig(BaseModel):
    """
    Deck.gl / Map 前端渲染配置

    长期演进原则��
    - 保留原有绘制层字段
    - 新增展示 / 交互 / 事实表映射语义
    - 为地图 click / bbox / compare 提供稳定协议
    """
    layer_id: str
    layer_type: str = Field(..., description="地图图层类型，如 polygon / point / heatmap")
    data_var: str = Field(..., description="指向的数据集变量名")

    # 原有视觉字段
    color_column: Optional[str] = None
    size_column: Optional[str] = None
    color_range: Optional[List[str]] = Field(default_factory=lambda: ["#0000ff", "#ff0000"])

    opacity: float = 0.8
    visible: bool = True
    params: Dict[str, Any] = Field(default_factory=dict, description="透传给前端图层的附加参数")

    # 动画相关
    is_animated: bool = Field(False, description="是否开启时间轴动画")
    animation_column: Optional[str] = Field(None, description="驱动动画的时间轴字段")

    # ------------------------------
    # 新增：展示层语义（Render Meta）
    # ------------------------------
    display_dimension: Optional[str] = Field(
        None,
        description="地图最终展示给用户识别/悬浮的空间维度字段，如 Zone / Borough / region_name"
    )
    display_metric: Optional[str] = Field(
        None,
        description="地图最终展示的指标字段，如 trip_count / amount"
    )

    # ------------------------------
    # 新增：交互层语义（Interaction Meta）
    # ------------------------------
    interaction_field: Optional[str] = Field(
        None,
        description="地图点击/框选后的主交互字段，如 LocationID / Zone"
    )
    interaction_key_field: Optional[str] = Field(
        None,
        description="地图交互时用于传递稳定 key 的字段，如 LocationID"
    )
    fact_link_field: Optional[str] = Field(
        None,
        description="最终作用到事实表上的过滤字段，如 PULocationID / DOLocationID"
    )
    selection_supported: Optional[List[str]] = Field(
        default_factory=list,
        description="该图层支持的交互集合，如 ['click', 'bbox']"
    )

    # ------------------------------
    # 新增：通用扩展元信息
    # ------------------------------
    render_meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="预留给生成器/执行器/前端的扩展渲染元信息"
    )


class ChartConfig(BaseModel):
    """
    图表配置

    长期演进原则：
    - target_* 表示“分析目标”
    - display_* / interaction_* 表示“最终呈现与交互语义”
    - 保持对旧字段 x_axis / y_axis / series_name 的兼容
    """
    chart_type: ChartType

    # ------------------------------
    # 原有：分析目标层（Planner Locked）
    # ------------------------------
    target_dimension: Optional[str] = Field(
        "auto",
        description="分析目标维度（不一定等于最终展示维度）"
    )
    target_metric: Optional[str] = Field(
        "auto",
        description="分析目标指标"
    )

    # ------------------------------
    # 原有：前端展示字段（兼容）
    # ------------------------------
    x_axis: Optional[str] = Field(
        None,
        description="图表显示的 X 轴字段（兼容旧前端逻辑）"
    )
    y_axis: Optional[List[str]] = Field(
        None,
        description="图表显示的 Y 轴字段列表"
    )
    series_name: Optional[str] = Field(
        None,
        description="系列名称字段/分类字段"
    )

    unit: Optional[str] = None
    stack: bool = False
    time_bucket: Optional[str] = None
    theme: Optional[str] = None

    animation_frame: Optional[str] = Field(None, description="Plotly 动画帧字段")
    animation_group: Optional[str] = Field(None, description="动画分组字段")
    play_speed: int = Field(500, description="动画播放速度")

    # ------------------------------
    # 新增：展示层语义（最终渲染）
    # ------------------------------
    display_dimension: Optional[str] = Field(
        None,
        description="最终展示给用户的维度字段，如 Zone / Borough"
    )
    display_metric: Optional[str] = Field(
        None,
        description="最终展示给用户的指标字段"
    )

    # ------------------------------
    # 新增：交互层语义
    # ------------------------------
    interaction_field: Optional[str] = Field(
        None,
        description="前端点击/选择时应上报的字段，如 Zone / Borough"
    )
    interaction_key_field: Optional[str] = Field(
        None,
        description="若 customdata 中保存稳定 key，则该字段表示交互 key，如 LocationID"
    )
    fact_link_field: Optional[str] = Field(
        None,
        description="最终作用到事实表上的过滤字段，如 PULocationID / DOLocationID"
    )

    # ------------------------------
    # 新增：交互范围与比较模式
    # ------------------------------
    selection_scope: Optional[str] = Field(
        None,
        description="图表当前所处选择范围，如 global / local / compare_left / compare_right"
    )
    compare_mode: Optional[str] = Field(
        None,
        description="��较模式，如 none / selection_vs_global / selection_vs_selection"
    )

    # ------------------------------
    # 原项目兼容扩展字段
    # ------------------------------
    primary_metric: Optional[str] = Field(
        None,
        description="由 planner 锁定的主指标（兼容旧版 planner 输出）"
    )
    primary_dimension: Optional[str] = Field(
        None,
        description="由 planner 锁定的主维度（兼容旧版 planner 输出）"
    )

    # ------------------------------
    # 新增：统一扩展元信息
    # ------------------------------
    render_meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="预留给生成器/执行器/前端的扩展渲染元信息"
    )


class InsightCard(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    summary: str
    detail: str = Field(
        ...,
        validation_alias=AliasChoices('detail', 'Description', 'description', 'content')
    )
    tags: List[str] = Field(default_factory=list)
    evidence: Optional[Union[Dict[str, Any], List[Any]]] = None


# --- 核心组件定义 ---

class DashboardComponent(BaseModel):
    id: str = Field(..., description="组件唯一ID")
    title: str = "分析组件"
    type: ComponentType
    layout: LayoutConfig

    data_payload: Optional[Union[Dict[str, Any], List[Any], str]] = Field(
        None,
        description="组件渲染后的前端负载（图表 Figure JSON / 地图协议 / 文本等）"
    )

    map_config: Optional[List[MapLayerConfig]] = None
    chart_config: Optional[ChartConfig] = None
    insight_config: Optional[InsightCard] = None
    timeline_config: Optional[TimelineConfig] = None

    links: List[ComponentLink] = Field(default_factory=list, description="该组件触发的联动规则")
    is_controllable: bool = True

    # 新增：组件级运行状态 / 前端局部状态扩展
    component_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="预留给交互态、比较态、局部高亮态等组件级状态"
    )


# --- 根协议 ---

class DashboardSchema(BaseModel):
    dashboard_id: str
    title: str
    description: Optional[str] = None

    initial_view_state: Dict[str, Any] = Field(
        default={
            "longitude": 0.0,
            "latitude": 0.0,
            "zoom": 10,
            "pitch": 45,
            "bearing": 0
        }
    )

    global_time_range: Optional[List[str]] = Field(
        None,
        description="看板当前的全局时间过滤范围 [start, end]"
    )

    components: List[DashboardComponent]

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "工作流与执行元数据。"
            "当前兼容字段包括 last_code, last_layout, snapshot_id, enriched_summaries, execution_stats；"
            "后续可扩展 interaction_state, component_render_meta 等。"
        )
    )