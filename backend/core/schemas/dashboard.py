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
    # [新增] 时间播放控制器，专门用于控制全局时间跨度
    TIMELINE_CONTROLLER = "timeline_controller"


class LayoutZone(str, Enum):
    """适配原型图的固定布局区域"""
    CENTER_MAIN = "center_main"  # 中间大地图区域
    RIGHT_SIDEBAR = "right_sidebar"  # 右侧图表区域 (可容纳多个)
    BOTTOM_INSIGHT = "bottom_insight"  # 下方数据洞察结果区域
    LEFT_HISTORY = "left_history"  # 左侧历史记录区
    TOP_NAV = "top_nav"  # 顶部导航/全局过滤器区


class ChartType(str, Enum):
    BAR = "bar"
    LINE = "line"  # 常用于时间趋势分析
    SCATTER = "scatter"
    PIE = "pie"
    HEATMAP = "heatmap"  # 2D 统计热力图
    TABLE = "table"
    TIMELINE_HEATMAP = "timeline_heatmap"
    AREA = "area"


class InteractionType(str, Enum):
    """交互行为类型"""
    BBOX = "bbox"
    CLICK = "click"
    FILTER = "filter"
    TIME = "time"
    # TIME_FILTER = "time_filter"


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


# [核心修复] 补齐缺失的时间轴契约配置类，防止 Pydantic 静默丢弃
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
    """Deck.gl 或 Mapbox 图层配置"""
    layer_id: str
    layer_type: str = Field(..., description="Deck.gl 图层类型")
    data_var: str = Field(..., description="指向的数据集变量名")

    color_column: Optional[str] = None
    size_column: Optional[str] = None
    color_range: Optional[List[str]] = Field(default_factory=lambda: ["#0000ff", "#ff0000"])

    opacity: float = 0.8
    visible: bool = True
    params: Dict[str, Any] = Field(default_factory=dict, description="透传给 Deck.gl 的其他参数")

    is_animated: bool = Field(False, description="是否开启时间轴动画")
    animation_column: Optional[str] = Field(None, description="驱动动画的时间轴字段")


class ChartConfig(BaseModel):
    chart_type: ChartType

    # [新增]: 接收 Planner 规划的图表映射维度，防止被静默丢弃
    target_dimension: Optional[str] = Field("auto", description="X轴/目标分类维度")
    target_metric: Optional[str] = Field("auto", description="Y轴/目标度量指标")

    x_axis: Optional[str] = None
    y_axis: Optional[List[str]] = None
    series_name: Optional[str] = None
    unit: Optional[str] = None
    stack: bool = False
    time_bucket: Optional[str] = None
    theme: Optional[str] = None

    animation_frame: Optional[str] = Field(None, description="Plotly 动画帧字段")
    animation_group: Optional[str] = Field(None, description="动画分组字段")
    play_speed: int = Field(500, description="动画播放速度")


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

    data_payload: Optional[Union[Dict[str, Any], List[Any], str]] = None

    map_config: Optional[List[MapLayerConfig]] = None
    chart_config: Optional[ChartConfig] = None
    insight_config: Optional[InsightCard] = None

    # [核心修复] 挂载时间轴配置，接通前后端数据链路
    timeline_config: Optional[TimelineConfig] = None

    links: List[ComponentLink] = Field(default_factory=list, description="该组件触发的联动规则")
    is_controllable: bool = True


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
        description="包含 last_code, data_hashes, dataset_vars 等"
    )