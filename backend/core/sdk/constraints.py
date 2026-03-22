# backend/core/sdk/constraints.py

# 视觉主题配置
THEME_CONFIG = {
    "template": "plotly_dark",
    "mapbox_style": "carto-darkmatter",
    "color_continuous_scale": "Viridis",
    "color_discrete_sequence": ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692", "#B6E880"],
    "marker_line_width": 0.2,
    "marker_line_color": "rgba(255,255,255,0.3)",
    "opacity": 0.8
}

# 图例与颜色轴配置 (严格遵守 Rule 13)
COLORBAR_STYLE = {
    "thickness": 15,
    "len": 0.8,
    "x": 1.02,
    "y": 0.5,
    "title": {"side": "top"}
}

# 性能防护阈值
MAX_SCATTER_POINTS = 50000
SKEWNESS_THRESHOLD = 10.0  # 判定是否需要自动对数缩放的倍数