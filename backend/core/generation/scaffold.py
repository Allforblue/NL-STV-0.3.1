# backend/core/generation/scaffold.py

from typing import List, Any


class STChartScaffold:
    def __init__(self):
        # ==========================================
        # 核心硬约束（压缩版）
        # ==========================================
        self.hard_constraints_core = """
=== HARD CONSTRAINTS ===
You MUST obey all rules below:

❌ NEVER import or use visualization libraries directly:
- plotly
- matplotlib
- seaborn
- bokeh
- altair
- folium
- pydeck
- keplergl

❌ NEVER write raw visualization code such as:
- px.bar(...)
- px.line(...)
- px.scatter_mapbox(...)
- px.choropleth_mapbox(...)
- go.Figure(...)
- fig.update_layout(...)
- fig.update_traces(...)
- fig.show(...)

❌ NEVER manually build map layers, geojson rendering logic, or frontend map configs.
❌ NEVER merge business df with GeoDataFrame manually for map rendering.
❌ NEVER use `dt.hour`, `dt.day`, `dt.month` for animated time columns.
❌ NEVER guess column names.
❌ NEVER wrap code inside `get_dashboard_data`.

✅ ALWAYS use the globally injected `stv` and `ops` objects.
✅ ALWAYS return the final `stv.xxx(...)` result directly.
✅ Map methods return protocol Dict.
✅ Chart methods return Plotly Figure.
"""

        # ==========================================
        # 核心工作范式（压缩版）
        # ==========================================
        self.coding_paradigm_core = """
=== 2-STEP CODING PARADIGM ===

[STEP 1: Data Preparation]
- Read data safely:
  `df = data_context.get('df_name').copy()`
- Use pandas + `ops` for filtering / groupby / resampling.
- Use `ops.safe_merge(...)` for lookup translation if needed.
- DO NOT manually merge GeoDataFrame with business df for map rendering.

[STEP 2: Rendering]
- For maps: call one `stv` map method and return the protocol Dict.
- For charts: call one `stv` chart method and return the Figure.
"""

        # ==========================================
        # 语义映射规则（保留）
        # ==========================================
        self.semantic_rules_core = """
=== SEMANTIC MAPPING RULES ===
NEVER guess column names. Use Metadata Context and semantic tags:

- `ST_LOC_ID` or `ID_KEY`:
  use for location keys, joins, `data_key`, `geo_key`
- `ST_TIME`:
  use for `time_col`, resampling, animation timeline
- `BIZ_METRIC`:
  use for `val_col`, aggregation target, chart metric, map metric
- `BIZ_CAT`:
  use for category dimensions and readable labels
"""

        # ==========================================
        # ops API（单独拆出）
        # ==========================================
        self.api_reference_ops = """
=== OPS API REFERENCE ===

1. `ops.safe_merge(left, right, on=None, how='left', left_on=None, right_on=None, **kwargs) -> DataFrame`
- Use this to merge business tables or lookup/dimension tables.
- Best for ID -> readable name translation.
- It automatically aligns key types and strips trailing `.0`.
- DO NOT use this to merge df with GeoDataFrame for map rendering.

2. `ops.safe_resample(df, time_col, val_col, freq='1h', agg_func='sum') -> DataFrame`
- Use this for time-series aggregation before line chart rendering.
- Supported agg_func:
  - 'sum'
  - 'mean'
  - 'size'
  - 'count'
- If agg_func is 'sum' or 'mean', you MUST provide a valid numeric val_col.
- If agg_func is 'size' or 'count', val_col may be omitted, but stable naming is recommended.
"""

        # ==========================================
        # map API（只给地图看）
        # ==========================================
        self.api_reference_maps = """
=== MAP API REFERENCE ===

FATAL MAP RULE:
For `stv.choropleth()` and `stv.animated_map()`, you MUST provide TWO explicit ID columns:
- `data_key`: ID column in business df
- `geo_key`: ID column in geographic gdf

1. `stv.choropleth(df, gdf, data_key, geo_key, val_col=None, hover_cols=None, label_map=None) -> Dict`
- Static polygon/area map
- Returns protocol Dict for center map rendering

2. `stv.animated_map(df, gdf, data_key, geo_key, time_col=None, val_col=None, hover_cols=None) -> Dict`
- Animated polygon/area map
- Returns protocol Dict
- `time_col` must be real continuous timestamp
- Use `dt.floor('h')`, NEVER `dt.hour`

3. `stv.scatter_map(df, lat_col=None, lon_col=None, val_col=None, size_col=None, hover_cols=None) -> Dict`
- Static point map
- Returns protocol Dict

4. `stv.heatmap(df, lat_col=None, lon_col=None, val_col=None) -> Dict`
- Heatmap
- Returns protocol Dict

5. `stv.animated_scatter(df, lat_col=None, lon_col=None, time_col=None, val_col=None, hover_cols=None) -> Dict`
- Animated point map
- Returns protocol Dict
- `time_col` must be real timestamp
"""

        # ==========================================
        # chart API（只给图表看）
        # ==========================================
        self.api_reference_charts = """
=== CHART API REFERENCE ===

FATAL CHART RULE:
For `stv.bar`, `stv.pie`, and `stv.line`, you MUST NOT pass raw unaggregated large tables.
You MUST aggregate first using pandas or `ops.safe_resample()`.

1. `stv.bar(df, x_val, y_cat, top_n=15, color_col=None) -> Figure`
- Input df must be aggregated
- One column categorical, one numeric

2. `stv.pie(df, names, values, top_n=8) -> Figure`
- Input df must be aggregated

3. `stv.line(df, time_col=None, val_col=None) -> Figure`
- Best used with resampled time series

4. `stv.periodic_bar(df, time_col=None, val_col=None, cycle='hour') -> Figure`
- Periodic distribution chart
- Can often use raw event-level data
"""

        # ==========================================
        # line chart 专属额外规则
        # ==========================================
        self.api_reference_line_extra = """
=== LINE CHART EXTRA RULES ===

COUNT-BASED TIME TREND RULE:
If the user asks for event counts / trip counts / order counts / row counts over time:

CORRECT pattern:
1. Keep only needed time column if possible
2. Create helper count column:
   `df['trip_count'] = 1`
3. Resample with:
   `ops.safe_resample(df, time_col='time_col', val_col='trip_count', freq='1h', agg_func='sum')`
4. Render with:
   `stv.line(df_ts, time_col='time_col', val_col='trip_count')`

WRONG pattern:
- `ops.safe_resample(..., val_col=None, agg_func='sum')`

NEVER assume a `count` column will be auto-created.
"""

        # ==========================================
        # map few-shot
        # ==========================================
        self.examples_maps_basic = """
=== MAP EXAMPLES ===

[Example: Static choropleth]
def get_area_map(data_context):
    df_orders = data_context.get('df_orders').copy()
    gdf_regions = data_context.get('gdf_regions').copy()
    df_agg = df_orders.groupby('region_id', as_index=False)['order_count'].sum()
    return stv.choropleth(
        df=df_agg,
        gdf=gdf_regions,
        data_key='region_id',
        geo_key='region_id',
        val_col='order_count'
    )

[Example: Animated choropleth]
def get_animated_area_map(data_context):
    df_orders = data_context.get('df_orders').copy()
    gdf_regions = data_context.get('gdf_regions').copy()
    df_orders['event_time'] = pd.to_datetime(df_orders['event_time']).dt.floor('h')
    df_agg = df_orders.groupby(['region_id', 'event_time'], as_index=False)['order_count'].sum()
    return stv.animated_map(
        df=df_agg,
        gdf=gdf_regions,
        data_key='region_id',
        geo_key='region_id',
        time_col='event_time',
        val_col='order_count'
    )
"""

        # ==========================================
        # bar few-shot
        # ==========================================
        self.examples_chart_bar = """
=== BAR CHART EXAMPLE ===

[Example: Bar chart with lookup translation]
def get_top_bar(data_context):
    df_fact = data_context.get('df_fact').copy()
    df_lookup = data_context.get('df_lookup').copy()
    df_agg = df_fact.groupby('zone_id', as_index=False)['amount'].sum()
    df_agg = ops.safe_merge(df_agg, df_lookup[['zone_id', 'zone_name']], on='zone_id')
    return stv.bar(
        df=df_agg,
        x_val='zone_name',
        y_cat='amount',
        top_n=15
    )
"""

        # ==========================================
        # line numeric few-shot
        # ==========================================
        self.examples_chart_line_numeric = """
=== NUMERIC LINE CHART EXAMPLE ===

[Example: Numeric time trend]
def get_time_line(data_context):
    df_orders = data_context.get('df_orders').copy()
    df_daily = ops.safe_resample(
        df=df_orders,
        time_col='order_time',
        val_col='amount',
        freq='1d',
        agg_func='sum'
    )
    return stv.line(
        df=df_daily,
        time_col='order_time',
        val_col='amount'
    )
"""

        # ==========================================
        # line count few-shot
        # ==========================================
        self.examples_chart_line_count = """
=== COUNT-BASED LINE CHART EXAMPLE ===

[Example: Count-based time trend]
def get_trip_count_trend(data_context):
    df_trips = data_context.get('df_trips')[['pickup_time']].copy()
    df_trips['trip_count'] = 1
    df_hourly = ops.safe_resample(
        df=df_trips,
        time_col='pickup_time',
        val_col='trip_count',
        freq='1h',
        agg_func='sum'
    )
    return stv.line(
        df=df_hourly,
        time_col='pickup_time',
        val_col='trip_count'
    )

[Wrong pattern]
Do NOT write:
ops.safe_resample(df, time_col='pickup_time', val_col=None, freq='1h', agg_func='sum')
"""

    def _build_component_hint(self, comp: Any) -> str:
        comp_hints = ""
        is_dict = isinstance(comp, dict)

        comp_id = comp.get('id') if is_dict else getattr(comp, 'id', 'unknown')
        c_type = comp.get('type') if is_dict else getattr(comp, 'type', 'unknown')
        c_type_str = str(c_type).split('.')[-1].lower()

        if c_type_str == 'chart':
            conf = comp.get('chart_config', {}) if is_dict else getattr(comp, 'chart_config', {})
            if hasattr(conf, "model_dump"):
                conf = conf.model_dump()
            if not conf:
                conf = {}

            dim = conf.get('target_dimension', 'auto')
            metric = conf.get('target_metric', 'auto')
            ctype = conf.get('chart_type', 'bar')

            comp_hints += f"Target Component ID: {comp_id}\n"
            comp_hints += f"Target Chart Type: {ctype}\n"
            comp_hints += f"Preferred Dimension: '{dim}'\n"
            comp_hints += f"Preferred Metric: '{metric}'\n"

            if str(ctype).lower() == 'line':
                comp_hints += "Hint: prefer `ops.safe_resample(...)` before `stv.line(...)`.\n"
                comp_hints += "Hint: if the metric is count-like, create a helper count column first.\n"
                comp_hints += "Hint: NEVER use `safe_resample(..., val_col=None, agg_func='sum')` for count trends.\n"
            elif str(ctype).lower() == 'pie':
                comp_hints += "Hint: aggregate by category first, then call `stv.pie(...)`.\n"
            elif str(ctype).lower() == 'periodic_bar':
                comp_hints += "Hint: `stv.periodic_bar(...)` can often work directly on raw event-level data.\n"
            else:
                comp_hints += "Hint: aggregate by dimension first, then call the corresponding `stv` chart method.\n"

        elif c_type_str == 'map':
            m_conf = comp.get('map_config', [{}]) if is_dict else getattr(comp, 'map_config', [{}])
            if isinstance(m_conf, list) and len(m_conf) > 0:
                first_layer = m_conf[0]
                if hasattr(first_layer, "model_dump"):
                    first_layer = first_layer.model_dump()
            else:
                first_layer = {}

            layer_type = first_layer.get('layer_type', 'point')
            anim = first_layer.get('is_animated', False)

            comp_hints += f"Target Component ID: {comp_id}\n"
            comp_hints += f"Target Map Layer Type: {layer_type}\n"
            comp_hints += f"Animated: {anim}\n"

            layer_type_str = str(layer_type).lower()

            if anim:
                if layer_type_str in ['polygon', 'choropleth', 'area']:
                    comp_hints += "You should prefer `stv.animated_map(...)`.\n"
                    comp_hints += "You MUST provide both `data_key` and `geo_key`.\n"
                    comp_hints += "You MUST use a real timestamp column, not `dt.hour`.\n"
                    comp_hints += "This method returns a protocol Dict for the center map renderer.\n"
                elif layer_type_str in ['point', 'scatter']:
                    comp_hints += "You should prefer `stv.animated_scatter(...)`.\n"
                    comp_hints += "Use a real timestamp column, not `dt.hour`.\n"
                    comp_hints += "This method returns a protocol Dict for the center map renderer.\n"
                else:
                    comp_hints += "Choose the most suitable animated map SDK method based on available columns.\n"
                    comp_hints += "Map methods return protocol Dict objects.\n"
            else:
                if layer_type_str in ['polygon', 'choropleth', 'area']:
                    comp_hints += "You should prefer `stv.choropleth(...)`.\n"
                    comp_hints += "You MUST provide both `data_key` and `geo_key`.\n"
                    comp_hints += "This method returns a protocol Dict for the center map renderer.\n"
                elif layer_type_str in ['heatmap']:
                    comp_hints += "You should prefer `stv.heatmap(...)`.\n"
                    comp_hints += "This method returns a protocol Dict for the center map renderer.\n"
                else:
                    comp_hints += "You should prefer `stv.scatter_map(...)` for point-based maps.\n"
                    comp_hints += "This method returns a protocol Dict for the center map renderer.\n"

        else:
            comp_hints += f"Target Component ID: {comp_id}\n"
            comp_hints += "Generate one SDK-based visualization function for this component.\n"

        return comp_hints

    def _compose_prompt_sections(self, comp: Any) -> str:
        """
        内部拼装器：按组件类型动态注入最小必要 Prompt 内容。
        对外接口保持不变，仅在 scaffold 内部减 token。
        """
        is_dict = isinstance(comp, dict)
        c_type = comp.get('type') if is_dict else getattr(comp, 'type', 'unknown')
        c_type_str = str(c_type).split('.')[-1].lower()

        sections = [
            self.hard_constraints_core,
            self.coding_paradigm_core,
            self.semantic_rules_core,
            self.api_reference_ops,
        ]

        if c_type_str == 'map':
            sections.extend([
                self.api_reference_maps,
                self.examples_maps_basic
            ])

        elif c_type_str == 'chart':
            conf = comp.get('chart_config', {}) if is_dict else getattr(comp, 'chart_config', {})
            if hasattr(conf, "model_dump"):
                conf = conf.model_dump()
            conf = conf or {}
            chart_type = str(conf.get('chart_type', 'bar')).lower()

            sections.append(self.api_reference_charts)

            if chart_type == 'line':
                sections.extend([
                    self.api_reference_line_extra,
                    self.examples_chart_line_numeric,
                    self.examples_chart_line_count
                ])
            elif chart_type == 'pie':
                sections.append(self.examples_chart_bar)
            else:
                sections.append(self.examples_chart_bar)

        else:
            sections.append(self.api_reference_charts)

        return "\n\n".join(sections)

    def get_system_prompt(self, context_str: str, component_plans: List[Any] = None) -> str:
        comp_id = "unknown"
        comp_hints = ""
        comp = None

        if component_plans and len(component_plans) > 0:
            comp = component_plans[0]
            is_dict = isinstance(comp, dict)
            comp_id = comp.get('id') if is_dict else getattr(comp, 'id', 'unknown')
            comp_hints = self._build_component_hint(comp)

        safe_func_name = f"get_{comp_id.replace('-', '_')}"
        prompt_body = self._compose_prompt_sections(comp) if comp is not None else "\n\n".join([
            self.hard_constraints_core,
            self.coding_paradigm_core,
            self.semantic_rules_core,
            self.api_reference_ops,
            self.api_reference_charts
        ])

        prompt = f"""
You are an Expert Python Spatio-Temporal Data Scientist.

=== DATA METADATA (YOUR ONLY SOURCE OF TRUTH) ===
{context_str}

{prompt_body}

=== YOUR SPECIFIC TASK (CRITICAL) ===
You are currently inside a concurrent worker loop.
Your task is to write code for ONE SINGLE COMPONENT ONLY.

{comp_hints}

Additional rules:
- DO NOT wrap your code inside a get_dashboard_data function.
- DO NOT return a dashboard dictionary.
- Return exactly one component result.
- If the selected SDK method is a map method, return the protocol Dict directly.
- If the selected SDK method is a chart method, return the Figure directly.
- For count-based line charts, ALWAYS create a helper count column first, then resample using that column.
- NEVER use `safe_resample(..., val_col=None, agg_func='sum')`.

=== FINAL OUTPUT FORMAT ===
Return ONLY valid Python code block:
```python
def {safe_func_name}(data_context):
    # Step 1: Prep
    # Step 2: Render
    return result
"""
        return prompt