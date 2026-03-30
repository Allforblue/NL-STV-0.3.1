import logging
import re
import json
from typing import Dict, Any, List, Optional

from core.generation.dashboard_script_utils import (
    parse_dashboard_script,
    assemble_dashboard_script,
)

logger = logging.getLogger(__name__)


class SelectionApplier:
    """
    SelectionApplier 第一阶段完整迁移版

    核心能力：
    1. 支持基础 UI 交互：
       - bbox
       - selected_ids
       - selected_values
       - time_range
    2. 始终返回完整 dashboard script
    3. 与当前 workflow / executor 契约兼容
    4. 修复 runtime patch 累积问题：每次 patch 前会清理旧 patch
    5. 从旧 viz_editor 中迁移全部 selection / runtime patch 逻辑
    6. 开始消费 interaction_state：
       - 优先基于 active selections 生成 patch
       - append 模式可兼容 selected_ids / selected_values / time_range
       - compare 先保守兼容，不强推双上下文执行
    """

    def __init__(self):
        pass

    # =========================================================
    # Layout / Component 辅助
    # =========================================================
    def _normalize_component_type(self, comp_type: Any) -> str:
        if comp_type is None:
            return "unknown"
        text = str(comp_type)
        return text.split(".")[-1].lower()

    def _get_layout_components(self, last_layout: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not last_layout:
            return []
        return last_layout.get("components", []) or []

    def _get_component_by_id(
        self,
        last_layout: Optional[Dict[str, Any]],
        component_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if not component_id or not last_layout:
            return None
        for comp in self._get_layout_components(last_layout):
            if comp.get("id") == component_id:
                return comp
        return None

    def _get_component_map(self, last_layout: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            c.get("id"): c
            for c in self._get_layout_components(last_layout)
            if c.get("id")
        }

    def _extract_links_from_component(self, active_component: Optional[Any]) -> List[Any]:
        if active_component is None:
            return []

        if hasattr(active_component, "links"):
            return getattr(active_component, "links") or []

        if isinstance(active_component, dict):
            return active_component.get("links", []) or []

        return []

    def _extract_component_config(self, active_component: Optional[Any]) -> Dict[str, Any]:
        if active_component is None:
            return {}

        if hasattr(active_component, "chart_config") and getattr(active_component, "chart_config", None) is not None:
            cfg = getattr(active_component, "chart_config")
            if hasattr(cfg, "model_dump"):
                return cfg.model_dump()
            if isinstance(cfg, dict):
                return cfg

        if hasattr(active_component, "map_config") and getattr(active_component, "map_config", None):
            map_cfg = getattr(active_component, "map_config")
            if isinstance(map_cfg, list) and map_cfg:
                layer = map_cfg[0]
                if hasattr(layer, "model_dump"):
                    return layer.model_dump()
                if isinstance(layer, dict):
                    return layer

        if isinstance(active_component, dict):
            if isinstance(active_component.get("chart_config"), dict):
                return active_component.get("chart_config") or {}
            map_cfg = active_component.get("map_config")
            if isinstance(map_cfg, list) and map_cfg:
                first = map_cfg[0]
                if isinstance(first, dict):
                    return first

        return {}

    def _resolve_target_component_ids(
        self,
        payload: Any,
        last_layout: Optional[Dict[str, Any]],
        active_component: Optional[Any] = None
    ) -> List[str]:
        """
        UI patch 目标组件解析：
        - 优先 active component
        - 再加 active component 的 links 指向组件
        - 如果都没有，则对全部 map/chart/table/kpi 做保守 patch
        """
        comp_map = self._get_component_map(last_layout)
        targets: List[str] = []

        active_component_id = getattr(payload, "active_component_id", None)

        if active_component_id and active_component_id in comp_map:
            targets.append(active_component_id)

        links = self._extract_links_from_component(active_component)
        for link in links:
            if hasattr(link, "model_dump"):
                link = link.model_dump()
            if not isinstance(link, dict):
                continue

            target_id = link.get("target_id")
            if target_id and target_id in comp_map and target_id not in targets:
                targets.append(target_id)

        if getattr(payload, "selected_values", None):
            for comp_id, comp in comp_map.items():
                ctype = self._normalize_component_type(comp.get("type"))
                if ctype in ["map", "chart", "table", "kpi"] and comp_id not in targets:
                    targets.append(comp_id)

        if not targets:
            for comp_id, comp in comp_map.items():
                ctype = self._normalize_component_type(comp.get("type"))
                if ctype in ["map", "chart", "table", "kpi"]:
                    targets.append(comp_id)

        return targets

    # =========================================================
    # interaction_state 辅助
    # =========================================================
    def _get_active_selection_items(self, interaction_state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not interaction_state or not isinstance(interaction_state, dict):
            return []

        selections = interaction_state.get("selections", []) or []
        active_ids = interaction_state.get("active_selection_ids", []) or []
        active_id_set = set(active_ids)

        result = []
        for item in selections:
            if not isinstance(item, dict):
                continue
            if item.get("selection_id") in active_id_set:
                result.append(item)
        return result

    def _get_effective_selection_items(self, interaction_state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        第一阶段保守策略：
        - 普通 replace / append：使用 active_selection_ids 对应 selections
        - compare 开启时：优先使用 left_selection_ids 对应 selections
          （仅作兼容执行，不做真正 compare-aware 双上下文）
        """
        if not interaction_state or not isinstance(interaction_state, dict):
            return []

        comparison_state = interaction_state.get("comparison_state", {}) or {}
        selections = interaction_state.get("selections", []) or []

        if comparison_state.get("enabled"):
            left_ids = comparison_state.get("left_selection_ids", []) or []
            left_id_set = set(left_ids)
            left_items = [
                item for item in selections
                if isinstance(item, dict) and item.get("selection_id") in left_id_set
            ]
            if left_items:
                return left_items

        return self._get_active_selection_items(interaction_state)

    def _merge_selection_items(
        self,
        selection_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        第一阶段合并策略：
        - bbox: 仅保留最后一个（保守）
        - selected_ids: 做并集
        - selected_values: 同 key 最近值覆盖
        - time_range: 保留最后一个
        """
        merged = {
            "bbox": None,
            "selected_ids": [],
            "selected_values": {},
            "time_range": None
        }

        selected_id_set = set()

        for item in selection_items or []:
            if not isinstance(item, dict):
                continue

            raw_payload = item.get("raw_payload", {}) or {}

            if raw_payload.get("bbox"):
                merged["bbox"] = raw_payload.get("bbox")

            if raw_payload.get("selected_ids"):
                for v in raw_payload.get("selected_ids", []) or []:
                    sv = str(v)
                    if sv not in selected_id_set:
                        selected_id_set.add(sv)
                        merged["selected_ids"].append(sv)

            if raw_payload.get("selected_values"):
                vals = raw_payload.get("selected_values", {}) or {}
                for k, v in vals.items():
                    merged["selected_values"][str(k)] = v

            if raw_payload.get("time_range"):
                merged["time_range"] = raw_payload.get("time_range")

        return merged

    def _build_effective_payload_from_state(
        self,
        payload: Any,
        interaction_state: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        selection_items = self._get_effective_selection_items(interaction_state)

        if not selection_items:
            return {
                "bbox": getattr(payload, "bbox", None),
                "selected_ids": getattr(payload, "selected_ids", None),
                "selected_values": getattr(payload, "selected_values", None),
                "time_range": getattr(payload, "time_range", None),
            }

        merged = self._merge_selection_items(selection_items)
        return merged

    # =========================================================
    # 数据集 / 字段候选辅助
    # =========================================================
    def _find_geo_dataset_var(self, summaries: List[Dict[str, Any]]) -> Optional[str]:
        for s in summaries:
            if s.get("is_geospatial") or s.get("basic_stats", {}).get("is_geospatial", False):
                return s.get("variable_name")
        return None

    def _find_candidate_id_columns(self, summary: Dict[str, Any]) -> List[str]:
        candidates = []
        col_meta = summary.get("semantic_analysis", {}).get("column_metadata", {})
        col_stats = summary.get("column_stats", {}) or summary.get("basic_stats", {}).get("column_stats", {})

        for col, meta in col_meta.items():
            if isinstance(meta, dict) and meta.get("semantic_tag") in ["ST_LOC_ID", "ID_KEY"]:
                candidates.append(col)

        for col in col_stats.keys():
            lc = str(col).lower()
            if "id" in lc and col not in candidates and "transaction" not in lc:
                candidates.append(col)

        return candidates

    def _find_common_filter_columns(
        self,
        summaries: List[Dict[str, Any]],
        preferred_keys: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        返回:
            {
                dataset_var: [candidate_filter_cols]
            }
        """
        result = {}
        preferred_keys = [k for k in (preferred_keys or []) if k]

        for s in summaries:
            var_name = s.get("variable_name")
            col_stats = s.get("column_stats", {}) or s.get("basic_stats", {}).get("column_stats", {})
            sem_candidates = self._find_candidate_id_columns(s)

            cols = list(col_stats.keys())
            final_cols: List[str] = []

            for key in preferred_keys:
                if key in cols and key not in final_cols:
                    final_cols.append(key)

            for c in sem_candidates:
                if c in cols and c not in final_cols:
                    final_cols.append(c)

            result[var_name] = final_cols

        return result

    def _extract_preferred_link_keys(self, active_component: Optional[Any]) -> List[str]:
        preferred_keys = []
        links = self._extract_links_from_component(active_component)

        for l in links:
            if hasattr(l, "model_dump"):
                l = l.model_dump()
            if not isinstance(l, dict):
                continue
            lk = l.get("link_key")
            if lk:
                preferred_keys.append(lk)

        cfg = self._extract_component_config(active_component)
        for key_name in ["interaction_key_field", "fact_link_field", "interaction_field"]:
            val = cfg.get(key_name)
            if val and val not in preferred_keys:
                preferred_keys.append(val)

        render_meta = cfg.get("render_meta", {}) or {}
        for key_name in ["interaction_key_field", "fact_link_field", "interaction_field"]:
            val = render_meta.get(key_name)
            if val and val not in preferred_keys:
                preferred_keys.append(val)

        return preferred_keys

    def _find_columns_for_selected_value_filter(
        self,
        summaries: List[Dict[str, Any]],
        active_component: Optional[Any] = None,
        selected_values: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        为 selected_values 构造更通用的列候选：
        1. 优先 selected_values 的 key 本身
        2. 再加 component config / render_meta 中的 interaction 字段
        """
        result = {}
        selected_keys = [str(k) for k in (selected_values or {}).keys()]

        cfg = self._extract_component_config(active_component)
        component_hints = []
        for key_name in ["interaction_field", "interaction_key_field", "fact_link_field"]:
            val = cfg.get(key_name)
            if val:
                component_hints.append(val)

        render_meta = cfg.get("render_meta", {}) or {}
        for key_name in ["interaction_field", "interaction_key_field", "fact_link_field"]:
            val = render_meta.get(key_name)
            if val and val not in component_hints:
                component_hints.append(val)

        for s in summaries:
            var_name = s.get("variable_name")
            col_stats = s.get("column_stats", {}) or s.get("basic_stats", {}).get("column_stats", {})
            cols = list(col_stats.keys())
            candidates = []

            for k in selected_keys:
                if k in cols and k not in candidates:
                    candidates.append(k)

            for h in component_hints:
                if h in cols and h not in candidates:
                    candidates.append(h)

            result[var_name] = candidates

        return result

    # =========================================================
    # Runtime Patch 清理 / 注入
    # =========================================================
    def _strip_runtime_patch_lines(self, lines: List[str]) -> List[str]:
        """
        删除函数头部历史注入的 runtime patch 块，避免 bbox / selected_ids /
        selected_values / time_range 累积。
        """
        if len(lines) <= 1:
            return lines

        header = lines[0]
        body = lines[1:]

        runtime_prefixes = (
            "    # --- Runtime BBox Filter (Stage-1) ---",
            "    # --- Runtime Selected IDs Filter (Stage-1) ---",
            "    # --- Runtime Selected Values Filter (Stage-1) ---",
            "    # --- Runtime Time Range Filter (Stage-1) ---",
        )

        cleaned_body: List[str] = []
        i = 0

        while i < len(body):
            line = body[i]

            if any(line.startswith(prefix) for prefix in runtime_prefixes):
                i += 1

                while i < len(body):
                    cur = body[i]

                    if any(cur.startswith(prefix) for prefix in runtime_prefixes):
                        break

                    stripped = cur.strip()

                    if stripped.startswith("#") and "Runtime" not in stripped:
                        break

                    if stripped.startswith("return "):
                        break
                    if stripped.startswith("df_") or stripped.startswith("gdf_"):
                        break
                    if stripped.startswith("data_context.get("):
                        break

                    i += 1

                continue

            cleaned_body.extend(body[i:])
            break

        return [header] + cleaned_body

    def _strip_existing_runtime_patches(self, func_code: str) -> str:
        lines = func_code.splitlines()
        if not lines:
            return func_code

        if not re.match(r"^def\s+get_[a-zA-Z0-9_]+\(data_context\):\s*$", lines[0].strip()):
            return func_code

        cleaned_lines = self._strip_runtime_patch_lines(lines)
        return "\n".join(cleaned_lines)

    def _inject_preamble_into_function(self, func_code: str, preamble_lines: List[str]) -> str:
        """
        将 preamble 注入到函数签名之后。
        注入前假设已经清理过旧 runtime patch。
        """
        if not preamble_lines:
            return func_code

        lines = func_code.splitlines()
        if not lines:
            return func_code

        if not re.match(r"^def\s+get_[a-zA-Z0-9_]+\(data_context\):\s*$", lines[0].strip()):
            return func_code

        new_lines = [lines[0]]
        for pl in preamble_lines:
            new_lines.append(f"    {pl}")
        if len(lines) > 1:
            new_lines.extend(lines[1:])

        return "\n".join(new_lines)

    # =========================================================
    # Runtime Filter Preamble 构造
    # =========================================================
    def _build_bbox_preamble(
        self,
        summaries: List[Dict[str, Any]],
        active_component: Optional[Any] = None,
        bbox: Optional[List[float]] = None
    ) -> List[str]:
        if not bbox or len(bbox) != 4:
            return []

        min_lon, min_lat, max_lon, max_lat = bbox
        geo_var = self._find_geo_dataset_var(summaries)

        preferred_keys = self._extract_preferred_link_keys(active_component)

        filter_cols_map = self._find_common_filter_columns(
            summaries=summaries,
            preferred_keys=preferred_keys
        )

        lines: List[str] = []
        lines.append("# --- Runtime BBox Filter (Stage-1) ---")
        lines.append("_runtime_valid_ids = set()")

        if geo_var:
            geo_summary = next((s for s in summaries if s.get("variable_name") == geo_var), None)
            geo_id_candidates = self._find_candidate_id_columns(geo_summary or {})

            lines.append(f"if '{geo_var}' in data_context:")
            lines.append(f"    _gdf_runtime = data_context['{geo_var}'].copy()")
            lines.append("    if hasattr(_gdf_runtime, 'crs') and str(_gdf_runtime.crs) != 'EPSG:4326':")
            lines.append("        _gdf_runtime = _gdf_runtime.to_crs(epsg=4326)")
            lines.append(f"    _gdf_runtime = _gdf_runtime.cx[{min_lon}:{max_lon}, {min_lat}:{max_lat}]")
            lines.append(f"    data_context['{geo_var}'] = _gdf_runtime")

            for c in geo_id_candidates[:2]:
                lines.append(f"    if '{c}' in _gdf_runtime.columns:")
                lines.append(f"        _runtime_valid_ids.update(_gdf_runtime['{c}'].dropna().astype(str).tolist())")
                lines.append(f"    elif _gdf_runtime.index.name == '{c}':")
                lines.append("        _runtime_valid_ids.update([str(v) for v in _gdf_runtime.index.tolist()])")

        for var_name, candidates in filter_cols_map.items():
            if not candidates:
                continue
            if geo_var and var_name == geo_var:
                continue

            match_col = candidates[0]
            lines.append(f"if '{var_name}' in data_context and _runtime_valid_ids:")
            lines.append(f"    _df_runtime = data_context['{var_name}'].copy()")
            lines.append(f"    if '{match_col}' in _df_runtime.columns:")
            lines.append(
                f"        data_context['{var_name}'] = _df_runtime[_df_runtime['{match_col}'].astype(str).isin(_runtime_valid_ids)]"
            )

        return lines

    def _build_selected_ids_preamble(
        self,
        summaries: List[Dict[str, Any]],
        active_component: Optional[Any] = None,
        selected_ids: Optional[List[Any]] = None
    ) -> List[str]:
        if not selected_ids:
            return []

        preferred_keys = self._extract_preferred_link_keys(active_component)

        filter_cols_map = self._find_common_filter_columns(
            summaries=summaries,
            preferred_keys=preferred_keys
        )

        lines: List[str] = []
        ids_json = json.dumps([str(v) for v in selected_ids], ensure_ascii=False)

        lines.append("# --- Runtime Selected IDs Filter (Stage-1) ---")
        lines.append(f"_runtime_selected_ids = set({ids_json})")

        for var_name, candidates in filter_cols_map.items():
            if not candidates:
                continue

            match_col = candidates[0]
            lines.append(f"if '{var_name}' in data_context and _runtime_selected_ids:")
            lines.append(f"    _df_runtime = data_context['{var_name}'].copy()")
            lines.append(f"    if '{match_col}' in _df_runtime.columns:")
            lines.append(
                f"        data_context['{var_name}'] = _df_runtime[_df_runtime['{match_col}'].astype(str).isin(_runtime_selected_ids)]"
            )

        return lines

    def _build_selected_values_preamble(
        self,
        summaries: List[Dict[str, Any]],
        active_component: Optional[Any] = None,
        selected_values: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Runtime selected_values filter（升级版）：
        1. 优先走通用列等值过滤
           - selected_values 的 key 命中数据表字段
           - active component 的 interaction_field / interaction_key_field / fact_link_field
           - render_meta 中的同名字段
        2. 若没有可用通用列，再 fallback 到 Taxi Zone/Borough/LocationID 逻辑
        """
        if not selected_values:
            return []

        values_json = json.dumps(selected_values, ensure_ascii=False)
        preferred_cols_map = self._find_columns_for_selected_value_filter(
            summaries=summaries,
            active_component=active_component,
            selected_values=selected_values
        )

        lines: List[str] = []
        lines.append("# --- Runtime Selected Values Filter (Stage-1) ---")
        lines.append(f"_runtime_selected_values = {values_json}")

        # 通用路径：优先直接列过滤
        lines.append("for _var_name, _df_runtime in list(data_context.items()):")
        lines.append("    if not hasattr(_df_runtime, 'columns'):")
        lines.append("        continue")
        lines.append("    _df_runtime = _df_runtime.copy()")
        lines.append("    _applied = False")

        for var_name, candidates in preferred_cols_map.items():
            if not candidates:
                continue

            lines.append(f"    if _var_name == '{var_name}':")
            for col in candidates:
                lines.append(f"        if '{col}' in _df_runtime.columns and '{col}' in _runtime_selected_values:")
                lines.append(f"            _df_runtime = _df_runtime[_df_runtime['{col}'].astype(str) == str(_runtime_selected_values['{col}'])]")
                lines.append("            _applied = True")

        lines.append("    if not _applied:")
        lines.append("        for _col, _val in _runtime_selected_values.items():")
        lines.append("            if _col in _df_runtime.columns:")
        lines.append("                _df_runtime = _df_runtime[_df_runtime[_col].astype(str) == str(_val)]")
        lines.append("                _applied = True")

        lines.append("    if _applied:")
        lines.append("        data_context[_var_name] = _df_runtime")

        # Taxi fallback：保留兼容
        lines.append("_runtime_lookup_df = None")
        lines.append("for _var_name, _df_runtime in list(data_context.items()):")
        lines.append("    if not hasattr(_df_runtime, 'columns'):")
        lines.append("        continue")
        lines.append("    _runtime_cols = set([str(c) for c in _df_runtime.columns])")
        lines.append("    if 'LocationID' in _runtime_cols and ('Zone' in _runtime_cols or 'Borough' in _runtime_cols):")
        lines.append("        _runtime_lookup_df = _df_runtime.copy()")
        lines.append("        break")

        lines.append("_runtime_location_ids = None")
        lines.append("if _runtime_lookup_df is not None:")
        lines.append("    if 'Zone' in _runtime_selected_values:")
        lines.append("        _val = _runtime_selected_values['Zone']")
        lines.append("        _runtime_location_ids = _runtime_lookup_df[_runtime_lookup_df['Zone'].astype(str) == str(_val)]['LocationID'].dropna().tolist()")
        lines.append("    elif 'Borough' in _runtime_selected_values:")
        lines.append("        _val = _runtime_selected_values['Borough']")
        lines.append("        _runtime_location_ids = _runtime_lookup_df[_runtime_lookup_df['Borough'].astype(str) == str(_val)]['LocationID'].dropna().tolist()")
        lines.append("    elif 'LocationID' in _runtime_selected_values:")
        lines.append("        _val = _runtime_selected_values['LocationID']")
        lines.append("        _runtime_location_ids = _runtime_lookup_df[_runtime_lookup_df['LocationID'].astype(str) == str(_val)]['LocationID'].dropna().tolist()")

        lines.append("if _runtime_location_ids is not None:")
        lines.append("    _runtime_location_ids = [str(v) for v in _runtime_location_ids]")

        lines.append("for _var_name, _df_runtime in list(data_context.items()):")
        lines.append("    if _runtime_location_ids is None:")
        lines.append("        break")
        lines.append("    if not hasattr(_df_runtime, 'columns'):")
        lines.append("        continue")
        lines.append("    _df_runtime = _df_runtime.copy()")
        lines.append("    _applied = False")
        lines.append("    if 'PULocationID' in _df_runtime.columns:")
        lines.append("        _df_runtime = _df_runtime[_df_runtime['PULocationID'].astype(str).isin(_runtime_location_ids)]")
        lines.append("        _applied = True")
        lines.append("    elif 'DOLocationID' in _df_runtime.columns:")
        lines.append("        _df_runtime = _df_runtime[_df_runtime['DOLocationID'].astype(str).isin(_runtime_location_ids)]")
        lines.append("        _applied = True")
        lines.append("    elif 'LocationID' in _df_runtime.columns:")
        lines.append("        _df_runtime = _df_runtime[_df_runtime['LocationID'].astype(str).isin(_runtime_location_ids)]")
        lines.append("        _applied = True")
        lines.append("    elif 'zone' in _df_runtime.columns and 'Zone' in _runtime_selected_values:")
        lines.append("        _df_runtime = _df_runtime[_df_runtime['zone'].astype(str) == str(_runtime_selected_values['Zone'])]")
        lines.append("        _applied = True")
        lines.append("    elif 'borough' in _df_runtime.columns and 'Borough' in _runtime_selected_values:")
        lines.append("        _df_runtime = _df_runtime[_df_runtime['borough'].astype(str) == str(_runtime_selected_values['Borough'])]")
        lines.append("        _applied = True")
        lines.append("    if _applied:")
        lines.append("        data_context[_var_name] = _df_runtime")

        return lines

    def _build_time_range_preamble(
        self,
        time_range: Optional[List[str]] = None
    ) -> List[str]:
        """
        Runtime time_range filter:
        - 对 data_context 中所有包含常见时间列名的数据表进行时间裁剪
        """
        if not time_range or len(time_range) != 2:
            return []

        start_time, end_time = time_range

        lines: List[str] = []
        lines.append("# --- Runtime Time Range Filter (Stage-1) ---")
        lines.append(f"_runtime_time_start = '{start_time}'")
        lines.append(f"_runtime_time_end = '{end_time}'")
        lines.append("for _var_name, _df_runtime in list(data_context.items()):")
        lines.append("    if not hasattr(_df_runtime, 'columns'):")
        lines.append("        continue")
        lines.append("    _df_runtime = _df_runtime.copy()")
        lines.append("    _applied = False")
        lines.append("    for _col in _df_runtime.columns:")
        lines.append("        _col_lower = str(_col).lower()")
        lines.append("        if any(_kw in _col_lower for _kw in ['time', 'date', 'datetime', 'timestamp']):")
        lines.append("            try:")
        lines.append("                _series = pd.to_datetime(_df_runtime[_col], errors='coerce')")
        lines.append("                _df_runtime = _df_runtime[(_series >= pd.to_datetime(_runtime_time_start)) & (_series <= pd.to_datetime(_runtime_time_end))]")
        lines.append("                _applied = True")
        lines.append("                break")
        lines.append("            except Exception:")
        lines.append("                pass")
        lines.append("    if _applied:")
        lines.append("        data_context[_var_name] = _df_runtime")

        return lines

    def _build_combined_preamble(
        self,
        summaries: List[Dict[str, Any]],
        active_component: Optional[Any] = None,
        effective_payload: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        effective_payload = effective_payload or {}
        lines: List[str] = []

        bbox = effective_payload.get("bbox")
        selected_ids = effective_payload.get("selected_ids")
        selected_values = effective_payload.get("selected_values")
        time_range = effective_payload.get("time_range")

        if bbox:
            lines.extend(self._build_bbox_preamble(
                summaries=summaries,
                active_component=active_component,
                bbox=bbox
            ))

        if selected_ids:
            lines.extend(self._build_selected_ids_preamble(
                summaries=summaries,
                active_component=active_component,
                selected_ids=selected_ids
            ))

        if selected_values:
            lines.extend(self._build_selected_values_preamble(
                summaries=summaries,
                active_component=active_component,
                selected_values=selected_values
            ))

        if time_range:
            lines.extend(self._build_time_range_preamble(
                time_range=time_range
            ))

        return lines

    # =========================================================
    # Runtime UI Patch 主逻辑
    # =========================================================
    def _patch_components_with_preamble(
        self,
        parsed: Dict[str, Any],
        target_component_ids: List[str],
        preamble_lines: List[str]
    ) -> str:
        component_functions = parsed["component_functions"].copy()
        updated_functions = {}

        for comp_id, info in component_functions.items():
            code = info["code"]

            if comp_id in target_component_ids:
                code = self._strip_existing_runtime_patches(code)
                code = self._inject_preamble_into_function(code, preamble_lines)

            updated_functions[comp_id] = {
                **info,
                "code": code
            }

        return assemble_dashboard_script(parsed, updated_functions)

    def _patch_from_effective_payload(
        self,
        parsed: Dict[str, Any],
        payload: Any,
        summaries: List[Dict[str, Any]],
        last_layout: Optional[Dict[str, Any]],
        active_component: Optional[Any],
        effective_payload: Dict[str, Any]
    ) -> str:
        target_ids = self._resolve_target_component_ids(payload, last_layout, active_component)
        preamble = self._build_combined_preamble(
            summaries=summaries,
            active_component=active_component,
            effective_payload=effective_payload
        )
        return self._patch_components_with_preamble(parsed, target_ids, preamble)

    def _patch_bbox_filter(
        self,
        parsed: Dict[str, Any],
        payload: Any,
        summaries: List[Dict[str, Any]],
        last_layout: Optional[Dict[str, Any]],
        active_component: Optional[Any] = None
    ) -> str:
        target_ids = self._resolve_target_component_ids(payload, last_layout, active_component)
        preamble = self._build_bbox_preamble(
            summaries=summaries,
            active_component=active_component,
            bbox=payload.bbox
        )
        return self._patch_components_with_preamble(parsed, target_ids, preamble)

    def _patch_selected_ids_filter(
        self,
        parsed: Dict[str, Any],
        payload: Any,
        summaries: List[Dict[str, Any]],
        last_layout: Optional[Dict[str, Any]],
        active_component: Optional[Any] = None
    ) -> str:
        target_ids = self._resolve_target_component_ids(payload, last_layout, active_component)
        preamble = self._build_selected_ids_preamble(
            summaries=summaries,
            active_component=active_component,
            selected_ids=payload.selected_ids
        )
        return self._patch_components_with_preamble(parsed, target_ids, preamble)

    def _patch_selected_values_filter(
        self,
        parsed: Dict[str, Any],
        payload: Any,
        summaries: List[Dict[str, Any]],
        last_layout: Optional[Dict[str, Any]],
        active_component: Optional[Any] = None
    ) -> str:
        target_ids = self._resolve_target_component_ids(payload, last_layout, active_component)

        logger.info(f"[SelectionApplier] selected_values patch target_ids = {target_ids}")
        logger.info(f"[SelectionApplier] selected_values payload = {getattr(payload, 'selected_values', None)}")

        preamble = self._build_selected_values_preamble(
            summaries=summaries,
            active_component=active_component,
            selected_values=payload.selected_values
        )

        return self._patch_components_with_preamble(parsed, target_ids, preamble)

    def _patch_time_range_filter(
        self,
        parsed: Dict[str, Any],
        payload: Any,
        last_layout: Optional[Dict[str, Any]],
        active_component: Optional[Any] = None
    ) -> str:
        target_ids = self._resolve_target_component_ids(payload, last_layout, active_component)
        preamble = self._build_time_range_preamble(
            time_range=payload.time_range
        )
        return self._patch_components_with_preamble(parsed, target_ids, preamble)

    # =========================================================
    # 主入口
    # =========================================================
    def apply_to_code(
        self,
        original_code: str,
        payload: Any,
        interaction_state: Dict[str, Any],
        summaries: List[Dict[str, Any]],
        last_layout: Optional[Dict[str, Any]] = None,
        active_component: Optional[Any] = None
    ) -> str:
        """
        第一阶段统一入口：
        - 优先基于 interaction_state 中的 active selections 生成 runtime patch
        - 若 state 不可用，则回退到当前 payload 驱动 patch
        - 始终返回完整 dashboard script
        """
        try:
            parsed = parse_dashboard_script(original_code)
        except Exception as e:
            logger.error(f"[SelectionApplier] Failed to parse dashboard script: {e}")
            return original_code

        try:
            effective_payload = self._build_effective_payload_from_state(
                payload=payload,
                interaction_state=interaction_state
            )

            if any([
                effective_payload.get("bbox"),
                effective_payload.get("selected_ids"),
                effective_payload.get("selected_values"),
                effective_payload.get("time_range"),
            ]):
                logger.info(
                    f"[SelectionApplier] Applying state-aware runtime patch | "
                    f"bbox={bool(effective_payload.get('bbox'))}, "
                    f"selected_ids={len(effective_payload.get('selected_ids') or [])}, "
                    f"selected_values_keys={list((effective_payload.get('selected_values') or {}).keys())}, "
                    f"time_range={bool(effective_payload.get('time_range'))}"
                )

                new_code = self._patch_from_effective_payload(
                    parsed=parsed,
                    payload=payload,
                    summaries=summaries,
                    last_layout=last_layout,
                    active_component=active_component,
                    effective_payload=effective_payload
                )
                return new_code or original_code

            if getattr(payload, "bbox", None):
                new_code = self._patch_bbox_filter(
                    parsed=parsed,
                    payload=payload,
                    summaries=summaries,
                    last_layout=last_layout,
                    active_component=active_component
                )
                return new_code or original_code

            if getattr(payload, "selected_ids", None):
                new_code = self._patch_selected_ids_filter(
                    parsed=parsed,
                    payload=payload,
                    summaries=summaries,
                    last_layout=last_layout,
                    active_component=active_component
                )
                return new_code or original_code

            if getattr(payload, "selected_values", None):
                new_code = self._patch_selected_values_filter(
                    parsed=parsed,
                    payload=payload,
                    summaries=summaries,
                    last_layout=last_layout,
                    active_component=active_component
                )
                return new_code or original_code

            if getattr(payload, "time_range", None):
                new_code = self._patch_time_range_filter(
                    parsed=parsed,
                    payload=payload,
                    last_layout=last_layout,
                    active_component=active_component
                )
                return new_code or original_code

            logger.info("[SelectionApplier] No actionable selection payload found, returning original code.")
            return original_code

        except Exception as e:
            logger.error(f"[SelectionApplier] apply_to_code failed: {e}", exc_info=True)
            return original_code