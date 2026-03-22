# backend/core/sdk/layers.py
import pandas as pd
import geopandas as gpd
from typing import Dict, Any, List


class MapLayerBuilder:
    """生成与前端 Deck.gl 映射的中立协议字典"""

    @staticmethod
    def scatter_layer(df: pd.DataFrame, lat_col: str, lon_col: str, color_col: str = None, id_suffix: str = "1") -> \
    Dict[str, Any]:
        df = df.dropna(subset=[lat_col, lon_col])
        # 仅提取需要的列发给前端，极大节省带宽
        cols_to_keep = [lat_col, lon_col]
        if color_col: cols_to_keep.append(color_col)

        return {
            "layer_type": "ScatterplotLayer",
            "layer_id": f"scatter_{id_suffix}",
            "data": df[cols_to_keep].to_dict(orient="records"),
            "mapping": {
                "longitude": lon_col,
                "latitude": lat_col,
                "color_value": color_col
            }
        }

    @staticmethod
    def geojson_layer(gdf: gpd.GeoDataFrame, fill_color: list = [255, 0, 0, 100], id_suffix: str = "1") -> Dict[
        str, Any]:
        """用于区域高亮或色块图底图"""
        import json
        return {
            "layer_type": "GeoJsonLayer",
            "layer_id": f"geojson_{id_suffix}",
            "data": json.loads(gdf.to_json()),
            "style": {
                "fillColor": fill_color,
                "stroked": True,
                "lineWidthMinPixels": 1
            }
        }