# core/generation/view_transformer.py

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

from core.generation.view_ir import (
    ViewIR,
    ViewIRPatch,
    merge_view_ir,
)


class ViewTransformer:
    """
    轻量 View IR 变换器（第一阶段）

    职责：
    1. 从用户 query 中提取结构化编辑 patch
    2. 将 patch 应用到当前 ViewIR
    3. 根据语义规则自动补全缺失字段
    4. 做轻量合法性修正
    5. 返回更新后的 ViewIR + warning 信息

    当前优先支持：
    - 改成折线图
    - 按月 / 按天 / 按小时展示
    - 改成占比图
    - 改成排名图

    设计原则：
    - 规则优先，不依赖 LLM
    - 只做 chart 组件语义变换
    - 不推翻现有 chart_config 体系，只作为中间语义层
    """

    def infer_patch_from_query(
        self,
        query: str,
        current_ir: Optional[ViewIR] = None
    ) -> ViewIRPatch:
        """
        从用户自然语言中提取结构化 ViewIRPatch。

        当前阶段：
        - 主要通过规则识别
        - 保持保守，避免误改 dimension / metric
        """
        query = query or ""
        q = query.strip().lower()

        patch_data: Dict[str, Any] = {}

        # ---------------------------------------------------------
        # 标题修改
        # ---------------------------------------------------------
        title_match = re.search(
            r"(?:标题改成|标题换成|title to|rename to)\s*[:：]?\s*(.+)$",
            query,
            re.IGNORECASE
        )
        if title_match:
            patch_data["title"] = title_match.group(1).strip()

        # ---------------------------------------------------------
        # 图类型 / 意图识别
        # 规则优先级：
        # 1. 明确图类型词
        # 2. 语义意图词
        # ---------------------------------------------------------

        # 折线 / 趋势
        if self._contains_any(q, [
            "折线图", "line chart", "line"
        ]):
            patch_data["chart_type"] = "line"
            patch_data["analysis_intent"] = "trend"

        # 柱状 / 条形
        elif self._contains_any(q, [
            "柱状图", "条形图", "bar chart", "bar"
        ]):
            patch_data["chart_type"] = "bar"

        # 饼图 / 占比
        elif self._contains_any(q, [
            "饼图", "pie chart", "pie"
        ]):
            patch_data["chart_type"] = "pie"
            patch_data["analysis_intent"] = "composition"

        # 热力图
        elif self._contains_any(q, [
            "热力图", "heatmap"
        ]):
            patch_data["chart_type"] = "heatmap"

        # 散点图
        elif self._contains_any(q, [
            "散点图", "scatter plot", "scatter"
        ]):
            patch_data["chart_type"] = "scatter"

        # 面积图
        elif self._contains_any(q, [
            "面积图", "area chart", "area"
        ]):
            patch_data["chart_type"] = "area"
            patch_data.setdefault("analysis_intent", "trend")

        # ---------------------------------------------------------
        # 基于语义意图词补充
        # ---------------------------------------------------------
        if self._contains_any(q, [
            "趋势", "变化", "随时间", "走势", "波动", "time trend", "over time"
        ]):
            patch_data.setdefault("analysis_intent", "trend")
            patch_data.setdefault("chart_type", "line")

        if self._contains_any(q, [
            "构成", "占比", "比例", "份额", "composition", "proportion", "share"
        ]):
            patch_data.setdefault("analysis_intent", "composition")
            patch_data.setdefault("chart_type", "pie")

        if self._contains_any(q, [
            "排名", "排行", "top", "bottom", "最多", "最少", "highest", "lowest", "rank"
        ]):
            patch_data.setdefault("analysis_intent", "ranking")
            patch_data.setdefault("chart_type", "bar")
            patch_data.setdefault("sort_order", "desc")

        # ---------------------------------------------------------
        # 时间粒度识别
        # ---------------------------------------------------------
        if self._contains_any(q, [
            "按月", "每月", "月份", "monthly", "by month"
        ]):
            patch_data["time_bucket"] = "1M"
            patch_data.setdefault("analysis_intent", "trend")

        elif self._contains_any(q, [
            "按天", "每天", "日趋势", "daily", "by day"
        ]):
            patch_data["time_bucket"] = "1D"
            patch_data.setdefault("analysis_intent", "trend")

        elif self._contains_any(q, [
            "按小时", "每小时", "小时", "hourly", "by hour"
        ]):
            patch_data["time_bucket"] = "1H"
            patch_data.setdefault("analysis_intent", "trend")

        # ---------------------------------------------------------
        # 排序方向进一步识别
        # ---------------------------------------------------------
        if self._contains_any(q, [
            "升序", "从小到大", "ascending", "asc"
        ]):
            patch_data["sort_order"] = "asc"

        elif self._contains_any(q, [
            "降序", "从大到小", "descending", "desc"
        ]):
            patch_data["sort_order"] = "desc"

        # ---------------------------------------------------------
        # limit 粗略识别（如 top 10）
        # ---------------------------------------------------------
        top_n = self._extract_top_n(q)
        if top_n is not None:
            patch_data["limit"] = top_n
            patch_data.setdefault("analysis_intent", "ranking")
            patch_data.setdefault("chart_type", "bar")
            patch_data.setdefault("sort_order", "desc")

        return ViewIRPatch(**patch_data)

    def transform(
        self,
        query: str,
        current_ir: ViewIR
    ) -> Dict[str, Any]:
        """
        主入口：
        current_ir + query -> patch -> resolved_ir

        返回：
        {
            "patch": {...},
            "view_ir": {...},
            "warnings": [...]
        }
        """
        patch = self.infer_patch_from_query(query=query, current_ir=current_ir)
        merged_ir = merge_view_ir(current_ir, patch)
        resolved_ir, warnings = self.validate_and_complete(merged_ir, query=query)

        return {
            "patch": patch.model_dump(exclude_none=True),
            "view_ir": resolved_ir.model_dump(),
            "warnings": warnings,
        }

    def validate_and_complete(
        self,
        view_ir: ViewIR,
        query: str = ""
    ) -> (ViewIR, List[str]):
        """
        对 ViewIR 做自动补全与轻量合法性修正。

        当前重点规则：
        1. 折线图 <-> 趋势
        2. time_bucket -> 趋势
        3. 占比图 -> 构成
        4. 排名图 -> bar + 排序
        5. 尽量不主动篡改 dimension / metric
        """
        ir = deepcopy(view_ir)
        warnings: List[str] = []
        q = (query or "").strip().lower()

        # ---------------------------------------------------------
        # 1. chart_type / intent 互相补全
        # ---------------------------------------------------------
        if ir.chart_type == "line":
            if ir.analysis_intent == "unknown":
                ir.analysis_intent = "trend"

        if ir.chart_type == "area":
            if ir.analysis_intent == "unknown":
                ir.analysis_intent = "trend"

        if ir.chart_type == "pie":
            ir.analysis_intent = "composition"

        if ir.analysis_intent == "trend":
            if not ir.chart_type or ir.chart_type in {"unknown"}:
                ir.chart_type = "line"

        if ir.analysis_intent == "composition":
            ir.chart_type = "pie"

        if ir.analysis_intent == "ranking":
            if not ir.chart_type or ir.chart_type in {"unknown", "line", "pie"}:
                ir.chart_type = "bar"

        # ---------------------------------------------------------
        # 2. time_bucket 驱动趋势语义
        # ---------------------------------------------------------
        if ir.time_bucket:
            if ir.analysis_intent == "unknown":
                ir.analysis_intent = "trend"

            # 对“按月/按天/按小时展示”这种编辑，默认更倾向折线图
            if ir.chart_type in {None, "unknown"}:
                ir.chart_type = "line"

            # 如果 query 明确带有趋势语义，也强制 line
            if self._contains_any(q, [
                "趋势", "变化", "随时间", "走势", "波动", "over time", "trend"
            ]):
                ir.chart_type = "line"

        # ---------------------------------------------------------
        # 3. composition 规则
        # ---------------------------------------------------------
        if ir.analysis_intent == "composition":
            ir.chart_type = "pie"

            # 占比图通常不保留时间粒度
            # 这里采取保守策略：
            # 若 query 明确在表达“改成占比图”，则清空 time_bucket
            # 避免 "按月看占比" 这类未来需求被完全覆盖
            if self._contains_any(q, [
                "占比", "构成", "比例", "份额", "pie", "composition", "proportion", "share"
            ]):
                if ir.time_bucket:
                    ir.time_bucket = None

            if not ir.dimension:
                warnings.append("composition chart may need a categorical dimension, but current view_ir.dimension is empty.")

            if not ir.metric:
                warnings.append("composition chart may need an aggregated metric or count-like value, but current view_ir.metric is empty.")

        # ---------------------------------------------------------
        # 4. ranking 规则
        # ---------------------------------------------------------
        if ir.analysis_intent == "ranking":
            ir.chart_type = "bar"

            if not ir.sort_order:
                ir.sort_order = "desc"

            if not ir.sort_by and ir.metric:
                ir.sort_by = ir.metric

            if ir.time_bucket and not self._contains_any(q, [
                "按月排名", "按天排名", "按小时排名",
                "monthly ranking", "daily ranking", "hourly ranking"
            ]):
                # 排名图多数情况下不是时间连续趋势图，这里给 warning，但不强制清空
                warnings.append(
                    "ranking chart still keeps time_bucket; make sure the regenerated code aggregates and ranks on the intended temporal slice."
                )

            if not ir.metric:
                warnings.append("ranking chart may need a metric for sorting, but current view_ir.metric is empty.")

        # ---------------------------------------------------------
        # 5. trend 规则
        # ---------------------------------------------------------
        if ir.analysis_intent == "trend":
            if not ir.chart_type or ir.chart_type in {"unknown", "pie"}:
                ir.chart_type = "line"

            # 若用户明确要求按月/按天/按小时，必须有 time_bucket
            # 已在 patch 中处理；此处只做保守 warning
            if self._contains_any(q, [
                "按月", "每月", "月份", "monthly", "by month",
                "按天", "每天", "daily", "by day",
                "按小时", "每小时", "hourly", "by hour"
            ]) and not ir.time_bucket:
                warnings.append("trend edit mentions a temporal granularity, but time_bucket is still empty.")

            # 趋势图若完全没有时间字段信息，先不强行伪造 time_field
            # 只给 warning，避免编造字段名
            if not ir.time_field and not self._looks_like_time_axis(ir.x):
                warnings.append(
                    "trend chart may need a time_field or time-like x axis, but current view_ir does not provide one explicitly."
                )

        # ---------------------------------------------------------
        # 6. 编码层轻量补全
        # ---------------------------------------------------------
        if not ir.x and ir.dimension:
            # 对 bar / pie，这个补全是合理的
            if ir.chart_type in {"bar", "pie", "heatmap", "scatter"}:
                ir.x = ir.dimension

        if not ir.y and ir.metric:
            if ir.chart_type in {"bar", "line", "area", "scatter"}:
                ir.y = ir.metric

        # 排名图一般按 dimension vs metric
        if ir.analysis_intent == "ranking":
            if not ir.x and ir.dimension:
                ir.x = ir.dimension
            if not ir.y and ir.metric:
                ir.y = ir.metric

        # 趋势图尽量让 y 对齐 metric
        if ir.analysis_intent == "trend" and ir.metric and not ir.y:
            ir.y = ir.metric

        # ---------------------------------------------------------
        # 7. 合法性收尾
        # ---------------------------------------------------------
        if ir.sort_order and ir.sort_order not in {"asc", "desc"}:
            warnings.append(f"invalid sort_order={ir.sort_order}, reset to None.")
            ir.sort_order = None

        if ir.limit is not None:
            try:
                if int(ir.limit) <= 0:
                    warnings.append("limit must be positive; reset to None.")
                    ir.limit = None
            except Exception:
                warnings.append("limit is not an integer; reset to None.")
                ir.limit = None

        return ir, warnings

    # =========================================================
    # 内部规则工具
    # =========================================================

    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        if not text:
            return False
        return any(k in text for k in keywords)

    def _extract_top_n(self, text: str) -> Optional[int]:
        """
        识别 top 10 / 前10 / 前 10 / top10 等表达
        """
        if not text:
            return None

        patterns = [
            r"\btop\s*(\d+)\b",
            r"前\s*(\d+)\s*",
            r"\btop(\d+)\b",
        ]

        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
        return None

    def _looks_like_time_axis(self, field_name: Optional[str]) -> bool:
        """
        非严格判断，只用于 warning 决策，不用于强推断字段。
        """
        if not field_name:
            return False

        f = str(field_name).strip().lower()
        time_hints = [
            "time", "date", "datetime", "timestamp",
            "hour", "day", "month", "week", "year",
            "时间", "日期"
        ]
        return any(h in f for h in time_hints)