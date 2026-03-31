# backend/core/execution/insight_extractor.py

import logging
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from core.llm.AI_client import AIClient
from core.schemas.dashboard import InsightCard

logger = logging.getLogger(__name__)


class InsightExtractor:
    """
    智能洞察提取器 (V4.1 Compare-Context Aware Edition)

    核心升级：
    1. 对齐 ResultSummarizer 输出结构：
       - headline_facts
       - ranking_insights
       - temporal_insights
       - spatial_insights
       - composition_insights
       - narrative_hints
    2. 支持 executor 注入的顶层 `_context`
       - selection_enabled
       - active_selection_labels
       - compare_enabled
       - baseline_mode
       - left_selection_labels
       - right_selection_labels
    3. compare 场景下切换为“描述性差异分析”提示策略
    4. 保持普通单看板洞察能力不受影响
    5. 保持 JSON-safe 序列化
    """

    def __init__(self, llm_client: AIClient):
        self.llm = llm_client

    def _sanitize_for_json(self, obj: Any) -> Any:
        """
        递归清洗对象，确保可被 json.dumps 正常序列化。
        """
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}

        elif isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(v) for v in obj]

        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)

        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return float(obj)

        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')

        elif isinstance(obj, np.datetime64):
            return pd.to_datetime(obj).strftime('%Y-%m-%d %H:%M:%S')

        elif isinstance(obj, np.ndarray):
            return self._sanitize_for_json(obj.tolist())

        elif isinstance(obj, pd.Series):
            return self._sanitize_for_json(obj.to_list())

        elif isinstance(obj, pd.DataFrame):
            return self._sanitize_for_json(obj.to_dict(orient="records"))

        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj

        return obj

    def _build_semantic_context(self, summaries: List[Dict[str, Any]]) -> str:
        context_blocks = []

        for s in summaries:
            var_name = s.get('variable_name')
            sem = s.get('semantic_analysis', {})
            domain = sem.get('dataset_domain', '通用领域')
            desc = sem.get('dataset_description', '')
            tags = sem.get('column_metadata', {})

            tags_str = ", ".join([
                f"{k}({v.get('concept_name')})"
                for k, v in tags.items()
                if isinstance(v, dict)
            ])

            context_blocks.append(
                f"数据集 `{var_name}` [{domain}]: {desc}\n"
                f"   - 字段含义: {tags_str}"
            )

        return "\n".join(context_blocks)

    def _extract_context_from_execution_stats(self, safe_execution_stats: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(safe_execution_stats, dict):
            return {}

        context = safe_execution_stats.get("_context", {})
        return context if isinstance(context, dict) else {}

    def _build_compare_context_text(self, context: Dict[str, Any]) -> str:
        if not context:
            return "当前无显式 selection/compare 上下文。"

        selection_enabled = bool(context.get("selection_enabled"))
        compare_enabled = bool(context.get("compare_enabled"))
        active_selection_count = context.get("active_selection_count", 0)
        active_selection_labels = context.get("active_selection_labels", []) or []

        if compare_enabled:
            baseline_mode = context.get("baseline_mode", "global")
            left_labels = context.get("left_selection_labels", []) or []
            right_labels = context.get("right_selection_labels", []) or []

            return (
                "当前处于比较分析上下文：\n"
                f"- compare_enabled: True\n"
                f"- baseline_mode: {baseline_mode}\n"
                f"- left_selection_labels: {left_labels}\n"
                f"- right_selection_labels: {right_labels}\n"
                f"- active_selection_labels: {active_selection_labels}\n"
            )

        if selection_enabled:
            return (
                "当前处于选区分析上下文：\n"
                f"- selection_enabled: True\n"
                f"- active_selection_count: {active_selection_count}\n"
                f"- active_selection_labels: {active_selection_labels}\n"
            )

        return "当前无选区上下文，属于普通整体分析。"

    def _build_component_overview(self, safe_execution_stats: Dict[str, Any]) -> str:
        """
        给 LLM 一个更容易理解的组件摘要清单，而不仅仅是原始 JSON。
        跳过顶层 _context。
        """
        if not isinstance(safe_execution_stats, dict):
            return "无组件摘要"

        lines = []
        for cid, stats in safe_execution_stats.items():
            if cid == "_context":
                continue
            if not isinstance(stats, dict):
                continue

            title = stats.get("title", cid)
            viz_type = stats.get("viz_type", "unknown")
            semantic_role = stats.get("semantic_role", "unknown")
            headline_facts = stats.get("headline_facts", {})

            line = f"- 组件 `{cid}` | 标题: {title} | 类型: {viz_type} | 语义角色: {semantic_role}"
            if headline_facts:
                line += f" | 关键事实: {json.dumps(headline_facts, ensure_ascii=False)}"
            lines.append(line)

        return "\n".join(lines) if lines else "无组件摘要"

    async def generate_insights(
        self,
        query: str,
        execution_stats: Dict[str, Any],
        summaries: List[Dict[str, Any]]
    ) -> InsightCard:
        """
        根据统一的执行摘要结果生成深度业务洞察。

        Args:
            query: 用户的原始问题
            execution_stats: Executor + ResultSummarizer 产出的结构化摘要
            summaries: 数据语义背景
        """

        # 1. 语义背景
        semantic_context = self._build_semantic_context(summaries)

        # 2. 清洗 execution_stats
        safe_execution_stats = self._sanitize_for_json(execution_stats)

        # 3. 提取顶层 interaction / compare context
        compare_context = self._extract_context_from_execution_stats(safe_execution_stats)
        compare_context_text = self._build_compare_context_text(compare_context)
        is_compare_mode = bool(compare_context.get("compare_enabled"))
        is_selection_mode = bool(compare_context.get("selection_enabled"))

        # 4. 组件级概览（帮助模型快速定位）
        component_overview = self._build_component_overview(safe_execution_stats)

        # 5. Prompt 组装
        if is_compare_mode:
            system_prompt = f"""
你是一位顶尖的时空数据分析专家和商业顾问。
你的任务是根据提供的【统一结构化执行摘要】、【交互上下文】和【语义上下文】，生成准确、具体、面向业务的“描述性差异分析”。

=== 数据语义背景 ===
{semantic_context}

=== 当前交互 / 比较上下文 ===
{compare_context_text}

=== 组件摘要总览 ===
{component_overview}

=== 摘要结构说明（必须理解） ===
execution_stats 中每个组件可能包含以下结构化字段：
- `headline_facts`: 最核心、最可信的关键结论
- `ranking_insights`: 排名类图表的排序结果（如 top_items_desc / bottom_items_asc）
- `temporal_insights`: 趋势图/时间图的峰值、谷值、起止点、趋势方向
- `spatial_insights`: 地图热点、冷点、区域排名
- `composition_insights`: 构成图的占比和集中度
- `narrative_hints`: 系统生成的解释提示，帮助你避免误读图表
- 顶层 `_context`: 当前选区/比较状态摘要

=== 比较分析准则 (CRITICAL) ===
1. 当前任务是“描述性差异分析”，不是普通单图解读。
2. 如果 `compare_enabled=True` 且 right_selection_labels 为空：
   - 将任务理解为“选区 vs 整体（global baseline）”
3. 如果 `compare_enabled=True` 且 left/right 均存在：
   - 将任务理解为“左侧区域 vs 右侧区域”
4. 你的重点不是编造因果，而是识别：
   - 哪些指标更高/更低
   - 哪些类别更集中
   - 哪些时间峰值更突出
   - 哪些热点区域更显著
5. **优先使用结构化摘要，而不是猜图**
   - 如果存在 `ranking_insights.top_items_desc`，必须使用其中第一个对象作为头部项依据。
   - 如果存在 `spatial_insights.top_regions_desc`，必须使用其中第一个对象作为热点依据。
   - 如果存在 `temporal_insights.peak_point`，必须用它作为峰值依据。
   - 如果存在 `composition_insights.top_items_desc`，必须用它作为构成最高项依据。
6. **禁止做重型因果推断**
   - 不要说“导致”“必然由……引起”
   - 只能做统计描述与业务解释建议
7. **严格避免空话**
   - 禁止使用“未提供具体数值”“需进一步分析”等推脱性表述。
8. **输出风格**
   - summary 必须点明“比较对象”或“差异对象”
   - detail 必须围绕“差异”展开
   - 必须输出自然语言段落，不能输出 Markdown 表格
"""
            user_prompt = f"""
用户的原始分析意图: "{query}"

以下是执行器输出的结构化组件摘要（JSON）：
{json.dumps(safe_execution_stats, indent=2, ensure_ascii=False)}

请直接输出 JSON 结果，严格符合以下结构：
{{
  "summary": "一句话核心比较结论（必须包含比较对象或差异对象，字数控制在25字内）",
  "detail": "描述性差异分析：请结合当前 compare context 和结构化摘要中的真实数值，从用户最相关的主图/主指标出发，分析选区与整体或左区域与右区域之间在空间分布、时间峰值、排名结构、构成结构上的差异，不少于150字，并给出1条具体业务建议。禁止使用表格，不做因果断言。",
  "tags": ["对比分析", "区域差异", "描述性洞察"]
}}
"""
        elif is_selection_mode:
            system_prompt = f"""
你是一位顶尖的时空数据分析专家和商业顾问。
你的任务是根据提供的【统一结构化执行摘要】、【交互上下文】和【语义上下文】，针对当前选区生成准确、具体、面向业务的洞察。

=== 数据语义背景 ===
{semantic_context}

=== 当前交互上下文 ===
{compare_context_text}

=== 组件摘要总览 ===
{component_overview}

=== 摘要结构说明（必须理解） ===
execution_stats 中每个组件可能包含以下结构化字段：
- `headline_facts`
- `ranking_insights`
- `temporal_insights`
- `spatial_insights`
- `composition_insights`
- `narrative_hints`
- 顶层 `_context`

=== 深度分析准则 (CRITICAL) ===
1. 当前任务是“基于当前选区”的分析，不是整体看板泛泛总结。
2. **优先使用结构化摘要，而不是猜图**
3. 必须围绕当前 active_selection_labels 对应的区域/对象展开描述。
4. 可以做业务解释，但不能做重型因果推断。
5. 严格避免空话与推脱性表述。
6. 输出必须是自然语言段落，不能输出 Markdown 表格。
"""
            user_prompt = f"""
用户的原始分析意图: "{query}"

以下是执行器输出的结构化组件摘要（JSON）：
{json.dumps(safe_execution_stats, indent=2, ensure_ascii=False)}

请直接输出 JSON 结果，严格符合以下结构：
{{
  "summary": "一句话核心结论（必须包含当前选区的对象或数值，字数控制在25字内）",
  "detail": "深度解析：请结合当前选区上下文与结构化摘要中的真实数值，从用户问题最相关的主图/主指标出发，综合时间趋势、空间分布、排名结构或构成结构进行不少于150字的分析，并给出1条具体的业务建议。禁止使用表格。",
  "tags": ["选区分析", "时空分析", "描述性洞察"]
}}
"""
        else:
            system_prompt = f"""
你是一位顶尖的时空数据分析专家和商业顾问。
你的任务是根据提供的【统一结构化执行摘要】和【语义上下文】，针对用户的【分析需求】生成准确、具体、面向业务的洞察。

=== 数据语义背景 ===
{semantic_context}

=== 当前交互上下文 ===
{compare_context_text}

=== 组件摘要总览 ===
{component_overview}

=== 摘要结构说明（必须理解） ===
execution_stats 中每个组件可能包含以下结构化字段：
- `headline_facts`: 最核心、最可信的关键结论
- `ranking_insights`: 排名类图表的排序结果（如 top_items_desc / bottom_items_asc）
- `temporal_insights`: 趋势图/时间图的峰值、谷值、起止点、趋势方向
- `spatial_insights`: 地图热点、冷点、区域排名
- `composition_insights`: 构成图的占比和集中度
- `narrative_hints`: 系统生成的解释提示，帮助你避免误读图表
- 顶层 `_context`: 当前选区/比较状态摘要

=== 深度分析准则 (CRITICAL) ===
1. **优先使用结构化摘要，而不是猜图**
   - 如果存在 `headline_facts`，优先引用其中的结论。
   - 如果存在 `ranking_insights.top_items_desc`，必须使用其中第一个对象作为“最高项/头部项”的依据。
   - 如果存在 `spatial_insights.top_regions_desc`，必须使用其中第一个对象作为热点区域依据。
   - 如果存在 `temporal_insights.peak_point`，必须用它作为时间峰值依据。
   - 如果存在 `composition_insights.top_items_desc`，必须用它作为构成最高项依据。

2. **绝对禁止误读柱状图视觉顺序**
   - 排名类条形图可能按升序显示，顶部条目不一定是最大值。
   - 你必须以 `ranking_insights.top_items_desc[0]` 为准，而不是根据视觉位置猜测最大值。
   - 如果 `narrative_hints` 提示“顶部条目不是最高值”，你必须遵守。

3. **时间维度解析**
   - 识别趋势、峰值、谷值和变化方向。
   - 如果 trend_direction 存在，必须在 detail 中体现。

4. **空间维度解析**
   - 明确指出热点区域/冷点区域。
   - 对区域热点给出业务解释，但不能脱离数值事实。

5. **构成维度解析**
   - 若是构成图，说明头部类别的占比和集中程度。
   - 若 concentration_level 较高，需指出“集中”。

6. **业务语言转化**
   - 将数字结论转化为业务洞察与建议。
   - 例如：高订单量可能意味着通勤需求旺盛；高金额可能意味着高消费或长途需求集中。

7. **严格避免空话**
   - 禁止使用“未提供具体数值”、“需进一步分析”、“由于缺乏详细数据”等推脱性表述。
   - 当前提供的结构化摘要已经足够支撑明确结论。

8. **输出风格**
   - 必须输出自然语言段落，不能输出 Markdown 表格。
   - summary 必须简洁、有数值、有对象。
   - detail 必须围绕用户问题，不要被不相关组件带偏。
"""
            user_prompt = f"""
用户的原始分析意图: "{query}"

以下是执行器输出的结构化组件摘要（JSON）：
{json.dumps(safe_execution_stats, indent=2, ensure_ascii=False)}

请直接输出 JSON 结果，严格符合以下结构：
{{
  "summary": "一句话核心结论（必须包含最核心的业务实体或数字，字数控制在25字内）",
  "detail": "深度解析：请结合结构化摘要中的真实数值，从用户问题最相关的主图/主指标出发，综合时间趋势、空间分布、排名结构或构成结构进行不少于150字的分析，并给出1条具体的业务建议。禁止使用表格。",
  "tags": ["标签1", "标签2", "标签3"]
}}
"""

        logger.info(">>> [Insight] 正在将结构化执行摘要转化为业务洞察...")

        try:
            ai_response = await self.llm.query_json_async(
                prompt=user_prompt,
                system_prompt=system_prompt
            )

            if isinstance(ai_response, dict):
                for alias in ["Description", "description", "content"]:
                    if alias in ai_response and "detail" not in ai_response:
                        ai_response["detail"] = ai_response.pop(alias)

                if "Metric" in ai_response or "Value" in ai_response:
                    m = ai_response.pop("Metric", "关键指标")
                    v = ai_response.pop("Value", "统计值")
                    table_lead = f"【{m}】: {v}"
                    original_detail = ai_response.get("detail", "")
                    ai_response["detail"] = f"{table_lead}\n{original_detail}".strip()

            if not ai_response.get("summary"):
                if is_compare_mode:
                    ai_response["summary"] = "区域比较分析已完成"
                elif is_selection_mode:
                    ai_response["summary"] = "选区分析已完成"
                else:
                    ai_response["summary"] = "时空分析执行完成"

            if not ai_response.get("detail"):
                if is_compare_mode:
                    ai_response["detail"] = (
                        "系统已根据 compare context 和结构化执行摘要提取出关键差异特征。"
                        "请重点关注左右区域或选区与整体之间在头部类别、热点区域、时间峰值和构成集中度上的差异，"
                        "并据此进行业务判断。"
                    )
                elif is_selection_mode:
                    ai_response["detail"] = (
                        "系统已根据当前选区上下文与结构化执行摘要提取出关键统计特征。"
                        "请重点关注当前选区中的头部类别、热点区域、折线峰值与主要构成项，以辅助业务判断。"
                    )
                else:
                    ai_response["detail"] = (
                        "系统已根据结构化执行摘要提取出关键统计特征。"
                        "请重点关注排名第一的头部类别、热点区域、折线峰值与主要构成项，以辅助业务判断。"
                    )

            if not ai_response.get("tags"):
                if is_compare_mode:
                    ai_response["tags"] = ["对比分析", "区域差异", "描述性洞察"]
                elif is_selection_mode:
                    ai_response["tags"] = ["选区分析", "时空分析", "描述性洞察"]
                else:
                    ai_response["tags"] = ["自动提取", "时空分析"]

            return InsightCard(**ai_response)

        except Exception as e:
            logger.error(f"❌ 洞察生成失败: {e}")

            found_keys = []
            if isinstance(safe_execution_stats, dict):
                found_keys = [k for k in safe_execution_stats.keys() if k != "_context"]

            if is_compare_mode:
                fallback_summary = "区域比较分析已完成"
                fallback_detail = (
                    f"系统已完成对组件 {found_keys} 的结构化摘要提取，并识别到当前处于比较分析上下文。"
                    f"虽然 AI 洞察文本生成暂未成功，但你仍可直接依据各组件中的 headline_facts、ranking_insights、"
                    f"spatial_insights、temporal_insights 与 composition_insights 判断左右区域或选区与整体之间的差异。"
                    f"建议优先关注头部类别变化、热点区域变化、时间峰值变化以及构成集中度差异。"
                )
                fallback_tags = ["系统回退", "对比分析", "结构化摘要"]
            elif is_selection_mode:
                fallback_summary = "选区分析已完成"
                fallback_detail = (
                    f"系统已完成对组件 {found_keys} 的结构化摘要提取，并识别到当前存在选区上下文。"
                    f"虽然 AI 洞察文本生成暂未成功，但你仍可直接依据各组件中的 headline_facts、ranking_insights、"
                    f"spatial_insights 与 temporal_insights 对当前选区进行业务判断。"
                    f"建议优先关注排名第一项、热点区域、最大占比项以及时间峰值点。"
                )
                fallback_tags = ["系统回退", "选区分析", "结构化摘要"]
            else:
                fallback_summary = "结构化分析已完成"
                fallback_detail = (
                    f"系统已完成对组件 {found_keys} 的结构化摘要提取。"
                    f"虽然 AI 洞察文本生成暂未成功，但你仍可直接依据各组件中的 headline_facts、ranking_insights、"
                    f"spatial_insights 与 temporal_insights 进行业务判断。"
                    f"建议优先关注排名第一项、热点区域、最大占比项以及时间峰值点。"
                )
                fallback_tags = ["系统回退", "结构化摘要"]

            return InsightCard(
                summary=fallback_summary,
                detail=fallback_detail,
                tags=fallback_tags
            )