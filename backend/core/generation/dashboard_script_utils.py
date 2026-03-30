import re
import textwrap
from typing import Dict, Any, List


def safe_func_name(component_id: str) -> str:
    """
    统一将 component_id 转为 dashboard 内部函数名。
    例如:
        main_map -> get_main_map
        chart-dynamic-1 -> get_chart_dynamic_1
    """
    safe_id = str(component_id or "unknown").replace("-", "_")
    return f"get_{safe_id}"


def indent_block(code: str, spaces: int = 4) -> str:
    """
    对代码块整体缩进。
    """
    return textwrap.indent(code.strip(), " " * spaces)


def parse_dashboard_script(code: str) -> Dict[str, Any]:
    """
    解析 generator 生成的 dashboard 脚本。

    预期结构：
        import ...
        import ...

        def get_dashboard_data(data_context):
            dashboard_results = {}

            def get_xxx(data_context):
                ...
                return ...

            def get_yyy(data_context):
                ...
                return ...

            # ================= Execution Block =================
            try:
                dashboard_results['xxx'] = get_xxx(data_context)
            except Exception as e:
                ...
            return dashboard_results

    返回结构：
    {
        "imports_block": str,
        "component_functions": {
            "main_map": {
                "component_id": "main_map",
                "func_name": "get_main_map",
                "code": "def get_main_map(data_context): ..."
            },
            ...
        },
        "execution_order": [
            {"component_id": "main_map", "func_name": "get_main_map"},
            ...
        ],
        "insight_components": ["ai_insight"]
    }
    """
    clean_code = textwrap.dedent(code).strip()

    wrapper_signature = "def get_dashboard_data(data_context):"
    if wrapper_signature not in clean_code:
        raise ValueError("Not a valid dashboard script: missing get_dashboard_data(data_context).")

    wrapper_start = clean_code.index(wrapper_signature)
    imports_block = clean_code[:wrapper_start].rstrip()

    wrapper_code = clean_code[wrapper_start:]
    wrapper_lines = wrapper_code.splitlines()

    component_functions: Dict[str, Dict[str, str]] = {}
    execution_order: List[Dict[str, str]] = []
    insight_components: List[str] = []

    # ---------------------------------------------------------
    # 1. 先从 execution block 中提取 component_id -> func_name 映射
    # ---------------------------------------------------------
    for line in wrapper_lines:
        m = re.search(
            r"dashboard_results\['([^']+)'\]\s*=\s*(get_[a-zA-Z0-9_]+)\(data_context\)",
            line
        )
        if m:
            comp_id = m.group(1)
            func_name = m.group(2)
            execution_order.append({
                "component_id": comp_id,
                "func_name": func_name
            })

    func_to_component = {
        item["func_name"]: item["component_id"]
        for item in execution_order
    }

    # ---------------------------------------------------------
    # 2. 提取 wrapper 内部所有组件函数
    # ---------------------------------------------------------
    i = 0
    while i < len(wrapper_lines):
        line = wrapper_lines[i]

        # 匹配 4 空格缩进的内层组件函数
        func_match = re.match(r"^\s{4}def\s+(get_[a-zA-Z0-9_]+)\(data_context\):\s*$", line)
        if func_match:
            func_name = func_match.group(1)
            block_lines = [line]

            i += 1
            while i < len(wrapper_lines):
                next_line = wrapper_lines[i]

                # 下一个内层 def，说明当前函数结束
                if re.match(r"^\s{4}def\s+(get_[a-zA-Z0-9_]+)\(data_context\):\s*$", next_line):
                    break

                # execution block 开始，说明函数区结束
                if next_line.strip() == "# ================= Execution Block =================":
                    break

                block_lines.append(next_line)
                i += 1

            func_code_indented = "\n".join(block_lines)
            func_code = textwrap.dedent(func_code_indented).rstrip()

            component_id = func_to_component.get(func_name)
            if not component_id:
                # 如果 execution block 中没找到映射，则保守退化
                guessed = func_name.replace("get_", "", 1)
                component_id = guessed

            component_functions[component_id] = {
                "component_id": component_id,
                "func_name": func_name,
                "code": func_code
            }

            # insight 占位识别：函数体最后 return None
            if re.search(r"return\s+None\s*$", func_code, flags=re.MULTILINE):
                insight_components.append(component_id)

            continue

        i += 1

    return {
        "imports_block": imports_block,
        "component_functions": component_functions,
        "execution_order": execution_order,
        "insight_components": insight_components
    }


def assemble_dashboard_script(
    parsed: Dict[str, Any],
    updated_component_functions: Dict[str, Dict[str, str]]
) -> str:
    """
    使用解析结果 + 更新后的组件函数，重新组装完整 dashboard script。
    """
    imports_block = (parsed.get("imports_block") or "").strip()
    execution_order = parsed.get("execution_order") or []

    lines: List[str] = []

    if imports_block:
        lines.append(imports_block)
        lines.append("")

    lines.append("def get_dashboard_data(data_context):")
    lines.append("    dashboard_results = {}")
    lines.append("")

    # ---------------------------------------------------------
    # 1. 按 execution_order 顺序插入组件函数
    # ---------------------------------------------------------
    added = set()

    for item in execution_order:
        comp_id = item["component_id"]
        func_info = updated_component_functions.get(comp_id)
        if not func_info:
            continue

        func_code = func_info["code"]
        lines.append(indent_block(func_code, spaces=4))
        lines.append("")
        added.add(comp_id)

    # ---------------------------------------------------------
    # 2. 补充 execution_order 外的函数（理论上极少）
    # ---------------------------------------------------------
    for comp_id, func_info in updated_component_functions.items():
        if comp_id in added:
            continue
        func_code = func_info["code"]
        lines.append(indent_block(func_code, spaces=4))
        lines.append("")

    # ---------------------------------------------------------
    # 3. 重新生成 execution block
    # ---------------------------------------------------------
    lines.append("    # ================= Execution Block =================")

    for item in execution_order:
        comp_id = item["component_id"]
        func_info = updated_component_functions.get(comp_id)
        if not func_info:
            continue

        func_name = func_info["func_name"]

        lines.append("    try:")
        lines.append(f"        dashboard_results['{comp_id}'] = {func_name}(data_context)")
        lines.append("    except Exception as e:")
        lines.append(f"        print(f'Error generating {comp_id}: {{str(e)}}')")
        lines.append(f"        dashboard_results['{comp_id}'] = None")
        lines.append("")

    lines.append("    return dashboard_results")

    return "\n".join(lines).strip() + "\n"