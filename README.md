### NL-STV 0.3系列版本
基于以下需求：
1. 理解数据，给出一些基础分析和可视化，等待用户问题
2. 用户问问题之后，能回答用户问题，相应给出可视化显示
3. 动态展示功能
4. 交互，地图上自主选择数据


### 下一步计划
1. 支持返回绘制图表的理由，这个dashboard_planner里已经写了传给前端就行
2. 时间部分还有点问题，聚合步长有点问题，在dashboard里决定1H 还是1D
3. 重构scaffold.py，目前提示词已经严重膨胀，需要将常用代码（如风格配置）封装成SDK
4. 处理scaffold硬编码问题，防止过拟合，更具有通用性 recipe优化
5. 交互，地图上自主选择数据，这一步是核心挑战
6. insight组件修复

#### 项目组织架构

```text
NL-STV-V0.3.1
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── chat.py
│   │   ├── data.py
│   │   └── session.py
│   └── core/
│       ├── data_sandbox/
│       ├── execution/
│       │   ├── __init__.py
│       │   ├── executor.py
│       │   └── insight_extractor.py
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── viz_generator.py
│       │   ├── dashboard_planner.py
│       │   ├── scaffold.py
│       │   ├── templates.py
│       │   └── viz_editor.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── ingestion.py
│       │   └── loader_factory.py
│       ├── llm/
│       │   ├── __init__.py
│       │   └── AI_client.py
│       ├── profiler/
│       │   ├── __init__.py
│       │   ├── basic_stats.py
│       │   ├── interaction_mapper.py
│       │   ├── relation_mapper.py
│       │   └── semantic_analyzer.py
│       ├── schemas/
│       │   ├── __init__.py
│       │   ├── dashboard.py
│       │   ├── interaction.py
│       │   └── state.py
│       ├── sdk/
│       │   ├── __init__.py
│       │   ├── constraints.py
│       │   ├── data_utils.py
│       │   ├── layers.py
│       │   ├── operators.py
│       │   └── visualizer.py
│       ├── services/
│       │   ├── __init__.py
│       │   ├── session_service.py
│       │   └── workflow.py
│       └── __init__.py
├── __init__.py
├── main.py
└── pytest.ini
```

### 下一步

- 编辑模式这玩意还没正确接入
- 框选功能还没实现，这个可以和editor一起弄，是同源的问题