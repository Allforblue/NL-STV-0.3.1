import logging
import uuid
import threading
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# --- 引入必要的模型与下层模块 ---
from core.ingestion.ingestion import IngestionManager
from core.profiler.basic_stats import get_dataset_fingerprint
from core.schemas.state import (
    SessionStateSnapshot,
    SessionStateStore,
    InteractionState,
)
from core.schemas.dashboard import DashboardSchema

logger = logging.getLogger(__name__)


class SessionManager:
    """
    增强型会话管理器（长期状态化兼容版）

    当前职责：
    1. 管理会话级数据上下文（采样 / 全量）
    2. 管理看板快照与历史回溯
    3. 维护 last_workflow_state，兼容现有 workflow / VizEditor
    4. 新增维护 interaction_state 与 component_render_meta
    """

    def __init__(self):
        # 内存存储结构:
        # {
        #   session_id: {
        #       "session_id": ...,
        #       "data_context": {...},
        #       "summaries": [...],
        #       "file_paths": [...],
        #       "is_full_data": bool,
        #       "state_store": SessionStateStore,
        #       "last_workflow_state": dict | None,
        #       "interaction_state": dict,
        #       "component_render_meta": dict,
        #       "lock": threading.Lock()
        #   }
        # }
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self.ingestion_manager = IngestionManager()

    # =========================================================
    # 会话初始化
    # =========================================================
    def create_session(self, session_id: str, file_paths: List[str]) -> Dict[str, Any]:
        """创建新会话并初始化画像"""
        logger.info(f">>> [Session] 正在初始化时空分析会话: {session_id}")

        # 1. 初始加载：采用采样模式，保障响应速度
        data_context = self.ingestion_manager.load_all_to_context(file_paths, use_full=False)

        # 2. 生成基础画像 (Summaries)
        summaries = []
        for var_name, df in data_context.items():
            try:
                matched_path = next(
                    (p for p in file_paths if Path(p).stem.lower() in var_name),
                    file_paths[0]
                )

                fingerprint = get_dataset_fingerprint(df)

                summaries.append({
                    "variable_name": var_name,
                    "file_info": {
                        "path": str(matched_path),
                        "name": Path(matched_path).name,
                        "rows_total": fingerprint.get("rows", 0)
                    },
                    "is_geospatial": fingerprint.get("is_geospatial", False),
                    "crs": fingerprint.get("crs", "Unknown"),
                    "column_stats": fingerprint.get("column_stats", {}),
                    "basic_stats": fingerprint,
                    "semantic_analysis": {
                        "description": f"数据源: {Path(matched_path).name}",
                        "dataset_type": "spatial" if fingerprint.get("is_geospatial") else "tabular",
                        "semantic_tags": {}
                    }
                })
            except Exception as e:
                logger.error(f"画像生成失败 ({var_name}): {e}")

        # 3. 初始化状态存储库
        state_store = SessionStateStore(
            session_id=session_id,
            active_files=[str(p) for p in file_paths],
            interaction_state=InteractionState(),
            component_render_meta={}
        )

        session_state = {
            "session_id": session_id,
            "data_context": data_context,
            "summaries": summaries,
            "file_paths": file_paths,
            "is_full_data": False,
            "state_store": state_store,
            "last_workflow_state": None,

            # 为便于 workflow 直接取用，也在 session 顶层保留一份镜像
            "interaction_state": state_store.interaction_state.model_dump(),
            "component_render_meta": state_store.component_render_meta,

            "lock": threading.Lock()
        }

        self._sessions[session_id] = session_state
        return session_state

    # =========================================================
    # 快照管理
    # =========================================================
    def save_snapshot(
        self,
        session_id: str,
        query: str,
        code: str,
        layout_data: DashboardSchema,
        summary: str = "",
        intent: Optional[str] = None,
        execution_time_ms: Optional[float] = None
    ) -> str:
        """保存当前看板状态为快照"""
        session = self.get_session(session_id)
        if not session:
            return ""

        snapshot_id = f"snap_{uuid.uuid4().hex[:8]}"

        store: SessionStateStore = session["state_store"]

        # 尝试读取当前交互状态与组件元信息
        interaction_state_obj = store.interaction_state if store else None
        component_render_meta = store.component_render_meta if store else {}

        new_snapshot = SessionStateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            user_query=query,
            intent=intent,
            code_snapshot=code,
            layout_data=layout_data,
            summary_text=summary or f"分析: {(query or '交互更新')[:15]}...",
            execution_time_ms=execution_time_ms,
            interaction_state=interaction_state_obj,
            component_render_meta=component_render_meta or {}
        )

        store.add_snapshot(new_snapshot)

        logger.info(f"✅ 快照已存档: {snapshot_id} (Session: {session_id})")
        return snapshot_id

    def get_snapshot(self, session_id: str, snapshot_id: str) -> Optional[SessionStateSnapshot]:
        """获取特定历史快照"""
        session = self.get_session(session_id)
        if session:
            return session["state_store"].get_snapshot(snapshot_id)
        return None

    def get_history_list(self, session_id: str) -> List[Dict[str, Any]]:
        """获取历史记录摘要列表"""
        session = self.get_session(session_id)
        if not session:
            return []

        return [
            {
                "snapshot_id": s.snapshot_id,
                "query": s.user_query,
                "time": s.timestamp.strftime("%H:%M:%S"),
                "summary": s.summary_text
            }
            for s in session["state_store"].snapshots
        ]

    # =========================================================
    # 交互状态管理
    # =========================================================
    def get_interaction_state(self, session_id: str) -> Dict[str, Any]:
        """获取当前会话的结构化交互状态（dict 形式，便于 workflow 直接使用）"""
        session = self.get_session(session_id)
        if not session:
            return InteractionState().model_dump()

        store: SessionStateStore = session["state_store"]
        if store and store.interaction_state:
            return store.interaction_state.model_dump()

        return session.get("interaction_state", InteractionState().model_dump())

    def update_interaction_state(self, session_id: str, interaction_state: Dict[str, Any]):
        """更新当前会话的结构化交互状态"""
        session = self.get_session(session_id)
        if not session:
            return

        try:
            state_obj = interaction_state
            if not isinstance(interaction_state, InteractionState):
                state_obj = InteractionState(**(interaction_state or {}))

            session["interaction_state"] = state_obj.model_dump()

            store: SessionStateStore = session["state_store"]
            store.interaction_state = state_obj

            logger.info(f"🧭 会话交互状态已更新: {session_id}")
        except Exception as e:
            logger.warning(f"[Session] update_interaction_state failed: {e}")

    # =========================================================
    # 组件渲染元信息管理
    # =========================================================
    def get_component_render_meta(self, session_id: str) -> Dict[str, Any]:
        """获取当前会话的组件 render meta"""
        session = self.get_session(session_id)
        if not session:
            return {}

        store: SessionStateStore = session["state_store"]
        if store:
            return store.component_render_meta or {}

        return session.get("component_render_meta", {})

    def update_component_render_meta(self, session_id: str, render_meta: Dict[str, Any]):
        """更新当前会话的组件 render meta"""
        session = self.get_session(session_id)
        if not session:
            return

        render_meta = render_meta or {}

        session["component_render_meta"] = render_meta

        store: SessionStateStore = session["state_store"]
        store.component_render_meta = render_meta

        logger.info(f"🧩 会话组件渲染元信息已更新: {session_id}")

    # =========================================================
    # 数据一致性维护
    # =========================================================
    def ensure_full_data_context(self, session_id: str):
        """
        切换至全量数据模式。
        确保在大规模数据加载过程中保持坐标系感知和内存隔离。
        """
        session = self.get_session(session_id)
        if not session:
            return

        if session.get("is_full_data"):
            return

        lock = session["lock"]
        with lock:
            if session.get("is_full_data"):
                return

            logger.info(f">>> [IO] 会话 {session_id} 正在执行全量数据切换 (Large Scale Loading)...")
            try:
                full_context = self.ingestion_manager.load_all_to_context(
                    session["file_paths"],
                    use_full=True
                )

                # 数据完整性检查：确保变量名未发生漂移
                for var in session["data_context"].keys():
                    if var not in full_context:
                        logger.warning(f"全量加载中缺失变量: {var}，保留采样副本。")
                        full_context[var] = session["data_context"][var]

                session["data_context"] = full_context
                session["is_full_data"] = True
                logger.info(f"✅ 会话 {session_id} 全量数据就绪。")
            except Exception as e:
                logger.error(f"全量加载过程中发生严重错误: {e}")
                # 失败时保持采样模式，不中断业务

    # =========================================================
    # 基础操作
    # =========================================================
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str):
        """清理会话资源，防止内存溢出"""
        if session_id in self._sessions:
            try:
                for var_name in list(self._sessions[session_id]["data_context"].keys()):
                    del self._sessions[session_id]["data_context"][var_name]
                self._sessions[session_id]["data_context"].clear()
            except Exception:
                pass

            del self._sessions[session_id]
            logger.info(f"🗑️ 会话 {session_id} 资源已释放。")

    def update_session_metadata(self, session_id: str, metadata: Dict[str, Any]):
        """
        更新会话执行状态，为 workflow / VizEditor 提供上下文。
        兼容旧逻辑，同时同步 interaction_state / component_render_meta（若存在）。
        """
        session = self.get_session(session_id)
        if not session:
            return

        session["last_workflow_state"] = metadata

        # 尝试同步 interaction_state
        if isinstance(metadata, dict) and metadata.get("interaction_state") is not None:
            try:
                self.update_interaction_state(session_id, metadata["interaction_state"])
            except Exception:
                pass

        # 尝试同步 component_render_meta
        if isinstance(metadata, dict) and metadata.get("component_render_meta") is not None:
            try:
                self.update_component_render_meta(session_id, metadata["component_render_meta"])
            except Exception:
                pass

        logger.info(f"💾 会话状态已同步 (Code/Layout/Interaction): {session_id}")


# 单例导出
session_service = SessionManager()