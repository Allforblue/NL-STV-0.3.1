import os
import logging
import shutil
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# --- 核心模块导入 ---
from api import chat, data, session
from core.llm.AI_client import AIClient
from core.services.workflow import AnalysisWorkflow
from core.services.session_service import session_service

from dotenv import load_dotenv
load_dotenv()

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- 路径配置 ---
BASE_DIR = Path(__file__).resolve().parent
# 沙箱路径定义在 core 目录下，用于存放用户 session 的数据副本
SANDBOX_PATH = BASE_DIR / "core" / "data_sandbox"


# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    在应用启动时初始化核心资源，并进行环境自检。
    """
    logger.info(">>> [NL-STV V2.2] 正在初始化高交互时空分析后端...")

    # 1. 确保数据沙箱目录存在且纯净
    if not SANDBOX_PATH.exists():
        SANDBOX_PATH.mkdir(parents=True, exist_ok=True)
        logger.info(f"已创建数据沙箱: {SANDBOX_PATH}")
    else:
        # [可选] 启动时清理旧的沙箱临时数据
        logger.info(f"检测到现有沙箱，正在执行启动自检...")

    # 2. 初始化核心引擎
    try:
        # A. 初始化 AI 客户端 (适配 DeepSeek API)
        # 建议从环境变量获取 API Key，此处保留默认注入逻辑
        client = AIClient(model_name="deepseek-chat")

        # B. 连通性检查
        if not client.is_alive():
            logger.warning("⚠️ 警告: 无法连接到 LLM 服务，分析引擎将受限。请检查 API Key 和网络。")
        else:
            logger.info("✅ LLM 服务 (DeepSeek API) 连接正常")

        # C. 初始化分析工作流 (Workflow)
        # 该实例封装了 Analyzer, Mapper, Planner, Generator, Executor 等全套模块
        app.state.workflow = AnalysisWorkflow(client)
        logger.info("✅ 时空分析工作流引擎 (AnalysisWorkflow V2.2) 挂载成功")

    except Exception as e:
        logger.error(f"❌ 系统核心模块初始化失败: {e}")
        import traceback
        traceback.print_exc()
        # 启动失败应抛出异常，防止服务带病运行
        raise e

    yield

    # 3. 停止时的清理逻辑
    logger.info(">>> 正在关闭 NL-STV 服务，回收内存资源...")
    # --- 新增：自动清空沙箱代码 ---
    sandbox_root = "core/data_sandbox"
    if os.path.exists(sandbox_root):
        try:
            # 遍历目录下的所有内容
            for filename in os.listdir(sandbox_root):
                file_path = os.path.join(sandbox_root, filename)
                # 如果是文件夹则递归删除，如果是文件则直接删除
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                else:
                    os.remove(file_path)
            logger.info(f"✅ 沙箱清理完成: {os.path.abspath(sandbox_root)}")
        except Exception as e:
            logger.error(f"❌ 清理沙箱失败: {e}")
    else:
        logger.info("ℹ️ 未发现沙箱目录，跳过清理")
    # 可以通过 session_service 遍历清理所有活跃 Session 的数据 context


# --- 创建 FastAPI 实例 ---
app = FastAPI(
    title="NL-STV Platform API",
    description="LLM 驱动的高交互时空分析平台后端 - 具备地理空间感知、交互联动与历史回溯能力",
    version="2.2.0",
    lifespan=lifespan
)

# --- 配置跨域 (CORS) ---
# 时空数据分析涉及大量 GeoJSON 与大数据量传输，允许标准的前端交互
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 开发环境允许所有源，生产环境建议指定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 挂载业务路由 ---
app.include_router(chat.router, prefix="/api/v1/chat", tags=["智能对话与交互"])
app.include_router(data.router, prefix="/api/v1/data", tags=["数据上传与画像"])
app.include_router(session.router, prefix="/api/v1/session", tags=["会话管理与回溯"])


# --- 基础健康检查 ---
@app.get("/", tags=["System"])
async def root():
    return {
        "status": "online",
        "service": "NL-STV Engine",
        "version": "2.2.0",
        "engine_state": "Spatio-Temporal Optimized",
        "features": [
            "Smart Spatial Sampling",
            "Auto-Focus ViewState",
            "Cross-filtering Logic",
            "Snapshot Persistence"
        ]
    }


if __name__ == "__main__":
    import uvicorn

    # 在本地开发模式下开启 reload=True
    # 对于时空数据处理，建议增大 timeout 阈值防止大数据量传输导致的超时
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )