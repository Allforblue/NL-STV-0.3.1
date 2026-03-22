import json
import logging
import os
from typing import Dict, List, Any
from openai import OpenAI, AsyncOpenAI, APIError, AuthenticationError, APIConnectionError  # [修改] 引入 AsyncOpenAI

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AIClient:
    """
    使用 OpenAI SDK 封装 DeepSeek API 的客户端。
    保持与原 OllamaClient 相同的方法签名，以确保业务代码兼容性。
    [新增] 支持异步调用以优化并发性能。
    """

    def __init__(self,
                 api_key="sk-5871f8d87ae6423ebfc34165a61aa1e9",  # os.getenv("DEEPSEEK_API_KEY"),  # 在这里输入KEY
                 model_name: str = "deepseek-chat",  # deepseek-chat (V3)
                 timeout: int = 120):
        """
        初始化 DeepSeek 客户端。

        Args:
            api_key: DeepSeek API Key
            model_name: 模型名称 (deepseek-chat 或 deepseek-reasoner)
            timeout: 请求超时时间
        """
        # 同步客户端 (保持兼容)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=timeout
        )
        # [新增] 异步客户端
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=timeout
        )
        self.model_name = model_name

        logger.info(f"AI Client (DeepSeek via OpenAI SDK) 初始化完成，使用模型: {self.model_name}")

    def is_alive(self) -> bool:
        """
        连通性测试。
        通过调用 list models 接口来验证 API Key 和网络连接。
        """
        try:
            self.client.models.list()
            return True
        except AuthenticationError:
            logger.error("API Key 无效")
            return False
        except APIConnectionError:
            logger.error("无法连接到 DeepSeek API 服务器")
            return False
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

    def chat(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """
        发送聊天请求 (同步版本)。
        """
        try:
            # 构造请求参数
            params = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "temperature": 0.0 if json_mode else 0.7,  # JSON 模式通常需要更确定的输出
            }

            # 启用 JSON Mode (DeepSeek 支持 OpenAI 格式的 json_object)
            if json_mode:
                params["response_format"] = {"type": "json_object"}

            # 发起请求
            response = self.client.chat.completions.create(**params)

            # 获取内容
            content = response.choices[0].message.content
            # 显示消耗
            usage = response.usage
            logger.info(
                f"Tokens used: {usage.total_tokens} (Prompt: {usage.prompt_tokens}, Completion: {usage.completion_tokens})")
            return content

        except APIError as e:
            logger.error(f"DeepSeek API 返回错误: {e}")
            raise ConnectionError(f"DeepSeek API Error: {e}")
        except Exception as e:
            logger.error(f"LLM 请求发生未知错误: {e}")
            raise e

    async def chat_async(self, messages: List[Dict[str, str]], json_mode: bool = False) -> str:
        """
        [新增] 发送聊天请求 (异步版本)。
        """
        try:
            # 构造请求参数
            params = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "temperature": 0.0 if json_mode else 0.7,
            }

            if json_mode:
                params["response_format"] = {"type": "json_object"}

            # 发起异步请求
            response = await self.async_client.chat.completions.create(**params)

            content = response.choices[0].message.content
            return content

        except APIError as e:
            logger.error(f"DeepSeek API (Async) 返回错误: {e}")
            raise ConnectionError(f"DeepSeek API Error: {e}")
        except Exception as e:
            logger.error(f"LLM (Async) 请求发生未知错误: {e}")
            raise e

    def query_json(self, prompt: str, system_prompt: str = "You are a helpful data assistant.") -> Dict[str, Any]:
        """
        获取 JSON 结构化数据的高级封装 (同步版本)。
        """
        # DeepSeek/OpenAI 要求：使用 json_mode 时，Prompt 中必须包含 "json" 字样
        if "json" not in system_prompt.lower() and "json" not in prompt.lower():
            system_prompt += " Please output the result strictly in JSON format."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # 调用 chat 获取原始字符串
        raw_response = self.chat(messages, json_mode=True)

        # 数据清洗 (防止 Markdown 包裹)
        clean_response = self._clean_markdown(raw_response)

        try:
            return json.loads(clean_response)
        except json.JSONDecodeError:
            logger.error(f"JSON 解析失败。原始返回: {raw_response}")
            raise ValueError("LLM 未返回有效的 JSON 格式")

    async def query_json_async(self, prompt: str, system_prompt: str = "You are a helpful data assistant.") -> Dict[
        str, Any]:
        """
        [新增] 获取 JSON 结构化数据的高级封装 (异步版本)。
        """
        if "json" not in system_prompt.lower() and "json" not in prompt.lower():
            system_prompt += " Please output the result strictly in JSON format."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # 调用异步 chat
        raw_response = await self.chat_async(messages, json_mode=True)

        clean_response = self._clean_markdown(raw_response)

        try:
            return json.loads(clean_response)
        except json.JSONDecodeError:
            logger.error(f"JSON (Async) 解析失败。原始返回: {raw_response}")
            raise ValueError("LLM 未返回有效的 JSON 格式")

    def _clean_markdown(self, text: str) -> str:
        """去除可能存在的 Markdown 代码块标记 (```json ... ```)"""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]
        return text.strip()


# --- 单元测试 ---
if __name__ == "__main__":
    import asyncio  # 引入 asyncio 运行测试

    print("正在测试 DeepSeek (OpenAI SDK) 连接...")

    # 请在此处填入您的真实 Key 进行测试
    client = AIClient(api_key="sk-60160407beb64fb989638a7e1aaadf12", model_name="deepseek-chat")

    # 1. 测试连通性
    if client.is_alive():
        print("✅ API 连接正常")
    else:
        print("❌ 无法连接，请检查 API Key 或网络")
        exit()

    # 2. 测试普通对话
    try:
        print("\n[普通对话测试] 9.11 和 9.9 哪个大？")
        reply = client.chat([{"role": "user", "content": "9.11 和 9.9 哪个大？只告诉我结果。"}])
        print(f"AI: {reply}")
    except Exception as e:
        print(f"普通对话失败: {e}")

    # 3. 测试 JSON
    try:
        print("\n[JSON 测试] 提取信息")
        res = client.query_json("""
        Analyze this file:
        Filename: nyc_taxi_2025.csv
        Columns: pickup_lat, pickup_lon, fare_amount
        Return JSON with "file_type" and "columns".
        """)
        print(f"JSON: {json.dumps(res, indent=2)}")
        try:
            if "file_type" in res:
                print("✅ JSON 测试通过")
        except Exception as e:
            print(f"JSON 测试失败: {e}")
    except Exception as e:
        print(f"JSON 测试失败: {e}")


    # 4. [新增] 测试异步调用
    async def test_async():
        print("\n[异步测试] 同时发起两个请求...")
        try:
            task1 = client.chat_async([{"role": "user", "content": "1+1=?"}])
            task2 = client.query_json_async("Return {'status': 'ok'}")

            res1, res2 = await asyncio.gather(task1, task2)
            print(f"AI Async Chat: {res1}")
            print(f"AI Async JSON: {res2}")
            print("✅ 异步测试通过")
        except Exception as e:
            print(f"异步测试失败: {e}")


    asyncio.run(test_async())
