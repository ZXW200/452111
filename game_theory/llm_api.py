"""
统一 LLM API 调用接口
Unified LLM API Interface

支持: OpenAI / Gemini / DeepSeek / 本地Ollama
"""

import os
import json
from typing import Optional, Dict, List

import requests

# 全局共享的 Session 实例，用于跨线程复用 TCP 连接
# requests.Session 是线程安全的，可以在多线程中共享使用
_shared_session: requests.Session = None


def get_shared_session() -> requests.Session:
    """
    获取全局共享的 requests.Session

    优化点：
    - 复用 TCP 连接，避免每次请求都进行 TCP 握手和 SSL 验证
    - 线程安全，可在 ThreadPoolExecutor 中安全使用
    - 单例模式，整个进程只维护一个连接池
    """
    global _shared_session
    if _shared_session is None:
        _shared_session = requests.Session()
        # 配置连接池大小，适应并发 API 请求
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100,  # 连接池数量
            pool_maxsize=100,      # 每个连接池的最大连接数
            max_retries=3          # 自动重试次数
        )
        _shared_session.mount('http://', adapter)
        _shared_session.mount('https://', adapter)
    return _shared_session

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "llm_config.json")

DEFAULT_CONFIG = {
    "default_provider": "deepseek",
    
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "gemini": {
        "api_key": "",
        "model": "gemini-1.5-flash",
    },
    "deepseek": {
        "api_key": "",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },

}


def load_config() -> Dict:
    """加载配置"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG


def save_config(config: Dict):
    """保存配置"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_api_key(provider: str) -> str:
    """获取 API Key（优先环境变量）"""
    env_keys = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
    }
    if provider in env_keys:
        env_val = os.environ.get(env_keys[provider])
        if env_val:
            return env_val
    config = load_config()
    return config.get(provider, {}).get("api_key", "")


class LLMClient:
    """
    统一 LLM 客户端

    Example:
        llm = LLMClient()  # 使用默认 provider
        llm = LLMClient(provider="openai")
        response = llm.chat("你好")

    优化：使用 requests.Session 复用 TCP 连接，
    避免每次请求都进行 TCP 握手和 SSL 验证（节省约 0.2~0.5秒/请求）
    """

    def __init__(self, provider: str = None, session: requests.Session = None):
        self.config = load_config()
        self.provider = provider or self.config.get("default_provider", "deepseek")
        # 默认使用全局共享的 Session，复用 TCP 连接
        # 也可传入自定义 session
        self.session = session or get_shared_session()
        
    def chat(self,
             prompt: str,
             system_prompt: str = None,
             temperature: float = 0.7,
             max_tokens: int = 500) -> str:
        """发送聊天请求"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.provider == "gemini":
            return self._call_gemini(messages, temperature, max_tokens)
        elif self.provider == "ollama":
            return self._call_ollama(messages, temperature, max_tokens)
        else:
            return self._call_openai_compatible(messages, temperature, max_tokens)
    
    def _call_openai_compatible(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """调用 OpenAI 兼容 API (OpenAI/DeepSeek/中转站)"""
        provider_config = self.config.get(self.provider, {})
        api_key = get_api_key(self.provider)
        base_url = provider_config.get("base_url", "https://api.openai.com/v1")
        model = provider_config.get("model", "gpt-4o-mini")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            resp = self.session.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[API Error: {e}]"
    
    def _call_gemini(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """调用 Gemini API"""
        api_key = get_api_key("gemini")
        model = self.config.get("gemini", {}).get("model", "gemini-1.5-flash")

        # 转换消息格式为 Gemini 格式
        contents = []
        system_instruction = None
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
            elif msg["role"] == "assistant":
                contents.append({"role": "model", "parts": [{"text": msg["content"]}]})

        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }
        if system_instruction:
            data["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

        try:
            resp = self.session.post(url, json=data, timeout=60)
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"[Gemini Error: {e}]"
    
    def _call_ollama(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """调用本地 Ollama"""
        ollama_config = self.config.get("ollama", {})
        base_url = ollama_config.get("base_url", "http://localhost:11434")
        model = ollama_config.get("model", "llama3")

        data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens}
        }

        try:
            resp = self.session.post(f"{base_url}/api/chat", json=data, timeout=120)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            return f"[Ollama Error: {e}]"
    
    def test_connection(self) -> bool:
        """测试连接"""
        try:
            response = self.chat("Say OK", max_tokens=10)
            return not response.startswith("[") and len(response) > 0
        except:
            return False


def chat(prompt: str, provider: str = None, **kwargs) -> str:
    """快速聊天接口"""
    client = LLMClient(provider=provider)
    return client.chat(prompt, **kwargs)


def setup_wizard():
    """配置向导"""
    print("\n" + "="*50)
    print("🔧 LLM API 配置向导")
    print("="*50)
    
    config = load_config()
    
    print("\n可用 Provider:")
    print("  1. deepseek  - ¥1/百万token，最便宜 (推荐)")
    print("  2. openai    - GPT-4o-mini")
    print("  3. gemini    - Gemini 1.5 Flash")
    print("  4. ollama    - 本地模型，免费")

    choice = input("\n选择 (1-4) [1]: ").strip() or "1"
    provider = {"1": "deepseek", "2": "openai", "3": "gemini", "4": "ollama"}.get(choice, "deepseek")
    config["default_provider"] = provider
    
    if provider != "ollama":
        api_key = input(f"\n输入 {provider.upper()} API Key: ").strip()
        if api_key:
            config[provider]["api_key"] = api_key
    
    save_config(config)
    print(f"\n✅ 配置已保存")
    
    print("\n测试连接...")
    client = LLMClient(provider=provider)
    if client.test_connection():
        print("✅ 连接成功!")
    else:
        print("❌ 连接失败，请检查 API Key")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        setup_wizard()
    else:
        print("运行 python llm_api.py setup 进行配置")
