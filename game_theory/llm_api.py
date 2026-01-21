"""
统一 LLM API 调用接口
Unified LLM API Interface

支持: OpenAI / Claude / DeepSeek / 本地Ollama
"""

import os
import json
from typing import Optional, Dict, List

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "llm_config.json")

DEFAULT_CONFIG = {
    "default_provider": "deepseek",
    
    "openai": {
        "api_key": "",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "claude": {
        "api_key": "",
        "model": "claude-3-haiku-20240307",
    },
    "deepseek": {
        "api_key": "",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3",
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
        "claude": "ANTHROPIC_API_KEY",
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
    """
    
    def __init__(self, provider: str = None):
        self.config = load_config()
        self.provider = provider or self.config.get("default_provider", "deepseek")
        
    def chat(self,
             prompt: str,
             system_prompt: str = None,
             temperature: float = 0.7,
             max_tokens: int = 500) -> str:
        """发送聊天请求"""
        try:
            import requests
        except ImportError:
            return "[Error: pip install requests]"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.provider == "claude":
            return self._call_claude(messages, temperature, max_tokens)
        elif self.provider == "ollama":
            return self._call_ollama(messages, temperature, max_tokens)
        else:
            return self._call_openai_compatible(messages, temperature, max_tokens)
    
    def _call_openai_compatible(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """调用 OpenAI 兼容 API (OpenAI/DeepSeek/中转站)"""
        import requests
        
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
            resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[API Error: {e}]"
    
    def _call_claude(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """调用 Claude API"""
        import requests
        
        api_key = get_api_key("claude")
        model = self.config.get("claude", {}).get("model", "claude-3-haiku-20240307")
        
        # 分离 system prompt
        system_prompt = None
        claude_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                claude_messages.append(msg)
        
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": model,
            "messages": claude_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_prompt:
            data["system"] = system_prompt
        
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=60)
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]
        except Exception as e:
            return f"[Claude Error: {e}]"
    
    def _call_ollama(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """调用本地 Ollama"""
        import requests
        
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
            resp = requests.post(f"{base_url}/api/chat", json=data, timeout=120)
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
    print("  3. claude    - Claude 3 Haiku")
    print("  4. ollama    - 本地模型，免费")
    
    choice = input("\n选择 (1-4) [1]: ").strip() or "1"
    provider = {"1": "deepseek", "2": "openai", "3": "claude", "4": "ollama"}.get(choice, "deepseek")
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
