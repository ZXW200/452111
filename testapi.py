from game_theory.llm_api import LLMClient

for provider in ["deepseek", "openai", "gemini"]:
    print(f"\n测试 {provider}...")
    client = LLMClient(provider=provider)
    response = client.chat("说OK")
    print(f"  响应: {response[:50]}")
