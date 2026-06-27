from openai import OpenAI
from anthropic import Anthropic

openai_client = OpenAI(base_url="http://localhost:8080/openai", api_key="sk-local-no-auth")
anthropic_client = Anthropic(base_url="http://localhost:8080/anthropic", api_key="<vk-or-dummy>")

openai_client.chat.completions.create(
    model="local-vllm/qwen2.5-3b-instruct",
    messages=[{"role": "user", "content": "What is vLLM?"}],
    max_tokens=80,
)
