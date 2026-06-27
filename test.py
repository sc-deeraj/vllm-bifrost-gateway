from openai import OpenAI
from anthropic import Anthropic

openai_client = OpenAI(base_url="http://localhost:8080/openai", api_key="sk-local-no-auth")
anthropic_client = Anthropic(base_url="http://localhost:8080/anthropic", api_key="sk-local-no-auth")

question = "What is capital of france?"

oa_resp = openai_client.chat.completions.create(
    model="local-vllm/qwen2.5-3b-instruct",
    messages=[{"role": "user", "content": question}],
    max_tokens=80,
)
print("[OpenAI SDK]   ", oa_resp.choices[0].message.content.strip())

an_resp = anthropic_client.messages.create(
    model="local-vllm/qwen2.5-3b-instruct",
    max_tokens=80,
    messages=[{"role": "user", "content": question}],
)
print("[Anthropic SDK]", an_resp.content[0].text.strip())
