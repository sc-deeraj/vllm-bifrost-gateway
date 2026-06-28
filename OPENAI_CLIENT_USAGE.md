# OpenAI-Compatible Client Usage Guide

A simple, reusable module for querying the vLLM Bifrost Gateway with OpenAI-compatible API.

## Installation

Copy `openai_client.py` to your project and install dependencies:

```bash
pip install openai
```

## Quick Start

### Simple Query

```python
from openai_client import query_model

response = query_model("What is Python?")
print(response)
```

### With Custom Configuration

```python
from openai_client import query_model

response = query_model(
    "What is Python?",
    model="local-vllm/qwen3.5-9b",
    max_tokens=512,
    base_url="http://localhost:8080/openai",
    api_key="your-api-key"
)
print(response)
```

## Usage Patterns

### 1. Simple Text Query

```python
from openai_client import query_model

answer = query_model("Explain machine learning in one sentence")
print(answer)
```

### 2. Using Direct Client

```python
from openai_client import get_client

client = get_client()
response = client.chat.completions.create(
    model="local-vllm/qwen3.5-9b",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    max_tokens=256
)
print(response.choices[0].message.content)
```

### 3. Query with Reasoning (for reasoning models)

```python
from openai_client import query_model_with_reasoning

answer, reasoning = query_model_with_reasoning(
    "Solve: What is 15 * 23?"
)
print(f"Answer: {answer}")
print(f"Thinking process: {reasoning}")
```

### 4. Vision Query

```python
from openai_client import query_model_with_vision

response = query_model_with_vision(
    prompt="Describe what you see",
    image_url="https://example.com/image.jpg"
)
print(response)
```

### 5. Streaming Response

```python
from openai_client import get_client

client = get_client()
stream = client.chat.completions.create(
    model="local-vllm/qwen3.5-9b",
    messages=[{"role": "user", "content": "Write a poem about AI"}],
    stream=True,
    max_tokens=512
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## Configuration

### Environment Variables

Set these environment variables to avoid passing them repeatedly:

```bash
export BIFROST_BASE_URL="http://localhost:8080/openai"
export BIFROST_API_KEY="your-api-key"
```

Then use in code:

```python
from openai_client import query_model
import os

response = query_model(
    "Hello",
    base_url=os.getenv("BIFROST_BASE_URL"),
    api_key=os.getenv("BIFROST_API_KEY")
)
```

### Using Config File

If you have a `bifrost/config.json` file:

```python
from openai_client import query_model
import pathlib

response = query_model(
    "Hello",
    config_path=pathlib.Path("/path/to/bifrost/config.json")
)
```

## API Reference

### `query_model(question, model, max_tokens, base_url, api_key, config_path)`

Simple function to query the model.

**Parameters:**
- `question` (str): The prompt to send
- `model` (str): Model ID (default: "local-vllm/qwen3.5-9b")
- `max_tokens` (int): Max response length (default: 512)
- `base_url` (str): Bifrost endpoint (default: "http://localhost:8080/openai")
- `api_key` (str, optional): API key/virtual key
- `config_path` (Path, optional): Path to bifrost config.json

**Returns:** str - Model response text

---

### `query_model_with_reasoning(question, model, max_tokens, base_url, api_key, config_path)`

Query model and extract reasoning (for reasoning-capable models).

**Returns:** tuple(str, str) - (answer, reasoning)

---

### `query_model_with_vision(prompt, image_url, model, max_tokens, base_url, api_key, config_path)`

Query multimodal model with image.

**Parameters:**
- `prompt` (str): Text prompt
- `image_url` (str): Image URL or data URI
- Other params same as `query_model`

**Returns:** str - Model response text

---

### `get_client(base_url, api_key, config_path)`

Get a configured OpenAI client for advanced usage.

**Returns:** OpenAI - Client instance for full OpenAI API compatibility

---

### `BifrostConfig` class

Advanced configuration management.

```python
from openai_client import BifrostConfig

config = BifrostConfig(
    base_url="http://localhost:8080/openai",
    api_key="your-key",
    provider="local-vllm",
    model_id="qwen3.5-9b"
)
print(config.model)  # "local-vllm/qwen3.5-9b"
```

## Examples

### Example 1: Chat Assistant

```python
from openai_client import get_client

def chat_with_model(messages):
    client = get_client()
    response = client.chat.completions.create(
        model="local-vllm/qwen3.5-9b",
        messages=messages,
        max_tokens=512
    )
    return response.choices[0].message.content

# Multi-turn conversation
messages = [
    {"role": "user", "content": "Hello, what's your name?"},
]
response = chat_with_model(messages)
print(f"Assistant: {response}")

messages.append({"role": "assistant", "content": response})
messages.append({"role": "user", "content": "Can you help me with Python?"})
response = chat_with_model(messages)
print(f"Assistant: {response}")
```

### Example 2: Batch Queries

```python
from openai_client import query_model

questions = [
    "What is AI?",
    "What is ML?",
    "What is DL?"
]

for q in questions:
    answer = query_model(q)
    print(f"Q: {q}\nA: {answer}\n")
```

### Example 3: With Error Handling

```python
from openai_client import query_model
from openai import APIError

try:
    response = query_model("What is quantum computing?")
    print(response)
except APIError as e:
    print(f"API Error: {e}")
except FileNotFoundError as e:
    print(f"Config Error: {e}")
```

## Troubleshooting

### Connection Error
```
Error: Cannot connect to http://localhost:8080/openai
```
**Solution:** Ensure Bifrost gateway is running and accessible.

### Authentication Error
```
Error: 401 Unauthorized
```
**Solution:** Check API key or virtual key is correct. Verify `x-bf-vk` header is being sent.

### Config File Not Found
```
FileNotFoundError: Config file not found: bifrost/config.json
```
**Solution:** Either provide `api_key` parameter or ensure `bifrost/config.json` exists at the correct path.

### Model Not Found
```
Error: Model not found
```
**Solution:** Ensure model is properly loaded in vLLM. Use correct format: `provider/model-id`
