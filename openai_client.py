"""
OpenAI-compatible client for vLLM Bifrost Gateway.

Usage:
    from openai_client import query_model, get_client

    # Simple usage
    response = query_model("What is Python?")
    print(response)

    # Advanced usage with custom client
    client = get_client(base_url="http://localhost:8080/openai", api_key="your-key")
    response = client.chat.completions.create(
        model="local-vllm/qwen3.5-9b",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=512
    )
    print(response.choices[0].message.content)
"""

import json
import pathlib
from typing import Optional

from openai import OpenAI


class BifrostConfig:
    """Configuration for Bifrost gateway connection."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080/openai",
        api_key: Optional[str] = None,
        config_path: Optional[pathlib.Path] = None,
        provider: str = "local-vllm",
        model_id: str = "qwen3.5-9b",
    ):
        """
        Initialize Bifrost configuration.

        Args:
            base_url: Bifrost gateway OpenAI endpoint URL
            api_key: API key or virtual key. If not provided, loads from config_path
            config_path: Path to bifrost config.json (used to extract virtual key)
            provider: Model provider prefix (default: local-vllm)
            model_id: Model identifier (default: qwen3.5-9b)
        """
        self.base_url = base_url
        self.provider = provider
        self.model_id = model_id
        self.config_path = config_path or pathlib.Path(__file__).parent / "bifrost" / "config.json"
        self.api_key = api_key

    @property
    def model(self) -> str:
        """Return full model identifier in provider/model format."""
        return f"{self.provider}/{self.model_id}"

def get_client(
    base_url: str = "http://localhost:8080/openai",
    api_key: Optional[str] = None,
    config_path: Optional[pathlib.Path] = None,
) -> OpenAI:
    """
    Create an OpenAI client configured for Bifrost gateway.

    Args:
        base_url: Bifrost gateway OpenAI endpoint URL
        api_key: API key or virtual key. If not provided, loads from config_path
        config_path: Path to bifrost config.json

    Returns:
        Configured OpenAI client instance

    Example:
        client = get_client()
        response = client.chat.completions.create(
            model="local-vllm/qwen3.5-9b",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """
    config = BifrostConfig(
        base_url=base_url,
        api_key=api_key,
        config_path=config_path,
    )

    return OpenAI(
        base_url=config.base_url,
        api_key=config.api_key,
        default_headers={"x-bf-vk": config.api_key},
    )


def query_model(
    question: str,
    model: str = "local-vllm/qwen3.5-9b",
    max_tokens: int = 512,
    base_url: str = "http://localhost:8080/openai",
    api_key: Optional[str] = None,
    config_path: Optional[pathlib.Path] = None,
) -> str:
    """
    Simple query function for the OpenAI-compatible endpoint.

    Args:
        question: The prompt/question to send
        model: Model identifier (default: local-vllm/qwen3.5-9b)
        max_tokens: Maximum tokens in response (default: 512)
        base_url: Bifrost gateway URL
        api_key: API key or virtual key
        config_path: Path to bifrost config.json

    Returns:
        Model response text

    Example:
        response = query_model("What is 2+2?")
        print(response)  # "2 + 2 equals 4"
    """
    client = get_client(base_url=base_url, api_key=api_key, config_path=config_path)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=max_tokens,
    )

    return (resp.choices[0].message.content or "").strip()


def query_model_with_reasoning(
    question: str,
    model: str = "local-vllm/qwen3.5-9b",
    max_tokens: int = 512,
    base_url: str = "http://localhost:8080/openai",
    api_key: Optional[str] = None,
    config_path: Optional[pathlib.Path] = None,
) -> tuple[str, str]:
    """
    Query model and extract both answer and reasoning (if available).

    Args:
        question: The prompt/question to send
        model: Model identifier
        max_tokens: Maximum tokens in response
        base_url: Bifrost gateway URL
        api_key: API key or virtual key
        config_path: Path to bifrost config.json

    Returns:
        Tuple of (answer, reasoning). Reasoning may be empty string if not available.

    Example:
        answer, reasoning = query_model_with_reasoning("Explain quantum computing")
        print(f"Answer: {answer}")
        print(f"Reasoning: {reasoning}")
    """
    client = get_client(base_url=base_url, api_key=api_key, config_path=config_path)

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=max_tokens,
    )

    msg = resp.choices[0].message
    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", "")
    answer = (msg.content or "").strip()

    return answer, (reasoning or "").strip()


def query_model_with_vision(
    prompt: str,
    image_url: str,
    model: str = "local-vllm/qwen3.5-9b",
    max_tokens: int = 512,
    base_url: str = "http://localhost:8080/openai",
    api_key: Optional[str] = None,
    config_path: Optional[pathlib.Path] = None,
) -> str:
    """
    Query vision model with text and image.

    Args:
        prompt: Text prompt for the image
        image_url: URL or data URI of the image
        model: Model identifier (must be multimodal)
        max_tokens: Maximum tokens in response
        base_url: Bifrost gateway URL
        api_key: API key or virtual key
        config_path: Path to bifrost config.json

    Returns:
        Model response text

    Example:
        response = query_model_with_vision(
            "Describe this image",
            "https://example.com/image.jpg"
        )
        print(response)
    """
    client = get_client(base_url=base_url, api_key=api_key, config_path=config_path)

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
    )

    return (resp.choices[0].message.content or "").strip()


if __name__ == "__main__":
    # Example usage
    API_KEY = "sk-bf-f3a27705a3f6c8af23a6a31d9b78f292c1eb65752d346837"
    print("=== Simple Query ===")
    answer = query_model("What is the capital of France?", api_key=API_KEY)
    print(f"Answer: {answer}\n")

    print("=== Query with Reasoning ===")
    answer = query_model("What is 2+2?", api_key=API_KEY)
    print(f"Answer: {answer}")

