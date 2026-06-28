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


class ConversationManager:
    """Manage multi-turn conversations with context history."""

    def __init__(
        self,
        model: str = "local-vllm/qwen3.5-9b",
        max_tokens: int = 512,
        base_url: str = "http://localhost:8080/openai",
        api_key: Optional[str] = None,
        config_path: Optional[pathlib.Path] = None,
    ):
        """
        Initialize conversation manager.

        Args:
            model: Model identifier
            max_tokens: Max tokens per response
            base_url: Bifrost gateway URL
            api_key: API key or virtual key
            config_path: Path to bifrost config.json
        """
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key = api_key
        self.config_path = config_path
        self.messages = []
        self.client = get_client(
            base_url=base_url,
            api_key=api_key,
            config_path=config_path,
        )

    def add_user_message(self, content: str) -> None:
        """Add user message to conversation history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add assistant message to conversation history."""
        self.messages.append({"role": "assistant", "content": content})

    def query(self, user_input: str) -> str:
        """
        Send a message and get a response, maintaining context.

        Args:
            user_input: User message to send

        Returns:
            Assistant response

        Example:
            conv = ConversationManager(api_key="your-key")
            answer1 = conv.query("What is Python?")
            answer2 = conv.query("Can you explain more?")  # Has context of first question
        """
        self.add_user_message(user_input)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            max_tokens=self.max_tokens,
        )

        content = (resp.choices[0].message.content or "").strip()
        self.add_assistant_message(content)
        return content

    def get_history(self) -> list[dict]:
        """Get the full conversation history."""
        return self.messages.copy()

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages = []

    def set_system_prompt(self, system_prompt: str) -> None:
        """Set system prompt at the beginning of conversation."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = system_prompt
        else:
            self.messages.insert(0, {"role": "system", "content": system_prompt})


if __name__ == "__main__":
    # Example usage
    API_KEY = "sk-bf-f3a27705a3f6c8af23a6a31d9b78f292c1eb65752d346837"

    print("=== Simple Query (No Context) ===")
    answer = query_model("What is the capital of France?", api_key=API_KEY)
    print(f"Answer: {answer}\n")

    print("=== Continued Chat (With Context) ===")
    conv = ConversationManager(api_key=API_KEY)

    # First question
    q1 = "What is Python?"
    print(f"User: {q1}")
    a1 = conv.query(q1)
    print(f"Assistant: {a1}\n")

    # Follow-up question (model remembers context)
    q2 = "Can you give me a simple example?"
    print(f"User: {q2}")
    a2 = conv.query(q2)
    print(f"Assistant: {a2}\n")

    # Another follow-up
    q3 = "What was my first question?"
    print(f"User: {q3}")
    a3 = conv.query(q3)
    print(f"Assistant: {a3}\n")

    print("=== Conversation History ===")
    for msg in conv.get_history():
        print(f"{msg['role'].upper()}: {msg['content'][:100]}...")

