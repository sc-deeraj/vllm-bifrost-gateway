"""
Practical examples for using openai_client module in other projects.

Run with: python examples_openai_client.py
"""

from openai_client import (
    get_client,
    query_model,
    query_model_with_reasoning,
    BifrostConfig,
)


def example_1_simple_query():
    """Example 1: Simple one-line query."""
    print("\n" + "=" * 60)
    print("Example 1: Simple Query")
    print("=" * 60)

    response = query_model("What is Python programming?")
    print(f"Answer: {response}")


def example_2_custom_model():
    """Example 2: Using different models."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Model and Max Tokens")
    print("=" * 60)

    response = query_model(
        "Explain recursion in 100 words",
        model="local-vllm/qwen3.5-9b",
        max_tokens=256,
    )
    print(f"Answer: {response}")


def example_3_client_usage():
    """Example 3: Using the client directly for advanced features."""
    print("\n" + "=" * 60)
    print("Example 3: Direct Client Usage")
    print("=" * 60)

    client = get_client()

    # Single message
    response = client.chat.completions.create(
        model="local-vllm/qwen3.5-9b",
        messages=[
            {"role": "user", "content": "What is 2+2?"},
        ],
        max_tokens=100,
    )
    print(f"Answer: {response.choices[0].message.content}")


def example_4_multi_turn_conversation():
    """Example 4: Multi-turn conversation."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-Turn Conversation")
    print("=" * 60)

    client = get_client()
    messages = []

    # Turn 1
    user_input = "Hello! What's your name?"
    print(f"User: {user_input}")
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="local-vllm/qwen3.5-9b",
        messages=messages,
        max_tokens=256,
    )
    assistant_response = response.choices[0].message.content
    print(f"Assistant: {assistant_response}")
    messages.append({"role": "assistant", "content": assistant_response})

    # Turn 2
    user_input = "Can you help me learn Python?"
    print(f"\nUser: {user_input}")
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="local-vllm/qwen3.5-9b",
        messages=messages,
        max_tokens=256,
    )
    assistant_response = response.choices[0].message.content
    print(f"Assistant: {assistant_response}")


def example_5_batch_processing():
    """Example 5: Process multiple queries."""
    print("\n" + "=" * 60)
    print("Example 5: Batch Processing")
    print("=" * 60)

    questions = [
        "What is AI?",
        "What is ML?",
        "What is DL?",
        "What is NLP?",
    ]

    for i, question in enumerate(questions, 1):
        response = query_model(question, max_tokens=100)
        print(f"{i}. Q: {question}")
        print(f"   A: {response}\n")


def example_6_reasoning():
    """Example 6: Extract reasoning from reasoning models."""
    print("\n" + "=" * 60)
    print("Example 6: Query with Reasoning")
    print("=" * 60)

    question = "What is 15 * 23?"
    answer, reasoning = query_model_with_reasoning(question)

    print(f"Question: {question}")
    print(f"Answer: {answer}")
    if reasoning:
        print(f"Reasoning: {reasoning}")
    else:
        print("(No reasoning available - try a reasoning model)")


def example_7_error_handling():
    """Example 7: Proper error handling."""
    print("\n" + "=" * 60)
    print("Example 7: Error Handling")
    print("=" * 60)

    try:
        response = query_model("Hello world", api_key="invalid-key")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

    print("\nNote: This example shows how to catch errors.")
    print("In production, handle authentication and connection errors gracefully.")


def example_8_custom_config():
    """Example 8: Using BifrostConfig for advanced configuration."""
    print("\n" + "=" * 60)
    print("Example 8: Custom Configuration")
    print("=" * 60)

    config = BifrostConfig(
        base_url="http://localhost:8080/openai",
        api_key="your-api-key",
        provider="local-vllm",
        model_id="qwen3.5-9b",
    )

    print(f"Base URL: {config.base_url}")
    print(f"Model: {config.model}")
    print(f"API Key configured: {bool(config.api_key)}")

    # Now use with client
    client = get_client(
        base_url=config.base_url,
        api_key=config.api_key,
    )
    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100,
    )
    print(f"Response: {response.choices[0].message.content}")


def example_9_streaming():
    """Example 9: Streaming responses."""
    print("\n" + "=" * 60)
    print("Example 9: Streaming Response")
    print("=" * 60)

    client = get_client()

    print("Streaming response:\n")
    stream = client.chat.completions.create(
        model="local-vllm/qwen3.5-9b",
        messages=[{"role": "user", "content": "Write a short poem about AI"}],
        stream=True,
        max_tokens=200,
    )

    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")


def example_10_function_with_api_key():
    """Example 10: Create a reusable function with custom config."""
    print("\n" + "=" * 60)
    print("Example 10: Reusable Function Pattern")
    print("=" * 60)

    def ask_question(question: str, max_tokens: int = 256) -> str:
        """Wrapper function for your specific use case."""
        return query_model(
            question,
            model="local-vllm/qwen3.5-9b",
            max_tokens=max_tokens,
            base_url="http://localhost:8080/openai",
            # api_key can be set once and reused
        )

    answer1 = ask_question("What is machine learning?")
    print(f"Q1: What is machine learning?\nA1: {answer1}\n")

    answer2 = ask_question("What is deep learning?")
    print(f"Q2: What is deep learning?\nA2: {answer2}\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OpenAI Client Examples")
    print("=" * 60)
    print("\nMake sure Bifrost gateway is running at http://localhost:8080")
    print("Before running, update api_key in examples if needed.\n")

    try:
        # Run examples
        example_1_simple_query()
        example_2_custom_model()
        example_3_client_usage()
        example_4_multi_turn_conversation()
        example_5_batch_processing()
        example_6_reasoning()
        example_7_error_handling()
        example_8_custom_config()
        # example_9_streaming()  # Uncomment to see streaming
        example_10_function_with_api_key()

        print("\n" + "=" * 60)
        print("All examples completed!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure:")
        print("1. Bifrost gateway is running at http://localhost:8080")
        print("2. vLLM model is loaded")
        print("3. API key/virtual key is correct")
