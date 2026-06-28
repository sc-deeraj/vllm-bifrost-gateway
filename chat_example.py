#!/usr/bin/env python3
"""
Interactive chat example using ConversationManager.

Run with: python chat_example.py
Then type your questions. Previous context is maintained automatically.
"""

from openai_client import ConversationManager
import sys
import time
import threading


class TypingIndicator:
    """Show animated typing indicator while waiting for response."""

    def __init__(self):
        self.stop = False
        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.frame_index = 0

    def animate(self):
        """Run animation in background thread."""
        while not self.stop:
            sys.stdout.write(f"\r{self.frames[self.frame_index % len(self.frames)]} Thinking...")
            sys.stdout.flush()
            self.frame_index += 1
            time.sleep(0.1)
        sys.stdout.write("\r")  # Clear line
        sys.stdout.flush()

    def start(self):
        """Start typing indicator."""
        self.stop = False
        thread = threading.Thread(target=self.animate, daemon=True)
        thread.start()
        return thread

    def end(self):
        """Stop typing indicator."""
        self.stop = True
        time.sleep(0.2)


def main():
    # Configuration
    API_KEY = "sk-bf-f3a27705a3f6c8af23a6a31d9b78f292c1eb65752d346837"

    print("\n" + "=" * 60)
    print("AI Chat with Context Management")
    print("=" * 60)
    print("\nCommands:")
    print("  Type your question - AI will respond with full context")
    print("  'history'  - Show conversation history")
    print("  'clear'    - Clear conversation history")
    print("  'exit'     - Exit chat")
    print("=" * 60 + "\n")

    # Create conversation manager
    conv = ConversationManager(
        api_key=API_KEY,
        max_tokens=512,
    )

    # Optional: Set system prompt for consistent behavior
    # conv.set_system_prompt(
    #     "You are a helpful assistant. Answer questions clearly and concisely."
    # )

    turn = 1
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle special commands
            if user_input.lower() == "exit":
                print("\nGoodbye!")
                break

            if user_input.lower() == "clear":
                conv.clear()
                print("✓ Conversation history cleared\n")
                turn = 1
                continue

            if user_input.lower() == "history":
                history = conv.get_history()
                if not history:
                    print("(No conversation history)\n")
                else:
                    print("\n--- Conversation History ---")
                    for i, msg in enumerate(history, 1):
                        role = msg["role"].upper()
                        content = msg["content"]
                        # Truncate long messages for display
                        if len(content) > 150:
                            content = content[:150] + "..."
                        print(f"{i}. [{role}] {content}")
                    print("---\n")
                continue

            # Regular query with full context
            print()  # New line
            indicator = TypingIndicator()
            indicator.start()

            try:
                response = conv.query(user_input)
            finally:
                indicator.end()

            print(f"Assistant: {response}")
            print()

            turn += 1

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            print("Make sure Bifrost gateway is running at http://localhost:8080\n")


def demo_conversation():
    """Run a demo conversation showing context is maintained."""
    print("\n" + "=" * 60)
    print("DEMO: Continuous Conversation with Context")
    print("=" * 60 + "\n")

    API_KEY = "sk-bf-f3a27705a3f6c8af23a6a31d9b78f292c1eb65752d346837"
    conv = ConversationManager(api_key=API_KEY, max_tokens=256)

    # Demo conversation
    demo_questions = [
        "What is machine learning?",
        "Can you give a simple example?",
        "What was my first question?",
    ]

    for i, question in enumerate(demo_questions, 1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {question}")

        indicator = TypingIndicator()
        indicator.start()

        try:
            response = conv.query(question)
            indicator.end()
            print(f"Assistant: {response}")
        except Exception as e:
            indicator.end()
            print(f"❌ Error: {e}")
            print("Make sure Bifrost gateway is running.")
            return

    # Show full history
    print("\n\n--- Full Conversation History ---")
    for msg in conv.get_history():
        role = msg["role"].upper()
        content = msg["content"]
        if len(content) > 100:
            content = content[:100] + "..."
        print(f"[{role}] {content}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_conversation()
    else:
        main()
