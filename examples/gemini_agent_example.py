"""
Gemini-based Agent example.

Steps to run:
1. Install dependencies:
   - pip install -e .
2. Create a .env file in the project root with:
   - GEMINI_API_KEY=your_api_key_here
   - GEMINI_MODEL=gemini-1.5-flash  (optional, has a default)
3. Run:
   - python examples/gemini_agent_example.py
"""

from dotenv import load_dotenv

from agentic import Agent, Runner, Message, MessageRole
from agentic.providers import GeminiModelProvider


def main() -> None:
    """Run a simple Gemini-backed agent."""
    # Load environment variables from .env
    load_dotenv()

    # Create Gemini model provider
    model_provider = GeminiModelProvider()

    # Create an agent without extra tools (pure LLM example)
    agent = Agent(
        name="gemini_agent",
        model_provider=model_provider,
        tools=[],
        system_prompt=(
            "You are a helpful assistant powered by Google Gemini. "
            "Answer concisely and clearly."
        ),
    )

    runner = Runner(enable_logging=True)

    message = Message(
        role=MessageRole.USER,
        content="Give me three bullet points on why modular agent frameworks are useful.",
    )

    print("=== Gemini Agent Example ===")
    print("User:", message.content)
    print()

    response = runner.run(agent, message)
    print("Agent:", response.content)


if __name__ == "__main__":
    main()

