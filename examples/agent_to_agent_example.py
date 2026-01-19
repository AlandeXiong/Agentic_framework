"""
Agent-to-Agent communication example using AgentCard.

This example shows:
- How to describe agents via AgentCard
- How to instantiate multiple agents from cards
- How to coordinate them with Runner.run_multi_agent
"""

from agentic import (
    AgentCard,
    Agent,
    Message,
    MessageRole,
    Runner,
)
from agentic.tools import CalculatorTool, WeatherTool
from agentic.providers.mock import MockModelProvider


def build_agents() -> list[Agent]:
    """Create two agents from AgentCards."""
    # Shared tools
    calculator = CalculatorTool()
    weather = WeatherTool()
    tool_registry = {
        calculator.name: calculator,
        weather.name: weather,
    }

    # Shared model provider (in real apps these can differ)
    model_provider = MockModelProvider(name="multi-agent-demo")

    # Planner agent: focuses on understanding and planning
    planner_card = AgentCard(
        name="planner",
        description="Understands user goals and proposes a plan.",
        system_prompt=(
            "You are a planning agent. "
            "Your job is to understand the user's goal and describe a plan. "
            "Do not execute tools yourself; describe what the executor should do."
        ),
        tool_names=[],  # planner does not call tools in this simple demo
        metadata={"role": "planner"},
    )

    # Executor agent: focuses on executing tools
    executor_card = AgentCard(
        name="executor",
        description="Executes concrete steps using tools.",
        system_prompt=(
            "You are an executor agent. "
            "Given a plan or instruction, call tools to get concrete results "
            "and answer the user as specifically as possible."
        ),
        tool_names=["calculator", "weather"],
        metadata={"role": "executor"},
    )

    planner = planner_card.create_agent(
        model_provider=model_provider,
        tool_registry=tool_registry,
    )
    executor = executor_card.create_agent(
        model_provider=model_provider,
        tool_registry=tool_registry,
    )
    return [planner, executor]


def round_robin_routing():
    """Create a simple round-robin routing strategy."""
    current_idx = {"value": 0}

    def strategy(message: Message, agents: list[Agent]) -> Agent:
        agent = agents[current_idx["value"] % len(agents)]
        current_idx["value"] += 1
        return agent

    return strategy


def main() -> None:
    """Run the agent-to-agent example."""
    agents = build_agents()
    runner = Runner(enable_logging=True)

    user_message = Message(
        role=MessageRole.USER,
        content=(
            "I want to know two things: "
            "1) What is 15 plus 27, and 2) what's the weather in San Francisco?"
        ),
    )

    print("=== Agent-to-Agent Example (Planner + Executor) ===")
    print("User:", user_message.content)
    print()

    messages = runner.run_multi_agent(
        agents=agents,
        initial_message=user_message,
        routing_strategy=round_robin_routing(),
    )

    for i, msg in enumerate(messages, start=1):
        print(f"Turn {i} - {msg.role}: {msg.content}")


if __name__ == "__main__":
    main()

