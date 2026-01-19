"""
Basic usage example of the Agentic framework.

This demonstrates:
- Creating a model provider
- Defining tools
- Creating an agent
- Running the agent with a runner
"""

from agentic import Agent, Runner, Message, MessageRole
from agentic.core.model import ModelProvider
from agentic.tools import CalculatorTool, WeatherTool
from agentic.providers.mock import MockModelProvider


def main():
    """Run a basic agent example."""
    # Create a model provider (mock for demonstration)
    model_provider = MockModelProvider(name="demo-model")
    
    # Create tools
    calculator = CalculatorTool()
    weather = WeatherTool()
    
    # Create an agent with tools
    agent = Agent(
        name="assistant",
        model_provider=model_provider,
        tools=[calculator, weather],
        system_prompt="You are a helpful assistant that can perform calculations and check weather.",
    )
    
    # Create a runner
    runner = Runner(enable_logging=True)
    
    # Run the agent
    print("=== Agentic Framework Demo ===\n")
    
    # Example 1: Calculator
    print("Example 1: Calculator")
    message1 = Message(
        role=MessageRole.USER,
        content="What is 15 plus 27?",
    )
    response1 = runner.run(agent, message1)
    print(f"User: {message1.content}")
    print(f"Agent: {response1.content}")
    if response1.tool_calls:
        print(f"Tool calls: {len(response1.tool_calls)}")
        for tool_call in response1.tool_calls:
            func = tool_call.get("function", {})
            print(f"  - {func.get('name')}({func.get('arguments')})")
    print()
    
    # Example 2: Weather
    print("Example 2: Weather")
    message2 = Message(
        role=MessageRole.USER,
        content="What's the weather in San Francisco?",
    )
    response2 = runner.run(agent, message2)
    print(f"User: {message2.content}")
    print(f"Agent: {response2.content}")
    if response2.tool_calls:
        print(f"Tool calls: {len(response2.tool_calls)}")
    print()
    
    # Show execution log
    print("Execution Log:")
    for entry in runner.get_execution_log():
        agent_info = entry.get('agent', {})
        agent_name = agent_info.get('name', 'N/A') if isinstance(agent_info, dict) else 'N/A'
        print(f"  {entry.get('event')}: {agent_name}")


if __name__ == "__main__":
    main()
