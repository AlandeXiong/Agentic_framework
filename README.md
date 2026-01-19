# Agentic Framework

A model-agnostic, deployment-agnostic, and tool-extensible agentic framework inspired by Google ADK's core design principles.

## Core Design Principles

- **Code-First**: Define agents and tools through code, not configuration
- **Modular Multi-Agent**: Support multiple agents working together
- **Tool Decoupling**: Tools are independent, composable modules
- **Deployment Agnostic**: Works with any model provider and deployment environment
- **Extensible**: Easy to add new tools and agent capabilities

## Architecture

### Core Components

1. **Agent Base Class**: Defines the core interface and behavior for agents
2. **Tool Interface**: Abstract interface for all tools
3. **Runner**: Orchestrates agent execution and coordination
4. **Workflow System**: Declarative orchestration of tool calls (Flow model)
5. **MCP Integration**: Adapter layer for MCP Server tools
6. **Message System**: Type-safe message passing between agents

### Key Features

- **Model Agnostic**: Works with any LLM provider (OpenAI, Anthropic, local models, etc.)
- **Deployment Agnostic**: Can run locally, in cloud, or hybrid environments
- **Tool Extensible**: Easy to create and register new tools
- **Workflow Support**: First-class support for code-defined workflows
- **MCP Support**: Call tools exposed by MCP servers as normal tools
- **Testable**: Modular design enables unit testing of components
- **Type Safe**: Full type hints for better IDE support and error detection

## Installation

```bash
pip install -e .
```

## Quick Start

### 1. Agent Model (LLM decides which tools to call)

```python
from agentic import Agent, Runner, Message, MessageRole
from agentic.tools import CalculatorTool
from agentic.providers.mock import MockModelProvider

# Create a model provider (or use your own implementation)
model_provider = MockModelProvider()

# Create an agent with tools
agent = Agent(
    name="calculator_agent",
    model_provider=model_provider,
    tools=[CalculatorTool()],
    system_prompt="You are a helpful calculator assistant.",
)

# Run the agent
runner = Runner()
message = Message(role=MessageRole.USER, content="What is 5 + 3?")
result = runner.run(agent, message)
print(result.content)
```

### 2. Flow Model (explicit workflow of tools)

```python
from agentic import (
    WorkflowStep,
    StepType,
    Workflow,
    WorkflowContext,
    WorkflowRunner,
)
from agentic.tools import CalculatorTool

# Define steps
step_calculate = WorkflowStep(
    id="calculate_sum",
    step_type=StepType.TOOL,
    name="Calculate Sum",
    tool_name="calculator",
    tool_params={"operation": "add", "a": 15, "b": 27},
    output_key="sum_result",
)

workflow = Workflow(
    id="sum_workflow",
    name="Sum Workflow",
    start_step_id="calculate_sum",
)
workflow.add_step(step_calculate)

# Tools registry
calculator = CalculatorTool()
tools = {calculator.name: calculator}

ctx = WorkflowContext()
runner = WorkflowRunner()
final_ctx = runner.run(workflow, tools=tools, context=ctx)
print(final_ctx.data["sum_result"])
```

### 3. MCP-based Tool (MCP Server integration)

```python
from typing import Dict, Any, Iterable, Optional

from agentic import (
    Agent,
    Message,
    MessageRole,
    Runner,
    MCPClient,
    MCPTool,
    MCPToolConfig,
    MCPAuthConfig,
)
from agentic.tools import CalculatorTool
from agentic.providers.mock import MockModelProvider


class LocalMCPClient(MCPClient):
    def __init__(self) -> None:
        self._handlers: Dict[tuple[str, str], Any] = {}

    def register_handler(self, server_name: str, tool_name: str, handler: Any) -> None:
        self._handlers[(server_name, tool_name)] = handler

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        auth: Optional[MCPAuthConfig] = None,
    ) -> Any:
        handler = self._handlers[(server_name, tool_name)]
        return handler(arguments)

    def stream_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        auth: Optional[MCPAuthConfig] = None,
    ) -> Iterable[Any]:
        yield self.call_tool(server_name, tool_name, arguments, auth=auth)


# Local calculator implementation
calculator = CalculatorTool()

client = LocalMCPClient()
client.register_handler(
    server_name="local-mcp",
    tool_name="calculator",
    handler=lambda args: calculator.execute(**args),
)

mcp_tool = MCPTool(
    config=MCPToolConfig(
        server_name="local-mcp",
        tool_name="calculator",
        description="Calculator via MCP",
        auth=MCPAuthConfig(auth_type="bearer", token="demo-token"),
    ),
    client=client,
)

agent = Agent(
    name="mcp_agent",
    model_provider=MockModelProvider(),
    tools=[mcp_tool],
)

runner = Runner()
message = Message(role=MessageRole.USER, content="What is 21 plus 21?")
result = runner.run(agent, message)
print(result.content)
```

## Architecture Details

### ModelProvider Interface

The `ModelProvider` abstraction allows you to use any LLM:

```python
from agentic.core.model import ModelProvider

class YourModelProvider(ModelProvider):
    def generate(self, messages, tools=None, **kwargs):
        # Your implementation
        pass
    
    def stream(self, messages, tools=None, **kwargs):
        # Your streaming implementation
        pass
```

### Tool Interface

Tools are decoupled, composable modules:

```python
from agentic import Tool

class MyTool(Tool):
    @property
    def name(self) -> str:
        return "my_tool"
    
    @property
    def description(self) -> str:
        return "Tool description"
    
    def _get_parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "param": {"type": "string"}
            },
            "required": ["param"]
        }
    
    def execute(self, **kwargs):
        # Your tool logic
        return result
```

### Runner and Multi-Agent Coordination

The `Runner` orchestrates single-agent and multi-agent execution:

```python
from agentic import Agent, Runner, Message, MessageRole

runner = Runner()

# Single agent
response = runner.run(agent, Message(role=MessageRole.USER, content="..."))

# Multi-agent
messages = runner.run_multi_agent(
    agents=[agent1, agent2],
    initial_message=Message(role=MessageRole.USER, content="..."),
    routing_strategy=your_routing_function,  # Optional
)
```

### Workflow System

The workflow layer lets you declare tool orchestration as data:

```python
from agentic import WorkflowStep, StepType, Workflow, WorkflowContext, WorkflowRunner

step = WorkflowStep(
    id="step1",
    step_type=StepType.TOOL,
    name="Do something",
    tool_name="my_tool",
    tool_params={"param": "value"},
    output_key="result1",
)

workflow = Workflow(
    id="my_workflow",
    name="My Workflow",
    start_step_id="step1",
)
workflow.add_step(step)

runner = WorkflowRunner()
ctx = runner.run(workflow, tools={"my_tool": my_tool}, context=WorkflowContext())
```

### MCP Integration

The MCP integration lets you treat MCP Server tools as normal tools:

- `MCPClient`: abstract client interface (you implement transport + auth)
- `MCPAuthConfig`: generic auth configuration (bearer/api_key/custom)
- `MCPToolConfig`: describes an MCP tool (server, tool name, schema, auth)
- `MCPTool`: adapts an MCP tool to the `Tool` interface and supports streaming

## Project Structure

```
agentic/
├── agentic/
│   ├── core/               # Core abstractions
│   │   ├── agent.py        # Agent base class
│   │   ├── tool.py         # Tool interface
│   │   ├── runner.py       # Runner orchestrator
│   │   ├── model.py        # ModelProvider interface
│   │   ├── message.py      # Message system
│   │   ├── workflow_step.py# Workflow step definition
│   │   ├── workflow.py     # Workflow and context
│   │   ├── workflow_runner.py # Workflow runner
│   │   └── mcp.py          # MCP integration (MCPClient, MCPTool, auth)
│   ├── tools/              # Example tool implementations
│   │   ├── calculator.py
│   │   └── weather.py
│   └── providers/          # Model provider implementations
│       └── mock.py         # Mock provider for testing
├── examples/               # Example usage
│   ├── basic_usage.py      # Agent model example
│   ├── workflow_usage.py   # Flow model example
│   └── mcp_agent_example.py# MCP-based agent example
└── tests/             # Unit tests (to be added)
```

## Core Design Philosophy

This framework extracts the core software engineering principles from Google ADK:

1. **Code-First Approach**: Agents and tools are defined through Python classes, not configuration files
2. **Modular Multi-Agent**: Support for multiple agents working together with clear interfaces
3. **Tool Decoupling**: Tools are independent, composable modules that can be mixed and matched
4. **Deployment Agnostic**: No assumptions about deployment environment (local, cloud, hybrid)
5. **Model Agnostic**: Works with any LLM through the ModelProvider abstraction

Unlike Google ADK, this framework:
- Removes Google ecosystem dependencies
- Provides pure abstractions that work with any model provider
- Focuses on software engineering best practices (modularity, testability, extensibility)

## License

MIT
