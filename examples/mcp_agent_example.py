"""
MCP-based Agent example.

This example shows how to:
- Implement a simple MCPClient
- Wrap an MCP Server tool as an MCPTool
- Let an Agent call that MCPTool via normal tool-calling flow

For demonstration we simulate an MCP Server that exposes a \"calculator\" tool.
The Agent uses MockModelProvider which already knows how to call a tool
named \"calculator\", so we route that to our MCP implementation instead
of using the local CalculatorTool directly.
"""

from typing import Any, Dict, Iterable, Optional

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
    """A minimal MCPClient implementation for demo purposes.

    It simulates an MCP server by routing calls to local handler functions.
    In a real project you would:
    - Use the official `mcp` Python SDK to connect to a real server, or
    - Implement your own transport (HTTP, gRPC, etc.)
    """

    def __init__(self) -> None:
        # Mapping: (server_name, tool_name) -> callable(arguments) -> Any
        self._handlers: Dict[tuple[str, str], Any] = {}

    def register_handler(
        self,
        server_name: str,
        tool_name: str,
        handler: Any,
    ) -> None:
        self._handlers[(server_name, tool_name)] = handler

    def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        auth: Optional[MCPAuthConfig] = None,
    ) -> Any:
        key = (server_name, tool_name)
        if key not in self._handlers:
            raise ValueError(f"No MCP handler registered for {server_name}.{tool_name}")

        # Demo: print auth info if provided
        if auth is not None:
            print(f"[LocalMCPClient] Using auth type={auth.auth_type}")

        handler = self._handlers[key]
        return handler(arguments)

    def stream_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        auth: Optional[MCPAuthConfig] = None,
    ) -> Iterable[Any]:
        # Simple demo: yield the full result as a single chunk
        yield self.call_tool(server_name, tool_name, arguments, auth=auth)


def build_agent_with_mcp_calculator() -> Agent:
    """Create an Agent that calls a calculator tool via MCP."""
    # 1. Underlying local calculator implementation
    calculator = CalculatorTool()

    # 2. MCP client that delegates to the local calculator
    client = LocalMCPClient()

    def calculator_handler(args: Dict[str, Any]) -> Any:
        # Directly reuse the CalculatorTool implementation
        return calculator.execute(**args)

    client.register_handler(
        server_name="local-mcp",
        tool_name="calculator",
        handler=calculator_handler,
    )

    # 3. MCPToolConfig + MCPTool wrapper
    mcp_calculator_tool = MCPTool(
        config=MCPToolConfig(
            server_name="local-mcp",
            tool_name="calculator",
            description="Calculator tool exposed via MCP",
            # Example auth config (optional)
            auth=MCPAuthConfig(auth_type="bearer", token="demo-token"),
        ),
        client=client,
    )

    # 4. Model provider that knows how to trigger \"calculator\" tool calls
    model_provider = MockModelProvider(name="mcp-agent-demo")

    # 5. Agent that only sees the MCP-wrapped calculator
    agent = Agent(
        name="mcp_calculator_agent",
        model_provider=model_provider,
        tools=[mcp_calculator_tool],
        system_prompt="You are a helpful assistant who uses an MCP-exposed calculator tool.",
    )
    return agent


def main() -> None:
    """Run the MCP-based Agent example."""
    agent = build_agent_with_mcp_calculator()
    runner = Runner(enable_logging=True)

    # Ask a question that will trigger a calculator tool call
    message = Message(
        role=MessageRole.USER,
        content="What is 21 plus 21?",
    )

    print("=== MCP Agent Example ===")
    print("User:", message.content)

    response = runner.run(agent, message)
    print("Agent:", response.content)


if __name__ == "__main__":
    main()

