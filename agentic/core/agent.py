"""Agent base class for modular, decoupled agent implementation."""

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from agentic.core.message import Message, MessageRole
from agentic.core.model import ModelProvider
from agentic.core.tool import Tool, ToolExecutionError


class AgentConfig(BaseModel):
    """Configuration for an agent."""

    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    max_iterations: int = Field(10, description="Maximum number of tool-calling iterations")
    temperature: float = Field(0.7, description="Model temperature")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Agent:
    """
    Base class for agents.
    
    Agents are modular, decoupled components that can:
    - Use tools to perform actions
    - Communicate with other agents
    - Maintain conversation state
    - Work with any model provider
    
    Design principles:
    - Code-first: Defined through Python classes, not configuration
    - Tool decoupling: Tools are independent, composable modules
    - Model agnostic: Works with any ModelProvider implementation
    - Deployment agnostic: No assumptions about deployment environment
    """

    def __init__(
        self,
        name: str,
        model_provider: ModelProvider,
        tools: Optional[List[Tool]] = None,
        description: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 10,
        temperature: float = 0.7,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize an agent.
        
        Args:
            name: Unique agent name
            model_provider: Model provider instance (model-agnostic)
            tools: List of tools available to this agent
            description: Agent description
            system_prompt: System prompt for the agent
            max_iterations: Maximum tool-calling iterations per request
            temperature: Model temperature
            metadata: Additional metadata
        """
        self.config = AgentConfig(
            name=name,
            description=description,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            temperature=temperature,
            metadata=metadata or {},
        )
        self.model_provider = model_provider
        self.tools: Dict[str, Tool] = {}
        self.conversation_history: List[Message] = []
        
        # Register tools
        if tools:
            for tool in tools:
                self.add_tool(tool)
        
        # Add system prompt if provided
        if self.config.system_prompt:
            self.conversation_history.append(
                Message(role=MessageRole.SYSTEM, content=self.config.system_prompt)
            )

    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent.
        
        Args:
            tool: Tool instance to add
        """
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self.tools[tool.name] = tool

    def remove_tool(self, tool_name: str) -> None:
        """
        Remove a tool from the agent.
        
        Args:
            tool_name: Name of the tool to remove
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' is not registered")
        del self.tools[tool_name]

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self.tools.get(tool_name)

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            Tool execution result
            
        Raises:
            ToolExecutionError: If tool execution fails
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ToolExecutionError(
                tool_name=tool_name,
                message=f"Tool '{tool_name}' not found",
            )
        
        if not tool.validate(**kwargs):
            raise ToolExecutionError(
                tool_name=tool_name,
                message="Tool parameter validation failed",
            )
        
        try:
            return tool.execute(**kwargs)
        except Exception as e:
            raise ToolExecutionError(
                tool_name=tool_name,
                message=str(e),
                cause=e,
            )

    def process_message(self, message: Message) -> Message:
        """
        Process a message and generate a response.
        
        This method handles the agent's reasoning loop:
        1. Add user message to conversation history
        2. Generate response from model (with tool calls if needed)
        3. Execute tool calls
        4. Continue until final response or max iterations
        
        Args:
            message: Input message from user or another agent
            
        Returns:
            Agent's response message
        """
        # Add user message to history
        self.conversation_history.append(message)
        
        # Prepare tools for model
        tools_list = list(self.tools.values()) if self.tools else None
        formatted_tools = (
            self.model_provider.format_tools_for_model(tools_list) if tools_list else None
        )
        
        iterations = 0
        while iterations < self.config.max_iterations:
            iterations += 1
            
            # Generate response from model
            response = self.model_provider.generate(
                messages=self.conversation_history,
                tools=formatted_tools,
                temperature=self.config.temperature,
            )
            
            # Add response to history
            self.conversation_history.append(response)
            
            # Check if model wants to call tools
            if response.tool_calls:
                # Execute all tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("function", {}).get("name")
                    tool_args = tool_call.get("function", {}).get("arguments", {})
                    tool_call_id = tool_call.get("id")
                    
                    if not tool_name:
                        continue
                    
                    try:
                        # Parse arguments if they're a string (JSON)
                        if isinstance(tool_args, str):
                            tool_args = json.loads(tool_args)
                        
                        # Execute tool
                        result = self.execute_tool(tool_name, **tool_args)
                        
                        # Create tool response message
                        tool_message = Message(
                            role=MessageRole.TOOL,
                            content=str(result),
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                        tool_results.append(tool_message)
                        self.conversation_history.append(tool_message)
                    except Exception as e:
                        # Create error message
                        error_message = Message(
                            role=MessageRole.TOOL,
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call_id,
                            name=tool_name,
                        )
                        tool_results.append(error_message)
                        self.conversation_history.append(error_message)
                
                # Continue loop to let model process tool results
                continue
            else:
                # No tool calls, return final response
                return response
        
        # Max iterations reached
        return Message(
            role=MessageRole.ASSISTANT,
            content="Maximum iterations reached. Please try a simpler request.",
        )

    def reset(self) -> None:
        """Reset agent's conversation history."""
        self.conversation_history = []
        if self.config.system_prompt:
            self.conversation_history.append(
                Message(role=MessageRole.SYSTEM, content=self.config.system_prompt)
            )

    def get_conversation_history(self) -> List[Message]:
        """Get the agent's conversation history."""
        return self.conversation_history.copy()

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"<Agent: {self.config.name}>"
