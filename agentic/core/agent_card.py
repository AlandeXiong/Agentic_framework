"""Agent card abstraction for describing and instantiating agents.

Agent cards capture the high-level identity and capabilities of an agent
and can be used to construct concrete `Agent` instances.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from agentic.core.agent import Agent
from agentic.core.model import ModelProvider
from agentic.core.tool import Tool


class AgentCard(BaseModel):
    """Declarative description of an agent."""

    name: str = Field(..., description="Agent name")
    description: Optional[str] = Field(None, description="High-level agent description")
    system_prompt: Optional[str] = Field(
        None, description="System prompt that defines the agent's behavior"
    )
    tool_names: List[str] = Field(
        default_factory=list,
        description="Names of tools this agent can use (keys in a tool registry)",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    def create_agent(
        self,
        model_provider: ModelProvider,
        tool_registry: Dict[str, Tool],
        max_iterations: int = 10,
        temperature: float = 0.7,
    ) -> Agent:
        """Instantiate an `Agent` from this card.

        Args:
            model_provider: Model provider to use
            tool_registry: Mapping from tool name to Tool instance
            max_iterations: Maximum tool-calling iterations
            temperature: Model temperature
        """
        tools: List[Tool] = []
        for name in self.tool_names:
            tool = tool_registry.get(name)
            if tool is not None:
                tools.append(tool)

        return Agent(
            name=self.name,
            model_provider=model_provider,
            tools=tools,
            description=self.description,
            system_prompt=self.system_prompt,
            max_iterations=max_iterations,
            temperature=temperature,
            metadata=self.metadata,
        )

