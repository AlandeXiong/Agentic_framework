"""
Agentic Framework - A model-agnostic, deployment-agnostic agentic framework.

Inspired by Google ADK's core design principles:
- Code-first approach
- Modular multi-agent architecture
- Tool decoupling
- Deployment agnostic
"""

from agentic.core.agent import Agent
from agentic.core.tool import Tool
from agentic.core.runner import Runner
from agentic.core.message import Message, MessageRole
from agentic.core.model import ModelProvider
from agentic.core.workflow_step import WorkflowStep, StepType
from agentic.core.workflow import Workflow, WorkflowContext
from agentic.core.workflow_runner import WorkflowRunner
from agentic.core.mcp import MCPAuthConfig, MCPClient, MCPTool, MCPToolConfig

__version__ = "0.1.0"
__all__ = [
    "Agent",
    "Tool",
    "Runner",
    "Message",
    "MessageRole",
    "ModelProvider",
    # Workflow-related
    "WorkflowStep",
    "StepType",
    "Workflow",
    "WorkflowContext",
    "WorkflowRunner",
    # MCP-related
    "MCPAuthConfig",
    "MCPClient",
    "MCPToolConfig",
    "MCPTool",
]
