"""Tool interface abstraction."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ToolSchema(BaseModel):
    """Schema definition for a tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="JSON schema for tool parameters"
    )


class Tool(ABC):
    """
    Abstract base class for all tools.

    Tools are decoupled, composable modules that agents can use to perform actions.
    Each tool must implement:
    - name: Unique identifier
    - description: What the tool does
    - schema: Parameter schema for validation
    - execute: The actual tool execution logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Return the tool description."""
        pass

    @property
    def schema(self) -> ToolSchema:
        """Return the tool schema for validation and LLM understanding."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self._get_parameters_schema(),
        )

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get JSON schema for tool parameters.

        Override this method to provide custom parameter schemas.
        Default implementation returns empty schema.
        """
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            Tool execution result

        Raises:
            ToolExecutionError: If tool execution fails
        """
        pass

    def validate(self, **kwargs: Any) -> bool:
        """
        Validate tool parameters before execution.

        Override this method to add custom validation logic.
        Default implementation always returns True.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if parameters are valid
        """
        return True

    def __repr__(self) -> str:
        """String representation of the tool."""
        return f"<Tool: {self.name}>"


class ToolExecutionError(Exception):
    """Exception raised when tool execution fails."""

    def __init__(self, tool_name: str, message: str, cause: Optional[Exception] = None):
        """Initialize tool execution error."""
        self.tool_name = tool_name
        self.message = message
        self.cause = cause
        super().__init__(f"Tool '{tool_name}' execution failed: {message}")
