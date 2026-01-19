"""Message types for agent communication."""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Message role types."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """A message in the agent conversation."""

    role: MessageRole = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The message content")
    name: Optional[str] = Field(None, description="Optional name identifier")
    tool_call_id: Optional[str] = Field(None, description="ID for tool call responses")
    tool_calls: Optional[list[Dict[str, Any]]] = Field(
        None, description="Tool calls made in this message"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        """Pydantic config."""

        use_enum_values = True
        frozen = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        result = {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        return cls(**data)
