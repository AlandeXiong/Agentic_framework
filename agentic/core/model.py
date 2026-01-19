"""Model provider abstraction for model-agnostic agent framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agentic.core.message import Message, MessageRole


class ModelProvider(ABC):
    """
    Abstract interface for model providers.
    
    This abstraction allows the framework to work with any LLM provider
    (OpenAI, Anthropic, local models, etc.) without coupling to specific implementations.
    """

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        **kwargs: Any
    ) -> Message:
        """
        Generate a response from the model.
        
        Args:
            messages: Conversation history
            tools: Available tools for the model to use (optional)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated message from the model
        """
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        **kwargs: Any
    ):
        """
        Stream responses from the model.
        
        Args:
            messages: Conversation history
            tools: Available tools for the model to use (optional)
            **kwargs: Additional model-specific parameters
            
        Yields:
            Message chunks as they are generated
        """
        pass

    def format_tools_for_model(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """
        Format tools into the format expected by the model.
        
        Override this method if your model provider requires a specific tool format.
        Default implementation assumes tools have a schema property.
        
        Args:
            tools: List of Tool instances
            
        Returns:
            List of tool definitions in model-specific format
        """
        return [tool.schema.model_dump() for tool in tools]
