"""Mock model provider for testing and demonstration."""

import json
from typing import Any, Dict, List, Optional

from agentic.core.message import Message, MessageRole
from agentic.core.model import ModelProvider


class MockModelProvider(ModelProvider):
    """
    Mock model provider for testing and demonstration.
    
    This provider simulates an LLM by:
    - Parsing tool calls from simple patterns
    - Returning mock responses
    - Demonstrating the model provider interface
    
    In production, replace this with actual model providers
    (OpenAI, Anthropic, local models, etc.)
    """

    def __init__(self, name: str = "mock-model"):
        """
        Initialize the mock model provider.
        
        Args:
            name: Model name identifier
        """
        self.name = name

    def generate(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        **kwargs: Any
    ) -> Message:
        """
        Generate a mock response.
        
        This is a simplified mock that:
        - Detects simple tool call patterns
        - Returns appropriate tool calls or responses
        """
        last_message = messages[-1] if messages else None
        
        if not last_message:
            return Message(role=MessageRole.ASSISTANT, content="Hello! How can I help you?")
        
        content = last_message.content.lower()
        
        # Simple pattern matching for tool calls
        if tools:
            tool_names = [tool.get("name", "") for tool in tools]
            
            # Check for calculator operations
            if "calculator" in tool_names:
                if any(op in content for op in ["add", "plus", "+"]):
                    numbers = self._extract_numbers(content)
                    if len(numbers) >= 2:
                        return Message(
                            role=MessageRole.ASSISTANT,
                            content="",
                            tool_calls=[
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "calculator",
                                        "arguments": json.dumps({
                                            "operation": "add",
                                            "a": numbers[0],
                                            "b": numbers[1],
                                        }),
                                    },
                                }
                            ],
                        )
            
            # Check for weather queries
            if "weather" in tool_names and "weather" in content:
                # Extract location (simplified)
                location = self._extract_location(content) or "San Francisco, CA"
                return Message(
                    role=MessageRole.ASSISTANT,
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": json.dumps({"location": location}),
                            },
                        }
                    ],
                )
        
        # Default response
        return Message(
            role=MessageRole.ASSISTANT,
            content=f"I understand you said: '{last_message.content}'. This is a mock response.",
        )

    def stream(
        self,
        messages: List[Message],
        tools: Optional[List[Any]] = None,
        **kwargs: Any
    ):
        """
        Stream mock responses.
        
        For simplicity, this just yields the full response.
        In a real implementation, this would yield chunks.
        """
        response = self.generate(messages, tools, **kwargs)
        yield response

    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numbers from text (simplified)."""
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        return [float(n) for n in numbers]

    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location from text (simplified)."""
        # Very simple extraction - in real scenario, use NLP
        if "san francisco" in text:
            return "San Francisco, CA"
        elif "new york" in text:
            return "New York, NY"
        elif "beijing" in text:
            return "Beijing, China"
        return None
