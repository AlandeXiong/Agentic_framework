"""Calculator tool example."""

from typing import Any, Dict
from agentic.core.tool import Tool


class CalculatorTool(Tool):
    """
    A simple calculator tool that performs basic arithmetic operations.
    
    This demonstrates how to create a tool with parameter validation
    and schema definition.
    """

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "calculator"

    @property
    def description(self) -> str:
        """Return the tool description."""
        return "Performs basic arithmetic operations: add, subtract, multiply, divide"

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform",
                },
                "a": {
                    "type": "number",
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "description": "Second number",
                },
            },
            "required": ["operation", "a", "b"],
        }

    def validate(self, **kwargs: Any) -> bool:
        """Validate tool parameters."""
        operation = kwargs.get("operation")
        b = kwargs.get("b")
        
        if operation == "divide" and b == 0:
            return False
        
        return True

    def execute(self, **kwargs: Any) -> Any:
        """Execute the calculator tool."""
        operation = kwargs["operation"]
        a = float(kwargs["a"])
        b = float(kwargs["b"])
        
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("Division by zero is not allowed")
            return a / b
        else:
            raise ValueError(f"Unknown operation: {operation}")
