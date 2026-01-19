"""Weather tool example."""

from typing import Any, Dict
from agentic.core.tool import Tool


class WeatherTool(Tool):
    """
    A weather tool that retrieves weather information.
    
    This is a mock implementation for demonstration purposes.
    In a real scenario, this would call an actual weather API.
    """

    @property
    def name(self) -> str:
        """Return the tool name."""
        return "weather"

    @property
    def description(self) -> str:
        """Return the tool description."""
        return "Gets the current weather for a given location"

    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters."""
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                    "default": "celsius",
                },
            },
            "required": ["location"],
        }

    def execute(self, **kwargs: Any) -> Any:
        """Execute the weather tool."""
        location = kwargs["location"]
        units = kwargs.get("units", "celsius")
        
        # Mock weather data
        # In a real implementation, this would call a weather API
        mock_data = {
            "location": location,
            "temperature": 22 if units == "celsius" else 72,
            "condition": "Sunny",
            "humidity": 65,
            "units": units,
        }
        
        return f"Weather in {location}: {mock_data['temperature']}°{units[0].upper()}, {mock_data['condition']}, Humidity: {mock_data['humidity']}%"
