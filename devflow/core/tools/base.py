"""Base tool interface for agent tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Result of a tool execution."""

    success: bool = Field(..., description="Whether the tool execution succeeded")
    output: Optional[Any] = Field(default=None, description="Tool output")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ToolDefinition(BaseModel):
    """Definition of a tool for function calling."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(..., description="JSON Schema for tool parameters")


class BaseTool(ABC):
    """Abstract base class for all tools."""

    def __init__(self):
        """Initialize the tool."""
        self._definition: Optional[ToolDefinition] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the tool name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Get the tool description."""
        pass

    @abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Get the JSON Schema for tool parameters.

        Returns:
            Dictionary with JSON Schema for parameters
        """
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    def get_definition(self) -> ToolDefinition:
        """Get the tool definition for function calling.

        Returns:
            ToolDefinition instance
        """
        if self._definition is None:
            self._definition = ToolDefinition(
                name=self.name,
                description=self.description,
                parameters=self.get_parameters_schema(),
            )
        return self._definition

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate tool parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            True if parameters are valid
        """
        schema = self.get_parameters_schema()
        required = schema.get("required", [])

        # Check required parameters
        for param in required:
            if param not in parameters:
                return False

        # Check parameter types
        properties = schema.get("properties", {})
        for param_name, param_value in parameters.items():
            if param_name not in properties:
                continue

            param_schema = properties[param_name]
            param_type = param_schema.get("type")

            if param_type == "string" and not isinstance(param_value, str):
                return False
            elif param_type == "number" and not isinstance(param_value, (int, float)):
                return False
            elif param_type == "integer" and not isinstance(param_value, int):
                return False
            elif param_type == "boolean" and not isinstance(param_value, bool):
                return False
            elif param_type == "array" and not isinstance(param_value, list):
                return False
            elif param_type == "object" and not isinstance(param_value, dict):
                return False

        return True
