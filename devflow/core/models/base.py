"""Base LLM interface for all model adapters."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    """A message in the conversation."""

    role: str = Field(..., description="Message role: system, user, assistant, tool")
    content: str = Field(..., description="Message content")
    tool_call_id: Optional[str] = Field(default=None, description="Tool call ID for tool responses")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool calls in this message")


class ToolDefinition(BaseModel):
    """Definition of a tool/function."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters schema")


class ChatResponse(BaseModel):
    """Response from a chat completion."""

    content: str = Field(..., description="Response content")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Tool calls made by the model")
    finish_reason: Optional[str] = Field(default=None, description="Reason for completion")
    tokens_used: Optional[Dict[str, int]] = Field(default=None, description="Token usage")


class Delta(BaseModel):
    """A delta in a streaming response."""

    content: Optional[str] = Field(default=None, description="Delta content")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(default=None, description="Delta tool calls")
    finish_reason: Optional[str] = Field(default=None, description="Finish reason")


class BaseLLM(ABC):
    """Abstract base class for all LLM adapters."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ):
        """Initialize the LLM adapter.

        Args:
            model_name: Name of the model
            api_key: API key for authentication
            base_url: Custom base URL for the API
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional model-specific parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = kwargs

    @abstractmethod
    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: List of messages in the conversation
            tools: Optional list of available tools
            **kwargs: Additional request parameters

        Returns:
            ChatResponse with the model's response
        """
        pass

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[Delta]:
        """Stream a chat completion request.

        Args:
            messages: List of messages in the conversation
            tools: Optional list of available tools
            **kwargs: Additional request parameters

        Yields:
            Delta objects with streaming response chunks
        """
        pass

    async def health_check(self) -> bool:
        """Check if the model API is accessible.

        Returns:
            True if the API is accessible, False otherwise
        """
        try:
            await self.chat([Message(role="user", content="Hello")])
            return True
        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }
