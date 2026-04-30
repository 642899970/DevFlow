"""OpenAI adapter for LLM interface."""

import os
from typing import AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from devflow.core.models.base import BaseLLM, ChatResponse, Delta, Message, ToolDefinition


class OpenAIModel(BaseLLM):
    """OpenAI model adapter."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ):
        """Initialize the OpenAI model.

        Args:
            model_name: Name of the OpenAI model (e.g., gpt-4o, gpt-4-turbo-preview)
            api_key: OpenAI API key
            base_url: Custom base URL (for OpenAI-compatible APIs)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        super().__init__(model_name, api_key, base_url, temperature, max_tokens, **kwargs)

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request to OpenAI.

        Args:
            messages: List of messages in the conversation
            tools: Optional list of available tools
            **kwargs: Additional request parameters

        Returns:
            ChatResponse with the model's response
        """
        openai_messages = [
            {"role": msg.role, "content": msg.content, "tool_call_id": msg.tool_call_id, "tool_calls": msg.tool_calls}
            for msg in messages
        ]

        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        if openai_tools:
            params["tools"] = openai_tools

        response = await self.client.chat.completions.create(**params)

        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return ChatResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            tokens_used={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
        )

    async def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[Delta]:
        """Stream a chat completion request from OpenAI.

        Args:
            messages: List of messages in the conversation
            tools: Optional list of available tools
            **kwargs: Additional request parameters

        Yields:
            Delta objects with streaming response chunks
        """
        openai_messages = [
            {"role": msg.role, "content": msg.content, "tool_call_id": msg.tool_call_id, "tool_calls": msg.tool_calls}
            for msg in messages
        ]

        openai_tools = None
        if tools:
            openai_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
        }

        if openai_tools:
            params["tools"] = openai_tools

        stream = await self.client.chat.completions.create(**params)

        async for chunk in stream:
            delta = chunk.choices[0].delta

            tool_calls = None
            if delta.tool_calls:
                tool_calls = [
                    {
                        "index": tc.index,
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in delta.tool_calls
                ]

            yield Delta(
                content=delta.content,
                tool_calls=tool_calls,
                finish_reason=chunk.choices[0].finish_reason,
            )
