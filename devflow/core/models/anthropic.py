"""Anthropic adapter for LLM interface."""

import os
from typing import AsyncIterator, Dict, List, Optional

from anthropic import AsyncAnthropic

from devflow.core.models.base import BaseLLM, ChatResponse, Delta, Message, ToolDefinition


class AnthropicModel(BaseLLM):
    """Anthropic Claude model adapter."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ):
        """Initialize the Anthropic model.

        Args:
            model_name: Name of the Anthropic model (e.g., claude-sonnet-4-20250514)
            api_key: Anthropic API key
            base_url: Custom base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        """
        super().__init__(model_name, api_key, base_url, temperature, max_tokens, **kwargs)

        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"),
        )

    async def chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request to Anthropic.

        Args:
            messages: List of messages in the conversation
            tools: Optional list of available tools
            **kwargs: Additional request parameters

        Returns:
            ChatResponse with the model's response
        """
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            elif msg.role == "user":
                anthropic_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                content = [{"type": "text", "text": msg.content}]
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": tc["function"]["name"],
                            "input": tc["function"]["arguments"],
                        })
                anthropic_messages.append({"role": "assistant", "content": content})
            elif msg.role == "tool":
                # Find the tool use this is responding to
                tool_use_id = msg.tool_call_id
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": msg.content,
                        }
                    ],
                })

        # Convert tools to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ]

        params = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
        }

        if system_message:
            params["system"] = system_message

        if anthropic_tools:
            params["tools"] = anthropic_tools

        response = await self.client.messages.create(**params)

        content = response.content[0]
        text_content = ""
        tool_calls = None

        if content.type == "text":
            text_content = content.text
        elif content.type == "tool_use":
            text_content = ""
            tool_calls = [
                {
                    "id": content.id,
                    "type": "tool_use",
                    "function": {
                        "name": content.name,
                        "arguments": content.input,
                    },
                }
            ]

        return ChatResponse(
            content=text_content,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            tokens_used={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
        )

    async def stream_chat(
        self,
        messages: List[Message],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[Delta]:
        """Stream a chat completion request from Anthropic.

        Args:
            messages: List of messages in the conversation
            tools: Optional list of available tools
            **kwargs: Additional request parameters

        Yields:
            Delta objects with streaming response chunks
        """
        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            elif msg.role == "user":
                anthropic_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                content = [{"type": "text", "text": msg.content}]
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        content.append({
                            "type": "tool_use",
                            "id": tc.get("id"),
                            "name": tc["function"]["name"],
                            "input": tc["function"]["arguments"],
                        })
                anthropic_messages.append({"role": "assistant", "content": content})
            elif msg.role == "tool":
                tool_use_id = msg.tool_call_id
                anthropic_messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": msg.content,
                        }
                    ],
                })

        # Convert tools to Anthropic format
        anthropic_tools = None
        if tools:
            anthropic_tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                for tool in tools
            ]

        params = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": True,
        }

        if system_message:
            params["system"] = system_message

        if anthropic_tools:
            params["tools"] = anthropic_tools

        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield Delta(content=text)

            # Get final response for finish reason
            response = await stream.get_final_message()
            yield Delta(finish_reason=response.stop_reason)
