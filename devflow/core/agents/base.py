"""Base agent with Thought-Action-Observation loop."""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from devflow.core.logging.logger import StructuredLogger
from devflow.core.logging.tracker import TokenTracker
from devflow.core.models.base import BaseLLM, Message, ToolDefinition
from devflow.core.tools.base import BaseTool, ToolResult
from devflow.exceptions.errors import AgentExecutionError, AgentMaxIterationsError
from devflow.schemas.agents import (
    AgentExecution,
    AgentMessage,
    AgentResponse,
    AgentState,
    AgentThought,
    ToolCall,
)


class BaseAgent:
    """Base agent with Thought-Action-Observation loop."""

    def __init__(
        self,
        name: str,
        model: BaseLLM,
        system_prompt: str,
        tools: Optional[List[BaseTool]] = None,
        max_iterations: int = 10,
        logger: Optional[StructuredLogger] = None,
        token_tracker: Optional[TokenTracker] = None,
    ):
        """Initialize the agent.

        Args:
            name: Agent name
            model: LLM model to use
            system_prompt: System prompt for the agent
            tools: List of available tools
            max_iterations: Maximum number of iterations
            logger: Structured logger instance
            token_tracker: Token tracker instance
        """
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.logger = logger or StructuredLogger()
        self.token_tracker = token_tracker or TokenTracker()

        self._execution: Optional[AgentExecution] = None
        self._iteration = 0

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Execute a task using the Thought-Action-Observation loop.

        Args:
            task: Task description
            context: Additional context for the task

        Returns:
            AgentResponse with execution result
        """
        # Initialize execution
        self._execution = AgentExecution(
            agent_name=self.name,
            task_id=context.get("task_id", "unknown") if context else "unknown",
            model_used=self.model.model_name,
            state=AgentState.THINKING,
            iterations=0,
            max_iterations=self.max_iterations,
            started_at=datetime.now().isoformat(),
        )

        self.logger.log_agent_start(self.name, self._execution.task_id, self.model.model_name)

        # Build initial messages
        messages = self._build_initial_messages(task, context)

        try:
            # Thought-Action-Observation loop
            while self._iteration < self.max_iterations:
                self._iteration += 1
                self._execution.iterations = self._iteration

                # Thought: Get response from model
                thought = await self._think(messages)
                self._execution.thoughts.append(thought)

                # Check if we're done
                if thought.action == "done":
                    self._execution.state = AgentState.COMPLETED
                    self._execution.result = {"final_answer": thought.observation or "Task completed"}
                    break

                # Action: Execute tool call if any
                if thought.tool_call:
                    self._execution.state = AgentState.ACTING
                    observation = await self._act(thought.tool_call)
                    thought.observation = observation

                    # Add tool response to messages
                    messages.append(self._create_tool_message(thought.tool_call, observation))
                else:
                    # Just continue with the response
                    messages.append(
                        AgentMessage(role="assistant", content=thought.observation or thought.thought)
                    )

                # Check for max iterations
                if self._iteration >= self.max_iterations:
                    self._execution.state = AgentState.FAILED
                    raise AgentMaxIterationsError(
                        f"Agent {self.name} exceeded maximum iterations ({self.max_iterations})"
                    )

            # Mark as completed
            self._execution.completed_at = datetime.now().isoformat()
            self._execution.duration_seconds = (
                datetime.fromisoformat(self._execution.completed_at)
                - datetime.fromisoformat(self._execution.started_at)
            ).total_seconds()

            self.logger.log_agent_end(self.name, self._execution.task_id, success=True)
            self.logger.save_execution(self._execution)

            return AgentResponse(
                success=True,
                message="Task completed successfully",
                result=self._execution.result,
                tokens_used=self._execution.tokens_used,
                execution=self._execution,
            )

        except Exception as e:
            self._execution.state = AgentState.FAILED
            self._execution.error = str(e)
            self._execution.completed_at = datetime.now().isoformat()

            self.logger.log_agent_end(self.name, self._execution.task_id, success=False)
            self.logger.log_error(str(e), {"agent": self.name, "task": task})

            return AgentResponse(
                success=False,
                message=f"Task failed: {str(e)}",
                error=str(e),
                tokens_used=self._execution.tokens_used,
                execution=self._execution,
            )

    async def _think(self, messages: List[AgentMessage]) -> AgentThought:
        """Get a thought from the model.

        Args:
            messages: Conversation messages

        Returns:
            AgentThought with model response
        """
        # Convert messages to model format
        model_messages = [
            Message(
                role=msg.role,
                content=msg.content,
                tool_call_id=msg.tool_call_id,
                tool_calls=msg.tool_calls,
            )
            for msg in messages
        ]

        # Get tool definitions
        tool_definitions = [tool.get_definition() for tool in self.tools]

        # Call the model
        response = await self.model.chat(model_messages, tools=tool_definitions)

        # Track token usage
        if response.tokens_used:
            self._execution.tokens_used += response.tokens_used.get("total_tokens", 0)
            self.token_tracker.record_usage(
                model_id=self.model.model_name,
                agent_name=self.name,
                prompt_tokens=response.tokens_used.get("prompt_tokens", 0),
                completion_tokens=response.tokens_used.get("completion_tokens", 0),
                total_tokens=response.tokens_used.get("total_tokens", 0),
            )

        # Create thought
        thought = AgentThought(
            step=self._iteration,
            thought=response.content,
        )

        # Check for tool calls
        if response.tool_calls:
            thought.action = "tool"
            thought.tool_call = ToolCall(
                name=response.tool_calls[0]["function"]["name"],
                arguments=self._parse_tool_arguments(response.tool_calls[0]["function"]["arguments"]),
            )
        else:
            thought.action = "done"
            thought.observation = response.content

        self.logger.log_thought(thought)

        return thought

    async def _act(self, tool_call: ToolCall) -> str:
        """Execute a tool call.

        Args:
            tool_call: Tool call to execute

        Returns:
            Tool result as string
        """
        # Find the tool
        tool = None
        for t in self.tools:
            if t.name == tool_call.name:
                tool = t
                break

        if tool is None:
            error_msg = f"Tool not found: {tool_call.name}"
            self.logger.log_error(error_msg)
            return error_msg

        # Execute the tool
        try:
            result = await tool.execute(**tool_call.arguments)

            self.logger.log_tool_call(tool_call.name, tool_call.arguments, result.output)

            if result.success:
                return str(result.output) if result.output is not None else "Tool executed successfully"
            else:
                return f"Tool failed: {result.error}"

        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            self.logger.log_error(error_msg)
            return error_msg

    def _build_initial_messages(self, task: str, context: Optional[Dict[str, Any]]) -> List[AgentMessage]:
        """Build initial messages for the conversation.

        Args:
            task: Task description
            context: Additional context

        Returns:
            List of initial messages
        """
        messages = []

        # System message
        messages.append(AgentMessage(role="system", content=self.system_prompt))

        # Task message
        task_message = f"Task: {task}"
        if context:
            task_message += f"\n\nContext: {context}"

        messages.append(AgentMessage(role="user", content=task_message))

        return messages

    def _create_tool_message(self, tool_call: ToolCall, result: str) -> AgentMessage:
        """Create a tool response message.

        Args:
            tool_call: Tool call that was executed
            result: Tool result

        Returns:
            AgentMessage with tool response
        """
        return AgentMessage(
            role="tool",
            content=result,
            tool_call_id=tool_call.name,  # Use tool name as ID for simplicity
        )

    def _parse_tool_arguments(self, arguments: str) -> Dict[str, Any]:
        """Parse tool arguments from JSON string.

        Args:
            arguments: JSON string of arguments

        Returns:
            Dictionary of arguments
        """
        import json

        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {}

    def get_execution(self) -> Optional[AgentExecution]:
        """Get the current execution record.

        Returns:
            AgentExecution or None if no execution is in progress
        """
        return self._execution
