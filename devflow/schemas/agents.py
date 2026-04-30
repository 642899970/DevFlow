"""Pydantic models for agents."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(str):
    """State of an agent."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolCall(BaseModel):
    """A tool call made by an agent."""

    name: str = Field(..., description="Name of the tool")
    arguments: Dict[str, Any] = Field(..., description="Arguments for the tool")
    result: Optional[Any] = Field(default=None, description="Result of the tool call")
    error: Optional[str] = Field(default=None, description="Error if tool call failed")


class AgentThought(BaseModel):
    """A single thought step in an agent's reasoning."""

    step: int = Field(..., description="Step number")
    thought: str = Field(..., description="Agent's thought/reasoning")
    action: Optional[str] = Field(default=None, description="Action to take")
    tool_call: Optional[ToolCall] = Field(default=None, description="Tool call if any")
    observation: Optional[str] = Field(default=None, description="Observation from action")


class AgentMessage(BaseModel):
    """A message in the agent conversation."""

    role: str = Field(..., description="Message role (system, user, assistant, tool)")
    content: str = Field(..., description="Message content")
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="Tool calls in this message")
    tool_call_id: Optional[str] = Field(default=None, description="Tool call ID for tool responses")


class AgentExecution(BaseModel):
    """Execution record of an agent."""

    agent_name: str = Field(..., description="Name of the agent")
    task_id: str = Field(..., description="ID of the task being executed")
    model_used: str = Field(..., description="Model used by the agent")
    thoughts: List[AgentThought] = Field(default_factory=list, description="Thought process")
    messages: List[AgentMessage] = Field(default_factory=list, description="Conversation messages")
    state: AgentState = Field(default=AgentState.IDLE, description="Current state")
    iterations: int = Field(default=0, description="Number of iterations")
    max_iterations: int = Field(default=10, description="Maximum iterations allowed")
    tokens_used: int = Field(default=0, description="Total tokens used")
    duration_seconds: float = Field(default=0.0, description="Execution duration in seconds")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Final result")
    error: Optional[str] = Field(default=None, description="Error if failed")
    started_at: Optional[str] = Field(default=None, description="Start timestamp")
    completed_at: Optional[str] = Field(default=None, description="Completion timestamp")


class AgentResponse(BaseModel):
    """Response from an agent."""

    success: bool = Field(..., description="Whether the agent succeeded")
    message: str = Field(..., description="Response message")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Result data")
    tokens_used: int = Field(default=0, description="Tokens used")
    execution: Optional[AgentExecution] = Field(default=None, description="Full execution record")
