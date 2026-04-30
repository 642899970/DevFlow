"""Custom exception hierarchy for DevFlow."""

from typing import Optional


class DevFlowError(Exception):
    """Base exception for all DevFlow errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ConfigurationError(DevFlowError):
    """Raised when configuration is invalid or missing."""

    pass


class ModelError(DevFlowError):
    """Base exception for model-related errors."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when a requested model is not found."""

    pass


class ModelInitializationError(ModelError):
    """Raised when a model fails to initialize."""

    pass


class ModelAPIError(ModelError):
    """Raised when a model API call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[dict] = None):
        self.status_code = status_code
        super().__init__(message, details)


class AgentError(DevFlowError):
    """Base exception for agent-related errors."""

    pass


class AgentNotFoundError(AgentError):
    """Raised when a requested agent is not found."""

    pass


class AgentExecutionError(AgentError):
    """Raised when an agent fails to execute a task."""

    pass


class AgentMaxIterationsError(AgentError):
    """Raised when an agent exceeds maximum iterations."""

    pass


class ToolError(DevFlowError):
    """Base exception for tool-related errors."""

    pass


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not found."""

    pass


class ToolExecutionError(ToolError):
    """Raised when a tool fails to execute."""

    pass


class PlanningError(DevFlowError):
    """Base exception for planning-related errors."""

    pass


class TaskPlanningError(PlanningError):
    """Raised when task planning fails."""

    pass


class DAGError(PlanningError):
    """Raised when DAG operations fail."""

    pass


class OrchestrationError(DevFlowError):
    """Base exception for orchestration-related errors."""

    pass


class TaskExecutionError(OrchestrationError):
    """Raised when a task fails to execute."""

    pass


class DependencyError(OrchestrationError):
    """Raised when task dependencies are invalid."""

    pass


class WorkspaceError(DevFlowError):
    """Base exception for workspace-related errors."""

    pass


class FileNotFoundError(WorkspaceError):
    """Raised when a file is not found in the workspace."""

    pass


class FileOperationError(WorkspaceError):
    """Raised when a file operation fails."""

    pass


class CLIError(DevFlowError):
    """Base exception for CLI-related errors."""

    pass


class CommandError(CLIError):
    """Raised when a CLI command fails."""

    pass
