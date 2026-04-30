"""Pydantic models for tasks and DAG."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskStatus(str):
    """Status of a task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class Task(BaseModel):
    """A single task in the workflow."""

    id: str = Field(..., description="Unique task identifier")
    type: str = Field(..., description="Task type (design, backend, frontend, test)")
    agent: str = Field(..., description="Agent responsible for this task")
    depends: List[str] = Field(default_factory=list, description="List of task IDs this task depends on")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current status of the task")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task execution result")
    error: Optional[str] = Field(default=None, description="Error message if task failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")


class TaskDAG(BaseModel):
    """Directed Acyclic Graph representing task dependencies."""

    project_name: str = Field(..., description="Name of the project")
    tasks: List[Task] = Field(default_factory=list, description="List of all tasks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional DAG metadata")

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        completed_ids = {task.id for task in self.tasks if task.status == TaskStatus.COMPLETED}
        ready = []

        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                if all(dep_id in completed_ids for dep_id in task.depends):
                    ready.append(task)

        return ready

    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks."""
        return [task for task in self.tasks if task.status == TaskStatus.FAILED]

    def is_complete(self) -> bool:
        """Check if all tasks are completed or failed."""
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for task in self.tasks
        )

    def has_failed(self) -> bool:
        """Check if any task has failed."""
        return any(task.status == TaskStatus.FAILED for task in self.tasks)


class TaskExecutionResult(BaseModel):
    """Result of a task execution."""

    task_id: str = Field(..., description="ID of the executed task")
    success: bool = Field(..., description="Whether the task succeeded")
    output: Optional[Dict[str, Any]] = Field(default=None, description="Task output")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    tokens_used: int = Field(default=0, description="Number of tokens used")
    duration_seconds: float = Field(default=0.0, description="Execution duration in seconds")
