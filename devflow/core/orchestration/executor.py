"""Task executor for running individual tasks."""

import asyncio
from typing import Dict, Optional

from devflow.core.agents.base import BaseAgent
from devflow.core.logging.logger import StructuredLogger
from devflow.core.logging.tracker import TokenTracker
from devflow.core.workspace.manager import WorkspaceManager
from devflow.exceptions.errors import TaskExecutionError
from devflow.schemas.agents import AgentResponse
from devflow.schemas.tasks import Task, TaskExecutionResult, TaskStatus


class TaskExecutor:
    """Executor for running individual tasks."""

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        workspace: Optional[WorkspaceManager] = None,
        logger: Optional[StructuredLogger] = None,
        token_tracker: Optional[TokenTracker] = None,
    ):
        """Initialize the task executor.

        Args:
            agents: Dictionary mapping agent names to agent instances
            workspace: Workspace manager instance
            logger: Structured logger instance
            token_tracker: Token tracker instance
        """
        self.agents = agents
        self.workspace = workspace or WorkspaceManager.get_instance()
        self.logger = logger or StructuredLogger()
        self.token_tracker = token_tracker or TokenTracker()

    async def execute_task(self, task: Task, context: Optional[Dict] = None) -> TaskExecutionResult:
        """Execute a single task.

        Args:
            task: Task to execute
            context: Additional context for the task

        Returns:
            TaskExecutionResult

        Raises:
            TaskExecutionError: If task execution fails
        """
        import time

        start_time = time.time()

        try:
            # Get the agent for this task
            agent = self.agents.get(task.agent)
            if not agent:
                raise TaskExecutionError(f"Agent not found: {task.agent}")

            # Build task description
            task_description = self._build_task_description(task, context)

            # Execute the task
            response = await agent.execute(
                task=task_description,
                context={"task_id": task.id, "task_type": task.type, **(context or {})},
            )

            # Calculate duration
            duration = time.time() - start_time

            if response.success:
                return TaskExecutionResult(
                    task_id=task.id,
                    success=True,
                    output=response.result,
                    tokens_used=response.tokens_used,
                    duration_seconds=duration,
                )
            else:
                return TaskExecutionResult(
                    task_id=task.id,
                    success=False,
                    error=response.error,
                    tokens_used=response.tokens_used,
                    duration_seconds=duration,
                )

        except Exception as e:
            duration = time.time() - start_time
            return TaskExecutionResult(
                task_id=task.id,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    def _build_task_description(self, task: Task, context: Optional[Dict]) -> str:
        """Build a task description for the agent.

        Args:
            task: Task to describe
            context: Additional context

        Returns:
            Task description
        """
        # Get task type description
        type_descriptions = {
            "design": "Design the system architecture and create necessary documentation",
            "backend": "Implement the backend functionality",
            "frontend": "Implement the frontend user interface",
            "test": "Create and run tests for the system",
        }

        description = type_descriptions.get(task.type, f"Complete task of type {task.type}")

        # Add context if available
        if context:
            context_str = "\n".join(f"  - {k}: {v}" for k, v in context.items())
            description += f"\n\nContext:\n{context_str}"

        return description

    async def execute_tasks_sequential(self, tasks: list[Task]) -> Dict[str, TaskExecutionResult]:
        """Execute multiple tasks sequentially.

        Args:
            tasks: List of tasks to execute

        Returns:
            Dictionary mapping task IDs to execution results
        """
        results = {}

        for task in tasks:
            result = await self.execute_task(task)
            results[task.id] = result

            # Stop if task failed
            if not result.success:
                self.logger.log_error(f"Task {task.id} failed, stopping sequential execution")
                break

        return results

    async def execute_tasks_parallel(self, tasks: list[Task]) -> Dict[str, TaskExecutionResult]:
        """Execute multiple tasks in parallel.

        Args:
            tasks: List of tasks to execute

        Returns:
            Dictionary mapping task IDs to execution results
        """
        # Create coroutines for each task
        coroutines = [self.execute_task(task) for task in tasks]

        # Execute all tasks in parallel
        results_list = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results
        results = {}
        for task, result in zip(tasks, results_list):
            if isinstance(result, Exception):
                results[task.id] = TaskExecutionResult(
                    task_id=task.id,
                    success=False,
                    error=str(result),
                )
            else:
                results[task.id] = result

        return results
