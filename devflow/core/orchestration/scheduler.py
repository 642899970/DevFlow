"""Task scheduler for DAG-based parallel execution."""

import asyncio
from typing import Dict, List, Optional

from devflow.core.agents.base import BaseAgent
from devflow.core.logging.logger import StructuredLogger
from devflow.core.logging.tracker import TokenTracker
from devflow.core.planning.dag import DAG
from devflow.core.workspace.manager import WorkspaceManager
from devflow.exceptions.errors import OrchestrationError, TaskExecutionError
from devflow.schemas.agents import AgentResponse
from devflow.schemas.tasks import Task, TaskDAG, TaskExecutionResult, TaskStatus


class TaskScheduler:
    """Scheduler for executing tasks based on DAG dependencies."""

    def __init__(
        self,
        agents: Dict[str, BaseAgent],
        workspace: Optional[WorkspaceManager] = None,
        logger: Optional[StructuredLogger] = None,
        token_tracker: Optional[TokenTracker] = None,
    ):
        """Initialize the task scheduler.

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

        self._dag: Optional[DAG] = None
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._task_results: Dict[str, TaskExecutionResult] = {}

    async def execute(self, task_dag: TaskDAG) -> Dict[str, TaskExecutionResult]:
        """Execute all tasks in the DAG.

        Args:
            task_dag: Task DAG to execute

        Returns:
            Dictionary mapping task IDs to execution results

        Raises:
            OrchestrationError: If execution fails
        """
        try:
            # Create DAG from task DAG
            self._dag = DAG.from_task_dag(task_dag)

            self.logger.log_info(f"Starting execution of {len(self._dag.tasks)} tasks")

            # Execute tasks until all are complete
            while not self._dag.is_complete():
                # Get ready tasks
                ready_tasks = self._dag.get_ready_tasks()

                if not ready_tasks:
                    if self._dag.has_failed():
                        raise OrchestrationError("Execution failed: some tasks failed")
                    else:
                        # No ready tasks but not complete - possible deadlock
                        raise OrchestrationError("Execution stalled: no ready tasks")

                # Execute ready tasks in parallel
                await self._execute_tasks_parallel(ready_tasks)

                # Check for failures
                if self._dag.has_failed():
                    failed_tasks = self._dag.get_failed_tasks()
                    raise OrchestrationError(
                        f"Execution failed: {len(failed_tasks)} tasks failed"
                    )

            self.logger.log_info("All tasks completed successfully")

            return self._task_results

        except Exception as e:
            self.logger.log_error(f"Execution failed: {str(e)}")
            raise OrchestrationError(f"Task execution failed: {str(e)}")

    async def _execute_tasks_parallel(self, tasks: List[Task]) -> None:
        """Execute multiple tasks in parallel.

        Args:
            tasks: List of tasks to execute
        """
        # Create coroutines for each task
        coroutines = [self._execute_task(task) for task in tasks]

        # Execute all tasks in parallel
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Process results
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                # Task failed
                self._dag.update_task_status(task.id, TaskStatus.FAILED, {"error": str(result)})
                self._task_results[task.id] = TaskExecutionResult(
                    task_id=task.id,
                    success=False,
                    error=str(result),
                )
                self.logger.log_error(f"Task {task.id} failed: {str(result)}")
            else:
                # Task succeeded
                self._dag.update_task_status(task.id, TaskStatus.COMPLETED, result.output)
                self._task_results[task.id] = result
                self.logger.log_info(f"Task {task.id} completed successfully")

    async def _execute_task(self, task: Task) -> TaskExecutionResult:
        """Execute a single task.

        Args:
            task: Task to execute

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
            task_description = self._build_task_description(task)

            # Execute the task
            response = await agent.execute(
                task=task_description,
                context={"task_id": task.id, "task_type": task.type},
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

    def _build_task_description(self, task: Task) -> str:
        """Build a task description for the agent.

        Args:
            task: Task to describe

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

        # Add context about dependencies
        if task.depends:
            dep_results = []
            for dep_id in task.depends:
                dep_result = self._task_results.get(dep_id)
                if dep_result and dep_result.success:
                    dep_results.append(f"Task {dep_id} completed successfully")
                else:
                    dep_results.append(f"Task {dep_id} failed")

            if dep_results:
                description += f"\n\nDependency status:\n" + "\n".join(f"  - {r}" for r in dep_results)

        return description

    def get_task_results(self) -> Dict[str, TaskExecutionResult]:
        """Get all task results.

        Returns:
            Dictionary mapping task IDs to execution results
        """
        return self._task_results.copy()

    def get_failed_tasks(self) -> List[str]:
        """Get IDs of failed tasks.

        Returns:
            List of failed task IDs
        """
        return [
            task_id
            for task_id, result in self._task_results.items()
            if not result.success
        ]

    def get_successful_tasks(self) -> List[str]:
        """Get IDs of successful tasks.

        Returns:
            List of successful task IDs
        """
        return [
            task_id
            for task_id, result in self._task_results.items()
            if result.success
        ]

    def get_total_tokens_used(self) -> int:
        """Get total tokens used across all tasks.

        Returns:
            Total tokens used
        """
        return sum(result.tokens_used for result in self._task_results.values())

    def get_total_duration(self) -> float:
        """Get total duration of execution.

        Returns:
            Total duration in seconds
        """
        return sum(result.duration_seconds for result in self._task_results.values())
