"""DAG (Directed Acyclic Graph) data structures and operations."""

from typing import Dict, List, Optional, Set

from devflow.exceptions.errors import DAGError
from devflow.schemas.tasks import Task, TaskDAG, TaskStatus


class DAG:
    """Directed Acyclic Graph for task dependencies."""

    def __init__(self, tasks: List[Task]):
        """Initialize the DAG.

        Args:
            tasks: List of tasks in the DAG
        """
        self.tasks = {task.id: task for task in tasks}
        self._adjacency: Dict[str, Set[str]] = {}
        self._reverse_adjacency: Dict[str, Set[str]] = {}

        self._build_graph()

    def _build_graph(self) -> None:
        """Build the adjacency lists from task dependencies."""
        for task_id, task in self.tasks.items():
            self._adjacency[task_id] = set()
            self._reverse_adjacency[task_id] = set()

        for task_id, task in self.tasks.items():
            for dep_id in task.depends:
                if dep_id not in self.tasks:
                    raise DAGError(f"Dependency task not found: {dep_id}")

                self._adjacency[dep_id].add(task_id)
                self._reverse_adjacency[task_id].add(dep_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.

        Args:
            task_id: Task ID

        Returns:
            Task or None if not found
        """
        return self.tasks.get(task_id)

    def get_dependencies(self, task_id: str) -> List[str]:
        """Get the dependencies of a task.

        Args:
            task_id: Task ID

        Returns:
            List of task IDs that this task depends on
        """
        task = self.get_task(task_id)
        return task.depends if task else []

    def get_dependents(self, task_id: str) -> List[str]:
        """Get the tasks that depend on this task.

        Args:
            task_id: Task ID

        Returns:
            List of task IDs that depend on this task
        """
        return list(self._adjacency.get(task_id, set()))

    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute.

        Returns:
            List of tasks whose dependencies are all completed
        """
        ready = []

        for task_id, task in self.tasks.items():
            if task.status != TaskStatus.PENDING:
                continue

            # Check if all dependencies are completed
            deps_completed = all(
                self.tasks.get(dep_id, Task(id=dep_id)).status == TaskStatus.COMPLETED
                for dep_id in task.depends
            )

            if deps_completed:
                ready.append(task)

        return ready

    def get_failed_tasks(self) -> List[Task]:
        """Get all failed tasks.

        Returns:
            List of failed tasks
        """
        return [task for task in self.tasks.values() if task.status == TaskStatus.FAILED]

    def is_complete(self) -> bool:
        """Check if all tasks are complete.

        Returns:
            True if all tasks are completed or failed
        """
        return all(
            task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED)
            for task in self.tasks.values()
        )

    def has_failed(self) -> bool:
        """Check if any task has failed.

        Returns:
            True if any task has failed
        """
        return any(task.status == TaskStatus.FAILED for task in self.tasks.values())

    def topological_sort(self) -> List[str]:
        """Get a topological ordering of tasks.

        Returns:
            List of task IDs in topological order

        Raises:
            DAGError: If the graph has a cycle
        """
        # Kahn's algorithm
        in_degree: Dict[str, int] = {task_id: 0 for task_id in self.tasks}

        for task_id, task in self.tasks.items():
            for dep_id in task.depends:
                in_degree[task_id] += 1

        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            task_id = queue.pop(0)
            result.append(task_id)

            for dependent in self._adjacency[task_id]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self.tasks):
            raise DAGError("Graph has a cycle")

        return result

    def get_parallel_groups(self) -> List[List[str]]:
        """Get groups of tasks that can be executed in parallel.

        Returns:
            List of lists, where each inner list contains task IDs that can be executed together
        """
        groups = []
        remaining = set(self.tasks.keys())

        while remaining:
            # Find tasks with no remaining dependencies
            ready = []
            for task_id in remaining:
                task = self.tasks[task_id]
                if all(dep not in remaining for dep in task.depends):
                    ready.append(task_id)

            if not ready:
                # This shouldn't happen if the DAG is valid
                raise DAGError("Cannot find ready tasks - possible cycle")

            groups.append(ready)
            remaining -= set(ready)

        return groups

    def update_task_status(self, task_id: str, status: TaskStatus, result: Optional[Dict] = None) -> None:
        """Update the status of a task.

        Args:
            task_id: Task ID
            status: New status
            result: Task result (if completed)
        """
        task = self.get_task(task_id)
        if task:
            task.status = status
            if result:
                task.result = result

    def to_task_dag(self, project_name: str) -> TaskDAG:
        """Convert to TaskDAG schema.

        Args:
            project_name: Name of the project

        Returns:
            TaskDAG instance
        """
        return TaskDAG(
            project_name=project_name,
            tasks=list(self.tasks.values()),
        )

    @classmethod
    def from_task_dag(cls, task_dag: TaskDAG) -> "DAG":
        """Create DAG from TaskDAG schema.

        Args:
            task_dag: TaskDAG instance

        Returns:
            DAG instance
        """
        return cls(task_dag.tasks)
