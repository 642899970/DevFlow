"""Task planner for converting natural language to task DAG."""

import json
from typing import Dict, List, Optional

from devflow.core.models.base import BaseLLM, Message
from devflow.core.planning.dag import DAG
from devflow.exceptions.errors import TaskPlanningError
from devflow.schemas.tasks import Task, TaskDAG, TaskStatus


class TaskPlanner:
    """Planner for converting natural language requirements to task DAGs."""

    def __init__(self, model: BaseLLM):
        """Initialize the task planner.

        Args:
            model: LLM model to use for planning
        """
        self.model = model

    async def plan(self, requirement: str, project_name: str = "project") -> TaskDAG:
        """Convert a natural language requirement to a task DAG.

        Args:
            requirement: Natural language requirement
            project_name: Name of the project

        Returns:
            TaskDAG with planned tasks

        Raises:
            TaskPlanningError: If planning fails
        """
        try:
            # Build the planning prompt
            prompt = self._build_planning_prompt(requirement)

            # Call the model
            response = await self.model.chat([Message(role="user", content=prompt)])

            # Parse the response
            tasks_data = self._parse_response(response.content)

            # Create tasks
            tasks = []
            for task_data in tasks_data:
                task = Task(
                    id=task_data["id"],
                    type=task_data["type"],
                    agent=task_data["agent"],
                    depends=task_data.get("depends", []),
                    status=TaskStatus.PENDING,
                )
                tasks.append(task)

            # Create and validate the DAG
            dag = DAG(tasks)

            # Return as TaskDAG
            return dag.to_task_dag(project_name)

        except Exception as e:
            raise TaskPlanningError(f"Failed to plan tasks: {str(e)}")

    def _build_planning_prompt(self, requirement: str) -> str:
        """Build the planning prompt.

        Args:
            requirement: Natural language requirement

        Returns:
            Planning prompt
        """
        return f"""You are a task planning assistant. Your job is to break down a software development requirement into a structured task DAG.

Requirement: {requirement}

Please analyze this requirement and create a task plan with the following structure:

1. Identify the main components needed (architecture, frontend, backend, testing)
2. Determine dependencies between tasks
3. Assign each task to an appropriate agent (architect, frontend, backend, tester)

Output a JSON array of tasks with this format:
[
  {{
    "id": "1",
    "type": "design",
    "agent": "architect",
    "depends": []
  }},
  {{
    "id": "2",
    "type": "backend",
    "agent": "backend",
    "depends": ["1"]
  }},
  {{
    "id": "3",
    "type": "frontend",
    "agent": "frontend",
    "depends": ["1"]
  }},
  {{
    "id": "4",
    "type": "test",
    "agent": "tester",
    "depends": ["2", "3"]
  }}
]

Task types:
- design: Architecture and system design
- backend: Backend implementation
- frontend: Frontend implementation
- test: Testing and quality assurance

Agents:
- architect: System architect
- frontend: Frontend engineer
- backend: Backend engineer
- tester: Test engineer

Ensure the task graph is a valid DAG (no cycles). Output ONLY the JSON array, no other text."""

    def _parse_response(self, response: str) -> List[Dict]:
        """Parse the model response into task data.

        Args:
            response: Model response

        Returns:
            List of task data dictionaries

        Raises:
            TaskPlanningError: If parsing fails
        """
        try:
            # Try to extract JSON from the response
            response = response.strip()

            # Find JSON array in the response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1

            if start_idx == -1 or end_idx == 0:
                raise TaskPlanningError("No JSON array found in response")

            json_str = response[start_idx:end_idx]
            tasks_data = json.loads(json_str)

            # Validate the structure
            for task_data in tasks_data:
                if "id" not in task_data or "type" not in task_data or "agent" not in task_data:
                    raise TaskPlanningError(f"Invalid task structure: {task_data}")

                if "depends" not in task_data:
                    task_data["depends"] = []

            return tasks_data

        except json.JSONDecodeError as e:
            raise TaskPlanningError(f"Failed to parse JSON response: {str(e)}")

    async def refine_plan(self, task_dag: TaskDAG, feedback: str) -> TaskDAG:
        """Refine an existing task plan based on feedback.

        Args:
            task_dag: Existing task DAG
            feedback: User feedback for refinement

        Returns:
            Refined TaskDAG
        """
        try:
            # Build the refinement prompt
            prompt = self._build_refinement_prompt(task_dag, feedback)

            # Call the model
            response = await self.model.chat([Message(role="user", content=prompt)])

            # Parse the response
            tasks_data = self._parse_response(response.content)

            # Create tasks
            tasks = []
            for task_data in tasks_data:
                task = Task(
                    id=task_data["id"],
                    type=task_data["type"],
                    agent=task_data["agent"],
                    depends=task_data.get("depends", []),
                    status=TaskStatus.PENDING,
                )
                tasks.append(task)

            # Create and validate the DAG
            dag = DAG(tasks)

            # Return as TaskDAG
            return dag.to_task_dag(task_dag.project_name)

        except Exception as e:
            raise TaskPlanningError(f"Failed to refine plan: {str(e)}")

    def _build_refinement_prompt(self, task_dag: TaskDAG, feedback: str) -> str:
        """Build the refinement prompt.

        Args:
            task_dag: Existing task DAG
            feedback: User feedback

        Returns:
            Refinement prompt
        """
        # Convert task DAG to string representation
        tasks_str = json.dumps(
            [
                {
                    "id": task.id,
                    "type": task.type,
                    "agent": task.agent,
                    "depends": task.depends,
                }
                for task in task_dag.tasks
            ],
            indent=2,
        )

        return f"""You are a task planning assistant. Please refine the following task plan based on user feedback.

Current task plan:
{tasks_str}

User feedback:
{feedback}

Please modify the task plan to address the feedback. Output a JSON array with the same format as before.
Ensure the task graph remains a valid DAG (no cycles). Output ONLY the JSON array, no other text."""
