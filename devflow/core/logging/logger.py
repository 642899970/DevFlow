"""Structured logger for DevFlow."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from devflow.schemas.agents import AgentExecution, AgentThought, AgentMessage


class StructuredLogger:
    """Structured logger for agent execution."""

    def __init__(self, log_dir: Optional[Path] = None):
        """Initialize the structured logger.

        Args:
            log_dir: Directory to store log files
        """
        if log_dir is None:
            log_dir = Path.cwd() / "logs"

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set up Python logging
        self.logger = logging.getLogger("devflow")
        self.logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler
        log_file = self.log_dir / f"devflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.current_session_id: Optional[str] = None
        self.current_execution: Optional[AgentExecution] = None

    def start_session(self, session_id: str) -> None:
        """Start a new logging session.

        Args:
            session_id: Unique session identifier
        """
        self.current_session_id = session_id
        self.logger.info(f"Starting session: {session_id}")

    def end_session(self) -> None:
        """End the current logging session."""
        if self.current_session_id:
            self.logger.info(f"Ending session: {self.current_session_id}")
            self.current_session_id = None

    def log_agent_start(self, agent_name: str, task_id: str, model: str) -> None:
        """Log the start of an agent execution.

        Args:
            agent_name: Name of the agent
            task_id: ID of the task being executed
            model: Model being used
        """
        self.logger.info(f"Agent {agent_name} starting task {task_id} with model {model}")

    def log_agent_end(self, agent_name: str, task_id: str, success: bool) -> None:
        """Log the end of an agent execution.

        Args:
            agent_name: Name of the agent
            task_id: ID of the task being executed
            success: Whether the execution succeeded
        """
        status = "succeeded" if success else "failed"
        self.logger.info(f"Agent {agent_name} {status} on task {task_id}")

    def log_thought(self, thought: AgentThought) -> None:
        """Log an agent thought.

        Args:
            thought: Agent thought to log
        """
        self.logger.debug(f"Thought {thought.step}: {thought.thought}")

    def log_tool_call(self, tool_name: str, arguments: Dict[str, Any], result: Any) -> None:
        """Log a tool call.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            result: Tool result
        """
        self.logger.debug(f"Tool call: {tool_name}({arguments}) -> {result}")

    def log_error(self, error: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error.

        Args:
            error: Error message
            context: Additional context
        """
        if context:
            self.logger.error(f"Error: {error} | Context: {json.dumps(context)}")
        else:
            self.logger.error(f"Error: {error}")

    def log_info(self, message: str) -> None:
        """Log an info message.

        Args:
            message: Message to log
        """
        self.logger.info(message)

    def log_debug(self, message: str) -> None:
        """Log a debug message.

        Args:
            message: Message to log
        """
        self.logger.debug(message)

    def save_execution(self, execution: AgentExecution) -> None:
        """Save an agent execution to a JSON file.

        Args:
            execution: Agent execution to save
        """
        if not self.current_session_id:
            return

        session_dir = self.log_dir / self.current_session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        execution_file = session_dir / f"{execution.agent_name}_{execution.task_id}.json"
        with open(execution_file, "w", encoding="utf-8") as f:
            json.dump(execution.model_dump(), f, indent=2, default=str)

        self.logger.debug(f"Saved execution to {execution_file}")

    def get_session_logs(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all logs for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of log entries
        """
        session_dir = self.log_dir / session_id
        if not session_dir.exists():
            return []

        logs = []
        for log_file in session_dir.glob("*.json"):
            with open(log_file, "r", encoding="utf-8") as f:
                logs.append(json.load(f))

        return logs
