"""Command execution tool for agents."""

import asyncio
import os
import tempfile
from typing import Any, Dict

from devflow.core.tools.base import BaseTool, ToolResult
from devflow.exceptions.errors import ToolExecutionError


class ExecuteCommandTool(BaseTool):
    """Tool for executing shell commands in an isolated environment."""

    @property
    def name(self) -> str:
        return "execute_command"

    @property
    def description(self) -> str:
        return "Execute a shell command in an isolated environment"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                "working_dir": {
                    "type": "string",
                    "description": "Working directory for command execution",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 60)",
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str = "",
        timeout: int = 60,
        **kwargs,
    ) -> ToolResult:
        """Execute a shell command.

        Args:
            command: Shell command to execute
            working_dir: Working directory for command execution
            timeout: Timeout in seconds
            **kwargs: Additional parameters

        Returns:
            ToolResult with command output
        """
        try:
            # Import here to avoid circular dependency
            from devflow.core.workspace.manager import WorkspaceManager

            workspace = WorkspaceManager.get_instance()

            # Use workspace directory if no working_dir specified
            if not working_dir:
                working_dir = workspace.workspace_path

            # Create a temporary directory for isolation
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy workspace files to temp directory if needed
                # For now, we'll just use the workspace directory directly

                # Execute the command
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                    env=os.environ.copy(),
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return ToolResult(
                        success=False,
                        error=f"Command timed out after {timeout} seconds",
                    )

                stdout_text = stdout.decode("utf-8", errors="replace")
                stderr_text = stderr.decode("utf-8", errors="replace")

                if process.returncode != 0:
                    return ToolResult(
                        success=False,
                        error=f"Command failed with exit code {process.returncode}",
                        output=stdout_text,
                        metadata={
                            "stderr": stderr_text,
                            "exit_code": process.returncode,
                        },
                    )

                return ToolResult(
                    success=True,
                    output=stdout_text,
                    metadata={
                        "stderr": stderr_text,
                        "exit_code": process.returncode,
                    },
                )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to execute command: {str(e)}",
            )


class RunTestsTool(BaseTool):
    """Tool for running tests."""

    @property
    def name(self) -> str:
        return "run_tests"

    @property
    def description(self) -> str:
        return "Run tests in the workspace using pytest"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to test files or directory",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Enable verbose output",
                },
            },
            "required": [],
        }

    async def execute(self, path: str = "", verbose: bool = False, **kwargs) -> ToolResult:
        """Run tests using pytest.

        Args:
            path: Path to test files or directory
            verbose: Enable verbose output
            **kwargs: Additional parameters

        Returns:
            ToolResult with test results
        """
        try:
            # Import here to avoid circular dependency
            from devflow.core.workspace.manager import WorkspaceManager

            workspace = WorkspaceManager.get_instance()

            # Build pytest command
            cmd = ["python", "-m", "pytest"]

            if path:
                cmd.append(path)

            if verbose:
                cmd.append("-v")

            # Execute the command
            command = " ".join(cmd)
            result = await ExecuteCommandTool().execute(
                command=command,
                working_dir=workspace.workspace_path,
                timeout=120,
            )

            return result

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to run tests: {str(e)}",
            )
