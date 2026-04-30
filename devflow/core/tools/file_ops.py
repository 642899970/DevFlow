"""File operation tools for agents."""

from typing import Any, Dict

from devflow.core.tools.base import BaseTool, ToolResult
from devflow.exceptions.errors import FileOperationError, FileNotFoundError


class ReadFileTool(BaseTool):
    """Tool for reading files from the workspace."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file from the workspace"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                },
            },
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs) -> ToolResult:
        """Read a file from the workspace.

        Args:
            path: Path to the file to read
            **kwargs: Additional parameters

        Returns:
            ToolResult with file contents
        """
        try:
            # Import here to avoid circular dependency
            from devflow.core.workspace.manager import WorkspaceManager

            workspace = WorkspaceManager.get_instance()
            content = workspace.read_file(path)

            return ToolResult(
                success=True,
                output=content,
                metadata={"path": path, "size": len(content)},
            )

        except FileNotFoundError as e:
            return ToolResult(
                success=False,
                error=str(e),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to read file: {str(e)}",
            )


class WriteFileTool(BaseTool):
    """Tool for writing files to the workspace."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file in the workspace"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs) -> ToolResult:
        """Write content to a file in the workspace.

        Args:
            path: Path to the file to write
            content: Content to write
            **kwargs: Additional parameters

        Returns:
            ToolResult with write outcome
        """
        try:
            # Import here to avoid circular dependency
            from devflow.core.workspace.manager import WorkspaceManager

            workspace = WorkspaceManager.get_instance()
            workspace.write_file(path, content)

            return ToolResult(
                success=True,
                output=f"Successfully wrote {len(content)} bytes to {path}",
                metadata={"path": path, "size": len(content)},
            )

        except FileOperationError as e:
            return ToolResult(
                success=False,
                error=str(e),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to write file: {str(e)}",
            )


class ListFilesTool(BaseTool):
    """Tool for listing files in the workspace."""

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return "List files and directories in the workspace"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to list (default: root directory)",
                },
            },
            "required": [],
        }

    async def execute(self, path: str = "", **kwargs) -> ToolResult:
        """List files in the workspace.

        Args:
            path: Path to list (default: root directory)
            **kwargs: Additional parameters

        Returns:
            ToolResult with file list
        """
        try:
            # Import here to avoid circular dependency
            from devflow.core.workspace.manager import WorkspaceManager

            workspace = WorkspaceManager.get_instance()
            files = workspace.list_files(path)

            return ToolResult(
                success=True,
                output=files,
                metadata={"path": path or "/", "count": len(files)},
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to list files: {str(e)}",
            )
