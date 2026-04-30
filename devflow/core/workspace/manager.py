"""Virtual file system for agent collaboration."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from devflow.exceptions.errors import FileOperationError, FileNotFoundError


class WorkspaceManager:
    """Manager for the virtual file system workspace."""

    _instance: Optional["WorkspaceManager"] = None

    def __init__(self, workspace_path: Optional[Path] = None):
        """Initialize the workspace manager.

        Args:
            workspace_path: Path to the workspace directory
        """
        if workspace_path is None:
            # Create a temporary directory for the workspace
            self.workspace_path = Path(tempfile.mkdtemp(prefix="devflow_workspace_"))
        else:
            self.workspace_path = Path(workspace_path)

        # Ensure workspace directory exists
        self.workspace_path.mkdir(parents=True, exist_ok=True)

        # Track file changes
        self._file_changes: List[Dict[str, str]] = []

    @classmethod
    def get_instance(cls, workspace_path: Optional[Path] = None) -> "WorkspaceManager":
        """Get the singleton instance of WorkspaceManager.

        Args:
            workspace_path: Path to the workspace directory (only used on first call)

        Returns:
            WorkspaceManager instance
        """
        if cls._instance is None:
            cls._instance = cls(workspace_path)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance."""
        if cls._instance is not None:
            # Clean up the workspace directory
            if cls._instance.workspace_path.exists():
                shutil.rmtree(cls._instance.workspace_path)
            cls._instance = None

    def read_file(self, path: str) -> str:
        """Read a file from the workspace.

        Args:
            path: Path to the file to read

        Returns:
            File contents

        Raises:
            FileNotFoundError: If file does not exist
            FileOperationError: If read operation fails
        """
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise FileOperationError(f"Failed to read file {path}: {str(e)}")

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file in the workspace.

        Args:
            path: Path to the file to write
            content: Content to write

        Raises:
            FileOperationError: If write operation fails
        """
        file_path = self._resolve_path(path)

        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            # Track the change
            self._file_changes.append({
                "action": "write",
                "path": path,
                "size": len(content),
            })
        except Exception as e:
            raise FileOperationError(f"Failed to write file {path}: {str(e)}")

    def delete_file(self, path: str) -> None:
        """Delete a file from the workspace.

        Args:
            path: Path to the file to delete

        Raises:
            FileNotFoundError: If file does not exist
            FileOperationError: If delete operation fails
        """
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            file_path.unlink()

            # Track the change
            self._file_changes.append({
                "action": "delete",
                "path": path,
            })
        except Exception as e:
            raise FileOperationError(f"Failed to delete file {path}: {str(e)}")

    def list_files(self, path: str = "") -> List[str]:
        """List files and directories in the workspace.

        Args:
            path: Path to list (default: root directory)

        Returns:
            List of file and directory paths
        """
        list_path = self._resolve_path(path)

        if not list_path.exists():
            return []

        files = []
        for item in list_path.iterdir():
            relative_path = item.relative_to(self.workspace_path)
            files.append(str(relative_path))

        return sorted(files)

    def file_exists(self, path: str) -> bool:
        """Check if a file exists in the workspace.

        Args:
            path: Path to check

        Returns:
            True if file exists, False otherwise
        """
        file_path = self._resolve_path(path)
        return file_path.exists()

    def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get information about a file.

        Args:
            path: Path to the file

        Returns:
            Dictionary with file information

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = self._resolve_path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        stat = file_path.stat()
        return {
            "path": path,
            "size": stat.st_size,
            "is_file": file_path.is_file(),
            "is_dir": file_path.is_dir(),
            "modified": stat.st_mtime,
        }

    def export_to_disk(self, output_path: Path) -> None:
        """Export the workspace to a directory on disk.

        Args:
            output_path: Path to export the workspace to

        Raises:
            FileOperationError: If export operation fails
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            # Copy all files from workspace to output directory
            for item in self.workspace_path.iterdir():
                if item.is_file():
                    shutil.copy2(item, output_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, output_path / item.name, dirs_exist_ok=True)
        except Exception as e:
            raise FileOperationError(f"Failed to export workspace: {str(e)}")

    def get_changes(self) -> List[Dict[str, str]]:
        """Get the list of file changes.

        Returns:
            List of file change records
        """
        return self._file_changes.copy()

    def clear_changes(self) -> None:
        """Clear the file change history."""
        self._file_changes.clear()

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path relative to the workspace.

        Args:
            path: Path to resolve

        Returns:
            Absolute path within the workspace
        """
        # Remove leading slash if present
        path = path.lstrip("/")

        # Join with workspace path
        resolved = self.workspace_path / path

        # Ensure the resolved path is within the workspace
        try:
            resolved.resolve().relative_to(self.workspace_path.resolve())
        except ValueError:
            raise FileOperationError(f"Path {path} is outside the workspace")

        return resolved

    def cleanup(self) -> None:
        """Clean up the workspace directory."""
        if self.workspace_path.exists():
            shutil.rmtree(self.workspace_path)
