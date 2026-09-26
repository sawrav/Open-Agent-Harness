import os
from tools.base import BaseTool


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read the full contents of a file from the local filesystem."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            }
        },
        "required": ["path"],
    }

    def run(self, path: str) -> str:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: path does not exist: {path}"
        if not os.path.isfile(path):
            return f"Error: not a file: {path}"
        with open(path, errors="replace") as f:
            return f.read()


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Write content to a file, creating the file and parent directories if needed."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to write to.",
            },
            "content": {
                "type": "string",
                "description": "Text content to write.",
            },
        },
        "required": ["path", "content"],
    }

    def run(self, path: str, content: str) -> str:
        path = os.path.expanduser(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"


class ListDirectoryTool(BaseTool):
    name = "list_directory"
    description = "List all files and subdirectories inside a directory."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list. Defaults to current directory.",
            }
        },
        "required": [],
    }

    def run(self, path: str = ".") -> str:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: path does not exist: {path}"
        if not os.path.isdir(path):
            return f"Error: not a directory: {path}"
        entries = sorted(os.listdir(path))
        return "\n".join(entries) if entries else "(empty directory)"
