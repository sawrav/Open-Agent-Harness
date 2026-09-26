import asyncio
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

    async def run(self, path: str) -> str:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: path does not exist: {path}"
        if not os.path.isfile(path):
            return f"Error: not a file: {path}"
        # Use asyncio thread executor to avoid blocking the event loop on I/O
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _read_sync, path)


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

    async def run(self, path: str, content: str) -> str:
        path = os.path.expanduser(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write_sync, path, content)
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

    async def run(self, path: str = ".") -> str:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: path does not exist: {path}"
        if not os.path.isdir(path):
            return f"Error: not a directory: {path}"
        entries = sorted(os.listdir(path))
        return "\n".join(entries) if entries else "(empty directory)"


# ── Sync helpers (run in executor) ────────────────────────────────────────────

def _read_sync(path: str) -> str:
    with open(path, errors="replace") as f:
        return f.read()


def _write_sync(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)
