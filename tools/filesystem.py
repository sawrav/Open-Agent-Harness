import asyncio
import base64
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
    description = (
        "Write content to a file, creating the file and parent directories if needed. "
        "The content field accepts plain UTF-8 text. "
        "For files containing quotes, backslashes, or code, encode the content as "
        "standard base64 and set encoding to \"base64\" — this avoids JSON escaping "
        "issues entirely. "
        "When writing source code or config files, always use base64 encoding."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to write to.",
            },
            "content": {
                "type": "string",
                "description": (
                    "File content. Plain UTF-8 text, or base64-encoded bytes "
                    "when encoding is set to \"base64\"."
                ),
            },
            "encoding": {
                "type": "string",
                "enum": ["utf-8", "base64"],
                "description": (
                    "How content is encoded. Use \"base64\" for any file that "
                    "contains quotes, backslashes, or code to avoid JSON escaping "
                    "problems. Defaults to \"utf-8\"."
                ),
            },
        },
        "required": ["path", "content"],
    }

    async def run(self, path: str, content: str, encoding: str = "utf-8") -> str:
        path = os.path.expanduser(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        # Decode base64 if the model chose that encoding
        if encoding == "base64":
            try:
                raw_bytes = base64.b64decode(content)
                decoded = raw_bytes.decode("utf-8")
            except Exception as e:
                return f"Error: could not decode base64 content: {e}"
            write_content = decoded
        else:
            write_content = content

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write_sync, path, write_content)
        return f"Wrote {len(write_content)} bytes to {path}"


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
