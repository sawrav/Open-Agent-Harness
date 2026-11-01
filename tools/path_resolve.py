import asyncio
import os

from tools.base import BaseTool


class PathResolveTool(BaseTool):
    name = "resolve_path"
    description = "Resolve a path to its absolute canonical form and check if it exists."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to resolve (relative or absolute).",
            }
        },
        "required": ["path"],
    }

    async def run(self, path: str) -> str:
        path = os.path.expanduser(path)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _resolve_sync, path)


def _resolve_sync(path: str) -> str:
    """Synchronously resolve path."""
    try:
        abs_path = os.path.abspath(path)
        real_path = os.path.realpath(abs_path)
        exists = os.path.exists(real_path)
        
        info = [
            f"Original path: {path}",
            f"Absolute path: {abs_path}",
            f"Real path: {real_path}",
            f"Exists: {exists}",
        ]
        
        if exists:
            if os.path.isdir(real_path):
                info.append(f"Type: Directory")
                try:
                    entries = os.listdir(real_path)
                    info.append(f"Entries: {len(entries)} items")
                except:
                    pass
            elif os.path.isfile(real_path):
                info.append(f"Type: File")
                info.append(f"Size: {os.path.getsize(real_path)} bytes")
            else:
                info.append(f"Type: Other")
        
        return "\n".join(info)
    except Exception as e:
        return f"Error resolving path: {e}"
