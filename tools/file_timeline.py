import asyncio
import os
from datetime import datetime

from tools.base import BaseTool


class FileTimelineTool(BaseTool):
    name = "file_timeline"
    description = "Get timeline information for files in a directory, showing creation/modification times."
    parameters = {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "Directory to analyze.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to analyze recursively. Defaults to false.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of files to show. Defaults to 50.",
            }
        },
        "required": ["directory"],
    }

    async def run(self, directory: str, recursive: bool = False, limit: int = 50) -> str:
        directory = os.path.expanduser(directory)
        if not os.path.exists(directory):
            return f"Error: path does not exist: {directory}"
        if not os.path.isdir(directory):
            return f"Error: not a directory: {directory}"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _timeline_sync, directory, recursive, limit)


def _timeline_sync(directory: str, recursive: bool, limit: int) -> str:
    """Synchronously get file timeline."""
    files = []
    
    try:
        if recursive:
            for root, dirs, filenames in os.walk(directory):
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules')]
                for filename in filenames:
                    if filename.startswith('.'):
                        continue
                    filepath = os.path.join(root, filename)
                    try:
                        stat = os.stat(filepath)
                        rel_path = os.path.relpath(filepath, directory)
                        files.append((stat.st_mtime, rel_path, stat.st_size))
                    except:
                        pass
        else:
            for filename in os.listdir(directory):
                if filename.startswith('.'):
                    continue
                filepath = os.path.join(directory, filename)
                if os.path.isfile(filepath):
                    try:
                        stat = os.stat(filepath)
                        files.append((stat.st_mtime, filename, stat.st_size))
                    except:
                        pass
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: x[0], reverse=True)
        
        output = [f"File timeline for {directory} (showing {min(limit, len(files))} of {len(files)} files):", ""]
        
        for mtime, path, size in files[:limit]:
            dt = datetime.fromtimestamp(mtime)
            output.append(f"{dt.strftime('%Y-%m-%d %H:%M:%S')} | {size:>10} bytes | {path}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error getting timeline: {e}"
