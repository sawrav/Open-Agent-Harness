import asyncio
import os
from datetime import datetime

from tools.base import BaseTool


class WatchFileTool(BaseTool):
    name = "watch_file"
    description = "Monitor a file for changes and report its current modification time and size."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to watch.",
            },
            "compare_path": {
                "type": "string",
                "description": "Optional path to compare against for detecting changes.",
            }
        },
        "required": ["path"],
    }

    async def run(self, path: str, compare_path: str = None) -> str:
        path = os.path.expanduser(path)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _watch_sync, path, compare_path)


def _watch_sync(path: str, compare_path: str) -> str:
    """Synchronously watch file."""
    try:
        if not os.path.exists(path):
            return f"Error: path does not exist: {path}"
        
        stat = os.stat(path)
        mtime = datetime.fromtimestamp(stat.st_mtime)
        size = stat.st_size
        
        info = [
            f"Path: {path}",
            f"Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Size: {size} bytes",
        ]
        
        if compare_path:
            compare_path = os.path.expanduser(compare_path)
            if os.path.exists(compare_path):
                compare_stat = os.stat(compare_path)
                compare_mtime = datetime.fromtimestamp(compare_stat.st_mtime)
                compare_size = compare_stat.st_size
                
                info.extend([
                    f"\nComparison with {compare_path}:",
                    f"Modified: {compare_mtime.strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Size: {compare_size} bytes",
                    f"Time diff: {(stat.st_mtime - compare_stat.st_mtime)} seconds",
                    f"Size diff: {size - compare_size} bytes",
                ])
        
        return "\n".join(info)
    except Exception as e:
        return f"Error watching file: {e}"
