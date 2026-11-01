import asyncio
import os

from tools.base import BaseTool


class BatchReadTool(BaseTool):
    name = "batch_read"
    description = "Read multiple files at once and return their contents combined."
    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to read.",
            },
            "separator": {
                "type": "string",
                "description": "Separator between file contents. Defaults to '---'.",
            }
        },
        "required": ["paths"],
    }

    async def run(self, paths: list, separator: str = "---") -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _batch_read_sync, paths, separator)


def _batch_read_sync(paths: list, separator: str) -> str:
    """Synchronously read multiple files."""
    results = []
    
    for path in paths:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            results.append(f"{path}\nError: path does not exist")
            continue
        if not os.path.isfile(path):
            results.append(f"{path}\nError: not a file")
            continue
        
        try:
            with open(path, errors="replace") as f:
                content = f.read()
            results.append(f"{path}\n{separator}\n{content}")
        except Exception as e:
            results.append(f"{path}\nError: {e}")
    
    return "\n\n".join(results)
