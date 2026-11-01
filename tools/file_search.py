import asyncio
import os
import fnmatch

from tools.base import BaseTool


class FileSearchTool(BaseTool):
    name = "search_files"
    description = "Search for files matching a pattern in a directory recursively."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Filename pattern to match (supports glob patterns like *.py, test_*).",
            },
            "directory": {
                "type": "string",
                "description": "Directory to search in. Defaults to current directory.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 50.",
            },
        },
        "required": ["pattern"],
    }

    async def run(self, pattern: str, directory: str = ".", max_results: int = 50) -> str:
        directory = os.path.expanduser(directory)
        if not os.path.exists(directory):
            return f"Error: path does not exist: {directory}"
        if not os.path.isdir(directory):
            return f"Error: not a directory: {directory}"

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, _search_sync, directory, pattern, max_results
        )
        
        if not results:
            return f"No files matching '{pattern}' found in {directory}"
        
        output = [f"Found {len(results)} file(s) matching '{pattern}' in {directory}:"]
        for path in results:
            output.append(f"  {path}")
        
        return "\n".join(output)


def _search_sync(directory: str, pattern: str, max_results: int) -> list[str]:
    """Synchronously search for files matching pattern."""
    matches = []
    try:
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and common cache dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]
            
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, directory)
                    matches.append(rel_path)
                    if len(matches) >= max_results:
                        return matches
    except PermissionError:
        pass  # Skip directories we can't access
    
    return matches
