import asyncio
import os

from tools.base import BaseTool


class GrepTool(BaseTool):
    name = "grep_files"
    description = "Search for text patterns inside files recursively in a directory."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Text pattern to search for (case-sensitive).",
            },
            "directory": {
                "type": "string",
                "description": "Directory to search in. Defaults to current directory.",
            },
            "file_pattern": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g., *.py, *.md). Defaults to all files.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matches to return. Defaults to 100.",
            },
        },
        "required": ["pattern"],
    }

    async def run(self, pattern: str, directory: str = ".", file_pattern: str = "*", max_results: int = 100) -> str:
        directory = os.path.expanduser(directory)
        if not os.path.exists(directory):
            return f"Error: path does not exist: {directory}"
        if not os.path.isdir(directory):
            return f"Error: not a directory: {directory}"

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, _grep_sync, directory, pattern, file_pattern, max_results
        )
        
        if not results:
            return f"No matches for '{pattern}' found in {directory}"
        
        output = [f"Found {len(results)} match(es) for '{pattern}':"]
        for filepath, line_num, line in results:
            output.append(f"  {filepath}:{line_num}: {line.strip()}")
        
        return "\n".join(output)


def _grep_sync(directory: str, pattern: str, file_pattern: str, max_results: int) -> list[tuple]:
    """Synchronously grep for pattern in files."""
    matches = []
    import fnmatch
    
    try:
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories and common cache dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]
            
            for filename in files:
                if not fnmatch.fnmatch(filename, file_pattern):
                    continue
                
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, directory)
                
                try:
                    with open(filepath, 'r', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            if pattern in line:
                                matches.append((rel_path, i, line))
                                if len(matches) >= max_results:
                                    return matches
                except (PermissionError, UnicodeDecodeError, OSError):
                    continue
    except PermissionError:
        pass
    
    return matches
