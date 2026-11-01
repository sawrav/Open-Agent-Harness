import asyncio
import os

from tools.base import BaseTool


class GlobTool(BaseTool):
    name = "glob_search"
    description = "Find files matching glob patterns recursively."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern to search for (e.g., *.py, test_*.py, **/*.md).",
            },
            "directory": {
                "type": "string",
                "description": "Directory to search in. Defaults to current directory.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return. Defaults to 100.",
            }
        },
        "required": ["pattern"],
    }

    async def run(self, pattern: str, directory: str = ".", max_results: int = 100) -> str:
        directory = os.path.expanduser(directory)
        if not os.path.exists(directory):
            return f"Error: path does not exist: {directory}"
        if not os.path.isdir(directory):
            return f"Error: not a directory: {directory}"

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, _glob_sync, directory, pattern, max_results
        )
        
        if not results:
            return f"No files matching '{pattern}' found in {directory}"
        
        output = [f"Found {len(results)} file(s) matching '{pattern}' in {directory}:"]
        for path in results:
            output.append(f"  {path}")
        
        return "\n".join(output)


def _glob_sync(directory: str, pattern: str, max_results: int) -> list[str]:
    """Synchronously find files matching glob pattern."""
    import fnmatch
    
    matches = []
    
    # Handle ** patterns for recursive search
    if '**' in pattern:
        # Simple recursive glob implementation
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules')]
            
            # Get base pattern without **/
            if pattern.startswith('**/'):
                base_pattern = pattern[3:]
            else:
                base_pattern = pattern.split('**/')[-1]
            
            for filename in files:
                if fnmatch.fnmatch(filename, base_pattern):
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, directory)
                    matches.append(rel_path)
                    if len(matches) >= max_results:
                        return matches
    else:
        # Simple pattern matching in current directory
        for root, dirs, files in os.walk(directory):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules')]
            
            for filename in files:
                if fnmatch.fnmatch(filename, pattern):
                    full_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(full_path, directory)
                    matches.append(rel_path)
                    if len(matches) >= max_results:
                        return matches
    
    return matches
