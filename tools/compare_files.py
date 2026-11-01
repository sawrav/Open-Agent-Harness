import asyncio
import os

from tools.base import BaseTool


class CompareFilesTool(BaseTool):
    name = "compare_files"
    description = "Compare two files and report differences in size, modification time, and content."
    parameters = {
        "type": "object",
        "properties": {
            "path1": {
                "type": "string",
                "description": "First file path.",
            },
            "path2": {
                "type": "string",
                "description": "Second file path.",
            },
            "compare_content": {
                "type": "boolean",
                "description": "Whether to compare file contents. Defaults to false.",
            }
        },
        "required": ["path1", "path2"],
    }

    async def run(self, path1: str, path2: str, compare_content: bool = False) -> str:
        path1 = os.path.expanduser(path1)
        path2 = os.path.expanduser(path2)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _compare_sync, path1, path2, compare_content)


def _compare_sync(path1: str, path2: str, compare_content: bool) -> str:
    """Synchronously compare two files."""
    if not os.path.exists(path1):
        return f"Error: path1 does not exist: {path1}"
    if not os.path.exists(path2):
        return f"Error: path2 does not exist: {path2}"
    
    try:
        stat1 = os.stat(path1)
        stat2 = os.stat(path2)
        
        from datetime import datetime
        mtime1 = datetime.fromtimestamp(stat1.st_mtime)
        mtime2 = datetime.fromtimestamp(stat2.st_mtime)
        
        output = [
            f"File 1: {path1}",
            f"  Size: {stat1.st_size} bytes",
            f"  Modified: {mtime1.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Type: {'Directory' if os.path.isdir(path1) else 'File'}",
            "",
            f"File 2: {path2}",
            f"  Size: {stat2.st_size} bytes",
            f"  Modified: {mtime2.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Type: {'Directory' if os.path.isdir(path2) else 'File'}",
            "",
            f"Comparison:",
            f"  Size difference: {abs(stat1.st_size - stat2.st_size)} bytes",
            f"  Time difference: {abs(stat1.st_mtime - stat2.st_mtime)} seconds",
        ]
        
        if compare_content and os.path.isfile(path1) and os.path.isfile(path2):
            # Quick content comparison
            try:
                with open(path1, 'rb') as f1, open(path2, 'rb') as f2:
                    content1 = f1.read(1024 * 1024)  # Read first 1MB
                    content2 = f2.read(1024 * 1024)
                    
                    if content1 == content2:
                        output.append(f"  Content: Identical (first 1MB)")
                    else:
                        output.append(f"  Content: Different")
            except Exception as e:
                output.append(f"  Content comparison failed: {e}")
        
        return "\n".join(output)
    except Exception as e:
        return f"Error comparing files: {e}"
