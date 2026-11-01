import asyncio
import os

from tools.base import BaseTool


class DirectoryStatsTool(BaseTool):
    name = "dir_stats"
    description = "Get statistics about a directory including file count, size, and breakdown by type."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to analyze.",
            },
            "recursive": {
                "type": "boolean",
                "description": "Whether to analyze recursively. Defaults to true.",
            }
        },
        "required": ["path"],
    }

    async def run(self, path: str, recursive: bool = True) -> str:
        path = os.path.expanduser(path)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _dir_stats_sync, path, recursive)


def _dir_stats_sync(path: str, recursive: bool) -> str:
    """Synchronously analyze directory statistics."""
    if not os.path.exists(path):
        return f"Error: path does not exist: {path}"
    if not os.path.isdir(path):
        return f"Error: not a directory: {path}"
    
    total_files = 0
    total_dirs = 0
    total_size = 0
    extensions = {}
    
    try:
        if recursive:
            for root, dirs, files in os.walk(path):
                # Skip hidden directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules')]
                total_dirs += len(dirs)
                for filename in files:
                    if filename.startswith('.'):
                        continue
                    total_files += 1
                    filepath = os.path.join(root, filename)
                    try:
                        size = os.path.getsize(filepath)
                        total_size += size
                        ext = os.path.splitext(filename)[1].lower() or 'no_ext'
                        extensions[ext] = extensions.get(ext, 0) + 1
                    except:
                        pass
        else:
            entries = os.listdir(path)
            for entry in entries:
                if entry.startswith('.'):
                    continue
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    total_dirs += 1
                elif os.path.isfile(full_path):
                    total_files += 1
                    try:
                        total_size += os.path.getsize(full_path)
                        ext = os.path.splitext(entry)[1].lower() or 'no_ext'
                        extensions[ext] = extensions.get(ext, 0) + 1
                    except:
                        pass
    
    except PermissionError:
        return f"Error: permission denied accessing {path}"
    
    # Format output
    output = [
        f"Directory: {path}",
        f"Recursive: {recursive}",
        f"Total directories: {total_dirs}",
        f"Total files: {total_files}",
        f"Total size: {total_size} bytes ({total_size / 1024:.2f} KB)",
        "",
        "File extensions:",
    ]
    
    # Sort extensions by count
    sorted_exts = sorted(extensions.items(), key=lambda x: x[1], reverse=True)
    for ext, count in sorted_exts[:20]:  # Top 20
        output.append(f"  {ext}: {count} files")
    
    if len(extensions) > 20:
        output.append(f"  ... and {len(extensions) - 20} more")
    
    return "\n".join(output)
