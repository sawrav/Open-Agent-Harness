import asyncio
import os

from tools.base import BaseTool


class FileInfoTool(BaseTool):
    name = "file_info"
    description = "Get detailed information about a file including size, type, and permissions."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file or directory to inspect.",
            }
        },
        "required": ["path"],
    }

    async def run(self, path: str) -> str:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: path does not exist: {path}"
        
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(None, _get_info_sync, path)
        return info


def _get_info_sync(path: str) -> str:
    """Get file information synchronously."""
    import stat
    from datetime import datetime
    
    try:
        file_stat = os.stat(path)
        
        is_dir = os.path.isdir(path)
        is_file = os.path.isfile(path)
        is_link = os.path.islink(path)
        
        info_lines = [
            f"Path: {path}",
            f"Type: {'Directory' if is_dir else 'File' if is_file else 'Other'}",
            f"Size: {file_stat.st_size} bytes" if is_file else f"Entries: {len(os.listdir(path)) if is_dir else 'N/A'}",
        ]
        
        if is_file:
            # File type
            mode = file_stat.st_mode
            if stat.S_ISREG(mode):
                info_lines.append("Regular file")
            elif stat.S_ISLNK(mode):
                info_lines.append("Symbolic link")
            elif stat.S_ISFIFO(mode):
                info_lines.append("FIFO/pipe")
            elif stat.S_ISSOCK(mode):
                info_lines.append("Socket")
            elif stat.S_ISCHR(mode):
                info_lines.append("Character device")
            elif stat.S_ISBLK(mode):
                info_lines.append("Block device")
        
        # Permissions
        permissions = oct(file_stat.st_mode)[-3:]
        info_lines.append(f"Permissions: {permissions}")
        
        # Timestamps
        modified = datetime.fromtimestamp(file_stat.st_mtime)
        created = datetime.fromtimestamp(file_stat.st_ctime)
        info_lines.append(f"Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}")
        info_lines.append(f"Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Owner info (if available)
        try:
            info_lines.append(f"Owner UID: {file_stat.st_uid}")
            info_lines.append(f"Group GID: {file_stat.st_gid}")
        except:
            pass
        
        return "\n".join(info_lines)
        
    except Exception as e:
        return f"Error getting file info: {e}"
