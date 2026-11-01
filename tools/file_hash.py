import asyncio
import os
import hashlib

from tools.base import BaseTool


class FileHashTool(BaseTool):
    name = "file_hash"
    description = "Calculate hash of a file for integrity checking or comparison."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file.",
            },
            "algorithm": {
                "type": "string",
                "enum": ["md5", "sha1", "sha256", "sha512"],
                "description": "Hash algorithm to use. Defaults to sha256.",
            }
        },
        "required": ["path"],
    }

    async def run(self, path: str, algorithm: str = "sha256") -> str:
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return f"Error: path does not exist: {path}"
        if not os.path.isfile(path):
            return f"Error: not a file: {path}"

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _hash_sync, path, algorithm)


def _hash_sync(path: str, algorithm: str) -> str:
    """Synchronously calculate file hash."""
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_obj.update(chunk)
        
        return f"{algorithm.upper()} hash of {path}:\n{hash_obj.hexdigest()}"
    except Exception as e:
        return f"Error calculating hash: {e}"
