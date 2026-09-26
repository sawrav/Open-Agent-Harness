import asyncio

from tools.base import BaseTool

TIMEOUT_SECONDS = 30


class ShellTool(BaseTool):
    name = "run_shell"
    description = (
        "Run a shell command on the local machine and return its stdout and stderr. "
        "Use with caution — destructive commands will execute immediately."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            }
        },
        "required": ["command"],
    }

    async def run(self, command: str) -> str:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return f"Error: command timed out after {TIMEOUT_SECONDS}s"

            output = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()
            return output if output else "(no output)"

        except Exception as e:
            return f"Error: {e}"
