import subprocess
from tools.base import BaseTool


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

    def run(self, command: str) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = (result.stdout + result.stderr).strip()
            return output if output else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 30 seconds"
        except Exception as e:
            return f"Error: {e}"
