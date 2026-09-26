import importlib
import inspect
import os
import re
from typing import Optional

from tools.base import BaseTool
import tools as _tools_pkg


def _parse_skill_md(path: str) -> dict | None:
    """
    Parse a SKILL.md file into a dict with keys: name, description, content.

    Optional YAML-style frontmatter between --- delimiters may declare:
        name: skill_name
        description: short description

    If no frontmatter is present, the folder name is used as the skill name.
    """
    try:
        with open(path, "r") as f:
            raw = f.read().strip()
    except OSError as e:
        print(f"[registry] Could not read skill {path}: {e}")
        return None

    # Fallback values
    name = os.path.basename(os.path.dirname(path))
    description = ""
    content = raw

    # Parse optional frontmatter block
    fm_match = re.match(r"^---\n(.*?)\n---\n?(.*)", raw, re.DOTALL)
    if fm_match:
        fm_block = fm_match.group(1)
        content = fm_match.group(2).strip()
        for line in fm_block.splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("description:"):
                description = line.split(":", 1)[1].strip()

    return {"name": name, "description": description, "content": content}


class ToolRegistry:
    """
    Autodiscovers and manages two kinds of extensions inside the tools/ package:

    - Python tools  (.py files with BaseTool subclasses) — exposed as callable
                    functions via the OpenAI tool-calling API.
    - Markdown skills (SKILL.md files) — injected as blocks into the system
                    prompt to give the model behavioral instructions.

    To add a new tool:  drop a .py file with a BaseTool subclass anywhere under tools/.
    To add a new skill: drop a folder with a SKILL.md inside tools/.
    No other files need to change.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._skills: list[dict] = []
        self._autodiscover()

    # ── Discovery ─────────────────────────────────────────────────────────────

    def _autodiscover(self):
        """Recursively scan tools/ for Python tools and SKILL.md files."""
        tools_dir = os.path.dirname(_tools_pkg.__file__)

        for root, dirs, files in os.walk(tools_dir):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != "__pycache__"]

            for filename in files:
                filepath = os.path.join(root, filename)

                if filename.endswith(".py") and filename not in ("__init__.py", "base.py"):
                    self._load_python_module(filepath)

                elif filename == "SKILL.md":
                    skill = _parse_skill_md(filepath)
                    if skill:
                        self._skills.append(skill)

    def _load_python_module(self, filepath: str):
        """Import a .py file and register any BaseTool subclasses found in it."""
        tools_dir = os.path.dirname(_tools_pkg.__file__)
        # Build dotted module name relative to the parent of tools/
        # e.g. /…/tools/git/tool.py  →  tools.git.tool
        rel = os.path.relpath(filepath, os.path.dirname(tools_dir))
        module_name = rel.replace(os.sep, ".")[:-3]  # strip .py

        try:
            module = importlib.import_module(module_name)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseTool) and obj is not BaseTool:
                    instance = obj()
                    self._tools[instance.name] = instance
        except Exception as e:
            print(f"[registry] Failed to load {module_name}: {e}")

    # ── Tool access ───────────────────────────────────────────────────────────

    def register(self, tool: BaseTool):
        """Manually register a tool instance (overrides autodiscovery if same name)."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def all_schemas(self) -> list[dict]:
        """Return OpenAI function-calling schemas for every registered tool."""
        return [t.to_openai_schema() for t in self._tools.values()]

    async def run(self, name: str, **kwargs) -> str:
        """Execute a tool by name asynchronously and return its string result."""
        tool = self.get(name)
        if not tool:
            return f"Error: unknown tool '{name}'"
        try:
            return await tool.run(**kwargs)
        except Exception as e:
            return f"Error running '{name}': {e}"

    def list_tools(self) -> list[str]:
        return sorted(self._tools.keys())

    # ── Skill access ──────────────────────────────────────────────────────────

    def skill_prompt_block(self) -> str:
        """
        Render all loaded skills as a single Markdown block suitable for
        injection into the system prompt.
        """
        if not self._skills:
            return ""
        parts = ["## Loaded Skills\n"]
        for skill in self._skills:
            parts.append(f"### {skill['name']}")
            if skill["description"]:
                parts.append(f"_{skill['description']}_\n")
            parts.append(skill["content"])
            parts.append("")
        return "\n".join(parts)

    def list_skills(self) -> list[str]:
        return sorted(s["name"] for s in self._skills)
