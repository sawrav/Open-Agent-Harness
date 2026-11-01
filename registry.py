import importlib
import inspect
import os
import re
import sys
from typing import Optional

from tools.base import BaseTool
import tools as _tools_pkg
import config


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

    Improvements:
    - Better error handling during module loading
    - Skip __pycache__ directories
    - More robust module name resolution
    - Enhanced skill parsing
    - Support for EXTRA_TOOL_PATHS from config
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._skills: list[dict] = []
        self._autodiscover()

    # ── Discovery ─────────────────────────────────────────────────────────────

    def _autodiscover(self):
        """Recursively scan tools/ for Python tools and SKILL.md files."""
        # Discover from default tools directory
        default_tools_dir = os.path.dirname(_tools_pkg.__file__)
        self._scan_directory(default_tools_dir, is_default=True)
        
        # Discover from extra paths
        for extra_path in config.EXTRA_TOOL_PATHS:
            extra_path = os.path.expanduser(extra_path)
            if os.path.exists(extra_path):
                self._scan_directory(extra_path, is_default=False)
            else:
                print(f"[registry] Extra tool path does not exist: {extra_path}")

    def _scan_directory(self, tools_dir: str, is_default: bool = True):
        """Scan a directory for tools and skills."""
        if not os.path.isdir(tools_dir):
            return
            
        source_label = "default" if is_default else f"extra ({tools_dir})"
        print(f"[registry] Scanning {source_label} tools directory: {tools_dir}")
        
        for root, dirs, files in os.walk(tools_dir):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != "__pycache__"]

            for filename in files:
                filepath = os.path.join(root, filename)

                if filename.endswith(".py") and filename not in ("__init__.py", "base.py"):
                    self._load_python_module(filepath, tools_dir)

                elif filename == "SKILL.md":
                    skill = _parse_skill_md(filepath)
                    if skill:
                        self._skills.append(skill)

    def _load_python_module(self, filepath: str, base_dir: str):
        """Import a .py file and register any BaseTool subclasses found in it."""
        # Build module name based on relative path
        try:
            rel_path = os.path.relpath(filepath, base_dir)
            module_name = rel_path.replace(os.sep, ".")[:-3]  # strip .py
            
            # For extra paths, we need to make them importable
            # Add base_dir to sys.path if it's not already there
            if base_dir not in sys.path:
                sys.path.insert(0, base_dir)
            
            # Create a unique module name to avoid conflicts
            if base_dir != os.path.dirname(_tools_pkg.__file__):
                # Extra path - use a unique prefix
                module_name = f"extra_tools.{module_name}"
            
            module = importlib.import_module(module_name)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, BaseTool) and obj is not BaseTool:
                    instance = obj()
                    if instance.name in self._tools:
                        print(f"[registry] Warning: tool '{instance.name}' already exists, overwriting")
                    self._tools[instance.name] = instance
                    print(f"[registry] Loaded tool: {instance.name} from {filepath}")
        except Exception as e:
            print(f"[registry] Failed to load {filepath}: {e}")

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
