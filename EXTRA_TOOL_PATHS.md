# Open Agent Harness - Extra Tool Paths Configuration

This document explains how to use EXTRA_TOOL_PATHS to discover tools and skills from custom directories.

## Environment Variable

Set EXTRA_TOOL_PATHS to a comma-separated list of directories:

```bash
EXTRA_TOOL_PATHS=/path/to/custom/tools,/another/path/to/skills python main.py
```

## Features

- Discover tools and skills from multiple custom directories
- Tools are loaded in addition to the default tools/ directory
- SKILL.md files are parsed and injected into system prompt
- Python tools must extend BaseTool class
- Paths can be absolute or relative to current directory
- Non-existent paths are logged but don't cause errors

## Example Setup

```bash
# Create custom tools directory
mkdir -p ~/.open-agent-tools

# Add custom tools
cat > ~/.open-agent-tools/my_tool.py << 'EOF'
from tools.base import BaseTool

class MyCustomTool(BaseTool):
    name = "my_custom_tool"
    description = "My custom tool"
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Input"}
        },
        "required": ["input"]
    }
    
    async def run(self, input: str) -> str:
        return f"Processed: {input}"
EOF

# Add custom skill
mkdir -p ~/.open-agent-tools/my_skill
cat > ~/.open-agent-tools/my_skill/SKILL.md << 'EOF'
---
name: my_skill
description: My custom skill
---

# My Custom Skill

Instructions for the model...
EOF

# Run with extra paths
EXTRA_TOOL_PATHS=~/.open-agent-tools python main.py
```

## Tool Discovery Priority

1. Default tools/ directory
2. Extra paths in order specified

If tools have the same name, later paths overwrite earlier ones with a warning.

## Limitations

- Extra paths must be importable Python modules
- Tools in extra paths need access to tools.base.BaseTool
- Circular imports may cause issues
- Best to keep extra tools self-contained

## Use Cases

- Personal custom tools without modifying main repo
- Team-shared tools in a common directory
- Plugin-style extensions
- Testing new tools without committing them
- Environment-specific tools
