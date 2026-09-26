# OpenAgentHarness

A lightweight, extensible agentic harness for running local LLMs via
[llama.cpp](https://github.com/ggerganov/llama.cpp). Drop in a Python file to
add a callable tool, or drop in a `SKILL.md` to inject behavioral instructions
into the system prompt — no other code changes required.

---

## Features

- **Autodiscovery** — tools and skills are picked up automatically from the `tools/` directory
- **Two extension types** — Python tools (callable functions) and Markdown skills (prompt instructions)
- **OpenAI-compatible** — works with any llama.cpp server via its `/v1` API
- **Interactive REPL** — with `/tools`, `/skills`, `/clear`, and `/help` commands

---

## Requirements

- Python 3.11+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) running in server mode
- A GGUF model (e.g. `Muse-Glimmer-30B-Q4_K_M.gguf`)

---

## Setup

```bash
# Clone the repo
git clone https://github.com/sawrav/Open-Agent-Harness.git
cd Open-Agent-Harness

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

**Step 1 — Start the llama.cpp server** (in a separate terminal):

```bash
llama-server \
  --model ./models/Muse-Glimmer-30B-Q4_K_M.gguf \
  --ctx-size 16384 \
  --n-gpu-layers 99 \
  --port 8080
```

**Step 2 — Run the agent:**

```bash
python main.py
```

**In-session commands:**

| Command   | Description                              |
|-----------|------------------------------------------|
| `/tools`  | List all available callable tools        |
| `/skills` | List all loaded Markdown skills          |
| `/clear`  | Clear conversation history               |
| `/help`   | Show command list                        |
| `/exit`   | Quit                                     |

---

## Project structure

```
Open-Agent-Harness/
├── main.py            # Entry point
├── agent.py           # Core agentic loop
├── llm.py             # llama.cpp server client
├── registry.py        # Tool + skill autodiscovery and routing
├── config.py          # Server URL, model params, system prompt
├── requirements.txt
└── tools/
    ├── __init__.py
    ├── base.py            # BaseTool abstract interface
    ├── filesystem.py      # read_file, write_file, list_directory
    ├── shell.py           # run_shell
    ├── web.py             # fetch_url
    ├── git/
    │   └── SKILL.md       # Git operation guidelines (prompt injection)
    └── summarize/
        └── SKILL.md       # Summarization guidelines (prompt injection)
```

---

## Adding a new tool

Create a `.py` file anywhere under `tools/` with a class that extends `BaseTool`:

```python
# tools/calculator.py
from tools.base import BaseTool

class CalculatorTool(BaseTool):
    name = "calculate"
    description = "Evaluate a mathematical expression safely"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string"}
        },
        "required": ["expression"],
    }

    def run(self, expression: str) -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}))
        except Exception as e:
            return f"Error: {e}"
```

Restart the agent — it's automatically registered. No other changes needed.

---

## Adding a new skill

Create a folder under `tools/` and add a `SKILL.md`:

```
tools/
└── my-skill/
    └── SKILL.md
```

```markdown
---
name: my_skill
description: Short description of what this skill does
---

Instructions for the model when handling this type of request...
```

The skill is injected into the system prompt at startup. No code changes needed.

---

## Configuration

Edit `config.py` to change the server URL, model name, or system prompt:

```python
LLM_BASE_URL = "http://localhost:8080/v1"
CTX_SIZE = 16384
SYSTEM_PROMPT = "You are a helpful assistant..."
```

---

## License

MIT
