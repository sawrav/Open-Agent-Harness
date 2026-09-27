# Open-Agent-Harness

A lightweight, extensible agentic harness for running local sympathetic LLMs (having mechanical sympathy!) via
[llama.cpp](https://github.com/ggerganov/llama.cpp). Fully async, token-streaming, with tool-calling, thinking mode control, and automatic memory compaction.

**Live site:** `https://github.com/sawrav/Open-Agent-Harness`

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [REPL commands](#repl-commands)
- [Thinking modes](#thinking-modes)
- [Built-in tools](#built-in-tools)
- [Built-in skills](#built-in-skills)
- [Memory compaction](#memory-compaction)
- [Adding a new tool](#adding-a-new-tool)
- [Adding a new skill](#adding-a-new-skill)
- [Configuration reference](#configuration-reference)
- [Project structure](#project-structure)
- [License](#license)

---

## Features

- **Async streaming** — tokens are written to the terminal the instant llama.cpp emits them via raw httpx SSE, with zero buffering
- **`<thinking>` block rendering** — reasoning tokens (`delta.reasoning_content`) streamed live in dim style, visually separated from the final response
- **Tool calling** — model can invoke local tools; multiple tool calls in one turn execute concurrently via `asyncio.gather`
- **Autodiscovery** — drop a `.py` file under `tools/` to add a callable tool; drop a `SKILL.md` to inject behavioral instructions into the system prompt
- **Thinking modes** — three research modes (`fast` / `medium` / `slow`) control the model's reasoning budget; switchable live with `/mode`
- **Memory compaction** — automatic context management: exact token counting via `/tokenize`, LLM-based summarisation of old messages, KV cache erasure via `/slots/0?action=erase`; manual `/compact` command also available
- **Graceful error handling** — malformed tool calls are caught before dispatch; HTTP errors (400, 500) are caught and surfaced without crashing the session; context overflow triggers automatic compaction + retry

---

## Requirements

- Python 3.11+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) built and available as `llama-server`
- A GGUF model (tested with `Muse-Glimmer-30B-Q4_K_M.gguf`)

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
  --threads 16 \
  --port 8080
```

Recommended flags for Apple Silicon (M-series):

| Flag | Value | Reason |
|---|---|---|
| `--n-gpu-layers` | `99` | Offload all layers to Metal |
| `--ctx-size` | `16384` | Must match `CTX_SIZE` in `config.py` |
| `--threads` | `16` | Large core count on M-series chips |
| `--defrag-thold` | `0.1` | Enable KV cache defrag for long sessions |

**Step 2 — Run the agent:**

```bash
python main.py
```

The startup banner shows active tools, loaded skills, and current thinking mode:

```
OpenAgentHarness ready.
  Tools :  fetch_url, list_directory, read_file, run_shell, write_file
  Skills:  git_operations, summarize
  Mode  :  medium (4096 budget tokens)
  Type /help for available commands.
```

---

## REPL commands

| Command | Description |
|---|---|
| `/mode <name>` | Switch thinking mode: `fast`, `medium`, or `slow` |
| `/mode` | Show current mode and all options |
| `/compact` | Manually compact memory and erase KV cache |
| `/tools` | List all available callable tools |
| `/skills` | List all loaded Markdown skills |
| `/clear` | Clear conversation history (keeps system prompt) |
| `/help` | Show command list with current mode |
| `/exit` | Quit |

---

## Thinking modes

Controls how many tokens the model is allowed to spend in `<thinking>` before
producing a response. Sent to llama.cpp as a `thinking.budget_tokens` field.
Switch modes live at any point during a session — no restart required.

| Mode | Budget tokens | Use case |
|---|---|---|
| `fast` | 512 | Quick lookups, simple Q&A, low latency |
| `medium` | 4096 | General use — balanced quality and speed (**default**) |
| `slow` | 16384 | Deep research, multi-step reasoning, complex analysis |

```
You: /mode slow
Mode → slow (16384 budget tokens)

You: Analyse the architecture of vLLM in detail
<thinking>
...deep reasoning streamed live, visually dimmed...
</thinking>

vLLM uses a PagedAttention mechanism that...
```

The `<thinking>` block renders in dim ANSI style and is automatically closed
when content tokens begin, or at end of stream.

---

## Built-in tools

Tools are Python classes under `tools/` that extend `BaseTool`. All tool
execution is async. Multiple tool calls in a single turn run concurrently.

| Tool | Description |
|---|---|
| `read_file` | Read the full contents of a local file |
| `write_file` | Write content to a file, creating parent directories if needed |
| `list_directory` | List files and subdirectories in a directory |
| `run_shell` | Execute a shell command and return stdout + stderr (30s timeout) |
| `fetch_url` | Fetch the text content of a URL (capped at 8000 chars) |

Tool call validation before dispatch:
- Incomplete delta accumulations (missing `name` or `id`) are skipped with a descriptive error returned to the model
- Required parameters are checked against the tool's JSON Schema before calling
- Missing required params return a clean error string to the model rather than raising

---

## Built-in skills

Skills are `SKILL.md` files under `tools/`. They are injected into the system
prompt at startup and give the model behavioral instructions. No code involved.

| Skill | Description |
|---|---|
| `git_operations` | Best practices for git operations via the shell tool |
| `summarize` | Instructions for summarising files, documents, and web pages |

Skills are discovered recursively alongside Python tools. A folder with only a
`SKILL.md` and no `.py` file is perfectly valid.

---

## Memory compaction

The harness tracks exact token usage via llama.cpp's `/tokenize` endpoint and
manages the context window automatically. No server restart is ever required.

### Token tracking

Usage is reported in 10K-token bands after each turn:

```
  📊 Context: ~12,450 tokens (76% of 16,384)
```

### Auto-compaction

Compaction is triggered automatically when either condition is met:

| Trigger | Default threshold |
|---|---|
| Context usage | ≥ 90% of `CTX_SIZE` |
| Accumulated tokens | Every 20,000 tokens (`CTX_COMPACT_INTERVAL`) |

### Compaction cycle

1. The LLM summarises the oldest message pairs into a compact block (keeps the last 4 user/assistant pairs verbatim)
2. `POST /slots/0?action=erase` erases the slot KV cache
3. The next request rebuilds the KV from the compacted history automatically

```
  ⟳  Compacting memory (auto, 90% threshold) — ~14,800 tokens…
  ✓  Compacted: 14,800 → 2,900 tokens (saved 11,900). KV cache erased. Context now at 18%
```

### Overflow recovery

When llama.cpp returns a `400 exceed_context_size_error`, the harness:
1. Compacts immediately
2. Erases the KV cache
3. Re-appends the failed user message
4. Retries the turn automatically — transparent to the user

### Manual compaction

```
You: /compact
  ⟳  Compacting memory (manual) — ~8,400 tokens…
  ✓  Compacted: 8,400 → 1,800 tokens (saved 6,600). KV cache erased. Context now at 11%
```

---

## Adding a new tool

Create a `.py` file anywhere under `tools/` with a class extending `BaseTool`.
The registry autodiscovers it on next startup — no other files need to change.

```python
# tools/calculator.py
from tools.base import BaseTool

class CalculatorTool(BaseTool):
    name = "calculate"
    description = "Evaluate a mathematical expression safely"
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "A Python math expression"}
        },
        "required": ["expression"],
    }

    async def run(self, expression: str) -> str:
        try:
            return str(eval(expression, {"__builtins__": {}}))
        except Exception as e:
            return f"Error: {e}"
```

Note: `run()` must be `async def`. Use `asyncio.create_subprocess_shell` for
subprocesses, `httpx.AsyncClient` for HTTP, and `loop.run_in_executor` for
blocking I/O — same patterns as the built-in tools.

---

## Adding a new skill

Create a folder under `tools/` with a `SKILL.md` file:

```
tools/
└── code-review/
    └── SKILL.md
```

```markdown
---
name: code_review
description: Guidelines for reviewing and critiquing code
---

When asked to review code:
- Check for correctness, edge cases, and error handling first
- Note performance issues separately from correctness issues
- Always suggest a concrete fix, not just a diagnosis
- Be direct — avoid softening language like "you might want to consider"
```

The frontmatter `name` and `description` fields are optional — the folder name
is used as the skill name if not specified. The skill is injected into the
system prompt at startup. No code changes needed.

---

## Configuration reference

All configuration is in `config.py`:

```python
# ── Server ────────────────────────────────────────────────────────────────────
LLM_BASE_URL = "http://localhost:8080/v1"   # llama.cpp server URL
LLM_API_KEY  = "not-needed"                 # API key (unused by default)
LLM_MODEL    = "local"                      # model name sent in requests
CTX_SIZE     = 16384                        # must match --ctx-size on the server

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are a helpful assistant..."

# ── Thinking modes ────────────────────────────────────────────────────────────
THINKING_MODES = {
    "fast":   512,     # minimal reasoning
    "medium": 4096,    # balanced default
    "slow":   16384,   # deep research
}
DEFAULT_THINKING_MODE = "medium"

# ── Memory compaction ─────────────────────────────────────────────────────────
CTX_COMPRESS_THRESHOLD  = 0.90    # compact at 90% context usage
CTX_COMPRESS_KEEP_RECENT = 4      # keep last N user/assistant pairs verbatim
CTX_COMPACT_INTERVAL    = 20_000  # also compact every 20K accumulated tokens
CHARS_PER_TOKEN         = 4       # fallback char-based token estimate
```

---

## Project structure

```
Open-Agent-Harness/
├── main.py               # Async entry point (asyncio.run)
├── agent.py              # Core async agent loop, REPL, /commands
├── llm.py                # Raw httpx SSE client, ToolCallAccumulator, StreamResult
├── registry.py           # Tool + skill autodiscovery and routing
├── context_manager.py    # Token counting, compaction, KV cache management
├── config.py             # All configuration
├── requirements.txt      # httpx, rich, anyio
└── tools/
    ├── __init__.py
    ├── base.py            # BaseTool abstract interface (async run())
    ├── filesystem.py      # read_file, write_file, list_directory
    ├── shell.py           # run_shell (asyncio subprocess)
    ├── web.py             # fetch_url (httpx.AsyncClient)
    ├── git/
    │   └── SKILL.md       # Git operation guidelines
    └── summarize/
        └── SKILL.md       # Summarisation guidelines
```

---

## License

MIT
