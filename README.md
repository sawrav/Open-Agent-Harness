# Open-Agent-Harness

A lightweight, extensible agentic harness for running open-weight LLMs locally via
[vLLM](https://github.com/vllm-project/vllm). Fully async, token-streaming, with
tool-calling, live reasoning output, and automatic memory compaction.

Default model: **[Muse Glimmer 30B](https://huggingface.co/meta-models/Muse-Glimmer-30B)** —
Meta's on-device agentic reasoning model. Any model served by vLLM works by setting
a single environment variable.

**Live site:** `https://github.com/sawrav/Open-Agent-Harness`

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Switching models](#switching-models)
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

- **Async streaming** — tokens are written to the terminal the instant vLLM emits them via raw httpx SSE, with zero buffering
- **`<thinking>` block rendering** — reasoning tokens (`delta.reasoning_content`) streamed live in dim style, visually separated from the final response; works with Muse Glimmer, QwQ, DeepSeek-R1, and other reasoning models
- **Tool calling** — model can invoke local tools; multiple tool calls in one turn execute concurrently via `asyncio.gather`
- **Autodiscovery** — drop a `.py` file under `tools/` to add a callable tool; drop a `SKILL.md` to inject behavioral instructions into the system prompt
- **Thinking modes** — three research modes (`fast` / `medium` / `slow`) control the model's reasoning budget; switchable live with `/mode`
- **Memory compaction** — automatic context management: char-based token estimation with vLLM `/v1/tokenize` probe, LLM-based summarisation of old messages, automatic KV cache reclaim on next prefill; manual `/compact` command also available
- **Graceful error handling** — malformed tool calls are caught before dispatch; HTTP errors are caught and surfaced without crashing the session; context overflow triggers automatic compaction + retry
- **Zero-edit model switching** — every config value is an environment variable; no source edits needed to run a different model

---

## Requirements

- Python 3.11+
- [vLLM](https://github.com/vllm-project/vllm) installed and accessible as `vllm`
- A Hugging Face model (default: `meta-models/Muse-Glimmer-30B`)
- A Hugging Face account with access to the model (run `huggingface-cli login` once)

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

**Step 1 — Start the vLLM server** (in a separate terminal):

```bash
vllm serve meta-models/Muse-Glimmer-30B \
  --max-model-len 131072 \
  --tensor-parallel-size 1
```

Recommended flags:

| Flag | Value | Reason |
|---|---|---|
| `--max-model-len` | `131072` | Muse Glimmer's native context window |
| `--tensor-parallel-size` | `1` or `2` | Match to your GPU count |
| `--dtype` | `bfloat16` | Full-precision inference |
| `--quantization` | `awq` or `fp8` | Reduce VRAM on smaller GPUs |
| `--enable-reasoning` | _(flag)_ | Required to surface `reasoning_content` in the stream |
| `--reasoning-parser` | `muse_glimmer` | Route thinking tokens to `delta.reasoning_content` |

**Step 2 — Run the agent:**

```bash
python main.py
```

The startup banner shows the active model, tools, loaded skills, and current thinking mode:

```
OpenAgentHarness ready.
  Model :  meta-models/Muse-Glimmer-30B
  Tools :  fetch_url, list_directory, read_file, run_shell, write_file
  Skills:  git_operations, summarize
  Mode  :  medium (2048 budget tokens)
  Type /help for available commands.
```

---

## Switching models

No code edits are needed. Set `VLLM_MODEL` (and any other relevant variables) before
running, then start a matching vLLM server.

```bash
# Meta Llama 3.1 8B (non-reasoning — disable thinking)
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct ENABLE_THINKING=false python main.py

# Meta Llama 3.3 70B
VLLM_MODEL=meta-llama/Llama-3.3-70B-Instruct ENABLE_THINKING=false python main.py

# Mistral 7B
VLLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3 ENABLE_THINKING=false python main.py

# Qwen 2.5 7B
VLLM_MODEL=Qwen/Qwen2.5-7B-Instruct ENABLE_THINKING=false python main.py

# QwQ 32B (reasoning model)
VLLM_MODEL=Qwen/QwQ-32B ENABLE_THINKING=true python main.py

# DeepSeek R1 (reasoning model)
VLLM_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B ENABLE_THINKING=true python main.py

# Remote vLLM server
VLLM_BASE_URL=http://10.0.0.5:8000/v1 VLLM_MODEL=meta-models/Muse-Glimmer-30B python main.py

# Base model (no tool calling)
VLLM_MODEL=meta-llama/Llama-3.1-8B TOOL_CALL_SUPPORTED=false ENABLE_THINKING=false python main.py
```

See [Configuration reference](#configuration-reference) for all available variables.

---

## REPL commands

| Command | Description |
|---|---|
| `/mode <name>` | Switch thinking mode: `fast`, `medium`, or `slow` |
| `/mode` | Show current mode and all options |
| `/compact` | Manually compact memory |
| `/tools` | List all available callable tools |
| `/skills` | List all loaded Markdown skills |
| `/clear` | Clear conversation history (keeps system prompt) |
| `/help` | Show command list with current mode |
| `/exit` | Quit |

---

## Thinking modes

Controls how many tokens the model is allowed to spend in `<thinking>` before
producing a response. Only active when `ENABLE_THINKING=true` (the default for
Muse Glimmer). Switch modes live at any point during a session — no restart required.

| Mode | Budget tokens | Use case |
|---|---|---|
| `fast` | 512 | Quick lookups, simple Q&A, low latency |
| `medium` | 2048 | General use — balanced quality and speed (**default**) |
| `slow` | 4096 | Deep research, multi-step reasoning, complex analysis |

```
You: /mode slow
Mode → slow (4096 budget tokens)

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
| `fetch_url` | Fetch the text content of a URL (capped at 3000 chars) |

Tool call validation before dispatch:
- Incomplete delta accumulations (missing `name` or `id`) are skipped with a descriptive error returned to the model
- Required parameters are checked against the tool's JSON Schema before calling
- Missing required params return a clean error string to the model rather than raising
- Tool calls are disabled entirely when `TOOL_CALL_SUPPORTED=false`

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

The harness tracks token usage and manages the context window automatically.
No server restart is ever required.

### Token tracking

Token counts are obtained via vLLM's `/v1/tokenize` endpoint (available since
vLLM v0.4.3). If the endpoint is unavailable the harness falls back to a
character-based estimate (`chars / 4`).

Usage is reported in 10K-token bands after each turn:

```
  📊 Context: ~12,450 tokens (10% of 131,072)
```

### Auto-compaction

Compaction is triggered automatically when either condition is met:

| Trigger | Default threshold |
|---|---|
| Context usage | ≥ 80% of `CTX_SIZE` |
| Accumulated tokens | Every 20,000 tokens (`CTX_COMPACT_INTERVAL`) |

### Compaction cycle

1. The LLM summarises the oldest message pairs into a compact block (keeps the last 4 user/assistant pairs verbatim)
2. The shortened message history is used on the next request — vLLM reclaims KV cache automatically on the next prefill

```
  ⟳  Compacting memory (auto, 80% threshold) — ~105,000 tokens…
  ✓  Compacted: 105,000 → 2,900 tokens (saved 102,100). vLLM will reclaim KV cache on next prefill. Context now at 2%
```

### Overflow recovery

When vLLM returns a context-exceeded error, the harness:
1. Compacts the historical messages immediately (preserving in-progress tool calls)
2. Retries the turn automatically — transparent to the user

### Manual compaction

```
You: /compact
  ⟳  Compacting memory (manual) — ~8,400 tokens…
  ✓  Compacted: 8,400 → 1,800 tokens (saved 6,600). vLLM will reclaim KV cache on next prefill. Context now at 1%
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

All configuration lives in `config.py`. Every value reads from an environment
variable of the same name first, falling back to the default shown below.

```bash
# ── Server ────────────────────────────────────────────────────────────────────
VLLM_BASE_URL=http://localhost:8000/v1   # vLLM OpenAI-compatible server URL
LLM_API_KEY=not-needed                   # Bearer token (set if vLLM uses --api-key)

# ── Model ─────────────────────────────────────────────────────────────────────
VLLM_MODEL=meta-models/Muse-Glimmer-30B  # HuggingFace model ID or local path
CTX_SIZE=131072                           # Must match --max-model-len on the server

# ── Model capability flags ────────────────────────────────────────────────────
ENABLE_THINKING=true          # true for reasoning models (Muse Glimmer, QwQ, DeepSeek-R1…)
                              # false for standard instruct models (Llama, Mistral, Qwen…)
TOOL_CALL_SUPPORTED=true      # false for base models not fine-tuned for function calling

# ── Thinking modes ────────────────────────────────────────────────────────────
# Configured in config.py (not env vars); switch live with /mode
# fast:   512 budget tokens
# medium: 2048 budget tokens  (default)
# slow:   4096 budget tokens

# ── Generation limits ─────────────────────────────────────────────────────────
# Configured in config.py
# fast:   1024 max tokens
# medium: 3072 max tokens
# slow:   6144 max tokens

# ── Memory compaction ─────────────────────────────────────────────────────────
# Configured in config.py
# CTX_COMPRESS_THRESHOLD=0.80   compact at 80% context usage
# CTX_COMPRESS_KEEP_RECENT=4    keep last N user/assistant pairs verbatim
# CTX_COMPACT_INTERVAL=20000    also compact every 20K accumulated tokens
```

---

## Project structure

```
Open-Agent-Harness/
├── main.py               # Async entry point (asyncio.run)
├── agent.py              # Core async agent loop, REPL, /commands
├── llm.py                # vLLM httpx SSE client, ToolCallAccumulator, StreamResult
├── registry.py           # Tool + skill autodiscovery and routing
├── context_manager.py    # Token counting, compaction, vLLM KV cache management
├── config.py             # All configuration (env-var overridable)
├── requirements.txt      # httpx, rich, anyio, vllm
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
