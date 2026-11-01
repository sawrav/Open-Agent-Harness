# Running OpenAgentHarness in a container

This setup containerizes the agent harness so it runs inside a Linux sandbox
via Finch (Docker-compatible). The harness connects out to a vLLM server
running on the host or on a remote machine.

## Prerequisites

- Finch installed and running (`finch version`)
- A running vLLM server reachable from the container (see [Starting vLLM](#starting-vllm))

## Starting vLLM

Run vLLM on your host machine (outside the container) before starting the harness:

```bash
vllm serve meta-models/Muse-Glimmer-30B \
  --max-model-len 131072 \
  --tensor-parallel-size 1
```

The default `VLLM_BASE_URL` inside the container is
`http://host.docker.internal:8000/v1`, which resolves to the host machine.
Override it if your server is on a different host or port (see environment
variables below).

## Build the image

```bash
finch build -t open-agent-harness:latest .
```

## Run interactively

```bash
finch run --rm -it \
  -v "$(pwd):/app" \
  -e VLLM_BASE_URL=http://host.docker.internal:8000/v1 \
  -e VLLM_MODEL=meta-models/Muse-Glimmer-30B \
  -e ENABLE_THINKING=true \
  open-agent-harness:latest
```

The container mounts the current directory into `/app` so tools like
`read_file`, `write_file`, `list_directory`, and `run_shell` operate on your
host files inside the sandbox.

## Run via docker-compose

```bash
finch compose up --build
```

All environment variables are pre-configured in `docker-compose.yml` with
shell-variable defaults. Override any of them before running:

```bash
VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct ENABLE_THINKING=false finch compose up --build
```

To attach an interactive REPL:

```bash
finch compose run --rm agent-harness
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://host.docker.internal:8000/v1` | vLLM server URL |
| `VLLM_MODEL` | `meta-models/Muse-Glimmer-30B` | HuggingFace model ID or local path |
| `CTX_SIZE` | `131072` | Context window — must match `--max-model-len` on the server |
| `ENABLE_THINKING` | `true` | Set `false` for non-reasoning instruct models |
| `TOOL_CALL_SUPPORTED` | `true` | Set `false` for base models without function calling |
| `LLM_API_KEY` | `not-needed` | Bearer token — set if vLLM was started with `--api-key` |

## Switching models

Pass a different `VLLM_MODEL` (and matching `ENABLE_THINKING`) at runtime.
No image rebuild is needed — only a matching vLLM server is required.

```bash
# Meta Llama 3.3 70B
VLLM_MODEL=meta-llama/Llama-3.3-70B-Instruct ENABLE_THINKING=false finch compose up

# QwQ 32B (reasoning model)
VLLM_MODEL=Qwen/QwQ-32B ENABLE_THINKING=true finch compose up

# Remote vLLM server
VLLM_BASE_URL=http://10.0.0.5:8000/v1 VLLM_MODEL=meta-models/Muse-Glimmer-30B finch compose up
```

## Notes

- The harness image uses Python 3.11 slim. System packages `git` and `curl` are
  installed for tool support.
- `PYTHONPATH=/app` ensures imports resolve correctly.
- For production, consider binding only specific host paths instead of mounting
  the whole repo.
- vLLM manages its own KV cache internally. After memory compaction the harness
  shortens the message history and vLLM reclaims the cache automatically on the
  next prefill — no manual flush is needed.
