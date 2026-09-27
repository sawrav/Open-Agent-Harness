# Running OpenAgentHarness in a Finch container

This setup containerizes the entire Agent harness so it runs inside a proper Linux sandbox via Finch (Docker-compatible).

## Prerequisites

- Finch installed and running (`finch version`)
- A running `llama-server` instance reachable from the container. By default it expects `http://host.docker.internal:8080/v1`. Adjust `config.py` or the `LLM_BASE_URL` environment variable if your server is elsewhere.

## Build the image

```bash
finch build -t open-agent-harness:latest .
```

## Run interactively

```bash
finch run --rm -it \
  -v "$(pwd):/app" \
  -e LLM_BASE_URL=http://host.docker.internal:8080/v1 \
  open-agent-harness:latest
```

The container mounts the current directory into `/app` so tools like `read_file`, `write_file`, `list_directory`, and `run_shell` operate on your host files inside the sandbox.

## Run via docker-compose

```bash
finch compose up --build
```

To attach an interactive REPL:

```bash
finch compose run --rm agent-harness
```

## Notes

- The harness uses Python 3.11 slim. System packages `git` and `curl` are installed for tool support.
- `PYTHONPATH=/app` ensures imports resolve correctly.
- If you need a different model server address, override `LLM_BASE_URL` at runtime or edit `config.py` before building.
- For production, consider binding only specific host paths instead of mounting the whole repo.
