LLM_BASE_URL = "http://localhost:8080/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "local"
CTX_SIZE = 16384          # must match --ctx-size passed to llama-server
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them when needed to answer the user's request accurately."
)

# ── Generation limits ─────────────────────────────────────────────────────────
# Max tokens the model may generate per response (thinking + content combined).
# -1 = unlimited (default llama.cpp behaviour, can cause very long thinking blocks).
# Set a sensible cap to prevent runaway generation.
MAX_TOKENS: dict[str, int] = {
    "fast":   1024,
    "medium": 4096,
    "slow":   8192,
}

# ── Thinking / research modes ─────────────────────────────────────────────────
THINKING_MODES: dict[str, int] = {
    "fast":   512,
    "medium": 4096,
    "slow":   16384,
}
DEFAULT_THINKING_MODE = "medium"

# ── Context window management ─────────────────────────────────────────────────
# Trigger compression when estimated token usage exceeds this fraction of CTX_SIZE.
CTX_COMPRESS_THRESHOLD = 0.90   # 90%

# Characters-per-token estimate used for cheap local token counting.
# GPT-family average is ~4; llama models are similar.
CHARS_PER_TOKEN = 4

# How many of the oldest non-system message pairs to summarise per compression pass.
# Keeps the most recent turns intact for coherence.
CTX_COMPRESS_KEEP_RECENT = 4    # keep last N user/assistant pairs verbatim

# Auto-compaction interval in tokens. Compaction is also triggered when
# usage exceeds CTX_COMPRESS_THRESHOLD (90%), whichever comes first.
CTX_COMPACT_INTERVAL = 20_000   # compact every 20K accumulated tokens

# Server-side context expansion — doubles CTX_SIZE when compression alone
# is not enough. Requires llama-server to be managed by the harness.
# Set to None to disable server restart entirely.
LLAMA_SERVER_CMD: list[str] | None = None   # e.g. ["llama-server", "--model", "..."]
CTX_SIZE_MAX = 131072           # hard cap — never expand beyond this
CTX_SIZE_MULTIPLIER = 2         # each expansion doubles the window
