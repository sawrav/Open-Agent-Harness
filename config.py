LLM_BASE_URL = "http://localhost:8080/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "local"
CTX_SIZE = 16384          # must match --ctx-size passed to llama-server
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them when needed to answer the user's request accurately."
)

# ── Thinking / research modes ─────────────────────────────────────────────────
# budget_tokens: max tokens the model may spend in <thinking> before responding.
# Must stay well under CTX_SIZE — the prompt + tool results also consume tokens.
# Rule of thumb: budget_tokens <= CTX_SIZE // 4 to leave room for prompt + output.
THINKING_MODES: dict[str, int] = {
    "fast":   512,
    "medium": 2048,
    "slow":   4096,   # was 16384, which equalled the entire context window
}
DEFAULT_THINKING_MODE = "medium"

# ── Generation limits ─────────────────────────────────────────────────────────
# Hard ceiling on total tokens generated per response (thinking + content).
# budget_tokens + max_tokens must leave room for the prompt in the context window.
# With CTX_SIZE=16384 and ~4-6K typical prompt, max_tokens should be <= 8-10K.
MAX_TOKENS: dict[str, int] = {
    "fast":   1024,
    "medium": 3072,
    "slow":   6144,
}

# ── Context window management ─────────────────────────────────────────────────
# Trigger message compression when token usage exceeds this fraction of CTX_SIZE.
CTX_COMPRESS_THRESHOLD = 0.80   # 80% — compress earlier to stay ahead of overflow

# Characters-per-token estimate for fallback token counting (when /tokenize unavailable).
CHARS_PER_TOKEN = 4

# Keep the last N user/assistant pairs verbatim during compression.
CTX_COMPRESS_KEEP_RECENT = 4

# Also trigger auto-compaction every N accumulated tokens regardless of threshold.
CTX_COMPACT_INTERVAL = 20_000
