"""
All configuration for OpenAgentHarness.

Every value can be overridden at runtime via an environment variable of the
same name, making the harness work against any open-weight model without
editing this file:

    VLLM_MODEL=meta-llama/Llama-3.1-8B-Instruct python main.py
    VLLM_MODEL=meta-llama/Muse-30B-Instruct VLLM_BASE_URL=http://10.0.0.5:8000/v1 python main.py
    VLLM_MODEL=Qwen/QwQ-32B ENABLE_THINKING=true python main.py
"""

import os

def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


# ── Server ────────────────────────────────────────────────────────────────────

# Base URL of the vLLM OpenAI-compatible server.  Must end in /v1.
VLLM_BASE_URL: str = _env("VLLM_BASE_URL", "http://localhost:8000/v1")

# API key sent as Bearer token.  Set to match --api-key on the vLLM server.
# Leave as "not-needed" for unauthenticated servers.
LLM_API_KEY: str = _env("LLM_API_KEY", "not-needed")

# ── Model ─────────────────────────────────────────────────────────────────────

# Hugging Face model ID (or local path) that matches what vLLM was started with.
# Examples:
#   meta-llama/Llama-3.1-8B-Instruct
#   meta-llama/Llama-3.3-70B-Instruct
#   meta-llama/Muse-30B-Instruct-FP8
#   mistralai/Mistral-7B-Instruct-v0.3
#   Qwen/Qwen2.5-7B-Instruct
#   Qwen/QwQ-32B                         (reasoning — set ENABLE_THINKING=true)
#   deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
VLLM_MODEL: str = _env("VLLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")

# Context window size in tokens.  Must match --max-model-len on the vLLM server
# (or the model default if --max-model-len is omitted).
CTX_SIZE: int = _env_int("CTX_SIZE", 131072)

# ── Model capability flags ────────────────────────────────────────────────────
# These guard model-specific API fields so other models are not sent parameters
# they do not support.

# Enable extended thinking / chain-of-thought reasoning tokens.
# Set to true for reasoning models: Qwen/QwQ-*, DeepSeek-R1-*, Llama-*-Thinker-*
# When false, no budget_tokens field is sent and <thinking> blocks are never shown.
ENABLE_THINKING: bool = _env_bool("ENABLE_THINKING", False)

# Set to false for base (non-instruct) models not fine-tuned for function calling.
# When false, no tools are advertised to the model at all.
TOOL_CALL_SUPPORTED: bool = _env_bool("TOOL_CALL_SUPPORTED", True)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT: str = (
    "You are a helpful assistant with access to tools. "
    "Use them when needed to answer the user's request accurately."
)

# ── Thinking / research modes ─────────────────────────────────────────────────
# budget_tokens: maximum tokens the model may spend inside <thinking>.
# Only applied when ENABLE_THINKING=true.
# Rule of thumb: budget_tokens <= CTX_SIZE // 8 to leave room for prompt + output.
THINKING_MODES: dict[str, int] = {
    "fast":   512,
    "medium": 2048,
    "slow":   4096,
}
DEFAULT_THINKING_MODE: str = "medium"

# ── Generation limits ─────────────────────────────────────────────────────────
# Hard ceiling on total tokens generated per response (thinking + content combined).
# budget_tokens + max_tokens must leave room for the prompt in the context window.
MAX_TOKENS: dict[str, int] = {
    "fast":   1024,
    "medium": 3072,
    "slow":   6144,
}

# ── Context window management ─────────────────────────────────────────────────
# Trigger message compression when token usage exceeds this fraction of CTX_SIZE.
CTX_COMPRESS_THRESHOLD: float = 0.80   # compact at 80% to stay ahead of overflow

# Keep the last N user/assistant pairs verbatim during compression.
CTX_COMPRESS_KEEP_RECENT: int = 4

# Also trigger auto-compaction every N accumulated tokens regardless of threshold.
CTX_COMPACT_INTERVAL: int = 20_000

# Characters-per-token estimate for fallback token counting.
# Used when the /v1/tokenize endpoint is unavailable.
CHARS_PER_TOKEN: int = 4
