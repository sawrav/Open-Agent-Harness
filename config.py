LLM_BASE_URL = "http://localhost:8080/v1"
LLM_API_KEY = "not-needed"
LLM_MODEL = "local"
CTX_SIZE = 16384
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use them when needed to answer the user's request accurately."
)

# ── Thinking / research modes ─────────────────────────────────────────────────
# Controls how many tokens the model is allowed to spend inside <thinking>
# before producing its response. Maps to the budget_tokens field sent to
# llama.cpp. Set to 0 to disable thinking entirely.
#
# fast   — minimal reasoning, lowest latency
# medium — balanced default
# slow   — deep chain-of-thought, best for research / complex tasks
THINKING_MODES: dict[str, int] = {
    "fast":   512,
    "medium": 4096,
    "slow":   16384,
}
DEFAULT_THINKING_MODE = "medium"
