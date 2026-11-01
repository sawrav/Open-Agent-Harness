"""
context_manager.py — Context window management for OpenAgentHarness.

Strategy:
  - Token counting:   exact counts via vLLM /v1/tokenize endpoint, falling back
                      to character-based estimation if the endpoint is unavailable
  - Context tracking: 10K-token increments reported to the user
  - Auto-compaction:  triggered at CTX_COMPRESS_THRESHOLD (80%) or every
                      CTX_COMPACT_INTERVAL *new* tokens since the last compact
  - Compaction:       summarise old messages via a dedicated httpx LLM call that
                      does NOT share the main LLMClient, so _last_result on the
                      main client is never overwritten during summarisation
  - Overflow fallback: if compaction summary itself would overflow, fall back
                       to hard truncation (keep system + recent N pairs only)

NOTE: vLLM exposes /v1/tokenize (added in v0.4.3).  This module probes that
      endpoint first.  If absent (older vLLM or a different server), token
      counting falls back silently to char-based estimation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
from rich.console import Console

import config

if TYPE_CHECKING:
    from llm import LLMClient

console = Console()


def _vllm_base() -> str:
    """Return the vLLM /v1 base URL (used for /tokenize and summarisation)."""
    return config.VLLM_BASE_URL.rstrip("/")


# ── Token counting ────────────────────────────────────────────────────────────

async def count_tokens(text: str, http: httpx.AsyncClient) -> int:
    """
    Exact token count via vLLM /v1/tokenize endpoint (added in vLLM v0.4.3).
    Falls back to char-based estimation if the endpoint is unavailable.
    """
    try:
        resp = await http.post(
            "/tokenize",
            json={"model": config.VLLM_MODEL, "prompt": text},
            timeout=5.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            if "tokens" in data:
                return len(data["tokens"])
            if "count" in data:
                return int(data["count"])
    except Exception:
        pass
    return len(text) // config.CHARS_PER_TOKEN


async def count_messages_tokens(messages: list[dict], http: httpx.AsyncClient) -> int:
    """Count total tokens across all messages. Adds 4 overhead tokens per message."""
    text = " ".join(
        (msg.get("content") or "") +
        " ".join(
            tc.get("function", {}).get("name", "") +
            tc.get("function", {}).get("arguments", "")
            for tc in msg.get("tool_calls", [])
        )
        for msg in messages
    )
    token_count = await count_tokens(text, http)
    return token_count + (len(messages) * 4)


def _messages_to_text(messages: list[dict]) -> str:
    """Render messages as plain readable text for the summarisation prompt."""
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content") or ""
        if msg.get("tool_calls"):
            calls = [
                f"{tc['function']['name']}({tc['function'].get('arguments', '')})"
                for tc in msg["tool_calls"]
            ]
            content = (content or "") + "\n[Tool calls: " + ", ".join(calls) + "]"
        lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


# ── KV cache note ─────────────────────────────────────────────────────────────

async def erase_kv_cache(http: httpx.AsyncClient, slot_id: int = 0) -> bool:
    """
    vLLM manages KV cache internally via continuous batching — no manual
    slot-erase API exists.  After compaction, vLLM reclaims KV automatically
    on the next prefill of the shortened history.
    """
    return True


# ── Isolated summarisation (does NOT share main LLMClient) ───────────────────

async def _summarise(old_text: str, http: httpx.AsyncClient) -> str:
    """
    Call /v1/chat/completions directly via a raw httpx POST so that the main
    LLMClient._last_result is never overwritten during compaction.

    Returns the summary string, or "" on any failure (caller hard-truncates).
    """
    payload = {
        "model": config.VLLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a concise summariser. "
                    "Summarise the following conversation into a compact paragraph "
                    "preserving all key facts, decisions, tool results, file paths, "
                    "code snippets, and context needed to continue the conversation. "
                    "Output only the summary, no preamble. Keep it under 200 words."
                ),
            },
            {
                "role": "user",
                "content": f"Summarise:\n\n{old_text}",
            },
        ],
        "max_tokens": 512,
        "stream": False,
    }
    try:
        resp = await http.post("/chat/completions", json=payload, timeout=60.0)
        if resp.status_code == 200:
            data = resp.json()
            return data["choices"][0]["message"].get("content", "").strip()
    except Exception:
        pass
    return ""


# ── Message compression ───────────────────────────────────────────────────────

def _hard_truncate(messages: list[dict]) -> list[dict]:
    """
    Emergency fallback: keep only the system prompt and the most recent
    CTX_COMPRESS_KEEP_RECENT user/assistant pairs. No LLM call needed.
    """
    if len(messages) <= 1:
        return messages
    system_msg = messages[0]
    rest = messages[1:]
    user_indices = [i for i, m in enumerate(rest) if m.get("role") == "user"]
    if not user_indices:
        return messages
    keep_from = user_indices[-min(config.CTX_COMPRESS_KEEP_RECENT, len(user_indices))]
    return [system_msg] + rest[keep_from:]


async def compress_messages(
    messages: list[dict],
    llm: "LLMClient",
    http: httpx.AsyncClient,
) -> list[dict]:
    """
    Summarise the oldest non-system message pairs to reduce token usage.
    Uses an isolated HTTP call so the main LLMClient._last_result is untouched.

    Falls back to hard truncation if:
    - Not enough history to split
    - The old messages are too large to summarise safely
    - The summary comes back empty
    """
    if len(messages) <= 1:
        return messages

    system_msg = messages[0]
    rest = messages[1:]

    user_indices = [i for i, m in enumerate(rest) if m.get("role") == "user"]

    if len(user_indices) <= config.CTX_COMPRESS_KEEP_RECENT:
        return _hard_truncate(messages)

    split_at = user_indices[-config.CTX_COMPRESS_KEEP_RECENT]
    old_messages = rest[:split_at]
    recent_messages = rest[split_at:]

    if not old_messages:
        return messages

    old_text = _messages_to_text(old_messages)
    old_tokens = await count_tokens(old_text, http)
    if old_tokens > int(config.CTX_SIZE * 0.60):
        console.print(
            f"  [dim yellow]⚠  Old messages too large to summarise "
            f"({old_tokens:,} tokens) — hard truncating[/dim yellow]"
        )
        return _hard_truncate(messages)

    max_summary_input_chars = config.CTX_SIZE * config.CHARS_PER_TOKEN // 2
    if len(old_text) > max_summary_input_chars:
        old_text = old_text[:max_summary_input_chars] + "\n[truncated]"

    # Use isolated HTTP call — does NOT touch the main LLMClient._last_result
    summary = await _summarise(old_text, http)

    if not summary:
        console.print("  [dim yellow]⚠  Summarisation returned empty — hard truncating[/dim yellow]")
        return _hard_truncate(messages)

    summary_msg = {
        "role": "assistant",
        "content": f"[Summary of {len(old_messages)} earlier messages]\n\n{summary}",
    }

    return [system_msg, summary_msg] + recent_messages


# ── ContextManager ────────────────────────────────────────────────────────────

class ContextManager:
    """
    Orchestrates token tracking, auto-compaction, and KV cache management.

    Token interval tracking: _tokens_since_compact accumulates the *delta*
    (new tokens added each turn), not the full history size, so the 20K
    interval fires only after 20K genuinely new tokens have been generated.
    """

    def __init__(self, llm: "LLMClient"):
        self._llm = llm
        # Shared client for /tokenize and summarisation — same server, no coupling
        # to LLMClient so compaction never overwrites LLMClient._last_result.
        self._http = httpx.AsyncClient(
            base_url=_vllm_base(),
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            timeout=httpx.Timeout(connect=5.0, read=60.0, write=5.0, pool=5.0),
        )
        self._last_reported_band = 0
        self._tokens_since_compact = 0   # delta since last compact, not total
        self._last_token_count = 0       # history size at previous turn

    # ── Public API ────────────────────────────────────────────────────────────

    async def after_turn(self, messages: list[dict]) -> tuple[list[dict], bool]:
        """
        Called after every completed turn.
        Updates counters, reports 10K bands, triggers auto-compaction.

        Returns (messages, compacted) where compacted=True means compaction
        fired this turn.  The caller uses this to decide whether to
        automatically continue without waiting for user input.

        _tokens_since_compact tracks the delta (new tokens this turn) so the
        20K interval fires only after 20K genuinely new tokens, not on the
        first turn that crosses the cumulative total of 20K.
        """
        token_count = await count_messages_tokens(messages, self._http)

        # Delta = growth since last measurement (floor at 0 to handle compaction)
        delta = max(0, token_count - self._last_token_count)
        self._tokens_since_compact += delta
        self._last_token_count = token_count

        band = (token_count // 10_000) * 10_000
        if band > self._last_reported_band:
            self._last_reported_band = band
            ratio = token_count / config.CTX_SIZE
            console.print(
                f"  [dim]📊 Context: ~{token_count:,} tokens "
                f"({ratio:.0%} of {config.CTX_SIZE:,})[/dim]"
            )

        compacted = False
        if token_count >= config.CTX_SIZE * config.CTX_COMPRESS_THRESHOLD:
            messages = await self.compact(
                messages, reason=f"auto ({config.CTX_COMPRESS_THRESHOLD:.0%} threshold)"
            )
            compacted = True
        elif self._tokens_since_compact >= config.CTX_COMPACT_INTERVAL:
            messages = await self.compact(messages, reason="auto (20K interval)")
            compacted = True

        return messages, compacted

    async def handle_overflow(self, messages: list[dict]) -> list[dict]:
        """Called immediately on exceed_context_size_error."""
        console.print(
            "  [bold red]Context overflow — compacting history…[/bold red]"
        )
        return await self.compact(messages, reason="overflow recovery")

    async def compact(
        self,
        messages: list[dict],
        reason: str = "manual",
    ) -> list[dict]:
        """
        Full compaction cycle:
          1. LLM-summarise old messages via isolated HTTP call
          2. Reset delta counter; update last_token_count to post-compact size
        """
        before = await count_messages_tokens(messages, self._http)
        console.print(
            f"  [yellow]⟳  Compacting memory ({reason}) — "
            f"~{before:,} tokens in history…[/yellow]"
        )

        messages = await compress_messages(messages, self._llm, self._http)

        after = await count_messages_tokens(messages, self._http)
        saved = before - after
        ratio = after / config.CTX_SIZE

        await erase_kv_cache(self._http)

        console.print(
            f"  [green]✓  Compacted: {before:,} → {after:,} tokens "
            f"(saved {saved:,}). vLLM will reclaim KV cache on next prefill. "
            f"Context now at {ratio:.0%}[/green]"
        )

        # Reset interval counter and baseline so delta tracking restarts cleanly
        self._tokens_since_compact = 0
        self._last_token_count = after
        self._last_reported_band = (after // 10_000) * 10_000

        return messages

    async def token_count(self, messages: list[dict]) -> int:
        return await count_messages_tokens(messages, self._http)

    async def aclose(self):
        await self._http.aclose()
