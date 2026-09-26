"""
context_manager.py — Context window management for OpenAgentHarness.

Strategy:
  - Token counting:   exact counts via llama.cpp /tokenize endpoint
  - Context tracking: 10K-token increments reported to the user
  - Auto-compaction:  triggered at CTX_COMPRESS_THRESHOLD (80%) or every
                      CTX_COMPACT_INTERVAL tokens, whichever comes first
  - Compaction:       summarise old messages via LLM, then erase the slot
                      KV cache via POST /slots/0?action=erase
  - Overflow fallback: if compaction summary itself would overflow, fall back
                       to hard truncation (keep system + recent N pairs only)

NOTE: /tokenize and /slots are at the server root, NOT under /v1.
      A separate httpx client without the /v1 base_url is used for these.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from rich.console import Console

import config

if TYPE_CHECKING:
    from llm import LLMClient

console = Console()


def _server_root() -> str:
    """Return the server root URL without the /v1 suffix."""
    base = config.LLM_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


# ── Token counting ────────────────────────────────────────────────────────────

async def count_tokens(text: str, http: httpx.AsyncClient) -> int:
    """
    Exact token count via llama.cpp /tokenize endpoint (server root, not /v1).
    Falls back to char-based estimate if the endpoint is unavailable.
    """
    try:
        resp = await http.post(
            "/tokenize",
            json={"content": text, "add_special": False},
            timeout=5.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return len(data.get("tokens", []))
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


# ── KV cache erase ────────────────────────────────────────────────────────────

async def erase_kv_cache(http: httpx.AsyncClient, slot_id: int = 0) -> bool:
    """
    Erase llama.cpp's KV cache for the given slot.
    Uses POST /slots/{id}?action=erase at the server root (not /v1).
    """
    try:
        resp = await http.post(
            f"/slots/{slot_id}?action=erase",
            timeout=10.0,
        )
        return resp.status_code == 200
    except Exception:
        return False


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

    Falls back to hard truncation if:
    - Not enough history to split
    - The summarisation LLM call itself would overflow (old_messages too large)
    - The summary comes back empty
    """
    if len(messages) <= 1:
        return messages

    system_msg = messages[0]
    rest = messages[1:]

    user_indices = [i for i, m in enumerate(rest) if m.get("role") == "user"]

    if len(user_indices) <= config.CTX_COMPRESS_KEEP_RECENT:
        # Not enough history to split — hard truncate as last resort
        return _hard_truncate(messages)

    split_at = user_indices[-config.CTX_COMPRESS_KEEP_RECENT]
    old_messages = rest[:split_at]
    recent_messages = rest[split_at:]

    if not old_messages:
        return messages

    # Safety check: if old_messages text alone exceeds 60% of CTX_SIZE,
    # the summarisation request itself would overflow. Hard-truncate instead.
    old_text = _messages_to_text(old_messages)
    old_tokens = await count_tokens(old_text, http)
    if old_tokens > int(config.CTX_SIZE * 0.60):
        console.print(
            f"  [dim yellow]⚠  Old messages too large to summarise "
            f"({old_tokens:,} tokens) — hard truncating[/dim yellow]"
        )
        return _hard_truncate(messages)

    # Truncate old_text to be safe before sending to LLM
    max_summary_input_chars = config.CTX_SIZE * config.CHARS_PER_TOKEN // 2
    if len(old_text) > max_summary_input_chars:
        old_text = old_text[:max_summary_input_chars] + "\n[truncated]"

    summary_request = [
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
    ]

    summary_parts: list[str] = []
    async for kind, token in llm.stream(summary_request, max_tokens=512):
        if kind == "content":
            summary_parts.append(token)

    summary = "".join(summary_parts).strip()
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
    Uses a dedicated httpx client pointing at the server root (not /v1) for
    /tokenize and /slots endpoints.
    """

    def __init__(self, llm: "LLMClient"):
        self._llm = llm
        # Separate client at server root for /tokenize and /slots
        self._server_http = httpx.AsyncClient(
            base_url=_server_root(),
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
        )
        self._last_reported_band = 0
        self._tokens_since_compact = 0
        self._total_tokens = 0

    # ── Public API ────────────────────────────────────────────────────────────

    async def after_turn(self, messages: list[dict]) -> list[dict]:
        """
        Called after every completed turn.
        Updates counters, reports 10K bands, triggers auto-compaction.
        """
        token_count = await count_messages_tokens(messages, self._server_http)
        self._total_tokens = token_count
        self._tokens_since_compact += token_count

        band = (token_count // 10_000) * 10_000
        if band > self._last_reported_band:
            self._last_reported_band = band
            ratio = token_count / config.CTX_SIZE
            console.print(
                f"  [dim]📊 Context: ~{token_count:,} tokens "
                f"({ratio:.0%} of {config.CTX_SIZE:,})[/dim]"
            )

        if token_count >= config.CTX_SIZE * config.CTX_COMPRESS_THRESHOLD:
            messages = await self.compact(messages, reason=f"auto ({config.CTX_COMPRESS_THRESHOLD:.0%} threshold)")
        elif self._tokens_since_compact >= config.CTX_COMPACT_INTERVAL:
            messages = await self.compact(messages, reason="auto (20K interval)")

        return messages

    async def handle_overflow(self, messages: list[dict]) -> list[dict]:
        """Called immediately on exceed_context_size_error."""
        console.print(
            "  [bold red]Context overflow — compacting and erasing KV cache…[/bold red]"
        )
        return await self.compact(messages, reason="overflow recovery")

    async def compact(
        self,
        messages: list[dict],
        reason: str = "manual",
    ) -> list[dict]:
        """
        Full compaction cycle:
          1. LLM-summarise old messages (with hard-truncation fallback)
          2. Erase KV cache
        """
        before = await count_messages_tokens(messages, self._server_http)
        console.print(
            f"  [yellow]⟳  Compacting memory ({reason}) — "
            f"~{before:,} tokens in history…[/yellow]"
        )

        messages = await compress_messages(messages, self._llm, self._server_http)

        after = await count_messages_tokens(messages, self._server_http)
        saved = before - after
        ratio = after / config.CTX_SIZE

        erased = await erase_kv_cache(self._server_http)
        kv_note = " KV cache erased." if erased else " (KV erase unavailable)"

        console.print(
            f"  [green]✓  Compacted: {before:,} → {after:,} tokens "
            f"(saved {saved:,}).{kv_note} "
            f"Context now at {ratio:.0%}[/green]"
        )

        self._tokens_since_compact = 0
        self._last_reported_band = (after // 10_000) * 10_000

        return messages

    async def token_count(self, messages: list[dict]) -> int:
        return await count_messages_tokens(messages, self._server_http)

    async def aclose(self):
        await self._server_http.aclose()
