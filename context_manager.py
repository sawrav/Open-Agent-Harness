"""
context_manager.py — Context window management for OpenAgentHarness.

Strategy:
  - Token counting:   exact counts via llama.cpp /tokenize endpoint
  - Context tracking: 10K-token increments reported to the user
  - Auto-compaction:  triggered every 20K tokens of accumulated history
                      (configurable via CTX_COMPACT_INTERVAL)
  - Compaction:       summarise old messages via LLM, then erase the slot
                      KV cache via POST /slots/0?action=erase so llama.cpp
                      rebuilds the cache from the compacted history
  - No server restart required at any point.

Dynamic context resize is not supported by llama.cpp without restart.
The correct on-the-fly approach is to keep the message history small
enough to fit within the fixed context window via compaction.
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


# ── Token counting ────────────────────────────────────────────────────────────

async def count_tokens(text: str, http: httpx.AsyncClient) -> int:
    """
    Exact token count via llama.cpp /tokenize endpoint.
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
    # Fallback: character estimate
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
    """Render messages as plain readable text for summarisation."""
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
    Erase llama.cpp's KV cache for the given slot via POST /slots/{id}?action=erase.
    Returns True on success. After this, the next request will rebuild the KV
    cache from scratch using the (compacted) message history.
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

async def compress_messages(
    messages: list[dict],
    llm: "LLMClient",
) -> list[dict]:
    """
    Summarise the oldest non-system message pairs to reduce token usage.

    Keeps:
      - messages[0]: system prompt (always)
      - Last CTX_COMPRESS_KEEP_RECENT user/assistant pairs verbatim

    Replaces the older pairs with a single compact summary assistant message.
    Returns the new shorter message list.
    """
    if len(messages) <= 1:
        return messages

    system_msg = messages[0]
    rest = messages[1:]

    user_indices = [i for i, m in enumerate(rest) if m.get("role") == "user"]

    if len(user_indices) <= config.CTX_COMPRESS_KEEP_RECENT:
        return messages  # not enough history to compress

    split_at = user_indices[-config.CTX_COMPRESS_KEEP_RECENT]
    old_messages = rest[:split_at]
    recent_messages = rest[split_at:]

    if not old_messages:
        return messages

    # Build summarisation request — no tools, no thinking budget
    summary_request = [
        {
            "role": "system",
            "content": (
                "You are a concise summariser. "
                "Summarise the following conversation into a compact paragraph "
                "preserving all key facts, decisions, tool results, file paths, "
                "code snippets, and context needed to continue the conversation. "
                "Be thorough but brief. Output only the summary, no preamble."
            ),
        },
        {
            "role": "user",
            "content": "Summarise:\n\n" + _messages_to_text(old_messages),
        },
    ]

    summary_parts: list[str] = []
    async for kind, token in llm.stream(summary_request):
        if kind == "content":
            summary_parts.append(token)

    summary = "".join(summary_parts).strip()
    if not summary:
        # Summarisation failed — hard-truncate instead
        return [system_msg] + recent_messages

    summary_msg = {
        "role": "assistant",
        "content": (
            f"[Summary of {len(old_messages)} earlier messages]\n\n{summary}"
        ),
    }

    return [system_msg, summary_msg] + recent_messages


# ── ContextManager ────────────────────────────────────────────────────────────

class ContextManager:
    """
    Orchestrates token tracking, auto-compaction, and KV cache management.

    Token tracking:
      - Uses exact /tokenize counts
      - Reports progress in 10K-token increments
      - Triggers auto-compaction every CTX_COMPACT_INTERVAL tokens

    Compaction (manual /compact or auto):
      1. Summarise old messages via LLM (compress_messages)
      2. Erase slot KV cache (erase_kv_cache)
      3. Next LLM request rebuilds KV from compacted history

    No server restart. No dynamic resize. Works within the fixed ctx window.
    """

    def __init__(self, llm: "LLMClient"):
        self._llm = llm
        self._http = llm._http          # reuse the same httpx client
        self._last_reported_band = 0    # last 10K band reported to user
        self._tokens_since_compact = 0  # accumulates toward compact interval
        self._total_tokens = 0          # lifetime token counter

    # ── Public API ────────────────────────────────────────────────────────────

    async def after_turn(self, messages: list[dict]) -> list[dict]:
        """
        Called after every completed turn.
        - Updates token counters
        - Reports 10K-band progress
        - Triggers auto-compaction at CTX_COMPACT_INTERVAL
        Returns (possibly compacted) message list.
        """
        token_count = await count_messages_tokens(messages, self._http)
        self._total_tokens = token_count
        self._tokens_since_compact += token_count  # approximate delta

        # Report 10K band transitions
        band = (token_count // 10_000) * 10_000
        if band > self._last_reported_band:
            self._last_reported_band = band
            ratio = token_count / config.CTX_SIZE
            console.print(
                f"  [dim]📊 Context: ~{token_count:,} tokens "
                f"({ratio:.0%} of {config.CTX_SIZE:,})[/dim]"
            )

        # Auto-compaction threshold
        if token_count >= config.CTX_SIZE * config.CTX_COMPRESS_THRESHOLD:
            messages = await self.compact(messages, reason="auto (90% threshold)")

        elif self._tokens_since_compact >= config.CTX_COMPACT_INTERVAL:
            messages = await self.compact(messages, reason="auto (20K interval)")

        return messages

    async def handle_overflow(self, messages: list[dict]) -> list[dict]:
        """
        Called immediately on exceed_context_size_error.
        Forces compaction and KV erase before the turn is retried.
        """
        console.print(
            "  [bold red]Context overflow — compacting and erasing KV cache…[/bold red]"
        )
        return await self.compact(messages, reason="overflow recovery", force_kv_erase=True)

    async def compact(
        self,
        messages: list[dict],
        reason: str = "manual",
        force_kv_erase: bool = False,
    ) -> list[dict]:
        """
        Full compaction cycle:
          1. LLM-summarise old messages
          2. Erase KV cache so llama.cpp rebuilds from compacted history
        """
        before = await count_messages_tokens(messages, self._http)
        console.print(
            f"  [yellow]⟳  Compacting memory ({reason}) — "
            f"~{before:,} tokens in history…[/yellow]"
        )

        messages = await compress_messages(messages, self._llm)

        after = await count_messages_tokens(messages, self._http)
        saved = before - after
        ratio = after / config.CTX_SIZE

        # Always erase KV cache after compaction — the cache is now stale
        erased = await erase_kv_cache(self._http)
        kv_note = " KV cache erased." if erased else " (KV erase failed — continuing anyway)"

        console.print(
            f"  [green]✓  Compacted: {before:,} → {after:,} tokens "
            f"(saved {saved:,}).{kv_note} "
            f"Context now at {ratio:.0%}[/green]"
        )

        # Reset interval counter
        self._tokens_since_compact = 0
        self._last_reported_band = (after // 10_000) * 10_000

        return messages

    async def token_count(self, messages: list[dict]) -> int:
        return await count_messages_tokens(messages, self._http)
