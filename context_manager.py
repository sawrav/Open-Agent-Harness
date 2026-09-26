"""
context_manager.py — Automatic context window management.

Two-stage strategy:
  1. Compression (primary)  — when token usage hits CTX_COMPRESS_THRESHOLD,
     summarise the oldest message pairs in-place using the LLM itself.
     Fully transparent: no server restart, no conversation break.

  2. Server expansion (fallback) — if compression still leaves usage above
     the threshold (e.g. a single huge tool result), double CTX_SIZE and
     restart llama-server in the background. Only active when
     config.LLAMA_SERVER_CMD is set.
"""

from __future__ import annotations

import asyncio
import json
import signal
import subprocess
from typing import TYPE_CHECKING

import httpx
from rich.console import Console

import config

if TYPE_CHECKING:
    from llm import LLMClient

console = Console()


# ── Token estimation ──────────────────────────────────────────────────────────

def _estimate_tokens(messages: list[dict]) -> int:
    """
    Cheap character-based token estimate.
    Counts all string content in the message list and divides by CHARS_PER_TOKEN.
    Adds 4 tokens per message for role/structural overhead.
    """
    total_chars = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            # Multi-part content (text + images etc.)
            for part in content:
                if isinstance(part, dict):
                    total_chars += len(part.get("text", ""))
        else:
            total_chars += len(content)

        # tool_calls field
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            total_chars += len(fn.get("name", "")) + len(fn.get("arguments", ""))

    return (total_chars // config.CHARS_PER_TOKEN) + (len(messages) * 4)


def usage_ratio(messages: list[dict], ctx_size: int | None = None) -> float:
    """Return estimated token usage as a fraction of CTX_SIZE."""
    return _estimate_tokens(messages) / (ctx_size or config.CTX_SIZE)


# ── Message compression ───────────────────────────────────────────────────────

async def compress(messages: list[dict], llm: "LLMClient") -> list[dict]:
    """
    Summarise the oldest non-system message pairs to free context space.

    Keeps:
      - The system prompt (index 0)
      - The most recent CTX_COMPRESS_KEEP_RECENT user/assistant pairs
      - All tool/tool_call messages in the recent window

    Replaces the older pairs with a single summary assistant message.
    Returns the new (shorter) message list.
    """
    if len(messages) <= 1:
        return messages

    system_msg = messages[0]

    # Separate non-system messages
    rest = messages[1:]

    # Identify the boundary: keep the last N user messages and everything after them
    user_indices = [i for i, m in enumerate(rest) if m.get("role") == "user"]

    if len(user_indices) <= config.CTX_COMPRESS_KEEP_RECENT:
        # Not enough history to compress
        return messages

    # Split: old (to summarise) vs recent (to keep verbatim)
    split_at = user_indices[-config.CTX_COMPRESS_KEEP_RECENT]
    old_messages = rest[:split_at]
    recent_messages = rest[split_at:]

    if not old_messages:
        return messages

    console.print(
        f"  [dim yellow]⟳  Context at {usage_ratio(messages):.0%} — "
        f"compressing {len(old_messages)} old messages…[/dim yellow]"
    )

    # Build a summarisation prompt — do NOT pass tools so model returns plain text
    summary_request = [
        {
            "role": "system",
            "content": (
                "You are a concise summariser. "
                "Summarise the following conversation history into a compact paragraph "
                "that preserves all key facts, decisions, tool results, and context "
                "needed to continue the conversation. Be thorough but brief."
            ),
        },
        {
            "role": "user",
            "content": "Summarise this conversation history:\n\n"
            + _messages_to_text(old_messages),
        },
    ]

    # Use a lightweight stream call — no tools, no budget
    summary_parts: list[str] = []
    async for kind, token in llm.stream(summary_request):
        if kind == "content":
            summary_parts.append(token)

    summary = "".join(summary_parts).strip()
    if not summary:
        # Summarisation failed — fall back to truncating old messages
        console.print("  [dim red]⚠  Summarisation returned empty — truncating oldest messages[/dim red]")
        return [system_msg] + recent_messages

    summary_msg = {
        "role": "assistant",
        "content": f"[Conversation summary — {len(old_messages)} messages compressed]\n\n{summary}",
    }

    new_messages = [system_msg, summary_msg] + recent_messages
    new_ratio = usage_ratio(new_messages)
    console.print(
        f"  [dim green]✓  Compressed to {new_ratio:.0%} of context window[/dim green]"
    )
    return new_messages


def _messages_to_text(messages: list[dict]) -> str:
    """Render messages as plain text for the summarisation prompt."""
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


# ── Server-side context expansion ─────────────────────────────────────────────

class ServerManager:
    """
    Manages the llama-server process lifetime for context expansion.

    Only active when config.LLAMA_SERVER_CMD is set.
    Doubles --ctx-size on each expansion, up to CTX_SIZE_MAX.
    """

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._current_ctx: int = config.CTX_SIZE

    def is_managed(self) -> bool:
        return bool(config.LLAMA_SERVER_CMD)

    async def expand(self) -> int | None:
        """
        Double the context size and restart the server.
        Returns the new ctx size, or None if expansion is not possible.
        """
        if not self.is_managed():
            return None

        new_ctx = self._current_ctx * config.CTX_SIZE_MULTIPLIER
        if new_ctx > config.CTX_SIZE_MAX:
            console.print(
                f"  [red]⚠  Cannot expand context further "
                f"(already at max {config.CTX_SIZE_MAX} tokens)[/red]"
            )
            return None

        console.print(
            f"  [yellow]⟳  Expanding context: {self._current_ctx} → {new_ctx} tokens. "
            f"Restarting server…[/yellow]"
        )

        await self._stop()
        await self._start(new_ctx)
        await self._wait_until_ready()

        self._current_ctx = new_ctx
        console.print(f"  [green]✓  Server restarted with ctx={new_ctx}[/green]")
        return new_ctx

    async def _stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.send_signal(signal.SIGTERM)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._proc.wait)
        self._proc = None

    async def _start(self, ctx_size: int):
        cmd = _inject_ctx_size(list(config.LLAMA_SERVER_CMD), ctx_size)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    async def _wait_until_ready(self, timeout: float = 60.0):
        """Poll /health until the server responds or timeout."""
        base = config.LLM_BASE_URL.rstrip("/").removesuffix("/v1")
        deadline = asyncio.get_event_loop().time() + timeout
        async with httpx.AsyncClient() as client:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    r = await client.get(f"{base}/health", timeout=2.0)
                    if r.status_code == 200:
                        return
                except Exception:
                    pass
                await asyncio.sleep(1.0)
        raise TimeoutError(f"llama-server did not become ready within {timeout}s")

    async def stop(self):
        await self._stop()


def _inject_ctx_size(cmd: list[str], ctx_size: int) -> list[str]:
    """Replace --ctx-size value in the command list, or append it if absent."""
    for i, arg in enumerate(cmd):
        if arg in ("--ctx-size", "-c") and i + 1 < len(cmd):
            cmd[i + 1] = str(ctx_size)
            return cmd
    return cmd + ["--ctx-size", str(ctx_size)]


# ── ContextManager — orchestrates both strategies ─────────────────────────────

class ContextManager:
    """
    Orchestrates context compression and server-side expansion.

    Call check_and_manage() after every completed turn to proactively
    keep context usage below the threshold. Also call handle_overflow()
    when a 400/exceed_context_size error is received mid-turn.
    """

    def __init__(self, llm: "LLMClient"):
        self._llm = llm
        self._server = ServerManager()
        self._ctx_size = config.CTX_SIZE

    async def check_and_manage(self, messages: list[dict]) -> list[dict]:
        """
        Check usage after a turn completes. If above threshold, compress.
        If still above after compression, expand server context.
        Returns the (possibly modified) message list.
        """
        ratio = usage_ratio(messages, self._ctx_size)
        if ratio < config.CTX_COMPRESS_THRESHOLD:
            return messages  # nothing to do

        # Stage 1: compress
        messages = await compress(messages, self._llm)
        ratio = usage_ratio(messages, self._ctx_size)

        # Stage 2: server expansion if still over threshold and server is managed
        if ratio >= config.CTX_COMPRESS_THRESHOLD:
            new_ctx = await self._server.expand()
            if new_ctx:
                self._ctx_size = new_ctx

        return messages

    async def handle_overflow(self, messages: list[dict]) -> list[dict]:
        """
        Called immediately when a context overflow error is received.
        More aggressive: compresses first, then expands if needed.
        Returns the updated message list ready to retry.
        """
        console.print(
            "  [bold red]Context overflow detected — compressing and retrying…[/bold red]"
        )
        messages = await compress(messages, self._llm)
        ratio = usage_ratio(messages, self._ctx_size)

        if ratio >= config.CTX_COMPRESS_THRESHOLD:
            new_ctx = await self._server.expand()
            if new_ctx:
                self._ctx_size = new_ctx

        return messages

    def current_ctx_size(self) -> int:
        return self._ctx_size

    async def stop_server(self):
        await self._server.stop()
