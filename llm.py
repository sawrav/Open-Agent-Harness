"""
llm.py — Async streaming LLM client for llama.cpp.

Uses httpx with raw SSE parsing instead of the OpenAI SDK so that tokens
are yielded the instant llama.cpp emits them, with no buffering layer in
between. Tool-call deltas are accumulated here and returned as a structured
list once the stream is complete.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

import config


@dataclass
class ToolCallAccumulator:
    """Accumulates streamed tool-call delta fragments into a complete call."""
    index: int
    id: str = ""
    name: str = ""
    arguments: str = ""  # built up character by character from deltas


@dataclass
class StreamResult:
    """What the LLM client returns after a full streaming turn."""
    content: str                          # full assistant text (may be empty)
    tool_calls: list[ToolCallAccumulator] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """
    Async streaming client for llama.cpp's OpenAI-compatible /v1/chat/completions.

    Yields (kind, token) tuples:
      - kind == "reasoning"  → delta.reasoning_content (<thinking> block)
      - kind == "content"    → delta.content (final response)
      - kind == "error"      → connection or server error, session stays alive
    """

    def __init__(self):
        self._http = httpx.AsyncClient(
            base_url=config.LLM_BASE_URL,
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=5.0),
        )
        self._last_result: StreamResult | None = None

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        budget_tokens: int | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """
        Yield (kind, token) tuples as they arrive from the server.

        budget_tokens controls how many tokens the model may spend thinking:
          - None  → no thinking field sent (model default)
          - 0     → thinking disabled
          - N > 0 → model may think for up to N tokens

        After the iterator is exhausted, .last_result holds the full StreamResult.
        """
        payload: dict = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        if budget_tokens is not None:
            if budget_tokens > 0:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
            else:
                payload["thinking"] = {"type": "disabled"}

        content_parts: list[str] = []
        tool_accumulators: dict[int, ToolCallAccumulator] = {}

        try:
            async with self._http.stream("POST", "/chat/completions", json=payload) as resp:

                # ── HTTP error ────────────────────────────────────────────────
                if resp.status_code >= 400:
                    body = await resp.aread()
                    error_text = body.decode(errors="replace")[:500]
                    yield ("error", f"Server error {resp.status_code}: {error_text}")
                    self._last_result = StreamResult(content="", tool_calls=[])
                    return

                # ── SSE stream ────────────────────────────────────────────────
                async for raw_line in resp.aiter_lines():
                    if not raw_line.startswith("data:"):
                        continue

                    data = raw_line[len("data:"):].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices")
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    # Reasoning token — delta.reasoning_content
                    reasoning_token = delta.get("reasoning_content")
                    if reasoning_token:
                        yield ("reasoning", reasoning_token)

                    # Content token — delta.content
                    token = delta.get("content")
                    if token:
                        content_parts.append(token)
                        yield ("content", token)

                    # Tool call delta — accumulate across chunks
                    for tc_delta in delta.get("tool_calls", []):
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_accumulators:
                            tool_accumulators[idx] = ToolCallAccumulator(index=idx)

                        acc = tool_accumulators[idx]

                        if tc_delta.get("id"):
                            acc.id += tc_delta["id"]

                        fn = tc_delta.get("function", {})
                        if fn.get("name"):
                            acc.name += fn["name"]
                        if fn.get("arguments"):
                            acc.arguments += fn["arguments"]

        except httpx.ConnectError:
            yield ("error", (
                f"Cannot connect to llama-server at {config.LLM_BASE_URL}. "
                "Is the server running? Start it with:\n"
                "  llama-server --model <model.gguf> --ctx-size 16384 "
                "--n-gpu-layers 99 --port 8080"
            ))
            self._last_result = StreamResult(content="", tool_calls=[])
            return

        except httpx.ReadTimeout:
            yield ("error", "Request timed out — the model may be overloaded or the prompt too large.")
            self._last_result = StreamResult(content="", tool_calls=[])
            return

        except Exception as e:
            yield ("error", f"Unexpected error: {e}")
            self._last_result = StreamResult(content="", tool_calls=[])
            return

        self._last_result = StreamResult(
            content="".join(content_parts),
            tool_calls=list(tool_accumulators.values()),
        )

    @property
    def last_result(self) -> StreamResult | None:
        """The StreamResult from the most recent stream() call."""
        return self._last_result

    async def aclose(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()
