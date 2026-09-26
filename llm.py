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

    Usage:
        client = LLMClient()
        async for token in client.stream(messages, tools):
            print(token, end="", flush=True)
        result = await client.last_result()   # StreamResult after stream is done
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
        Yield (kind, token) tuples as they arrive from the server:
          - kind == "reasoning"  → token from <thinking> block (delta.reasoning_content)
          - kind == "content"    → token from the final response (delta.content)

        budget_tokens controls how many tokens the model may spend thinking:
          - None  → no thinking field sent (model default)
          - 0     → thinking disabled
          - N > 0 → model may think for up to N tokens

        After the iterator is exhausted, .last_result holds the full StreamResult
        including any accumulated tool calls.
        """
        payload: dict = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # llama.cpp thinking budget — passed as top-level thinking object
        # mirroring the Anthropic extended-thinking API shape that llama.cpp adopts.
        if budget_tokens is not None:
            if budget_tokens > 0:
                payload["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": budget_tokens,
                }
            else:
                payload["thinking"] = {"type": "disabled"}

        content_parts: list[str] = []
        # Map from index → accumulator for multi-tool-call responses
        tool_accumulators: dict[int, ToolCallAccumulator] = {}

        async with self._http.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()

            async for raw_line in resp.aiter_lines():
                # SSE lines are "data: <json>" or "data: [DONE]"
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

                # ── Reasoning token (<thinking>) ──────────────────────────────
                # llama.cpp emits thinking tokens in delta.reasoning_content,
                # separate from the final response in delta.content.
                reasoning_token = delta.get("reasoning_content")
                if reasoning_token:
                    yield ("reasoning", reasoning_token)

                # ── Text token ────────────────────────────────────────────────
                token = delta.get("content")
                if token:
                    content_parts.append(token)
                    yield ("content", token)

                # ── Tool call delta ───────────────────────────────────────────
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
