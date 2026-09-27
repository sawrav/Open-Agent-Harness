"""
llm.py — vLLM OpenAI-compatible async streaming client for OpenAgentHarness.

Uses httpx with raw SSE parsing against a vLLM OpenAI server
http://localhost:8000/v1/chat/completions for true token-by-token
streaming callbacks and native function calling. API is kept compatible
with the previous llama.cpp client so agent.py needs no changes.
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
    arguments: str = ""


@dataclass
class StreamResult:
    """What the LLM client returns after a full streaming turn."""
    content: str
    tool_calls: list[ToolCallAccumulator] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    """
    Async streaming client for vLLM OpenAI-compatible /v1/chat/completions.

    Yields (kind, token) tuples:
      - kind == "reasoning"  → delta.reasoning_content if model emits it
      - kind == "content"    → delta.content
      - kind == "error"      → connection/server error
    """

    def __init__(self):
        # vLLM OpenAI server default; override via config if needed
        base_url = getattr(config, "VLLM_BASE_URL", config.LLM_BASE_URL)
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=5.0),
        )
        self._last_result: StreamResult | None = None

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        model_name = getattr(config, "VLLM_MODEL", getattr(config, "LLM_MODEL", "local"))
        payload: dict = {
            "model": model_name,
            "messages": messages,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        # vLLM supports OpenAI style reasoning params for some models
        if budget_tokens is not None:
            if budget_tokens > 0:
                payload["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": budget_tokens}}
            else:
                payload["extra_body"] = {"thinking": {"type": "disabled"}}

        content_parts: list[str] = []
        tool_accumulators: dict[int, ToolCallAccumulator] = {}

        try:
            async with self._http.stream("POST", "/chat/completions", json=payload) as resp:
                if resp.status_code >= 400:
                    body = await resp.aread()
                    error_text = body.decode(errors="replace")[:500]
                    yield ("error", f"Server error {resp.status_code}: {error_text}")
                    self._last_result = StreamResult(content="", tool_calls=[])
                    return

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

                    # Reasoning token
                    reasoning_token = delta.get("reasoning_content")
                    if reasoning_token:
                        yield ("reasoning", reasoning_token)

                    # Content token
                    token = delta.get("content")
                    if token:
                        content_parts.append(token)
                        yield ("content", token)

                    # Tool call deltas
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
                f"Cannot connect to vLLM server at {self._http.base_url}. "
                "Start vLLM OpenAI server with:\n"
                f"  vllm serve {model_name} --port 8000"
            ))
            self._last_result = StreamResult(content="", tool_calls=[])
            return
        except httpx.ReadTimeout:
            yield ("error", "Request timed out — model overloaded or prompt too large.")
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
        return self._last_result

    async def aclose(self):
        await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()
