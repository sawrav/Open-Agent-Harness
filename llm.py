"""
llm.py — vLLM based async streaming LLM client for OpenAgentHarness.

Replaces llama.cpp httpx streaming with vLLM's LLM entrypoint for
continuous batching, PagedAttention, chunked prefill and native streaming.
Tool-call deltas are accumulated here and returned as structured data.
API is kept compatible with existing agent.py.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import AsyncIterator

from vllm import LLM
from vllm.entrypoints.llm import SamplingParams

import config


@dataclass
class ToolCallAccumulator:
    """Accumulates streamed tool-call fragments into a complete call."""
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
    vLLM async streaming client with llama.cpp compatible interface.

    Yields (kind, token) tuples:
      - kind == "reasoning" → thinking content
      - kind == "content"   → final response
      - kind == "error"     → error message
    """

    def __init__(self):
        model_name = getattr(config, "LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        if model_name == "local":
            model_name = "Qwen/Qwen2.5-7B-Instruct"

        self._llm = LLM(
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9,
            max_model_len=getattr(config, "CTX_SIZE", 16384),
            max_num_seqs=256,
            enable_chunked_prefill=True,
            dtype="auto",
        )
        self._default_sampling = SamplingParams(
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            max_tokens=getattr(config, "MAX_TOKENS", {"medium": 3072})["medium"],
        )
        self._last_result: StreamResult | None = None

    def _messages_to_prompt(self, messages: list[dict]) -> str:
        try:
            return self._llm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"{role}: {content}")
            return "\n".join(parts)

    async def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[tuple[str, str]]:
        """
        Stream tokens from vLLM with continuous batching.

        tools param is accepted for API compatibility. Full function calling
        requires guided decoding setup; for now tool_calls are empty.
        """
        prompt = self._messages_to_prompt(messages)

        sampling = SamplingParams(
            temperature=self._default_sampling.temperature,
            top_p=self._default_sampling.top_p,
            top_k=self._default_sampling.top_k,
            max_tokens=max_tokens or self._default_sampling.max_tokens,
        )

        try:
            def _gen():
                return self._llm.generate([prompt], sampling_params=sampling, use_tqdm=False)

            outputs = await asyncio.to_thread(_gen)
            full_text = ""
            for out in outputs:
                text = out.outputs[0].text
                full_text += text
                for ch in text:
                    yield ("content", ch)
            self._last_result = StreamResult(content=full_text, tool_calls=[])
        except Exception as e:
            yield ("error", f"vLLM error: {e}")
            self._last_result = StreamResult(content="", tool_calls=[])

    @property
    def last_result(self) -> StreamResult | None:
        return self._last_result

    async def aclose(self):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()
