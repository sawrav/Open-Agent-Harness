"""
agent.py — Async agentic loop with live token streaming.

Tokens are printed to the terminal the instant llama.cpp emits them.
Tool calls are accumulated from streaming deltas, executed concurrently
(when multiple tools are called in one turn), then results are fed back
for the next LLM turn — all within a single asyncio event loop.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from rich.console import Console
from rich.markdown import Markdown

import config
from llm import LLMClient, ToolCallAccumulator
from registry import ToolRegistry
from context_manager import ContextManager

console = Console()

COMMANDS = {
    "/tools":       "List all available callable tools",
    "/skills":      "List all loaded Markdown skills",
    "/mode <name>": "Switch thinking mode: fast | medium | slow",
    "/compact":     "Manually compact memory and erase KV cache",
    "/clear":       "Clear the conversation history (keeps system prompt)",
    "/help":        "Show this help message",
    "/exit":        "Quit the agent",
}


class Agent:
    """
    Async agent that streams LLM tokens directly to stdout and executes
    tool calls asynchronously between streaming turns.
    """

    def __init__(self):
        self.llm = LLMClient()
        self.registry = ToolRegistry()
        self.ctx_manager = ContextManager(self.llm)
        self.messages: list[dict] = []
        self._mode: str = config.DEFAULT_THINKING_MODE
        self._reset_messages()

        console.print(
            f"\n[bold green]OpenAgentHarness ready.[/bold green]\n"
            f"  [dim]Tools :[/dim]  [cyan]{', '.join(self.registry.list_tools()) or 'none'}[/cyan]\n"
            f"  [dim]Skills:[/dim]  [magenta]{', '.join(self.registry.list_skills()) or 'none'}[/magenta]\n"
            f"  [dim]Mode  :[/dim]  [yellow]{self._mode}[/yellow] "
            f"[dim]({config.THINKING_MODES[self._mode]} budget tokens)[/dim]\n"
            f"  [dim]Type [bold]/help[/bold] for available commands.[/dim]\n"
        )

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        parts = [
            config.SYSTEM_PROMPT,
            f"Current working directory: {os.getcwd()}",
            f"Callable tools: {', '.join(self.registry.list_tools()) or 'none'}",
        ]
        skill_block = self.registry.skill_prompt_block()
        if skill_block:
            parts.append("\n" + skill_block)
        return "\n".join(parts)

    def _reset_messages(self):
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ]

    # ── Core async loop ───────────────────────────────────────────────────────

    async def run_once(self, user_input: str) -> None:
        """
        Process one user turn end-to-end:
          1. Stream the LLM response token-by-token to stdout.
          2. If the model called tools, run them all concurrently.
          3. Feed results back and repeat until the model returns plain text.

        Caps at MAX_TOOL_ROUNDS tool-call rounds per turn to break circular loops.
        """
        MAX_TOOL_ROUNDS = 10
        tool_rounds = 0

        self.messages.append({"role": "user", "content": user_input})

        while True:
            printed_anything = False
            in_reasoning = False

            # ── Stream tokens live ─────────────────────────────────────────────
            # Use raw sys.stdout throughout — rich buffers per-line and will
            # swallow the visual separation between thinking and content blocks.
            async for kind, token in self.llm.stream(
                self.messages,
                tools=self.registry.all_schemas(),
                budget_tokens=config.THINKING_MODES[self._mode],
            ):
                if kind == "error":
                    # Server-side error — print it and abort this turn cleanly
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                    # Detect context overflow (400 exceed_context_size_error)
                    if "exceed_context_size" in token or "exceeds the available context" in token:
                        console.print(f"[red]Error:[/red] {token}")
                        # Pop the user message before compressing so it can be retried
                        if self.messages and self.messages[-1]["role"] == "user":
                            user_msg = self.messages.pop()
                        else:
                            user_msg = None
                        self.messages = await self.ctx_manager.handle_overflow(self.messages)
                        # Re-append user message and retry the turn
                        if user_msg:
                            self.messages.append(user_msg)
                        continue  # retry the while True loop
                    else:
                        console.print(f"[red]Error:[/red] {token}")
                        if self.messages and self.messages[-1]["role"] == "user":
                            self.messages.pop()
                        return
                elif kind == "reasoning":
                    if not in_reasoning:
                        sys.stdout.write("\n\033[2m<thinking>\n")  # dim on
                        in_reasoning = True
                    sys.stdout.write(token)
                    sys.stdout.flush()
                else:  # "content"
                    if in_reasoning:
                        sys.stdout.write("\n</thinking>\033[0m\n\n")  # dim off
                        sys.stdout.flush()
                        in_reasoning = False
                    sys.stdout.write(token)
                    sys.stdout.flush()
                printed_anything = True

            # Guard: close thinking block if stream ends mid-reasoning
            if in_reasoning:
                sys.stdout.write("\n</thinking>\033[0m\n\n")
                sys.stdout.flush()

            result = self.llm.last_result

            if printed_anything:
                sys.stdout.write("\n")
                sys.stdout.flush()

            if result is None:
                break

            # Record the assistant turn in message history
            assistant_msg = self._build_assistant_message(result)
            self.messages.append(assistant_msg)

            # ── No tool calls → done ──────────────────────────────────────────
            if not result.has_tool_calls:
                # Update token counters and trigger auto-compaction if needed
                self.messages = await self.ctx_manager.after_turn(self.messages)
                break

            # ── Execute tool calls concurrently ──────────────────────────────
            tool_rounds += 1
            if tool_rounds >= MAX_TOOL_ROUNDS:
                console.print(
                    f"  [yellow]⚠  Reached {MAX_TOOL_ROUNDS} tool-call rounds — "
                    f"nudging model to respond.[/yellow]"
                )
                # Inject a system nudge as a user message to break the loop
                self.messages.append({
                    "role": "user",
                    "content": (
                        "You have now gathered enough information. "
                        "Stop calling tools and provide your final response directly."
                    ),
                })
                # One final LLM call with no tools offered
                async for kind, token in self.llm.stream(self.messages, tools=None):
                    if kind in ("content", "reasoning"):
                        sys.stdout.write(token)
                        sys.stdout.flush()
                sys.stdout.write("\n")
                sys.stdout.flush()
                self.messages = await self.ctx_manager.after_turn(self.messages)
                return

            tool_results = await self._execute_tool_calls(result.tool_calls)
            self.messages.extend(tool_results)

    async def _execute_tool_calls(
        self, tool_calls: list[ToolCallAccumulator]
    ) -> list[dict]:
        """Run all tool calls concurrently and return tool result messages."""

        async def run_one(tc: ToolCallAccumulator) -> dict:
            # Guard: skip tool calls with incomplete delta accumulation
            if not tc.name or not tc.id:
                console.print(f"  [red]⚠  Skipping malformed tool call (missing name or id)[/red]")
                return {
                    "role": "tool",
                    "tool_call_id": tc.id or "unknown",
                    "content": "Error: tool call was malformed (missing name or id) and was not executed.",
                }

            try:
                fn_args = json.loads(tc.arguments) if tc.arguments.strip() else {}
            except json.JSONDecodeError:
                console.print(f"  [red]⚠  Could not parse arguments for {tc.name}: {tc.arguments!r}[/red]")
                fn_args = {}

            # Guard: check required parameters are present before dispatching
            tool = self.registry.get(tc.name)
            if tool:
                required = tool.parameters.get("required", [])
                missing = [p for p in required if p not in fn_args]
                if missing:
                    msg = f"Error: missing required parameter(s): {', '.join(missing)}"
                    console.print(f"  [red]⚠  {tc.name}: {msg}[/red]")
                    return {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": msg,
                    }

            console.print(f"  [dim]⚙  {tc.name}({fn_args})[/dim]")
            result = await self.registry.run(tc.name, **fn_args)

            return {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            }

        return list(await asyncio.gather(*[run_one(tc) for tc in tool_calls]))

    @staticmethod
    def _build_assistant_message(result) -> dict:
        """Convert a StreamResult into the assistant message dict for history."""
        msg: dict = {"role": "assistant", "content": result.content}
        if result.has_tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in result.tool_calls
            ]
        return msg

    # ── REPL ──────────────────────────────────────────────────────────────────

    async def chat(self):
        """Async interactive REPL — reads input in a thread to stay non-blocking."""
        loop = asyncio.get_event_loop()

        while True:
            try:
                # Read input without blocking the event loop
                user_input = await loop.run_in_executor(
                    None, lambda: console.input("[bold]You:[/bold] ")
                )
                user_input = user_input.strip()

                if not user_input:
                    continue

                if user_input.lower() in ("/exit", "/quit"):
                    console.print("[yellow]Bye.[/yellow]")
                    break

                elif user_input.lower() == "/tools":
                    tools = self.registry.list_tools()
                    console.print(f"[cyan]Tools:[/cyan] {', '.join(tools) if tools else 'none'}")
                    continue

                elif user_input.lower() == "/skills":
                    skills = self.registry.list_skills()
                    console.print(f"[magenta]Skills:[/magenta] {', '.join(skills) if skills else 'none'}")
                    continue

                elif user_input.lower().startswith("/mode"):
                    parts = user_input.strip().split()
                    if len(parts) == 1:
                        # Print current mode and available options
                        console.print(
                            f"[yellow]Current mode:[/yellow] [bold]{self._mode}[/bold] "
                            f"[dim]({config.THINKING_MODES[self._mode]} budget tokens)[/dim]"
                        )
                        console.print("[dim]Available modes:[/dim]")
                        for name, tokens in config.THINKING_MODES.items():
                            marker = " ◀" if name == self._mode else ""
                            console.print(f"  [bold]{name}[/bold]  {tokens} budget tokens{marker}")
                    else:
                        requested = parts[1].lower()
                        if requested not in config.THINKING_MODES:
                            valid = ", ".join(config.THINKING_MODES.keys())
                            console.print(f"[red]Unknown mode '{requested}'. Valid modes: {valid}[/red]")
                        else:
                            self._mode = requested
                            budget = config.THINKING_MODES[self._mode]
                            console.print(
                                f"[yellow]Mode → [bold]{self._mode}[/bold][/yellow] "
                                f"[dim]({budget} budget tokens)[/dim]"
                            )
                    continue

                elif user_input.lower() == "/clear":
                    self._reset_messages()
                    console.print("[dim]Conversation cleared.[/dim]")
                    continue

                elif user_input.lower() == "/compact":
                    self.messages = await self.ctx_manager.compact(
                        self.messages, reason="manual"
                    )
                    continue

                elif user_input.lower() == "/help":
                    for cmd, desc in COMMANDS.items():
                        console.print(f"  [bold]{cmd}[/bold]  {desc}")
                    console.print(
                        f"\n  [dim]Current mode:[/dim] [bold yellow]{self._mode}[/bold yellow] "
                        f"[dim]({config.THINKING_MODES[self._mode]} budget tokens)[/dim]"
                    )
                    continue

                await self.run_once(user_input)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[yellow]Bye.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")

    async def aclose(self):
        await self.llm.aclose()
