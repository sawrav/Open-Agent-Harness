import json
import os

from rich.console import Console
from rich.markdown import Markdown

import config
from llm import LLMClient
from registry import ToolRegistry

console = Console()

COMMANDS = {
    "/tools":  "List all available callable tools",
    "/skills": "List all loaded Markdown skills",
    "/clear":  "Clear the conversation history (keeps system prompt)",
    "/help":   "Show this help message",
    "/exit":   "Quit the agent",
}


class Agent:
    """
    Core agentic loop.

    - Sends user messages to the LLM.
    - Detects tool calls in the response and routes them through the ToolRegistry.
    - Loops until the model returns a plain text response (no more tool calls).
    - Skills from SKILL.md files are injected into the system prompt at startup.
    """

    def __init__(self):
        self.llm = LLMClient()
        self.registry = ToolRegistry()
        self.messages: list[dict] = []
        self._reset_messages()

        console.print(
            f"\n[bold green]OpenAgentHarness ready.[/bold green]\n"
            f"  [dim]Tools :[/dim]  [cyan]{', '.join(self.registry.list_tools()) or 'none'}[/cyan]\n"
            f"  [dim]Skills:[/dim]  [magenta]{', '.join(self.registry.list_skills()) or 'none'}[/magenta]\n"
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

    # ── Core loop ─────────────────────────────────────────────────────────────

    def run_once(self, user_input: str) -> str:
        """
        Process a single user turn. May invoke tools multiple times before
        returning the model's final plain-text response.
        """
        self.messages.append({"role": "user", "content": user_input})

        while True:
            msg = self.llm.chat(self.messages, tools=self.registry.all_schemas())
            self.messages.append(msg)

            # Model is done — return its text response
            if not msg.tool_calls:
                return msg.content or ""

            # Execute every tool call in this turn
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                console.print(f"  [dim]⚙  {fn_name}({fn_args})[/dim]")

                result = self.registry.run(fn_name, **fn_args)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    # ── REPL ──────────────────────────────────────────────────────────────────

    def chat(self):
        """Start the interactive REPL."""
        while True:
            try:
                user_input = console.input("[bold]You:[/bold] ").strip()

                if not user_input:
                    continue

                # Built-in slash commands
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

                elif user_input.lower() == "/clear":
                    self._reset_messages()
                    console.print("[dim]Conversation cleared.[/dim]")
                    continue

                elif user_input.lower() == "/help":
                    for cmd, desc in COMMANDS.items():
                        console.print(f"  [bold]{cmd}[/bold]  {desc}")
                    continue

                response = self.run_once(user_input)
                console.print(Markdown(response))

            except KeyboardInterrupt:
                console.print("\n[yellow]Bye.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")
