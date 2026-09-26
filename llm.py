from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
import config


class LLMClient:
    """Thin wrapper around the llama.cpp OpenAI-compatible server."""

    def __init__(self):
        self.client = OpenAI(
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> ChatCompletionMessage:
        """
        Send messages to the LLM and return the assistant's response message.

        If tools are provided, the model may respond with tool_calls instead
        of plain text content — the caller is responsible for handling both cases.
        """
        kwargs: dict = {
            "model": config.LLM_MODEL,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message
