from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Base interface that every callable tool must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name the model uses to call this tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description shown to the model."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema object describing the tool's parameters."""
        ...

    @abstractmethod
    def run(self, **kwargs) -> str:
        """Execute the tool and return a string result."""
        ...

    def to_openai_schema(self) -> dict:
        """Convert this tool to the OpenAI function-calling schema format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
