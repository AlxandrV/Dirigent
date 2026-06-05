from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Message:
    """Represents a single message in a conversation."""

    role: str  # "user" | "assistant" | "system"
    content: str


@dataclass
class LLMResponse:
    """Standardized response returned by any LLM backend."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = True
    error: Optional[str] = None


@dataclass
class LLMConfig:
    """Configuration for a LLM backend."""

    model: str
    base_url: str
    timeout: int = 120
    extra: dict = field(default_factory=dict)  # type: ignore[type-arg]


class BaseLLMClient(ABC):
    """
    Common interface for all LLM backends.

    Adding a new backend means:
    - Inherit from this class
    - Implement generate(), chat(), is_available()
    - Register it in config.yaml

    Nothing else changes.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.model = config.model
        self.base_url = config.base_url.rstrip("/")
        self.timeout = config.timeout

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        """
        Single-turn generation.
        One prompt in, one response out.
        """
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
    ) -> LLMResponse:
        """
        Multi-turn conversation.
        Takes the full message history, returns the next response.
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """
        Health check — returns True if the backend is reachable.
        Called before starting any task.
        """
        raise NotImplementedError

    def build_error(self, error: str) -> LLMResponse:
        """Builds a failed LLMResponse. Available to all subclasses."""
        return LLMResponse(
            content="",
            model=self.model,
            success=False,
            error=error,
        )
