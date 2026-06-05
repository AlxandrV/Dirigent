import pytest
from typing import Optional
from dirigent.llm.base import BaseLLMClient, LLMConfig, LLMResponse, Message


# --- Minimal concrete implementation for testing ---


class ConcreteClient(BaseLLMClient):
    """Minimal implementation used to test BaseLLMClient behavior."""

    def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
        return LLMResponse(content=f"response to: {prompt}", model=self.model)

    def chat(self, messages: list[Message], system: Optional[str] = None) -> LLMResponse:
        last = messages[-1].content if messages else ""
        return LLMResponse(content=f"chat response to: {last}", model=self.model)

    def is_available(self) -> bool:
        return True


# --- Fixtures ---


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(
        model="test-model",
        base_url="http://localhost:11434/",
        timeout=30,
    )


@pytest.fixture
def client(config: LLMConfig) -> ConcreteClient:
    return ConcreteClient(config)


# --- LLMConfig ---


class TestLLMConfig:
    def test_default_timeout(self) -> None:
        config = LLMConfig(model="m", base_url="http://localhost")
        assert config.timeout == 120

    def test_extra_defaults_to_empty_dict(self) -> None:
        config = LLMConfig(model="m", base_url="http://localhost")
        assert config.extra == {}

    def test_custom_values(self) -> None:
        config = LLMConfig(model="gemma4", base_url="http://host:11434", timeout=60)
        assert config.model == "gemma4"
        assert config.timeout == 60


# --- LLMResponse ---


class TestLLMResponse:
    def test_defaults_to_success(self) -> None:
        r = LLMResponse(content="ok", model="m")
        assert r.success is True
        assert r.error is None
        assert r.prompt_tokens == 0
        assert r.completion_tokens == 0

    def test_failed_response(self) -> None:
        r = LLMResponse(content="", model="m", success=False, error="timeout")
        assert r.success is False
        assert r.error == "timeout"


# --- Message ---


class TestMessage:
    def test_message_fields(self) -> None:
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"


# --- BaseLLMClient ---


class TestBaseLLMClient:
    def test_base_url_trailing_slash_stripped(self, config: LLMConfig) -> None:
        client = ConcreteClient(config)
        assert not client.base_url.endswith("/")

    def test_model_and_timeout_set(self, client: ConcreteClient) -> None:
        assert client.model == "test-model"
        assert client.timeout == 30

    def test_generate_returns_response(self, client: ConcreteClient) -> None:
        response = client.generate("write a function")
        assert isinstance(response, LLMResponse)
        assert response.success is True
        assert "write a function" in response.content

    def test_chat_returns_response(self, client: ConcreteClient) -> None:
        messages = [Message(role="user", content="hello")]
        response = client.chat(messages)
        assert isinstance(response, LLMResponse)
        assert response.success is True

    def test_chat_uses_last_message(self, client: ConcreteClient) -> None:
        messages = [
            Message(role="user", content="first"),
            Message(role="assistant", content="ok"),
            Message(role="user", content="last"),
        ]
        response = client.chat(messages)
        assert "last" in response.content

    def test_is_available_returns_bool(self, client: ConcreteClient) -> None:
        assert client.is_available() is True

    def test_build_error_returns_failed_response(self, client: ConcreteClient) -> None:
        response = client.build_error("connection refused")
        assert response.success is False
        assert response.error == "connection refused"
        assert response.content == ""
        assert response.model == "test-model"

    def test_chat_empty_messages(self, client: ConcreteClient) -> None:
        response = client.chat([])
        assert isinstance(response, LLMResponse)

    def test_generate_with_system_prompt(self, client: ConcreteClient) -> None:
        response = client.generate("prompt", system="you are an assistant")
        assert isinstance(response, LLMResponse)

    def test_chat_with_system_prompt(self, client: ConcreteClient) -> None:
        messages = [Message(role="user", content="hi")]
        response = client.chat(messages, system="you are an assistant")
        assert isinstance(response, LLMResponse)


# --- Abstract enforcement ---


class TestAbstractEnforcement:
    def test_cannot_instantiate_base_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseLLMClient(LLMConfig(model="m", base_url="http://localhost"))  # type: ignore[abstract]

    def test_partial_implementation_raises(self) -> None:
        class Partial(BaseLLMClient):
            def generate(self, prompt: str, system: Optional[str] = None) -> LLMResponse:
                return LLMResponse(content="", model=self.model)

        with pytest.raises(TypeError):
            Partial(LLMConfig(model="m", base_url="http://localhost"))  # type: ignore[abstract]
