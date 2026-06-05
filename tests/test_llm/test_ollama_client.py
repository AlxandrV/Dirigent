import json as json_lib

import httpx
import pytest
from pytest_mock import MockerFixture

from dirigent.llm.base import LLMConfig, Message
from dirigent.llm.ollama_client import OllamaClient

# --- Fixtures ---


@pytest.fixture
def config() -> LLMConfig:
    return LLMConfig(
        model="gemma4",
        base_url="http://localhost:11434",
        timeout=30,
    )


@pytest.fixture
def client(config: LLMConfig) -> OllamaClient:
    return OllamaClient(config)


# --- Helpers ---


def make_response(
    status_code: int, data: dict[str, object] | None = None, text: str = ""
) -> httpx.Response:
    request = httpx.Request("POST", "http://localhost:11434")
    if data is not None:
        content = json_lib.dumps(data).encode()
        headers = {"content-type": "application/json"}
    else:
        content = text.encode()
        headers = {"content-type": "text/plain"}
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers=headers,
        request=request,
    )


def make_generate_response(content: str = "hello") -> dict[str, object]:
    return {
        "response": content,
        "prompt_eval_count": 10,
        "eval_count": 5,
    }


def make_chat_response(content: str = "hello") -> dict[str, object]:
    return {
        "message": {"role": "assistant", "content": content},
        "prompt_eval_count": 10,
        "eval_count": 5,
    }


# --- generate() ---


class TestGenerate:
    def test_success(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch(
            "httpx.post", return_value=make_response(200, data=make_generate_response("ok"))
        )
        response = client.generate("write a test")
        assert response.success is True
        assert response.content == "ok"
        assert response.model == "gemma4"

    def test_prompt_tokens_parsed(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch("httpx.post", return_value=make_response(200, data=make_generate_response()))
        response = client.generate("prompt")
        assert response.prompt_tokens == 10
        assert response.completion_tokens == 5

    def test_with_system_prompt(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mock = mocker.patch(
            "httpx.post", return_value=make_response(200, data=make_generate_response())
        )
        client.generate("prompt", system="you are an assistant")
        payload = mock.call_args.kwargs["json"]
        assert payload["system"] == "you are an assistant"

    def test_without_system_prompt(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mock = mocker.patch(
            "httpx.post", return_value=make_response(200, data=make_generate_response())
        )
        client.generate("prompt")
        payload = mock.call_args.kwargs["json"]
        assert "system" not in payload

    def test_stream_is_false(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mock = mocker.patch(
            "httpx.post", return_value=make_response(200, data=make_generate_response())
        )
        client.generate("prompt")
        payload = mock.call_args.kwargs["json"]
        assert payload["stream"] is False

    def test_timeout_returns_error(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch("httpx.post", side_effect=httpx.TimeoutException("timeout"))
        response = client.generate("prompt")
        assert response.success is False
        assert "Timeout" in (response.error or "")

    def test_http_error_returns_error(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch("httpx.post", return_value=make_response(500, text="server error"))
        response = client.generate("prompt")
        assert response.success is False
        assert "500" in (response.error or "")

    def test_unexpected_exception_returns_error(
        self, client: OllamaClient, mocker: MockerFixture
    ) -> None:
        mocker.patch("httpx.post", side_effect=RuntimeError("unexpected"))
        response = client.generate("prompt")
        assert response.success is False
        assert "unexpected" in (response.error or "")

    def test_content_is_stripped(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch(
            "httpx.post", return_value=make_response(200, data=make_generate_response("  hello  "))
        )
        response = client.generate("prompt")
        assert response.content == "hello"


# --- chat() ---


class TestChat:
    def test_success(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch("httpx.post", return_value=make_response(200, data=make_chat_response("hi")))
        messages = [Message(role="user", content="hello")]
        response = client.chat(messages)
        assert response.success is True
        assert response.content == "hi"

    def test_messages_serialized(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mock = mocker.patch(
            "httpx.post", return_value=make_response(200, data=make_chat_response())
        )
        messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        client.chat(messages)
        payload = mock.call_args.kwargs["json"]
        assert payload["messages"] == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

    def test_with_system_prompt(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mock = mocker.patch(
            "httpx.post", return_value=make_response(200, data=make_chat_response())
        )
        client.chat([Message(role="user", content="hi")], system="be concise")
        payload = mock.call_args.kwargs["json"]
        assert payload["system"] == "be concise"

    def test_empty_messages(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch("httpx.post", return_value=make_response(200, data=make_chat_response()))
        response = client.chat([])
        assert isinstance(response.content, str)

    def test_timeout_returns_error(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch("httpx.post", side_effect=httpx.TimeoutException("timeout"))
        response = client.chat([Message(role="user", content="hi")])
        assert response.success is False
        assert "Timeout" in (response.error or "")

    def test_http_error_returns_error(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch("httpx.post", return_value=make_response(500, text="error"))
        response = client.chat([Message(role="user", content="hi")])
        assert response.success is False

    def test_unexpected_exception_returns_error(
        self, client: OllamaClient, mocker: MockerFixture
    ) -> None:
        mocker.patch("httpx.post", side_effect=RuntimeError("boom"))
        response = client.chat([Message(role="user", content="hi")])
        assert response.success is False

    def test_content_is_stripped(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch(
            "httpx.post", return_value=make_response(200, data=make_chat_response("  trimmed  "))
        )
        response = client.chat([Message(role="user", content="hi")])
        assert response.content == "trimmed"


# --- is_available() ---


class TestIsAvailable:
    def test_returns_true_when_reachable(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch("httpx.get", return_value=make_response(200))
        assert client.is_available() is True

    def test_returns_false_when_unreachable(
        self, client: OllamaClient, mocker: MockerFixture
    ) -> None:
        mocker.patch("httpx.get", side_effect=httpx.ConnectError("refused"))
        assert client.is_available() is False

    def test_returns_false_on_non_200(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch("httpx.get", return_value=make_response(503))
        assert client.is_available() is False

    def test_returns_false_on_timeout(self, client: OllamaClient, mocker: MockerFixture) -> None:
        mocker.patch("httpx.get", side_effect=httpx.TimeoutException("timeout"))
        assert client.is_available() is False
