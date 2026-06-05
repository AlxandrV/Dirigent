import httpx

from dirigent.llm.base import BaseLLMClient, LLMResponse, Message


class OllamaClient(BaseLLMClient):
    """
    Connector for Ollama (http://localhost:11434 by default).
    Compatible with any model available via `ollama pull`.
    """

    def generate(self, prompt: str, system: str | None = None) -> LLMResponse:
        """Single-turn generation via /api/generate."""
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data: dict[str, object] = response.json()
            return LLMResponse(
                content=str(data.get("response", "")).strip(),
                model=self.model,
                prompt_tokens=int(str(data.get("prompt_eval_count", 0))),
                completion_tokens=int(str(data.get("eval_count", 0))),
            )
        except httpx.TimeoutException:
            return self.build_error("Timeout — model took too long to respond.")
        except httpx.HTTPStatusError as e:
            return self.build_error(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            return self.build_error(str(e))

    def chat(
        self,
        messages: list[Message],
        system: str | None = None,
    ) -> LLMResponse:
        """Multi-turn conversation via /api/chat."""
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if system:
            payload["system"] = system

        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data: dict[str, object] = response.json()
            message = data.get("message", {})
            content = str(message.get("content", "")).strip() if isinstance(message, dict) else ""
            return LLMResponse(
                content=content,
                model=self.model,
                prompt_tokens=int(str(data.get("prompt_eval_count", 0))),
                completion_tokens=int(str(data.get("eval_count", 0))),
            )
        except httpx.TimeoutException:
            return self.build_error("Timeout — model took too long to respond.")
        except httpx.HTTPStatusError as e:
            return self.build_error(f"HTTP error {e.response.status_code}: {e.response.text}")
        except Exception as e:
            return self.build_error(str(e))

    def is_available(self) -> bool:
        """Returns True if Ollama is reachable and responding."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
