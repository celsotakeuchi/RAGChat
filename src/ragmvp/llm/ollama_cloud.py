"""LLM adapter for Ollama: the cloud API (https://ollama.com, needs a key) or a local server. Implements `LLMPort`."""
from __future__ import annotations

from typing import Iterator

from ragmvp.core.models import Usage


class OllamaGateway:
    def __init__(
        self,
        model: str,
        api_key: str = "",
        host: str = "https://ollama.com",
        temperature: float = 0.1,
        timeout: float | None = None,
    ):
        if not api_key and "ollama.com" in host:
            raise ValueError("OLLAMA_API_KEY is not set")
        from ollama import Client

        self.model = model
        self._temperature = temperature
        headers = {"Authorization": "Bearer " + api_key} if api_key else None
        self._client = Client(host=host, headers=headers, timeout=timeout)

    def stream(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> Iterator[str | Usage]:
        options: dict = {"temperature": self._temperature}
        if max_tokens:
            options["num_predict"] = max_tokens
        usage = Usage()
        for part in self._client.chat(model=self.model, messages=messages, stream=True, options=options):
            if part.message and part.message.content:
                yield part.message.content
            if part.done:
                usage = Usage(part.prompt_eval_count or 0, part.eval_count or 0)
        yield usage


OllamaCloudGateway = OllamaGateway
