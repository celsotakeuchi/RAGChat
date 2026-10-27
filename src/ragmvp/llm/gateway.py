"""LLM adapter through LiteLLM. Swap provider (Ollama, private cloud, commercial) by changing CHAT_MODEL.

Implements `ragmvp.core.ports.LLMPort`; the core never imports this module directly.
"""
from __future__ import annotations

from typing import Iterator

from ragmvp.core.models import Usage


class LiteLLMGateway:
    def __init__(self, model: str, temperature: float = 0.1):
        self.model = model
        self._temperature = temperature

    def stream(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> Iterator[str | Usage]:
        import litellm

        response = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        for part in response:
            usage = getattr(part, "usage", None)
            if usage:
                yield Usage(usage.prompt_tokens or 0, usage.completion_tokens or 0)
            if part.choices:
                delta = part.choices[0].delta.content
                if delta:
                    yield delta
