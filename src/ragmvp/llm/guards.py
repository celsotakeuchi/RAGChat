"""LLM wrappers that add operational behaviour around any LLMPort."""
from __future__ import annotations

import logging
import threading
from typing import Iterator

from ragmvp.core.models import LLMBusy, Usage
from ragmvp.core.ports import LLMPort

log = logging.getLogger("ragmvp")


class LimitedLLM:
    """Caps simultaneous generations (a small local model) and waits a bounded time for a free slot."""

    def __init__(self, inner: LLMPort, max_concurrent: int, wait_s: float):
        self._inner = inner
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._wait_s = wait_s
        self.model = inner.model

    def stream(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> Iterator[str | Usage]:
        if not self._slots.acquire(timeout=self._wait_s):
            raise LLMBusy(self.model)
        try:
            yield from self._inner.stream(messages, max_tokens)
        finally:
            self._slots.release()


class FallbackLLM:
    """Uses the secondary model only if the primary fails before producing any text."""

    def __init__(self, primary: LLMPort, secondary: LLMPort):
        self._primary = primary
        self._secondary = secondary
        self.model = primary.model

    def stream(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> Iterator[str | Usage]:
        started = False
        try:
            for piece in self._primary.stream(messages, max_tokens):
                if isinstance(piece, str):
                    started = True
                yield piece
            return
        except LLMBusy:
            raise
        except Exception:
            if started:
                raise
            log.warning("primary LLM %s failed, using fallback %s", self._primary.model, self._secondary.model, exc_info=True)
        for piece in self._secondary.stream(messages, max_tokens):
            if isinstance(piece, Usage):
                piece = Usage(piece.prompt_tokens, piece.completion_tokens, self._secondary.model)
            yield piece
