"""Ports: the only interfaces the core depends on. Adapters implement them and can be swapped freely."""
from __future__ import annotations

from typing import Iterator, Protocol

from ragmvp.core.models import Hit, Request, Response, Usage


class LLMPort(Protocol):
    """Text generation. Swap Ollama, a private cloud model or a commercial API behind this."""

    model: str

    def stream(self, messages: list[dict[str, str]], max_tokens: int | None = None) -> Iterator[str | Usage]:
        """Yield text pieces; optionally a final Usage if the provider reports it."""
        ...


class RetrieverPort(Protocol):
    """Finds the knowledge-base chunks relevant to a question (live KB version unless specified)."""

    def retrieve(
        self, kb_id: str, audience: str, question: str, top_k: int, version: str | None = None
    ) -> list[Hit]: ...

    def retrieve_scope(self, audiences: list[str], question: str, top_k: int) -> list[Hit]:
        """Search every live KB whose audience is in `audiences`."""
        ...


class RerankerPort(Protocol):
    """Reorders retrieved hits for a question. Must keep each hit's original (vector) score."""

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]: ...


class KBLifecyclePort(Protocol):
    """Writing, versioning and promoting knowledge-base versions (used by `kbadmin`, not by the core)."""

    def write_version(self, kb_id, version, audience, chunks, embeddings, embedding_model) -> None: ...

    def live_version(self, kb_id: str) -> str | None: ...

    def versions(self, kb_id: str) -> list[dict]: ...

    def promote(self, kb_id: str, version: str) -> None: ...

    def rollback(self, kb_id: str) -> str: ...

    def retire(self, kb_id: str, actor: str = "", note: str = "") -> str | None: ...


class OpsPort(Protocol):
    """Operational records: interaction log, feedback and token usage."""

    def log_interaction(self, request: Request, response: Response, kb_id: str, kb_version: str) -> None: ...

    def record_usage(self, user_id: str, model: str, usage: Usage) -> None: ...

    def tokens_used_today(self, user_id: str) -> int: ...

    def tokens_used_this_month(self) -> int: ...
