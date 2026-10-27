"""Plain data types shared by the core and its adapters (no infrastructure imports)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""  # set by a wrapper when a different model than the primary answered (fallback)

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMBusy(Exception):
    """No generation slot became free in time (typical for a small local model)."""


@dataclass
class StoredChunk:
    id: str
    title: str
    text: str
    metadata: dict[str, str]
    lang: str = "es"
    block: str = ""
    translation_of: str | None = None


@dataclass
class Hit:
    chunk: StoredChunk
    score: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Request:
    """A channel-neutral question. Any front end (web, voice gateway, other systems) produces this."""

    user_id: str
    session_id: str
    text: str
    audience: str
    lang: str | None = None
    department: str = ""


STATUS_ANSWERED = "answered"
STATUS_UNKNOWN = "unknown"
STATUS_BLOCKED = "blocked"
STATUS_BUDGET = "budget_exceeded"
STATUS_BUSY = "busy"


@dataclass
class Response:
    message_id: str
    answer: str
    status: str
    sources: list[dict] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    model: str = ""
    prompt_version: str = ""
    latency_ms: int = 0
    usage: Usage = field(default_factory=Usage)
    raw_answer: str = ""  # exactly what the LLM returned; set whenever the LLM was called, logged for audit
