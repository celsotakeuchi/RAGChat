"""The core: takes a channel-neutral Request and returns a Response.

Flow: input guardrails -> retrieval -> relevance check -> token budget -> LLM -> output guardrails -> log.
It depends only on ports (LLMPort, RetrieverPort, OpsPort), never on HTTP, a UI or a specific provider.

An audience `Profile` bundles what differs between internal and public users: behaviour (prompt,
guardrails, messages), which KB audiences may be searched, the model and the token limits.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Iterator

from ragmvp.config import KBConfig
from ragmvp.core.budget import BudgetExceeded, TokenBudget, estimate_tokens
from ragmvp.core.models import (
    STATUS_ANSWERED,
    STATUS_BLOCKED,
    STATUS_BUDGET,
    STATUS_BUSY,
    STATUS_UNKNOWN,
    LLMBusy,
    Request,
    Response,
    Usage,
)
from ragmvp.core.ports import LLMPort, OpsPort, RetrieverPort
from ragmvp.core.prompting import PromptEngine


@dataclass
class Profile:
    name: str
    kb: KBConfig  # behaviour: prompt, guardrails, retrieval, UI, messages
    prompts: PromptEngine
    llm: LLMPort
    budget: TokenBudget
    search_audiences: list[str] | None = None  # None = the single KB `kb.id` (legacy single-KB mode)


class Orchestrator:
    def __init__(
        self,
        kb: KBConfig,
        retriever: RetrieverPort,
        llm: LLMPort,
        ops: OpsPort,
        budget: TokenBudget,
        prompts: PromptEngine | None = None,
        profiles: dict[str, Profile] | None = None,
    ):
        self.kb = kb
        self.retriever = retriever
        self.llm = llm
        self.ops = ops
        self.budget = budget
        self.prompts = prompts or PromptEngine(kb)
        self.profiles = profiles or {}
        self._default = Profile("default", kb, self.prompts, llm, budget)

    def profile_for(self, audience: str) -> Profile:
        """With profiles configured, an unknown audience is an error (never fall back to a wider one)."""
        if not self.profiles:
            return self._default
        try:
            return self.profiles[audience]
        except KeyError:
            raise PermissionError(f"no profile for audience '{audience}'") from None

    def handle(self, req: Request) -> Iterator[str | Response]:
        """Yield answer text pieces, then the final Response."""
        start = time.perf_counter()
        profile = self.profile_for(req.audience)
        kb, prompts, llm, budget = profile.kb, profile.prompts, profile.llm, profile.budget
        lang = req.lang or kb.source_language
        message_id = str(uuid.uuid4())

        def finish(response: Response) -> Response:
            response.latency_ms = int((time.perf_counter() - start) * 1000)
            response.prompt_version = prompts.version
            response.model = response.model or llm.model
            self.ops.log_interaction(req, response, kb.id, kb.version)
            return response

        def refuse(status: str, message_key: str, flag: str, retrieved: list[dict] | None = None):
            text = kb.message(message_key, lang)
            yield text
            yield finish(Response(message_id, text, status, retrieved=retrieved or [], flags=[flag]))

        check = prompts.check_input(req.text)
        if not check.allowed:
            yield from refuse(STATUS_BLOCKED, "blocked", f"input_{check.reason}")
            return

        if profile.search_audiences is None:
            hits = self.retriever.retrieve(kb.id, req.audience, check.text, kb.retrieval.top_k)
        else:
            hits = self.retriever.retrieve_scope(profile.search_audiences, check.text, kb.retrieval.top_k)
        retrieved = [
            {"id": h.chunk.id, "score": round(h.score, 4), **({"kb": h.extra["kb_id"]} if "kb_id" in h.extra else {})}
            for h in hits
        ]
        if not hits or max(h.score for h in hits) < kb.retrieval.min_score:
            yield from refuse(STATUS_UNKNOWN, "unknown", "below_threshold", retrieved)
            return

        context = prompts.select_context(hits)
        messages = prompts.build_messages(context, check.text)
        estimated_input = sum(estimate_tokens(m["content"]) for m in messages)
        try:
            max_output = budget.check(req.user_id, estimated_input)
        except BudgetExceeded as exc:
            yield from refuse(STATUS_BUDGET, "budget", f"budget_{exc.reason}", retrieved)
            return

        parts: list[str] = []
        usage = Usage()
        strict = kb.guardrails.enforce_grounding
        try:
            for piece in llm.stream(messages, max_tokens=max_output):
                if isinstance(piece, Usage):
                    usage = piece
                else:
                    parts.append(piece)
                    if not strict:
                        yield piece
        except LLMBusy:
            if parts:
                raise
            yield from refuse(STATUS_BUSY, "busy", "llm_busy", retrieved)
            return
        answer = "".join(parts)
        if usage.total == 0:
            usage = Usage(estimated_input, estimate_tokens(answer), usage.model)
        used_model = usage.model or llm.model
        budget.record(req.user_id, used_model, usage)

        out = prompts.check_output(answer, context)
        if strict and ("uncited" in out.flags or "invalid_citation" in out.flags):
            text = kb.message("unknown", lang)
            yield text
            yield finish(
                Response(
                    message_id=message_id,
                    answer=text,
                    status=STATUS_UNKNOWN,
                    retrieved=retrieved,
                    invalid_citations=out.invalid,
                    flags=["ungrounded", *out.flags],
                    usage=usage,
                    raw_answer=answer,
                    model=used_model,
                )
            )
            return
        if strict:
            yield answer
        yield finish(
            Response(
                message_id=message_id,
                answer=answer,
                status=STATUS_ANSWERED,
                sources=out.sources,
                retrieved=retrieved,
                cited=out.cited,
                invalid_citations=out.invalid,
                flags=out.flags,
                usage=usage,
                raw_answer=answer,
                model=used_model,
            )
        )

    def answer(self, req: Request) -> Response:
        """Non-streaming convenience (used by evaluation and tests)."""
        result = None
        for item in self.handle(req):
            if isinstance(item, Response):
                result = item
        assert result is not None
        return result
