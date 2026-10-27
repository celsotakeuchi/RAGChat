"""Prompt-engineering layer: input guardrails, prompt/context assembly, output guardrails.

Everything here is driven by the KB config and prompt file (tone, rules, limits), so another
KB or customer changes configuration, not code.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ragmvp.config import KBConfig
from ragmvp.core.models import Hit
from ragmvp.core.text import extract_passage



@dataclass
class InputCheck:
    allowed: bool
    text: str
    reason: str = ""


@dataclass
class OutputCheck:
    cited: list[str]
    invalid: list[str]
    sources: list[dict]
    flags: list[str]


class PromptEngine:
    def __init__(self, kb: KBConfig):
        self.kb = kb
        self.system_prompt = kb.prompt_text()
        self.version = hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()[:8]
        self._blocked = [re.compile(p, re.I) for p in kb.guardrails.blocked_patterns]
        self._citation = re.compile(kb.guardrails.citation_pattern)

    def check_input(self, text: str) -> InputCheck:
        text = text.strip()
        if not text:
            return InputCheck(False, text, "empty")
        if len(text) > self.kb.guardrails.max_question_chars:
            return InputCheck(False, text, "too_long")
        if any(p.search(text) for p in self._blocked):
            return InputCheck(False, text, "blocked_pattern")
        return InputCheck(True, text)

    def select_context(self, hits: list[Hit]) -> list[Hit]:
        """Keep the best distinct sources: within the score margin, chunk count and size limits."""
        ctx = self.kb.context
        if not hits:
            return []
        best = max(h.score for h in hits)
        chosen: list[Hit] = []
        seen: set[str] = set()
        total = 0
        for h in hits:
            if h.chunk.id in seen:
                continue
            if ctx.score_margin and h.score < best - ctx.score_margin:
                continue
            if len(chosen) >= ctx.max_chunks:
                break
            size = min(len(h.chunk.text), ctx.max_chars_per_source)
            if chosen and total + size > ctx.max_total_chars:
                break
            seen.add(h.chunk.id)
            chosen.append(h)
            total += size
        return chosen

    def format_context(self, hits: list[Hit], question: str = "") -> str:
        limit = self.kb.context.max_chars_per_source
        blocks = []
        for h in hits:
            m = h.chunk.metadata
            meta = f"fiabilidad: {m['fiabilidad']} | revisar_antes_de: {m.get('revisar_antes_de', '')}\n" if m.get("fiabilidad") else ""
            if m.get("documento") and m.get("pagina"):
                meta = f"documento: {m['documento']} | página: {m['pagina']}\n"
            blocks.append(
                f"[{h.chunk.id}] {h.chunk.title}\n"
                f"{meta}"
                f"{extract_passage(h.chunk.text, question, limit)}"
            )
        return "\n\n".join(blocks)

    def build_messages(self, hits: list[Hit], question: str) -> list[dict[str, str]]:
        user = f"FUENTES:\n{self.format_context(hits, question)}\n\nPREGUNTA: {question}"
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]

    def check_output(self, answer: str, hits: list[Hit]) -> OutputCheck:
        allowed = {h.chunk.id for h in hits}
        cited: list[str] = []
        invalid: list[str] = []
        for cid in self._citation.findall(answer):
            bucket = cited if cid in allowed else invalid
            if cid not in bucket:
                bucket.append(cid)
        by_id = {h.chunk.id: h for h in hits}
        flags: list[str] = []
        if invalid:
            flags.append("invalid_citation")
        if self.kb.guardrails.require_citation and not cited:
            flags.append("uncited")
        return OutputCheck(cited, invalid, [self._source_entry(by_id[c]) for c in cited], flags)

    def _source_entry(self, hit: Hit) -> dict:
        m = hit.chunk.metadata
        document = m.get("documento") or hit.extra.get("kb_id") or self.kb.id
        return {
            "id": hit.chunk.id,
            "title": hit.chunk.title,
            "document": document,
            "page": m.get("pagina", ""),
            "reliability": m.get("fiabilidad", ""),
            "review_before": m.get("revisar_antes_de", ""),
            "references": [u.strip() for u in m.get("fuentes", "").split(" ; ") if u.strip()],
        }
