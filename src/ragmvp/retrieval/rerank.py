"""Reranking: optional second pass over the vector candidates (reorders only; scores stay vector scores)."""
from __future__ import annotations

from ragmvp.core.models import Hit
from ragmvp.core.ports import RerankerPort, RetrieverPort
from ragmvp.core.text import tokens


class LexicalReranker:
    """Boosts hits that literally contain the question's words (names, products, numbers). No dependencies."""

    def __init__(self, weight: float = 0.1):
        self.weight = weight

    def rerank(self, question: str, hits: list[Hit]) -> list[Hit]:
        q = tokens(question)
        if not q:
            return hits

        def key(h: Hit) -> float:
            overlap = len(q & tokens(f"{h.chunk.title} {h.chunk.text}")) / len(q)
            return h.score + self.weight * overlap

        return sorted(hits, key=key, reverse=True)


class RerankingRetriever:
    """Wraps any retriever: fetch more candidates, rerank, keep top_k."""

    def __init__(self, inner: RetrieverPort, reranker: RerankerPort, candidates: int):
        self._inner = inner
        self._reranker = reranker
        self._candidates = candidates

    def retrieve(
        self, kb_id: str, audience: str, question: str, top_k: int, version: str | None = None
    ) -> list[Hit]:
        hits = self._inner.retrieve(kb_id, audience, question, max(self._candidates, top_k), version)
        return self._reranker.rerank(question, hits)[:top_k]

    def retrieve_scope(self, audiences: list[str], question: str, top_k: int) -> list[Hit]:
        hits = self._inner.retrieve_scope(audiences, question, max(self._candidates, top_k))
        return self._reranker.rerank(question, hits)[:top_k]


RERANKERS = {"lexical": LexicalReranker}


def wrap_with_reranker(retriever: RetrieverPort, name: str, candidates: int, weight: float) -> RetrieverPort:
    if name in ("", "none"):
        return retriever
    if name not in RERANKERS:
        raise ValueError(f"Unknown reranker '{name}'. Available: none, {', '.join(RERANKERS)}")
    return RerankingRetriever(retriever, RERANKERS[name](weight), candidates)
