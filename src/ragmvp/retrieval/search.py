"""Retriever adapter: embeds the question and searches a KB version (live unless specified)."""
from __future__ import annotations

from ragmvp.core.models import Hit
from ragmvp.embeddings import Embedder
from ragmvp.retrieval.store import KBStore


class Retriever:
    def __init__(self, store: KBStore, embedder: Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(
        self, kb_id: str, audience: str, question: str, top_k: int, version: str | None = None
    ) -> list[Hit]:
        version = version or self.store.live_version(kb_id)
        if version is None:
            return []
        indexed_with = self.store.embedding_model(kb_id, version)
        if indexed_with and indexed_with != self.embedder.name:
            raise RuntimeError(
                f"KB '{kb_id}' v{version} was embedded with '{indexed_with}' but the query uses "
                f"'{self.embedder.name}'. Rebuild the KB or change the embedding settings."
            )
        return self.store.search(kb_id, audience, self.embedder.embed_query(question), top_k, version)

    def retrieve_scope(self, audiences: list[str], question: str, top_k: int) -> list[Hit]:
        for t in self.store.live_targets(audiences):
            if t["embedding_model"] != self.embedder.name:
                raise RuntimeError(
                    f"KB '{t['kb_id']}' v{t['version']} was embedded with '{t['embedding_model']}' but the query "
                    f"uses '{self.embedder.name}'. Rebuild the KB or change the embedding settings."
                )
        return self.store.search_scope(audiences, self.embedder.embed_query(question), top_k)
