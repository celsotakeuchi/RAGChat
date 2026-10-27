"""Retrieval backend registry: pick the vector store + retrieval model with RETRIEVER_BACKEND.

A backend provides the lifecycle store (build/promote/rollback) and a retriever for the core.
Add one (pgvector, a managed vector DB...) by calling `register_backend` with a builder; the core,
API and `kbadmin` do not change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ragmvp.config import Settings
from ragmvp.core.ports import KBLifecyclePort, RetrieverPort
from ragmvp.embeddings import Embedder, build_embedder


@dataclass
class RetrievalBackend:
    lifecycle: KBLifecyclePort
    retriever: RetrieverPort
    embedder: Embedder


Builder = Callable[[Settings], RetrievalBackend]
BACKENDS: dict[str, Builder] = {}


def register_backend(name: str, builder: Builder) -> None:
    BACKENDS[name] = builder


def _sqlite(settings: Settings) -> RetrievalBackend:
    from ragmvp.retrieval.search import Retriever
    from ragmvp.retrieval.store import KBStore

    store = KBStore(settings.kb_db_file)
    embedder = build_embedder(settings)
    return RetrievalBackend(store, Retriever(store, embedder), embedder)


register_backend("sqlite", _sqlite)


def build_backend(settings: Settings) -> RetrievalBackend:
    try:
        builder = BACKENDS[settings.retriever_backend]
    except KeyError:
        raise ValueError(
            f"Unknown RETRIEVER_BACKEND '{settings.retriever_backend}'. Available: {', '.join(BACKENDS)}"
        ) from None
    return builder(settings)
