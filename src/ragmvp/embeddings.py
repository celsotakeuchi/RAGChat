"""Embedding backends: local (fastembed, no API key) or any LiteLLM embedding model."""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Protocol

import numpy as np

from ragmvp.config import Settings, get_settings


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.clip(norms, 1e-12, None)).astype(np.float32)


class LocalEmbedder:
    def __init__(self, model_name: str):
        from fastembed import TextEmbedding

        self.name = f"local:{model_name}"
        self._e5 = "e5" in model_name.lower()
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        prefix = "passage: " if self._e5 else ""
        return _normalize(np.array(list(self._model.embed([prefix + t for t in texts]))))

    def embed_query(self, text: str) -> np.ndarray:
        prefix = "query: " if self._e5 else ""
        return _normalize(np.array(list(self._model.embed([prefix + text]))))[0]


class LiteLLMEmbedder:
    def __init__(self, model_name: str):
        self.name = f"litellm:{model_name}"
        self._model = model_name

    def _call(self, texts: list[str]) -> np.ndarray:
        import litellm

        resp = litellm.embedding(model=self._model, input=texts)
        return _normalize(np.array([d["embedding"] for d in resp["data"]]))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._call(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._call([text])[0]


EMBEDDERS: dict[str, Callable[[Settings], Embedder]] = {
    "local": lambda s: LocalEmbedder(s.local_embedding_model),
    "litellm": lambda s: LiteLLMEmbedder(s.embedding_model),
}


def register_embedder(name: str, builder: Callable[[Settings], Embedder]) -> None:
    EMBEDDERS[name] = builder


def build_embedder(settings: Settings | None = None) -> Embedder:
    s = settings or get_settings()
    try:
        return EMBEDDERS[s.embedding_backend](s)
    except KeyError:
        raise ValueError(f"Unknown EMBEDDING_BACKEND '{s.embedding_backend}'. Available: {', '.join(EMBEDDERS)}") from None


@lru_cache
def default_embedder() -> Embedder:
    return build_embedder()
