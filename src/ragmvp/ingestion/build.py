"""Build a KB version from its source (Markdown file or folder of PDFs): parse, embed, write (not yet live)."""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np

from ragmvp.config import KBConfig
from ragmvp.core.models import StoredChunk
from ragmvp.core.ports import KBLifecyclePort
from ragmvp.embeddings import Embedder
from ragmvp.ingestion.parse_md import parse_chunks
from ragmvp.ingestion.parse_pdf import IngestError, parse_pdf_dir


def embedding_text(chunk: StoredChunk) -> str:
    return f"{chunk.title}\n{chunk.text}"


def load_chunks(kb: KBConfig, warn: Callable[[str], None] | None = None) -> list[StoredChunk]:
    if kb.source_type == "pdf":
        if not kb.source_dir:
            raise IngestError(f"KB {kb.id}: source_dir is required for source_type 'pdf'")
        parsed = parse_pdf_dir(kb.resolve(kb.source_dir), kb.chunk_chars, kb.chunk_overlap)
        for w in parsed.warnings:
            (warn or (lambda _: None))(w)
        chunks = parsed.chunks
    elif kb.source_type == "md":
        source = kb.resolve(kb.source_file)
        chunks = [
            StoredChunk(
                id=c.id,
                title=c.title,
                text=c.text,
                metadata={"documento": source.name, **c.metadata},
                block=c.block,
            )
            for c in parse_chunks(source)
        ]
    else:
        raise IngestError(f"KB {kb.id}: unknown source_type '{kb.source_type}'")
    for c in chunks:
        c.lang = kb.source_language
    return chunks


def _embed(store: KBLifecyclePort, embedder: Embedder, chunks: list[StoredChunk]) -> np.ndarray:
    """Embed the chunks, reusing vectors already computed for identical text with the same model."""
    texts = [embedding_text(c) for c in chunks]
    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in texts]
    cached = store.cached_embeddings(embedder.name, hashes) if hasattr(store, "cached_embeddings") else {}
    missing = [i for i, h in enumerate(hashes) if h not in cached]
    if missing:
        fresh = embedder.embed_documents([texts[i] for i in missing])
        new = {hashes[i]: fresh[n] for n, i in enumerate(missing)}
        if hasattr(store, "cache_embeddings"):
            store.cache_embeddings(embedder.name, new)
        cached = {**cached, **new}
    return np.vstack([cached[h] for h in hashes]) if hashes else np.zeros((0, 0), dtype=np.float32)


def build_version(
    kb: KBConfig, store: KBLifecyclePort, embedder: Embedder, warn: Callable[[str], None] | None = None
) -> int:
    chunks = load_chunks(kb, warn)
    if not chunks:
        raise IngestError(f"KB {kb.id}: the source produced no chunks")
    store.write_version(kb.id, kb.version, kb.audience, chunks, _embed(store, embedder, chunks), embedder.name)
    return len(chunks)
