"""Knowledge-base storage: versioned chunks + embeddings, promotion history and the live alias.

Separate from operational data (see ragmvp.ops.store) so the KB can be rebuilt, promoted or
rolled back independently of the running service. `KBStore` is the adapter behind the retriever;
a pgvector implementation can replace it with the same methods.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

from ragmvp.core.models import Hit, StoredChunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    kb_id TEXT NOT NULL,
    version TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    audience TEXT NOT NULL,
    lang TEXT NOT NULL,
    translation_of TEXT,
    title TEXT NOT NULL,
    block TEXT,
    text TEXT NOT NULL,
    metadata TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (kb_id, version, chunk_id, lang)
);
CREATE TABLE IF NOT EXISTS kb_versions (
    kb_id TEXT NOT NULL,
    version TEXT NOT NULL,
    status TEXT NOT NULL,              -- built | live | retired
    embedding_model TEXT NOT NULL,
    chunk_count INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (kb_id, version)
);
CREATE TABLE IF NOT EXISTS kb_aliases (
    kb_id TEXT PRIMARY KEY,
    version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embedding_cache (
    model TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    embedding BLOB NOT NULL,
    PRIMARY KEY (model, text_hash)
);
CREATE TABLE IF NOT EXISTS kb_promotions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kb_id TEXT NOT NULL,
    version TEXT NOT NULL,
    promoted_at TEXT NOT NULL DEFAULT (datetime('now')),
    action TEXT NOT NULL DEFAULT 'promote',   -- promote | rollback
    actor TEXT NOT NULL DEFAULT '',
    gate TEXT NOT NULL DEFAULT '',
    note TEXT NOT NULL DEFAULT ''
);
"""

PROMOTION_MIGRATIONS = {"action": "'promote'", "actor": "''", "gate": "''", "note": "''"}


class VersionError(Exception):
    pass


class _Block:
    """All chunks of one live KB version, kept in memory as a matrix so a query is one matrix product."""

    def __init__(self, kb_id: str, version: str, rows: list[sqlite3.Row]):
        self.kb_id, self.version = kb_id, version
        self.matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        self.chunks = [
            StoredChunk(
                id=r["chunk_id"],
                title=r["title"],
                text=r["text"],
                metadata=json.loads(r["metadata"]),
                lang=r["lang"],
                block=r["block"] or "",
                translation_of=r["translation_of"],
            )
            for r in rows
        ]


class KBStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], _Block] = {}
        self._cache_lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(kb_promotions)")}
            for name, default in PROMOTION_MIGRATIONS.items():
                if name not in columns:
                    self._conn.execute(f"ALTER TABLE kb_promotions ADD COLUMN {name} TEXT NOT NULL DEFAULT {default}")

    # --- build -----------------------------------------------------------------------------
    def _check_unique_ids(self, kb_id: str, ids: set[str]) -> None:
        """A cited id must identify one chunk across every live KB, so another KB may not reuse it."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT c.kb_id, c.chunk_id FROM chunks c JOIN kb_aliases a "
                "ON a.kb_id=c.kb_id AND a.version=c.version WHERE c.kb_id != ?",
                (kb_id,),
            ).fetchall()
        clashes = sorted(f"{r['chunk_id']} (in {r['kb_id']})" for r in rows if r["chunk_id"] in ids)
        if clashes:
            raise VersionError(f"chunk ids already used by another live KB: {', '.join(clashes[:5])}")

    def write_version(
        self,
        kb_id: str,
        version: str,
        audience: str,
        chunks: list[StoredChunk],
        embeddings: np.ndarray,
        embedding_model: str,
    ) -> None:
        """Write a new (not yet live) version. A live version is immutable: bump the version instead."""
        if self.live_version(kb_id) == version:
            raise VersionError(f"{kb_id} v{version} is live; build a new version number instead")
        self._check_unique_ids(kb_id, {c.id for c in chunks})
        seen: set[tuple[str, str]] = set()
        for c in chunks:
            if (c.id, c.lang) in seen:
                raise VersionError(f"duplicate chunk id {c.id} in {kb_id} v{version}")
            seen.add((c.id, c.lang))
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks WHERE kb_id=? AND version=?", (kb_id, version))
            self._conn.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        kb_id,
                        version,
                        c.id,
                        audience,
                        c.lang,
                        c.translation_of,
                        c.title,
                        c.block,
                        c.text,
                        json.dumps(c.metadata, ensure_ascii=False),
                        embedding_model,
                        embeddings[i].astype(np.float32).tobytes(),
                    )
                    for i, c in enumerate(chunks)
                ],
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO kb_versions (kb_id, version, status, embedding_model, chunk_count) "
                "VALUES (?,?,?,?,?)",
                (kb_id, version, "built", embedding_model, len(chunks)),
            )

    # --- lifecycle -------------------------------------------------------------------------
    def live_version(self, kb_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT version FROM kb_aliases WHERE kb_id=?", (kb_id,)).fetchone()
        return row["version"] if row else None

    def versions(self, kb_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT version, status, embedding_model, chunk_count, created_at FROM kb_versions "
                "WHERE kb_id=? ORDER BY created_at, version",
                (kb_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def promote(
        self, kb_id: str, version: str, actor: str = "", gate: str = "", note: str = "", action: str = "promote"
    ) -> None:
        """Make `version` the one the retriever serves. The previous live version is kept for rollback."""
        with self._lock:
            ids = {
                r["chunk_id"]
                for r in self._conn.execute("SELECT chunk_id FROM chunks WHERE kb_id=? AND version=?", (kb_id, version))
            }
        self._check_unique_ids(kb_id, ids)
        with self._lock, self._conn:
            if not self._conn.execute(
                "SELECT 1 FROM kb_versions WHERE kb_id=? AND version=?", (kb_id, version)
            ).fetchone():
                raise VersionError(f"{kb_id} v{version} has not been built")
            self._conn.execute(
                "UPDATE kb_versions SET status='retired' WHERE kb_id=? AND status='live'", (kb_id,)
            )
            self._conn.execute(
                "UPDATE kb_versions SET status='live' WHERE kb_id=? AND version=?", (kb_id, version)
            )
            self._conn.execute(
                "INSERT INTO kb_aliases (kb_id, version) VALUES (?, ?) "
                "ON CONFLICT(kb_id) DO UPDATE SET version=excluded.version",
                (kb_id, version),
            )
            self._conn.execute(
                "INSERT INTO kb_promotions (kb_id, version, action, actor, gate, note) VALUES (?,?,?,?,?,?)",
                (kb_id, version, action, actor, gate, note),
            )

    def retire(self, kb_id: str, actor: str = "", note: str = "") -> str | None:
        """Take a KB out of service: it stops being searched. Versions stay stored; `promote` brings it back."""
        current = self.live_version(kb_id)
        if current is None:
            return None
        with self._lock, self._conn:
            self._conn.execute("UPDATE kb_versions SET status='retired' WHERE kb_id=? AND status='live'", (kb_id,))
            self._conn.execute("DELETE FROM kb_aliases WHERE kb_id=?", (kb_id,))
            self._conn.execute(
                "INSERT INTO kb_promotions (kb_id, version, action, actor, gate, note) VALUES (?,?,?,?,?,?)",
                (kb_id, current, "retire", actor, "", note),
            )
        with self._cache_lock:
            for stale in [k for k in self._cache if k[0] == kb_id]:
                del self._cache[stale]
        return current

    def promotions(self, kb_id: str | None = None, limit: int = 20) -> list[dict]:
        where, params = ("WHERE kb_id=?", (kb_id,)) if kb_id else ("", ())
        with self._lock:
            rows = self._conn.execute(
                f"SELECT kb_id, version, action, actor, gate, note, promoted_at FROM kb_promotions {where} "
                "ORDER BY id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def rollback(self, kb_id: str, actor: str = "", note: str = "") -> str:
        """Re-promote the version that was live before the current one."""
        current = self.live_version(kb_id)
        with self._lock:
            rows = self._conn.execute(
                "SELECT version FROM kb_promotions WHERE kb_id=? ORDER BY id DESC", (kb_id,)
            ).fetchall()
        previous = next((r["version"] for r in rows if r["version"] != current), None)
        if previous is None:
            raise VersionError(f"no earlier version of {kb_id} to roll back to")
        self.promote(kb_id, previous, actor=actor, note=note or f"rollback from {current}", action="rollback")
        return previous

    def embedding_model(self, kb_id: str, version: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT embedding_model FROM kb_versions WHERE kb_id=? AND version=?", (kb_id, version)
            ).fetchone()
        return row["embedding_model"] if row else None

    # --- embedding cache (rebuilds only embed new or changed text) ---------------------------------
    def cached_embeddings(self, model: str, hashes: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        unique = list(set(hashes))
        with self._lock:
            for i in range(0, len(unique), 500):
                batch = unique[i : i + 500]
                rows = self._conn.execute(
                    f"SELECT text_hash, embedding FROM embedding_cache WHERE model=? "
                    f"AND text_hash IN ({','.join('?' * len(batch))})",
                    (model, *batch),
                ).fetchall()
                found.update({r["text_hash"]: np.frombuffer(r["embedding"], dtype=np.float32) for r in rows})
        return found

    def cache_embeddings(self, model: str, vectors: dict[str, np.ndarray]) -> None:
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO embedding_cache (model, text_hash, embedding) VALUES (?,?,?)",
                [(model, h, v.astype(np.float32).tobytes()) for h, v in vectors.items()],
            )

    # --- read ------------------------------------------------------------------------------
    def search(
        self, kb_id: str, audience: str, query: np.ndarray, top_k: int, version: str | None = None
    ) -> list[Hit]:
        version = version or self.live_version(kb_id)
        if version is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE kb_id=? AND version=? AND audience=?",
                (kb_id, version, audience),
            ).fetchall()
        if not rows:
            return []
        matrix = np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
        scores = matrix @ query.astype(np.float32)
        hits = []
        for i in np.argsort(-scores)[:top_k]:
            r = rows[int(i)]
            chunk = StoredChunk(
                id=r["chunk_id"],
                title=r["title"],
                text=r["text"],
                metadata=json.loads(r["metadata"]),
                lang=r["lang"],
                block=r["block"] or "",
                translation_of=r["translation_of"],
            )
            hits.append(Hit(chunk=chunk, score=float(scores[int(i)])))
        return hits

    # --- cross-KB read (cached) ----------------------------------------------------------------
    def live_targets(self, audiences: list[str]) -> list[dict]:
        """Live (kb_id, version, embedding_model) of every KB whose audience is in `audiences`."""
        marks = ",".join("?" * len(audiences))
        with self._lock:
            rows = self._conn.execute(
                "SELECT a.kb_id, a.version, v.embedding_model FROM kb_aliases a "
                "JOIN kb_versions v ON v.kb_id=a.kb_id AND v.version=a.version "
                f"WHERE EXISTS (SELECT 1 FROM chunks c WHERE c.kb_id=a.kb_id AND c.version=a.version "
                f"AND c.audience IN ({marks}))",
                tuple(audiences),
            ).fetchall()
        return [dict(r) for r in rows]

    def _block(self, kb_id: str, version: str) -> _Block | None:
        key = (kb_id, version)
        with self._cache_lock:
            block = self._cache.get(key)
        if block is not None:
            return block
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE kb_id=? AND version=?", (kb_id, version)
            ).fetchall()
        if not rows:
            return None
        block = _Block(kb_id, version, rows)
        with self._cache_lock:
            # a promotion makes older versions of this KB unreachable: drop them from memory
            for stale in [k for k in self._cache if k[0] == kb_id and k[1] != version]:
                del self._cache[stale]
            self._cache[key] = block
        return block

    def search_scope(self, audiences: list[str], query: np.ndarray, top_k: int) -> list[Hit]:
        """Best chunks across all live KBs of the given audiences. Vectors are loaded once per live version."""
        query = query.astype(np.float32)
        scored: list[tuple[float, _Block, int]] = []
        for target in self.live_targets(audiences):
            block = self._block(target["kb_id"], target["version"])
            if block is None:
                continue
            scores = block.matrix @ query
            for i in np.argsort(-scores)[:top_k]:
                scored.append((float(scores[int(i)]), block, int(i)))
        scored.sort(key=lambda t: -t[0])
        return [
            Hit(chunk=block.chunks[i], score=score, extra={"kb_id": block.kb_id, "version": block.version})
            for score, block, i in scored[:top_k]
        ]

    def close(self) -> None:
        self._conn.close()
