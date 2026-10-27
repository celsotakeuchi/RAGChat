"""Retrieval evaluation against the golden questions; used by the eval script and the promotion gate."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from ragmvp.config import KBConfig
from ragmvp.core.ports import RetrieverPort
from ragmvp.ingestion.parse_md import EvalItem


def _matches(chunk_id: str, expected: list[str]) -> bool:
    """Exact id, or a document code (`MANUAL-VACACIONES` matches `MANUAL-VACACIONES-P3-1`)."""
    return any(chunk_id == e or chunk_id.startswith(e + "-P") for e in expected)


def load_golden(path: Path) -> list[EvalItem]:
    """Golden questions for KBs without a Block H: a YAML list of {question, expected: [id or DOCCODE], lang?}."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    return [
        EvalItem(
            question=d["question"],
            expected_ids=list(d.get("expected", [])),
            must_abstain=not d.get("expected"),
            lang=d.get("lang", "es"),
        )
        for d in data
    ]


def retrieval_eval(
    retriever: RetrieverPort, kb: KBConfig, items: list[EvalItem], k: int = 5, version: str | None = None
) -> tuple[list[dict], dict[str, dict]]:
    rows: list[dict] = []
    per_lang: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        hits = retriever.retrieve(kb.id, kb.audience, item.question, k, version)
        rank = next((i for i, h in enumerate(hits, 1) if _matches(h.chunk.id, item.expected_ids)), None)
        row = {
            "question": item.question,
            "expected_ids": item.expected_ids,
            "must_abstain": item.must_abstain,
            "lang": item.lang,
            "rank": rank,
            "top": [h.chunk.id for h in hits],
            "best_score": round(hits[0].score, 3) if hits else None,
        }
        rows.append(row)
        per_lang[item.lang].append(row)
    summary = {}
    for lang, all_rows in per_lang.items():
        rs = [r for r in all_rows if not r["must_abstain"]]  # abstention questions have no expected chunk
        n = len(rs) or 1
        summary[lang] = {
            "n": len(all_rows),
            f"recall@{k}": round(sum(1 for r in rs if r["rank"]) / n, 3),
            "recall@1": round(sum(1 for r in rs if r["rank"] == 1) / n, 3),
            "mrr": round(sum(1 / r["rank"] for r in rs if r["rank"]) / n, 3),
        }
    return rows, summary
