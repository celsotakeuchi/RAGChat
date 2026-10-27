"""Evaluation: golden set from Block H, retrieval metrics, optional generation checks.

Usage:
  python eval/run_eval.py                 # retrieval only (no LLM key needed)
  python eval/run_eval.py --generate      # also call the LLM: citation validity + abstention
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ragmvp.config import load_kb_config  # noqa: E402
from ragmvp.evaluation import retrieval_eval  # noqa: E402
from ragmvp.ingestion.parse_md import parse_eval_items  # noqa: E402

ABSTAIN_RE = re.compile(r"no (lo )?s[eé]\b|no (he )?encontrad|no (hay|dispongo|cuento|tengo)|desconozc", re.I)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    from ragmvp.factory import build_orchestrator
    from ragmvp.core.models import Request

    kb = load_kb_config()
    core = build_orchestrator(kb=kb)
    items = parse_eval_items(kb.resolve(kb.source_file))
    rows, summary = retrieval_eval(core.retriever, kb, items, args.k)

    if args.generate:
        by_q = {r["question"]: r for r in rows}
        for item in items:
            res = core.answer(Request("eval", "eval", item.question, kb.audience, item.lang))
            row = by_q[item.question]
            row["abstained"] = res.status != "answered" or bool(ABSTAIN_RE.search(res.answer))
            row["invalid_citations"] = res.invalid_citations
            row["has_citation"] = bool(res.cited)
            row["answer"] = res.answer
        for lang, s in summary.items():
            rs = [r for r in rows if r["lang"] == lang]
            ans = [r for r in rs if not r["must_abstain"]]
            must = [r for r in rs if r["must_abstain"]]
            s["abstention_correct"] = round(sum(r["abstained"] for r in must) / len(must), 3) if must else None
            s["false_abstention"] = round(sum(r["abstained"] for r in ans) / len(ans), 3) if ans else None
            s["citation_present"] = round(sum(r["has_citation"] for r in ans) / max(len(ans), 1), 3)
            s["invalid_citations"] = sum(len(r["invalid_citations"]) for r in rs)

    print(f"\nKB={kb.id} v{kb.version} k={args.k}\n")
    print(f"{'rank':>4}  {'score':>6}  {'expected':<18} question")
    for r in rows:
        print(f"{str(r['rank'] or '-'):>4}  {str(r['best_score']):>6}  {','.join(r['expected_ids']):<18} {r['question']}")
    print("\nSummary per language:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    out_dir = ROOT / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(
        json.dumps({"kb": kb.id, "version": kb.version, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
