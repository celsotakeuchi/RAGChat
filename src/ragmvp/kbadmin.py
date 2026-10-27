"""Knowledge-base lifecycle, independent of the running service. Works on one KB or on the whole catalog.

  python -m ragmvp.kbadmin status                          # every KB in kb/: audience, live version, configured version
  python -m ragmvp.kbadmin build [--kb kb/x.yaml | --all]  # parse + embed into a new version (not live)
  python -m ragmvp.kbadmin list [--kb ... | --all]         # versions and which one is live
  python -m ragmvp.kbadmin promote [--kb ... | --all] [--version V] [--min-recall 0.9] [--skip-gate] [--note TEXT]
  python -m ragmvp.kbadmin rollback [--kb ...]             # go back to the previously live version
  python -m ragmvp.kbadmin retire --kb kb/archive/x.yaml   # take a KB out of service (it is no longer searched)
  python -m ragmvp.kbadmin history [--kb ... | --all]      # who promoted/rolled back what, with the gate result

Promotion runs the golden questions against the candidate version and only switches the live
alias if recall is high enough. The service keeps answering from the old version until then.
A KB without golden questions (no Block H, no `golden_file`) is promoted without a quality gate; that is recorded.
"""
from __future__ import annotations

import argparse
import getpass
import sys

from ragmvp.config import KBConfig, get_settings, load_catalog, load_kb_config
from ragmvp.core.ports import KBLifecyclePort
from ragmvp.evaluation import load_golden, retrieval_eval
from ragmvp.ingestion.build import build_version
from ragmvp.ingestion.parse_md import parse_eval_items
from ragmvp.ingestion.parse_pdf import IngestError
from ragmvp.retrieval.backends import RetrievalBackend, build_backend
from ragmvp.retrieval.rerank import wrap_with_reranker
from ragmvp.retrieval.store import VersionError


def cmd_build(kb: KBConfig, backend: RetrievalBackend) -> int:
    count = build_version(kb, backend.lifecycle, backend.embedder, warn=lambda w: print(f"  warning: {w}"))
    print(f"Built {kb.id} v{kb.version}: {count} chunks, embedded with {backend.embedder.name} (not live yet)")
    return 0


def cmd_list(kb: KBConfig, store: KBLifecyclePort) -> int:
    live = store.live_version(kb.id)
    print(f"KB {kb.id} [{kb.audience}] (live: {live or 'none'})")
    for v in store.versions(kb.id):
        mark = "*" if v["version"] == live else " "
        print(f" {mark} v{v['version']:<8} {v['status']:<8} {v['chunk_count']:>5} chunks  {v['embedding_model']}  {v['created_at']}")
    return 0


def cmd_status(catalog: list[KBConfig], store: KBLifecyclePort) -> int:
    print(f"{'KB':<28} {'audience':<9} {'department':<12} {'type':<4} {'live':<8} configured")
    for kb in catalog:
        live = store.live_version(kb.id) or "-"
        flag = "" if live == kb.version else "  <- not live"
        print(f"{kb.id:<28} {kb.audience:<9} {(kb.department or '-'):<12} {kb.source_type:<4} {live:<8} {kb.version}{flag}")
    return 0


def _gate_items(kb: KBConfig):
    if kb.golden_file:
        return load_golden(kb.resolve(kb.golden_file))
    if kb.source_type == "md":
        return parse_eval_items(kb.resolve(kb.source_file))
    return []


def cmd_promote(
    kb: KBConfig, retriever, store: KBLifecyclePort, version: str, min_recall: float, skip_gate: bool, actor: str, note: str
) -> int:
    gate = "skipped (--skip-gate)"
    if not skip_gate:
        items = _gate_items(kb)
        if not items:
            gate = "no golden questions: promoted without a quality gate"
            print(f"  note: {kb.id} has no golden questions; no quality gate")
        else:
            k = kb.retrieval.top_k
            _, summary = retrieval_eval(retriever, kb, items, k, version)
            failing = {lang: s[f"recall@{k}"] for lang, s in summary.items() if s[f"recall@{k}"] < min_recall}
            print(f"Gate on {kb.id} v{version}: {summary}")
            if failing:
                print(f"Refused: recall@{k} below {min_recall} for {failing}. The live version is unchanged.")
                return 1
            gate = f"passed: {summary}"
    store.promote(kb.id, version, actor=actor, gate=gate, note=note)
    print(f"{kb.id} v{version} is now live")
    return 0


def cmd_rollback(kb: KBConfig, store: KBLifecyclePort, actor: str) -> int:
    print(f"{kb.id} rolled back to v{store.rollback(kb.id, actor=actor)}")
    return 0


def cmd_retire(kb: KBConfig, store: KBLifecyclePort, actor: str, note: str) -> int:
    version = store.retire(kb.id, actor=actor, note=note)
    print(f"{kb.id} was not live" if version is None else f"{kb.id} v{version} retired: it is no longer searched")
    return 0


def cmd_history(kb: KBConfig, store) -> int:
    print(f"KB {kb.id}")
    for p in store.promotions(kb.id):
        print(f"  {p['promoted_at']} {p['action']:<8} v{p['version']:<8} by {p['actor'] or '?'}  {p['note']}  [{p['gate'][:80]}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kbadmin")
    parser.add_argument("--kb", default=None, help="path to a KB yaml (default: KB_CONFIG)")
    parser.add_argument("--all", action="store_true", help="apply to every KB of the catalog folder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("build")
    sub.add_parser("list")
    sub.add_parser("history")
    p = sub.add_parser("promote")
    p.add_argument("--version", default=None, help="default: the version in the KB yaml")
    p.add_argument("--min-recall", type=float, default=None)
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument("--actor", default=None, help="who is promoting (default: the OS user)")
    p.add_argument("--note", default="")
    p = sub.add_parser("rollback")
    p.add_argument("--actor", default=None)
    p = sub.add_parser("retire")
    p.add_argument("--actor", default=None)
    p.add_argument("--note", default="")
    args = parser.parse_args(argv)

    settings = get_settings()
    catalog = load_catalog()
    kbs = catalog if args.all or args.cmd == "status" else [load_kb_config(args.kb)]
    backend = build_backend(settings)
    store = backend.lifecycle
    actor = getattr(args, "actor", None) or getpass.getuser()
    failures = 0
    if args.cmd == "status":
        return cmd_status(catalog, store)
    for kb in kbs:
        retriever = wrap_with_reranker(
            backend.retriever, kb.retrieval.rerank, kb.retrieval.candidates, kb.retrieval.rerank_weight
        )
        try:
            if args.cmd == "build":
                failures += cmd_build(kb, backend)
            elif args.cmd == "list":
                failures += cmd_list(kb, store)
            elif args.cmd == "history":
                failures += cmd_history(kb, store)
            elif args.cmd == "promote":
                failures += cmd_promote(
                    kb, retriever, store, args.version or kb.version,
                    args.min_recall or kb.promotion.min_recall, args.skip_gate, actor, args.note,
                )
            elif args.cmd == "retire":
                failures += cmd_retire(kb, store, actor, args.note)
            else:
                failures += cmd_rollback(kb, store, actor)
        except (VersionError, IngestError, FileNotFoundError) as exc:
            print(f"Error ({kb.id}): {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
