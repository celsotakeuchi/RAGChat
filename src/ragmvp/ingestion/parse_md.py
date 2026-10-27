"""Parse the Markdown knowledge base into chunks and extract the evaluation table."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING = re.compile(r"^## \[?([A-Z]+-\d{2,3})\]?\s+(.+?)\s*$")
BLOCK = re.compile(r"^# BLOQUE ([A-Z])\.")
META = re.compile(r"^- ([a-z_]+):\s*(.*)$")
ID_RE = re.compile(r"[A-Z]{3}-\d{3}")
EVAL_BLOCK = "H"
META_KEYS = ("categoria", "producto", "fiabilidad", "fuentes", "revisar_antes_de")


@dataclass
class Chunk:
    id: str
    title: str
    block: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class EvalItem:
    question: str
    expected_ids: list[str]
    must_abstain: bool
    lang: str = "es"


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    block = ""
    current: Chunk | None = None
    body: list[str] = []

    def close() -> None:
        nonlocal current, body
        if current is not None:
            current.text = "\n".join(body).strip()
            chunks.append(current)
        current, body = None, []

    for line in _read_lines(path):
        m_block = BLOCK.match(line)
        if m_block:
            close()
            block = m_block.group(1)
            continue
        if block == EVAL_BLOCK:
            continue
        m_head = HEADING.match(line)
        if m_head:
            close()
            current = Chunk(id=m_head.group(1), title=m_head.group(2), block=block, text="")
            continue
        if line.strip() == "---":
            close()
            continue
        if current is None:
            continue
        m_meta = META.match(line)
        if m_meta and m_meta.group(1) in META_KEYS and not any(b.strip() for b in body):
            current.metadata[m_meta.group(1)] = m_meta.group(2).strip()
        else:
            body.append(line)
    close()
    return chunks


def parse_eval_items(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    in_eval = False
    for line in _read_lines(path):
        m_block = BLOCK.match(line)
        if m_block:
            in_eval = m_block.group(1) == EVAL_BLOCK
            continue
        if not in_eval or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0].lower() == "pregunta" or set(cells[0]) <= {"-", " "}:
            continue
        expected = cells[1]
        items.append(
            EvalItem(
                question=cells[0],
                expected_ids=ID_RE.findall(expected),
                must_abstain=re.search(r"no (lo )?sabe", expected.lower()) is not None,
            )
        )
    return items
