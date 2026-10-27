"""PDF ingestion: one folder of PDFs becomes page-aware chunks with stable, citable ids.

Chunk id = DOCCODE-P<page>-<n>, where DOCCODE is the upper-cased file-name slug (so `manual vacaciones.pdf`
gives `MANUAL-VACACIONES-P3-1`). Text only: scanned PDFs need OCR, which is out of scope; they are reported.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ragmvp.core.models import StoredChunk


class IngestError(Exception):
    pass


@dataclass
class PdfParse:
    chunks: list[StoredChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def doc_code(path: Path) -> str:
    text = unicodedata.normalize("NFKD", path.stem).encode("ascii", "ignore").decode("ascii")
    code = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").upper()
    if not code:
        raise IngestError(f"cannot derive a document code from file name '{path.name}'")
    return code if code[0].isalpha() else f"DOC-{code}"


def extract_pages(path: Path) -> list[str]:
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(path))
        return [(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # corrupt or encrypted file
        raise IngestError(f"cannot read {path.name}: {exc}") from exc


def _normalize(text: str) -> str:
    text = re.sub(r"-\n(?=[a-záéíóúñ])", "", text)  # words hyphenated at a line break
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Remove running headers/footers (short lines on most pages) and bare page numbers."""
    if len(pages) < 4:
        return [re.sub(r"(?m)^\s*(?:página\s+)?\d{1,4}(?:\s*/\s*\d{1,4})?\s*$", "", p) for p in pages]
    seen = Counter()
    for p in pages:
        seen.update({ln.strip() for ln in p.splitlines() if 0 < len(ln.strip()) <= 80})
    common = {ln for ln, n in seen.items() if n >= 0.6 * len(pages)}
    page_number = re.compile(r"^\s*(?:página\s+)?\d{1,4}(?:\s*/\s*\d{1,4})?\s*$", re.I)
    return ["\n".join(ln for ln in p.splitlines() if ln.strip() not in common and not page_number.match(ln)) for p in pages]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraphs/sentences into chunks of about `size` characters, repeating the tail as overlap."""
    text = _normalize(text)
    overlap = min(overlap, size // 2)
    if not text:
        return []
    if len(text) <= size:
        return [text]
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        if len(para) <= size:
            units.append(para.strip())
        else:
            units.extend(s.strip() for s in re.split(r"(?<=[.!?;:])\s+", para) if s.strip())
    chunks: list[str] = []
    current = ""
    for unit in units:
        while len(unit) > size:  # an unbroken block longer than a chunk
            if current:
                chunks.append(current)
                current = ""
            chunks.append(unit[:size])
            unit = unit[size - overlap :]
        if current and len(current) + len(unit) + 1 > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = (tail + " " + unit).strip() if tail else unit
        else:
            current = f"{current}\n{unit}".strip() if current else unit
    if current:
        chunks.append(current)
    return chunks


def parse_pdf_dir(directory: Path, size: int = 1200, overlap: int = 150) -> PdfParse:
    files = sorted(directory.glob("*.pdf"), key=lambda p: p.name.lower())
    if not files:
        raise IngestError(f"no PDF files in {directory}")
    by_code: dict[str, list[str]] = {}
    for f in files:
        by_code.setdefault(doc_code(f), []).append(f.name)
    duplicates = {c: names for c, names in by_code.items() if len(names) > 1}
    if duplicates:
        raise IngestError(f"file names give the same document code: {duplicates}")

    result = PdfParse()
    for f in files:
        code = doc_code(f)
        pages = strip_repeated_lines(extract_pages(f))
        empty = [i + 1 for i, p in enumerate(pages) if not p.strip()]
        if len(empty) == len(pages):
            result.warnings.append(f"{f.name}: no extractable text (scanned? OCR is not supported); skipped")
            continue
        if empty:
            result.warnings.append(f"{f.name}: {len(empty)} page(s) without text: {empty[:10]}")
        title = re.sub(r"\s+", " ", f.stem.replace("_", " ")).strip()
        for number, text in enumerate(pages, start=1):
            for n, piece in enumerate(chunk_text(text, size, overlap), start=1):
                result.chunks.append(
                    StoredChunk(
                        id=f"{code}-P{number}-{n}",
                        title=f"{title} (p. {number})",
                        text=piece,
                        metadata={"documento": f.name, "pagina": str(number), "doc_code": code},
                        block=code,
                    )
                )
    return result
