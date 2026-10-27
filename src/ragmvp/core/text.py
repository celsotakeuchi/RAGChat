"""Small text helpers shared by the prompt layer and rerankers."""
from __future__ import annotations

import re
import unicodedata

_WORD = re.compile(r"\w+")


def tokens(text: str) -> set[str]:
    """Lowercase, accent-free words longer than 3 characters."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return {w for w in _WORD.findall(folded) if len(w) > 3}


def extract_passage(text: str, question: str, limit: int) -> str:
    """Return `text` if it fits; otherwise the lines most related to the question, in original order."""
    if len(text) <= limit:
        return text
    lines = text.split("\n")
    q = tokens(question)
    scored = sorted(
        range(1, len(lines)),
        key=lambda i: (-len(q & tokens(lines[i])), i),
    )
    keep = {0}
    used = len(lines[0])
    for i in scored:
        if used + len(lines[i]) + 1 > limit:
            continue
        keep.add(i)
        used += len(lines[i]) + 1
    out: list[str] = []
    prev = -1
    for i in sorted(keep):
        if i != prev + 1:
            out.append("[…]")
        out.append(lines[i])
        prev = i
    if prev != len(lines) - 1:
        out.append("[…]")
    return "\n".join(out)[: limit + 10]
