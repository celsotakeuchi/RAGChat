from collections import Counter

import pytest

from ragmvp.config import load_kb_config
from ragmvp.ingestion.parse_md import parse_chunks, parse_eval_items

KB = load_kb_config()
PATH = KB.resolve(KB.source_file)


@pytest.fixture(scope="module")
def chunks():
    return parse_chunks(PATH)


def test_ids_unique_and_expected_entries_present(chunks):
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert {"EMP-002", "COC-002", "PRO-004", "LAG-001", "AVI-001"} <= set(ids)


def test_every_chunk_has_metadata_and_text(chunks):
    for c in chunks:
        assert set(c.metadata) >= {"categoria", "producto", "fiabilidad", "revisar_antes_de"}, c.id
        assert c.text, c.id
        assert "---" not in c.text.splitlines(), c.id


def test_eval_block_not_indexed(chunks):
    assert "H" not in {c.block for c in chunks}
    assert all("Pregunta" not in c.text for c in chunks)


def test_eval_items():
    items = parse_eval_items(PATH)
    assert len(items) == 11
    by_q = {i.question: i for i in items}
    assert by_q["¿Cómo doy de baja mi póliza?"].expected_ids == ["PRO-003"]
    abstain = [i for i in items if i.must_abstain]
    assert len(abstain) == 2
    assert Counter(i.lang for i in items) == {"es": 11}
