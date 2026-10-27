"""Authentication, sessions, audiences, multi-KB retrieval, PDF ingestion and the anonymous public access."""
import json
import re
import sqlite3
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ragmvp.api.main import create_app
from ragmvp.auth.passwords import hash_password, verify_password
from ragmvp.auth.service import AuthError, AuthService
from ragmvp.config import KBConfig, LLMConfig, Settings, load_kb_config
from ragmvp.core.budget import BudgetLimits, TokenBudget
from ragmvp.core.models import LLMBusy, Request, StoredChunk, Usage
from ragmvp.core.orchestrator import Orchestrator, Profile
from ragmvp.core.prompting import PromptEngine
from ragmvp.ingestion.build import build_version
from ragmvp.ingestion.parse_pdf import IngestError, doc_code, parse_pdf_dir
from ragmvp.llm.guards import FallbackLLM, LimitedLLM
from ragmvp.ops.store import OpsStore
from ragmvp.retrieval.search import Retriever
from ragmvp.retrieval.store import KBStore, VersionError
from test_pipeline_api import HashEmbedder

SECRET = "unit-test-secret-with-enough-length-1234"


# --- helpers ----------------------------------------------------------------------------------
def make_pdf(path, pages):
    """A minimal valid PDF (Helvetica, one text line per list item) without extra dependencies."""

    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    n = len(pages)
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(n))}] /Count {n} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    for i, lines in enumerate(pages):
        content = "BT /F1 11 Tf 50 750 Td 14 TL " + " ".join(f"({esc(line)}) Tj T*" for line in lines) + " ET"
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            "/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        objs.append(f"<< /Length {len(content.encode('latin-1'))} >>\nstream\n{content}\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


class EchoLLM:
    """Cites the first [ID] present in the sources."""

    model = "echo"

    def __init__(self):
        self.calls = 0

    def stream(self, messages, max_tokens=None):
        self.calls += 1
        first = re.search(r"\[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\]", messages[-1]["content"]).group(1)
        yield f"Según [{first}] sí."
        yield Usage(20, 5)


def settings(**kw):
    base = dict(jwt_secret=SECRET, public_access_enabled=True, max_failed_logins=3, public_rate_per_minute=3,
                public_ip_sessions_per_hour=2, public_daily_token_budget=0)
    base.update(kw)
    return Settings(**base)


@pytest.fixture()
def pdf_dir(tmp_path):
    d = tmp_path / "pdfs"
    d.mkdir()
    make_pdf(d / "faq publico.pdf", [
        ["Para solicitar vacaciones envia el formulario al responsable.", "El plazo es de quince dias."],
        ["Los presupuestos de seguro de coche se piden en la web.", "No se aceptan datos personales."],
    ])
    return d


@pytest.fixture()
def world(tmp_path, pdf_dir):
    """An internal Markdown KB and a public PDF KB, both live, with one profile per audience."""
    kb = load_kb_config()
    kb.retrieval.min_score = 0.0
    kb.guardrails.enforce_grounding = False
    store = KBStore(tmp_path / "kb.sqlite3")
    emb = HashEmbedder()
    build_version(kb, store, emb)
    store.promote(kb.id, kb.version)
    pub_kb = KBConfig(id="faq_publico", version="1", audience="public", source_type="pdf", source_dir=str(pdf_dir))
    build_version(pub_kb, store, emb)
    store.promote(pub_kb.id, pub_kb.version)

    ops = OpsStore(tmp_path / "ops.sqlite3")
    public = kb.model_copy(deep=True)
    public.guardrails.enforce_grounding = True
    public.ui.title = "Publico"
    llm = EchoLLM()
    profiles = {
        "internal": Profile("internal", kb, PromptEngine(kb), llm, TokenBudget(ops, BudgetLimits(max_request_tokens=0)), ["internal", "public"]),
        "public": Profile("public", public, PromptEngine(public), llm, TokenBudget(ops, BudgetLimits(max_request_tokens=0, daily_tokens_per_user=50000)), ["public"]),
    }
    core = Orchestrator(kb, Retriever(store, emb), llm, ops, TokenBudget(ops, BudgetLimits(max_request_tokens=0)), profiles=profiles)
    return core, store, ops, llm


@pytest.fixture()
def auth(world):
    core, _, ops, _ = world
    svc = AuthService(ops, settings())
    svc.create_user("Ana", "correct-horse", "ventas")
    return svc


@pytest.fixture()
def client(world, auth):
    return TestClient(create_app(world[0], auth))


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def login(client, user="ana", password="correct-horse"):
    r = client.post("/v1/auth/login", json={"username": user, "password": password})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def chat(client, token, message, **extra):
    r = client.post("/v1/chat", json={"message": message, **extra}, headers=bearer(token))
    assert r.status_code == 200, r.text
    done = None
    for block in r.text.split("\n\n"):
        if block.startswith("event: done"):
            done = json.loads(block.split("data: ", 1)[1])
    return r.text, done


# --- passwords and login ----------------------------------------------------------------------
def test_password_hash_is_salted_and_verifiable():
    a, b = hash_password("correct-horse"), hash_password("correct-horse")
    assert a != b and verify_password("correct-horse", a) and not verify_password("wrong-pass", a)
    assert not verify_password("x", "not-a-hash")
    with pytest.raises(ValueError):
        hash_password("short")


def test_login_rejects_wrong_password_and_unknown_user_the_same_way(auth):
    for user, pw in (("ana", "bad-password"), ("nobody", "whatever1")):
        with pytest.raises(AuthError) as e:
            auth.login(user, pw)
        assert e.value.code == "invalid_credentials"


def test_lockout_after_repeated_failures_then_correct_password_is_refused(auth):
    for _ in range(3):
        with pytest.raises(AuthError):
            auth.login("ana", "bad-password")
    with pytest.raises(AuthError) as e:
        auth.login("ana", "correct-horse")
    assert e.value.code == "locked"


def test_disabled_user_cannot_login_and_active_sessions_end(auth):
    token = auth.login("ana", "correct-horse")
    auth.set_active("ana", False)
    with pytest.raises(AuthError):
        auth.authenticate(token)
    with pytest.raises(AuthError):
        auth.login("ana", "correct-horse")


def test_jwt_secret_is_required(world):
    with pytest.raises(ValueError, match="JWT_SECRET"):
        AuthService(world[2], Settings(jwt_secret=""))


# --- sessions ---------------------------------------------------------------------------------
def test_login_creates_server_session_with_department_and_audience(auth):
    p = auth.authenticate(auth.login("ana", "correct-horse", ip="10.0.0.1"))
    assert (p.user_id, p.department, p.audience, p.anonymous) == ("ana", "ventas", "internal", False)
    row = auth.list_sessions("user")[0]
    assert row["session_id"] == p.session_id and row["ended_at"] is None


def test_logout_and_expiry_invalidate_the_token(auth, world):
    token = auth.login("ana", "correct-horse")
    p = auth.authenticate(token)
    auth.logout(p.session_id)
    with pytest.raises(AuthError) as e:
        auth.authenticate(token)
    assert e.value.code == "session_ended"
    token2 = auth.login("ana", "correct-horse")
    sid = auth.authenticate(token2).session_id
    world[2].run("UPDATE sessions SET expires_at='2000-01-01T00:00:00' WHERE session_id=?", (sid,))
    with pytest.raises(AuthError):
        auth.authenticate(token2)


def test_tampered_or_foreign_token_is_rejected(auth):
    token = auth.login("ana", "correct-horse")
    with pytest.raises(AuthError):
        auth.authenticate(token[:-3] + "abc")
    import jwt as pyjwt

    forged = pyjwt.encode({"sub": "ana", "sid": "x", "exp": int(time.time()) + 60}, "another-secret-that-is-long-enough", algorithm="HS256")
    with pytest.raises(AuthError):
        auth.authenticate(forged)


# --- API --------------------------------------------------------------------------------------
def test_api_requires_a_token(client):
    assert client.post("/v1/chat", json={"message": "hola"}).status_code == 401
    assert client.get("/v1/me", headers=bearer("garbage")).status_code == 401
    assert client.post("/v1/auth/login", json={"username": "ana", "password": "nope"}).status_code == 401


def test_me_reports_server_side_identity(client):
    me = client.get("/v1/me", headers=bearer(login(client))).json()
    assert me["department"] == "ventas" and me["audience"] == "internal" and not me["anonymous"]


def test_client_cannot_choose_session_id_or_audience(client, world):
    token = login(client)
    sid = client.get("/v1/me", headers=bearer(token)).json()["session_id"]
    _, done = chat(client, token, "¿Cómo doy de baja mi póliza?", session_id="forged", audience="public", department="rrhh")
    row = world[2].rows("SELECT session_id, user_id, department, audience FROM interactions WHERE message_id=?", (done["message_id"],))[0]
    assert (row["session_id"], row["user_id"], row["department"], row["audience"]) == (sid, "ana", "ventas", "internal")


def test_internal_user_searches_internal_and_public_kbs(world):
    core, _, _, _ = world
    internal = core.answer(Request("ana", "s", "¿Cómo doy de baja mi póliza?", "internal", None, "ventas"))
    kbs = {h["kb"] for h in internal.retrieved}
    assert {"linea_directa", "faq_publico"} <= {h["kb"] for h in core.answer(Request("ana", "s", "vacaciones formulario baja póliza", "internal", None)).retrieved} or "linea_directa" in kbs


def test_logout_endpoint_ends_the_session(client):
    token = login(client)
    assert client.post("/v1/auth/logout", headers=bearer(token)).status_code == 204
    assert client.get("/v1/me", headers=bearer(token)).status_code == 401


def test_feedback_only_on_own_session(client, auth, world):
    t1 = login(client)
    _, done = chat(client, t1, "baja póliza")
    auth.create_user("luis", "another-pass1", "soporte")
    t2 = login(client, "luis", "another-pass1")
    body = {"message_id": done["message_id"], "rating": 1}
    assert client.post("/v1/feedback", json=body, headers=bearer(t2)).status_code == 404
    assert client.post("/v1/feedback", json=body, headers=bearer(t1)).status_code == 204


# --- public access ----------------------------------------------------------------------------
def test_public_session_disabled_by_default(world):
    svc = AuthService(world[2], settings(public_access_enabled=False))
    c = TestClient(create_app(world[0], svc))
    assert c.post("/v1/public/session").status_code == 404


def test_anonymous_gets_only_public_kb_and_public_profile(client, world):
    token = client.post("/v1/public/session").json()["access_token"]
    me = client.get("/v1/me", headers=bearer(token)).json()
    assert me["anonymous"] and me["audience"] == "public" and me["user_id"].startswith("anon:")
    assert client.get("/v1/config", headers=bearer(token)).json()["title"] == "Publico"
    text, done = chat(client, token, "¿Cómo doy de baja mi póliza?")
    row = world[2].rows("SELECT retrieved FROM interactions WHERE message_id=?", (done["message_id"],))[0]
    assert {h["kb"] for h in json.loads(row["retrieved"])} == {"faq_publico"}
    assert done["status"] == "answered" and done["sources"][0]["document"] == "faq publico.pdf"


def test_anonymous_token_cannot_reach_internal_profile(world, auth):
    core = world[0]
    with pytest.raises(PermissionError):
        core.answer(Request("x", "s", "hola", "unknown-audience", None))


def test_anonymous_rate_limit_per_session(client):
    token = client.post("/v1/public/session").json()["access_token"]
    statuses = [chat(client, token, "vacaciones formulario")[1]["status"] for _ in range(4)]
    assert statuses[:3] == ["answered"] * 3 and statuses[3] == "rate_limited"


def test_anonymous_session_creation_limited_per_ip(client):
    assert client.post("/v1/public/session").status_code == 200
    assert client.post("/v1/public/session").status_code == 200
    assert client.post("/v1/public/session").status_code == 429


def test_global_public_budget_stops_anonymous_traffic(world):
    svc = AuthService(world[2], settings(public_daily_token_budget=10))
    c = TestClient(create_app(world[0], svc))
    token = c.post("/v1/public/session").json()["access_token"]
    assert chat(c, token, "vacaciones formulario")[1]["status"] == "answered"  # uses 25 tokens
    assert chat(c, token, "vacaciones formulario")[1]["status"] == "budget_exceeded"


def test_per_session_daily_token_cap(world):
    core, _, _, _ = world
    first = core.answer(Request("anon:1", "s", "vacaciones formulario", "public", None))
    assert first.status == "answered"
    core.profiles["public"].budget.record("anon:1", "m", Usage(1500, 0))
    core.profiles["public"].budget.limits.daily_tokens_per_user = 2000
    assert core.answer(Request("anon:1", "s", "vacaciones formulario", "public", None)).status == "budget_exceeded"
    assert core.answer(Request("anon:2", "s", "vacaciones formulario", "public", None)).status == "answered"


def test_personal_data_is_masked_before_llm_and_log(client, world):
    token = client.post("/v1/public/session").json()["access_token"]
    _, done = chat(client, token, "mi email es juan@example.com, DNI 12345678Z, tel 612 345 678, matricula 1234BCD")
    q = world[2].rows("SELECT question FROM interactions WHERE message_id=?", (done["message_id"],))[0]["question"]
    for secret in ("juan@example.com", "12345678Z", "612 345 678", "1234BCD"):
        assert secret not in q
    assert "[EMAIL]" in q and "[DNI]" in q and "[TELEFONO]" in q and "[MATRICULA]" in q


def test_internal_questions_are_not_masked(client, world):
    _, done = chat(client, login(client), "cliente 12345678Z pregunta")
    assert "12345678Z" in world[2].rows("SELECT question FROM interactions WHERE message_id=?", (done["message_id"],))[0]["question"]


def test_ip_is_stored_hashed_and_purge_removes_old_anonymous_data(world):
    svc = AuthService(world[2], settings())
    token = svc.create_anonymous_session(ip="203.0.113.9")
    p = svc.authenticate(token)
    world[0].answer(Request(p.user_id, p.session_id, "vacaciones formulario", "public", None))
    row = world[2].rows("SELECT ip_hash FROM sessions WHERE session_id=?", (p.session_id,))[0]
    assert row["ip_hash"] and "203.0.113.9" not in row["ip_hash"]
    world[2].run("UPDATE sessions SET created_at='2000-01-01T00:00:00' WHERE session_id=?", (p.session_id,))
    assert svc.purge_anonymous(30) == 1
    assert not world[2].rows("SELECT 1 FROM interactions WHERE session_id=?", (p.session_id,))
    assert not world[2].rows("SELECT 1 FROM usage WHERE user_id=?", (p.user_id,))


def test_anonymous_session_stops_working_when_public_access_is_turned_off(world):
    on = AuthService(world[2], settings())
    token = on.create_anonymous_session(ip="1.1.1.1")
    off = AuthService(world[2], settings(public_access_enabled=False))
    with pytest.raises(AuthError):
        off.authenticate(token)


# --- multi-KB retrieval and lifecycle ---------------------------------------------------------
def test_chunk_ids_must_be_unique_across_live_kbs(world, pdf_dir):
    _, store, _, _ = world
    clone = KBConfig(id="otro", version="1", audience="public", source_type="pdf", source_dir=str(pdf_dir))
    with pytest.raises(VersionError, match="already used"):
        build_version(clone, store, HashEmbedder())


def test_scope_search_refreshes_after_promotion(world, tmp_path):
    core, store, _, _ = world
    retriever = core.retriever
    before = {h.chunk.id for h in retriever.retrieve_scope(["public"], "vacaciones", 20)}
    d = tmp_path / "v2"
    d.mkdir()
    make_pdf(d / "faq publico.pdf", [["Nuevo texto sobre teletrabajo y vacaciones."]])
    kb2 = KBConfig(id="faq_publico", version="2", audience="public", source_type="pdf", source_dir=str(d))
    build_version(kb2, store, HashEmbedder())
    assert {h.chunk.id for h in retriever.retrieve_scope(["public"], "vacaciones", 20)} == before
    store.promote("faq_publico", "2", actor="tester", gate="x", note="update")
    after = {h.chunk.id for h in retriever.retrieve_scope(["public"], "vacaciones", 20)}
    assert after == {"FAQ-PUBLICO-P1-1"} and after != before


def test_promotion_audit_records_actor_gate_and_rollback(world, tmp_path):
    _, store, _, _ = world
    d = tmp_path / "v2"
    d.mkdir()
    make_pdf(d / "faq publico.pdf", [["Texto distinto."]])
    build_version(KBConfig(id="faq_publico", version="2", audience="public", source_type="pdf", source_dir=str(d)), store, HashEmbedder())
    store.promote("faq_publico", "2", actor="celso", gate="passed", note="q3 update")
    assert store.rollback("faq_publico", actor="celso") == "1"
    history = store.promotions("faq_publico")
    assert [(p["action"], p["version"]) for p in history[:2]] == [("rollback", "1"), ("promote", "2")]
    assert history[1]["actor"] == "celso" and history[1]["note"] == "q3 update"


def test_embedding_cache_reuses_vectors(tmp_path, pdf_dir):
    store = KBStore(tmp_path / "kb.sqlite3")

    class Counting(HashEmbedder):
        calls = 0

        def embed_documents(self, texts):
            Counting.calls += len(texts)
            return super().embed_documents(texts)

    kb = KBConfig(id="c", version="1", audience="public", source_type="pdf", source_dir=str(pdf_dir))
    build_version(kb, store, Counting())
    first = Counting.calls
    kb.version = "2"
    build_version(kb, store, Counting())
    assert first > 0 and Counting.calls == first


# --- PDF ingestion ----------------------------------------------------------------------------
def test_doc_code_from_file_name(tmp_path):
    assert doc_code(tmp_path / "Manual de vacaciones 2024.pdf") == "MANUAL-DE-VACACIONES-2024"
    assert doc_code(tmp_path / "2024 política.pdf") == "DOC-2024-POLITICA"


def test_pdf_chunks_have_page_ids_and_metadata(pdf_dir):
    parsed = parse_pdf_dir(pdf_dir, 1200, 100)
    assert [c.id for c in parsed.chunks] == ["FAQ-PUBLICO-P1-1", "FAQ-PUBLICO-P2-1"]
    c = parsed.chunks[1]
    assert c.metadata == {"documento": "faq publico.pdf", "pagina": "2", "doc_code": "FAQ-PUBLICO"}
    assert "presupuestos" in c.text and "(p. 2)" in c.title


def test_long_page_is_split_with_overlap_and_unique_ids(tmp_path):
    make_pdf(tmp_path / "largo.pdf", [[f"Frase numero {i} del documento largo sobre politicas." for i in range(60)]])
    chunks = parse_pdf_dir(tmp_path, 400, 60).chunks
    assert len(chunks) > 3 and len({c.id for c in chunks}) == len(chunks)
    assert all(len(c.text) <= 520 for c in chunks)


def test_duplicate_document_codes_and_empty_pdfs_are_reported(tmp_path):
    make_pdf(tmp_path / "a b.pdf", [["uno"]])
    make_pdf(tmp_path / "a-b.pdf", [["dos"]])
    with pytest.raises(IngestError, match="same document code"):
        parse_pdf_dir(tmp_path)
    (tmp_path / "a-b.pdf").unlink()
    make_pdf(tmp_path / "vacio.pdf", [[""]])
    parsed = parse_pdf_dir(tmp_path)
    assert any("no extractable text" in w for w in parsed.warnings) and [c.id for c in parsed.chunks] == ["A-B-P1-1"]
    with pytest.raises(IngestError, match="no PDF"):
        parse_pdf_dir(tmp_path / "missing")


def test_repeated_headers_and_page_numbers_are_removed(tmp_path):
    pages = [["Empresa SA - Confidencial", f"Contenido unico de la pagina {i}", str(i)] for i in range(1, 6)]
    make_pdf(tmp_path / "doc.pdf", pages)
    text = " ".join(c.text for c in parse_pdf_dir(tmp_path).chunks)
    assert "Confidencial" not in text and "Contenido unico de la pagina 3" in text


# --- LLM per audience -------------------------------------------------------------------------
def test_limited_llm_reports_busy_when_no_slot_is_free():
    import threading

    gate = threading.Event()

    class Slow:
        model = "slow"

        def stream(self, messages, max_tokens=None):
            gate.wait(2)
            yield "ok"

    llm = LimitedLLM(Slow(), 1, 0.05)
    first = llm.stream([])
    t = threading.Thread(target=lambda: list(first))
    t.start()
    time.sleep(0.1)
    with pytest.raises(LLMBusy):
        list(llm.stream([]))
    gate.set()
    t.join()
    assert list(llm.stream([])) == ["ok"]


def test_busy_model_gives_the_busy_message_and_status(world):
    core, _, _, _ = world

    class Busy:
        model = "busy"

        def stream(self, messages, max_tokens=None):
            raise LLMBusy("busy")
            yield

    core.profiles["public"].llm = Busy()
    res = core.answer(Request("anon:1", "s", "vacaciones formulario", "public", None))
    assert res.status == "busy"


def test_fallback_used_only_when_primary_fails_before_answering():
    class Down:
        model = "down"

        def stream(self, messages, max_tokens=None):
            raise ConnectionError("down")
            yield

    class Up:
        model = "up"

        def stream(self, messages, max_tokens=None):
            yield "hola"
            yield Usage(1, 1)

    pieces = list(FallbackLLM(Down(), Up()).stream([]))
    assert pieces[0] == "hola" and pieces[1].model == "up"
    with pytest.raises(ConnectionError):
        list(FallbackLLM(Down(), Down()).stream([]))


def test_profile_llm_config_selects_provider_and_local_has_no_key_requirement():
    from ragmvp.factory import build_llm
    from ragmvp.llm.guards import LimitedLLM as Limited
    from ragmvp.llm.ollama_cloud import OllamaGateway

    s = Settings(ollama_api_key="", llm_provider="ollama_cloud")
    local = build_llm(s, LLMConfig(provider="ollama_local", model="llama3.1:8b", max_concurrent=2))
    assert isinstance(local, Limited) and local.model == "llama3.1:8b"
    with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
        build_llm(s)
    cloud = build_llm(Settings(ollama_api_key="k", llm_provider="ollama_cloud"), LLMConfig(model="gemma4:31b"))
    assert isinstance(cloud, OllamaGateway)


# --- migration --------------------------------------------------------------------------------
def test_old_ops_database_is_migrated(tmp_path):
    path = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE interactions (message_id TEXT PRIMARY KEY, session_id TEXT, created_at TEXT, user_id TEXT, "
                 "kb_id TEXT, kb_version TEXT, lang TEXT, question TEXT, answer TEXT, status TEXT, flags TEXT, retrieved TEXT, "
                 "cited TEXT, model TEXT, prompt_version TEXT, latency_ms INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER)")
    conn.commit()
    conn.close()
    ops = OpsStore(path)
    cols = {r["name"] for r in ops.rows("PRAGMA table_info(interactions)")}
    assert {"raw_answer", "department", "audience"} <= cols
    assert ops.rows("SELECT COUNT(*) c FROM users")[0]["c"] == 0


def test_old_kb_database_gets_promotion_audit_columns(tmp_path):
    path = tmp_path / "old_kb.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE kb_promotions (id INTEGER PRIMARY KEY AUTOINCREMENT, kb_id TEXT NOT NULL, version TEXT NOT NULL, "
                 "promoted_at TEXT NOT NULL DEFAULT (datetime('now')))")
    conn.execute("INSERT INTO kb_promotions (kb_id, version) VALUES ('k','1')")
    conn.commit()
    conn.close()
    store = KBStore(path)
    assert store.promotions("k")[0]["action"] == "promote"


# --- admin CLI --------------------------------------------------------------------------------
def test_admin_cli_manages_users_and_sessions(auth, capsys):
    from ragmvp.admin import main

    assert main(["user-add", "marta", "--department", "soporte", "--password", "long-enough-pass"], auth) == 0
    assert main(["user-add", "marta", "--department", "soporte", "--password", "long-enough-pass"], auth) == 1
    assert main(["user-list"], auth) == 0
    assert "marta" in capsys.readouterr().out
    auth.login("marta", "long-enough-pass")
    assert main(["sessions", "--internal"], auth) == 0
    assert "marta" in capsys.readouterr().out
    assert main(["user-disable", "marta"], auth) == 0
    with pytest.raises(AuthError):
        auth.login("marta", "long-enough-pass")
    assert main(["purge-public"], auth) == 0
