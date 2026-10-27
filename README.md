# RAG MVP: Business Rules assistant

Python RAG assistant with an optional anonymous public page. It supports Markdown and PDF knowledge bases (currently the public KB `air_passenger_rights_public.md`, EU air passenger rights, English; the previous insurance KB is archived in `kb/archive/` and used as a test fixture), login with server-side sessions, departments, and a separate LLM profile per audience.

## System architecture

The system is a RAG assistant built as a **core with ports and adapters**. The core holds the business flow. Everything else (UI, LLM, search, storage) plugs in through interfaces, so each part can change independently. The model is never retrained: it answers only from the retrieved documents.

```
Channels (adapters)          Core                         Adapters behind ports
-------------------          ----                         ---------------------
Chainlit UI (agents)  --\                              /-- LLMPort: Ollama cloud/local, LiteLLM
Public web page       ---+--> HTTP API (/v1) --> Orchestrator --+-- RetrieverPort: SQLite + vectors
(future: telephony,   --/    auth, sessions          |      \-- OpsPort: SQLite logs, usage, users
 other apps)                 rate limits             |
                                                     +--> Prompt engine, guardrails, token budget
```

### Layers
1. **Channels.** Chainlit (call-centre agents) and a static public page. Both only call the HTTP API. A new channel, such as telephony, is one more client.
2. **HTTP API (`/v1`).** Login, anonymous public sessions, streamed chat, feedback and config. It validates the bearer token and re-checks the session row on every request. Audience and department come from the server-side session; the client cannot choose them. It applies PII masking, rate limits and the global public token budget to anonymous users.
3. **Orchestrator (core).** Chooses a **profile** from the caller's audience (`internal` or `public`). Each profile has its own KBs to search, prompts, LLM, limits and grounding rules.
4. **Prompt engine.** Builds the prompt (tone, strict-grounding rules, excerpts, citation format) and checks the output.
5. **LLM layer.** One interface with providers `ollama_cloud`, `ollama_local` and `litellm`. A concurrency guard returns a "busy" message when the model is saturated; an optional fallback model is available.
6. **Retrieval and knowledge bases.** Each KB (Markdown or PDF) is chunked, embedded and stored in SQLite as an immutable version. A version goes live only after promotion, which runs a recall gate on a golden question set; rollback and an audit trail are available. Search runs across the KBs the profile may use (internal: internal + public; public: public only).
7. **Operations store.** Users, sessions, interactions (question, answer, raw answer, sources, status, department, audience), feedback and token usage.

### How a question is handled
1. The user signs in, or the public page requests an anonymous session; the server creates a session row.
2. The channel sends `POST /v1/chat` with the message and the bearer token.
3. The API identifies the principal and applies anonymous limits and PII masking.
4. The orchestrator selects the profile and runs input guardrails.
5. The retriever finds the top passages in the KBs the profile may search.
6. If relevance is too low, the system answers that it doesn't know and the LLM is not called.
7. The token budget checks request size and daily and monthly limits.
8. The LLM streams the answer from the passages only.
9. Output guardrails check citations. With strict grounding, an unsupported answer is withheld and replaced by "I don't know"; the raw text is still logged.
10. The interaction, usage, sources and status are logged, and the answer with its sources is returned.
11. The user can send feedback, accepted only for their own session.

### Operations
- `kbadmin`: build, promote, rollback, status and history of KBs.
- `admin`: users, sessions, disabling accounts and the public-session purge.
- `eval`: Recall@k, MRR, citation and abstention checks.
- Answers are improved by fixing the documents and promoting a new KB version.

### Why it scales
A new channel needs only an adapter; a new model is a config change; new KBs or departments are new YAML files plus build and promote; moving to pgvector means re-implementing `RetrieverPort` while the core stays the same.

## Architecture (details)

```
 Clients (any)                    Core (orchestrator)                       Adapters
 ─────────────                    ───────────────────                       ────────
 Chainlit web UI  ┐                1. input guardrails      ┐
 Voice gateway    ├─ HTTP /v1 ──>  2. retrieve (RetrieverPort) ──────────>  Retriever + KBStore (versioned KB)
 Other systems    ┘  (text)        3. relevance check          │ prompt
                                   4. token budget             │ engineering   TokenBudget + OpsStore (usage, logs)
                                   5. LLM (LLMPort) ───────────────────────>  LiteLLMGateway: Ollama / private / commercial
                                   6. output checks (citations)┘
                                   7. log every outcome
```

- **Core** (`src/ragmvp/core/`): `Orchestrator` takes a channel-neutral `Request` and returns a `Response`. It depends only on ports (`ports.py`), never on HTTP, a UI or a provider.
- **Channels:** the API is text-in/text-out. A voice system transcribes and calls the same `POST /v1/chat`; no channel logic lives in the core.
- **Prompt engineering** (`core/prompting.py`, `prompts/_base.md`, `prompts/*.md`, `kb/_defaults.yaml`, `kb/*.yaml`): a shared layer plus per-KB overrides. `kb/_defaults.yaml` (guardrails, injection patterns, context limits, promotion gate, refusal messages) and `prompts/_base.md` (general rules) apply to every KB; a KB's yaml overrides scalars and extends lists (e.g. extra `blocked_patterns`), and its own prompt is appended after the base prompt. The layer covers input guardrails, context extraction and output checks (citations verified against the retrieved context, flags `invalid_citation` / `uncited`).
- **Document-only answers:** the model only receives the retrieved passages and no tools or web search. `guardrails.enforce_grounding: true` (on in `kb/_defaults.yaml`) additionally withholds any answer that has no citation, or cites a source that was not retrieved, and shows the "unknown" message instead (status `unknown`, flag `ungrounded`). The original text is kept in `interactions.raw_answer` for review. In this mode the answer is checked before it is sent, so it arrives in one piece instead of streaming word by word; set it to `false` in a KB yaml to get streaming back.
- **Context extraction:** `context.max_chunks`, `context.score_margin` (drop sources far below the best), duplicate removal, and, for long sources, only the lines most related to the question (`[…]` marks cuts).
- **Retrieval backends** (`retrieval/backends.py`): `RETRIEVER_BACKEND` (default `sqlite`) and `EMBEDDING_BACKEND` select the vector store and embedding model from registries; `register_backend` / `register_embedder` add new ones (pgvector, managed vector DB) without touching the core, API or `kbadmin`. Optional reranking per KB: `retrieval.rerank: lexical` (`retrieval/rerank.py`); it is **off** by default because it lowered MRR on the current KB (0.91 → 0.84).
- **LLM abstraction** (`llm/gateway.py`): swap model with `CHAT_MODEL` only (e.g. `ollama/llama3.1` for on-prem, then a private commercial model). Nothing else changes.
- **Token control** (`core/budget.py`): per-request cap, max output tokens, daily tokens per user, monthly global budget. `0` = unlimited (use for Ollama). Exceeding a limit returns a `budget_exceeded` status without calling the LLM. Usage is stored per call (provider-reported, otherwise estimated).
- **Versioned KB** (`retrieval/store.py`, `kbadmin.py`): a version is built and embedded independently, evaluated, then promoted by switching a live alias. The service keeps answering from the old version until then. Live versions are immutable; rollback re-promotes the previous one.
- **Ops data** (`ops/store.py`): interactions (with status, flags, KB and prompt version, latency, tokens), feedback and usage, in a separate DB from the KB.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[local-embeddings,ui,dev]"
copy .env.example .env        # set CHAT_MODEL and the provider key
```

- **Embeddings:** `EMBEDDING_BACKEND=local` uses `intfloat/multilingual-e5-large` through fastembed (no key; ~2 GB download on first use). Use `litellm` for a hosted embedding model. Changing the embedding model requires building a new KB version; the retriever refuses to query a version embedded with another model.
- **LLM:** any LiteLLM model string in `CHAT_MODEL` plus the provider key in `.env`.
- **Token limits:** `MAX_OUTPUT_TOKENS`, `MAX_REQUEST_TOKENS`, `DAILY_TOKENS_PER_USER`, `MONTHLY_TOKEN_BUDGET` (see `.env.example`).

## Knowledge base lifecycle

```powershell
$env:PYTHONPATH="src"
python -m ragmvp.kbadmin build                     # parse + embed the Markdown as the version in kb/*.yaml (not live)
python -m ragmvp.kbadmin promote                   # runs the Block H gate (recall >= promotion.min_recall), then goes live
python -m ragmvp.kbadmin list                      # versions and which is live
python -m ragmvp.kbadmin rollback                  # back to the previous live version
python -m ragmvp.kbadmin --kb kb/archive/xxxx.yaml retire   # take a KB out of service (no longer searched; versions kept)
```

To update the KB: edit the Markdown, bump `version` in `kb/air_passenger_rights.yaml`, then `build` and `promote`. Use `--skip-gate` only deliberately. The running API picks up a promotion on the next request; no restart.

## Run

```powershell
$env:PYTHONPATH="src"
python -m uvicorn ragmvp.api.main:app --port 8000      # API
chainlit run ui/chainlit_app.py --port 8001            # agent UI (set API_URL if needed)
```

## Evaluate

```powershell
python eval/run_eval.py              # Recall@k / MRR on the Block H questions (no LLM needed)
python eval/run_eval.py --generate   # also checks citations and abstention (needs the LLM key)
pytest                               # parser, KB lifecycle, prompt layer, budgets, orchestrator, API contract
```

Results are saved to `eval/results/`. Baseline with `multilingual-e5-large`: Recall@5 = 1.0, Recall@1 = 0.82, MRR = 0.91 (11 questions).

## API contract (`/v1`)

| Endpoint | Purpose |
|---|---|
| `POST /auth/login` `{username, password}` / `POST /auth/logout` | Bearer token / end session |
| `GET /me` | Current principal |
| `POST /public/session` | Anonymous token (404 unless enabled) |
| `POST /chat` `{message, lang?}` | SSE: `token` events, then `done` `{message_id, status, sources[], flags[], invalid_citations[]}`, or `error` |
| `POST /feedback` `{message_id, rating: 1\|-1, comment?}` | Stores feedback (204; own session only) |
| `GET /config` | Title, disclaimer, languages for the caller's profile |
| `GET /health` | Liveness |

`status` is `answered`, `unknown`, `blocked`, `budget_exceeded`, `busy` or `rate_limited`. Session id, KB and audience are never accepted from the client; all endpoints except health/login/public session need a bearer token.

## Adding another KB or language
- New KB: add `kb/<id>.yaml` + a prompt in `prompts/`, then `python -m ragmvp.kbadmin --kb kb/<id>.yaml build` and `promote`.
- New language: add it to `languages`, add `messages.{unknown,blocked,budget}.<lang>`, add translated chunks if official wording is needed (`lang`, `translation_of` fields exist), and add questions with that `lang` to the eval set. Metrics are reported per language.

## Authentication, users and sessions

Login is server-side: the JWT carries only `sub` and `sid`; audience and department come from the session row, which is re-checked on every request (logout, expiry or disabling a user takes effect immediately). Each user belongs to one department. Set `JWT_SECRET` (>= 16 chars) in `.env`; Chainlit also needs `CHAINLIT_AUTH_SECRET`. `AUTH_ENABLED=false` gives a dev principal (never in production).

```powershell
python -m ragmvp.admin user-add ana --department ventas --role agent    # prompts for a password
python -m ragmvp.admin user-list
python -m ragmvp.admin user-passwd ana
python -m ragmvp.admin user-disable ana            # also ends her sessions (user-enable to undo)
python -m ragmvp.admin sessions [--public|--internal]
python -m ragmvp.admin end-session <session_id>
python -m ragmvp.admin purge-public                # retention for anonymous sessions
```

Every interaction is logged with `session_id`, `user_id`, `department` and `audience`. Failed logins lock the account (`MAX_FAILED_LOGINS`, `LOCKOUT_MINUTES`).

## Many KBs (department catalog, PDFs)

Put one YAML per KB in `kb/` (see `KB_CATALOG_DIR`), e.g. `source_type: pdf`, `source_dir: data/ventas`, `department: ventas`, `golden_file: eval/ventas.yaml`. Chunk ids are `DOCCODE-P<page>-<n>` and must be unique across live KBs. Scanned/empty PDFs give a warning (no OCR).

```powershell
python -m ragmvp.kbadmin status                        # all KBs
python -m ragmvp.kbadmin build --all                   # embeddings are cached, rebuilds are incremental
python -m ragmvp.kbadmin promote --all --actor celso --note "q3 update"
python -m ragmvp.kbadmin history
```

PDFs need the `local-embeddings` extra (fastembed); about 20,000 chunks are slow on CPU the first time. Promotions and rollbacks are audited in `kb_promotions`.

## Profiles: internal vs public

`kb/_profiles.yaml` defines one profile per audience: which KBs it searches (`internal` -> internal + public; `public` -> public only), its prompt, limits, strict grounding and its LLM. Both use the global Ollama cloud model by default. To serve the public side locally:

```yaml
public:
  llm: {provider: ollama_local, model: gemma3:4b, host: http://localhost:11434, max_concurrent: 2, queue_wait_s: 10}
```

Providers: `ollama_cloud`, `ollama_local`, `litellm`; optional `fallback`. When the model is saturated the user gets the `busy` message.

## Anonymous public page

Off by default. Enable with `PUBLIC_ACCESS_ENABLED=true` and `PUBLIC_CORS_ORIGINS`. `POST /v1/public/session` issues an anonymous session (limits per IP per hour, per-session requests per minute, per-session and global daily token budgets, PII masking of input; IPs are stored hashed with `IP_HASH_SALT`). Serve `web/public/` statically (set `window.PUBLIC_API_URL`) and put the API behind a reverse proxy/TLS that forwards the client IP.

## Production hardening
Use a secrets manager, TLS and a WAF; keep `data/` and `.venv` outside OneDrive (SQLite locking); rotate any API key that was ever pasted in chat.

## Known limits
- SQLite + numpy search is fine for thousands of chunks; implement `RetrieverPort` over pgvector when you scale. Docker/Postgres was not available on this machine.
- Retrieval is vector-only (no hybrid search or reranker yet).
- Token counts before the call are estimates (about 4 characters per token); the provider-reported usage is what gets recorded.
- The KB's own gaps (Block G) mean "I don't know" is expected for exclusions, deductibles, claims and several products.

## Limiting token usage

Where each limit is set. `0` means unlimited (use it for local Ollama). A request over a limit is refused before the model is called, with status `budget_exceeded` ("usage limit" message).

| Goal | Parameter | Where | Default |
|---|---|---|---|
| Shorter answers | `MAX_OUTPUT_TOKENS` | `.env` (profile: `limits.max_output_tokens` in `kb/_profiles.yaml`) | 700 (public 400) |
| Cap the prompt size (instructions + question + retrieved passages) | `MAX_REQUEST_TOKENS` | `.env` (profile: `limits.max_request_tokens`) | 6000 (public 5000) |
| Cap one user or session per day | `DAILY_TOKENS_PER_USER` | `.env` (profile: `limits.daily_tokens_per_user`) | 100000 (public 20000) |
| Cap the whole system per month | `MONTHLY_TOKEN_BUDGET` | `.env` | 0 |
| Cap all anonymous traffic per day | `PUBLIC_DAILY_TOKEN_BUDGET` | `.env` | 200000 |
| Slow down abuse | `PUBLIC_RATE_PER_MINUTE`, `PUBLIC_IP_SESSIONS_PER_HOUR` | `.env` | 10, 20 |
| Smaller prompts (the main cost driver) | `context.max_chunks`, `context.max_total_chars`, `context.max_chars_per_source` | `kb/_defaults.yaml` or a KB yaml | 5, 12000, 2500 |
| Fewer retrieved passages | `retrieval.top_k` | a KB yaml | 5 |
| Shorter questions | `guardrails.max_question_chars` | `kb/_defaults.yaml` (public: `kb/_profiles.yaml`) | 1000 (public 500) |

Notes:
- `MAX_REQUEST_TOKENS` limits the prompt, not the answer. Setting it too low (for example 600) rejects every request, because a prompt with retrieved passages is about 1,500 to 3,000 tokens. Keep it at 6000 or more.
- Profile limits in `kb/_profiles.yaml` override the `.env` values for that audience.
- Restart the API after changing `.env`, `config.py` or profiles. Token counts before the call are estimates (about 4 characters per token); the provider's reported usage is what is recorded and counted against the daily and monthly limits.

## Managing knowledge bases: replace and update

Run from the project folder. `--kb` goes **before** the subcommand.

```powershell
cd "C:\Users\Celso Augusto\OneDrive\Documents\CopilotChat"
$env:PYTHONPATH="src"
$py=".\.venv\Scripts\python.exe"
```

### Replace a knowledge base

1. **Prepare the source.** Markdown: one file with `## ID Title` headings and `---` between chunks. PDF: a folder of PDFs inside the project, one document code per file.
2. **Create `kb/<new_id>.yaml`:** `id`, `version`, `audience` (`public`, `internal` or a department), `source_language`, `languages`, `source_file` (md) or `source_type: pdf` + `source_dir`, `system_prompt` (a file in `prompts/`), `golden_file` and the `messages` for each language.
3. **Write golden questions** in `eval/<id>.yaml`: `question` and `expected` ids. A question with no `expected` must be refused; it is excluded from recall and MRR.
4. **Build** the new version (not live yet; the old KB keeps answering):
   ```powershell
   & $py -m ragmvp.kbadmin --kb kb/<new_id>.yaml build
   ```
5. **Promote** it. It goes live only if the recall gate passes (`promotion.min_recall`, 0.9 by default):
   ```powershell
   & $py -m ragmvp.kbadmin --kb kb/<new_id>.yaml promote --note "replace old KB"
   & $py -m ragmvp.kbadmin --kb kb/<new_id>.yaml promote --skip-gate --note "reason"   # only deliberately
   ```
6. **Detach the old KB.** Move its yaml to `kb/archive/`, then retire it (no longer searched; data kept):
   ```powershell
   & $py -m ragmvp.kbadmin --kb kb/archive/<old_id>.yaml retire --note "replaced by <new_id>"
   ```
7. **Point the system at the new KB:** set `base` and `search_audiences` in `kb/_profiles.yaml`, update `prompts/public.md` if the topic changed, and set `KB_CONFIG` in `.env`.
8. **Check:**
   ```powershell
   & $py -m ragmvp.kbadmin status
   & $py -m ragmvp.kbadmin --kb kb/<new_id>.yaml list
   & $py -m ragmvp.kbadmin --kb kb/<new_id>.yaml history
   & $py -m pytest -q
   & $py eval/run_eval.py
   ```
9. **Restart the API** to reload profiles and config (promotions and retirements are picked up without a restart):
   ```powershell
   & $py -m uvicorn ragmvp.api.main:app --port 8000
   ```

Undo:
```powershell
& $py -m ragmvp.kbadmin --kb kb/archive/<old_id>.yaml promote --note "restore"   # bring a retired KB back
& $py -m ragmvp.kbadmin --kb kb/<id>.yaml rollback                               # previous version of the same KB
```

### Update a knowledge base by adding a file

Live versions are immutable, so every change is a new version.

- **PDF KB:** copy the PDF into `source_dir`. Chunk ids are `DOCCODE-P<page>-<n>` and must be unique across all live KBs. Scanned PDFs without text give a warning (no OCR).
- **Markdown KB:** it is one file, so append sections with new unique ids (for example `AIR-18 Title`), or create a separate KB for a different topic or audience.
- Never put a public document in an internal KB (or the reverse): the KB's audience decides who can retrieve it.

Then bump `version` in the KB yaml, add golden questions for the new content (for PDFs, the document code alone matches all its pages) and run:

```powershell
& $py -m ragmvp.kbadmin --kb kb/<id>.yaml build        # unchanged text reuses cached embeddings
& $py -m ragmvp.kbadmin --kb kb/<id>.yaml promote --note "added <file>"
& $py -m ragmvp.kbadmin --kb kb/<id>.yaml list         # confirm the new version is live
```

For every KB of the catalog:
```powershell
& $py -m ragmvp.kbadmin build --all
& $py -m ragmvp.kbadmin promote --all --actor <name> --note "q4 update"
& $py -m ragmvp.kbadmin history --all
```

To remove a document: delete it from the source, bump the version, then build and promote. PDF KBs need the `local-embeddings` extra (`& $py -m pip install -e ".[local-embeddings]"`).



