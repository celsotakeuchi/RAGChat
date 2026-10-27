"""Settings (environment) and KB configuration (YAML)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    llm_provider: str = "litellm"  # "litellm" (uses chat_model) or "ollama_cloud"
    chat_model: str = "gpt-4o-mini"
    ollama_api_key: str = ""
    ollama_model: str = "gemma4:31b"
    ollama_host: str = "https://ollama.com"
    embedding_backend: str = "local"  # "local" (fastembed) or "litellm"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "intfloat/multilingual-e5-large"
    kb_db_path: str = "data/kb.sqlite3"
    ops_db_path: str = "data/ops.sqlite3"
    kb_config: str = "kb/air_passenger_rights.yaml"
    kb_defaults: str = "kb/_defaults.yaml"  # shared by every KB; each KB yaml overrides it
    retriever_backend: str = "sqlite"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8001"])
    api_url: str = "http://localhost:8000"
    # Authentication and sessions
    auth_enabled: bool = True  # false = fixed development principal (local experiments only)
    jwt_secret: str = ""
    session_hours: int = 8
    max_failed_logins: int = 5
    lockout_minutes: int = 15
    # Anonymous public access (off by default)
    public_access_enabled: bool = False
    public_session_hours: int = 2
    public_rate_per_minute: int = 10  # chat requests per session
    public_ip_sessions_per_hour: int = 20  # anonymous sessions one IP may open
    public_daily_token_budget: int = 200000  # global cap for all anonymous traffic (0 = unlimited)
    public_retention_days: int = 30
    ip_hash_salt: str = ""
    public_cors_origins: list[str] = Field(default_factory=list)
    profiles_config: str = "kb/_profiles.yaml"
    kb_catalog_dir: str = "kb"
    # Token limits (0 = unlimited, e.g. local Ollama)
    max_output_tokens: int = 700
    max_request_tokens: int = 6000
    daily_tokens_per_user: int = 100000
    monthly_token_budget: int = 0

    @staticmethod
    def _abs(path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else ROOT / p

    @property
    def kb_db_file(self) -> Path:
        return self._abs(self.kb_db_path)

    @property
    def ops_db_file(self) -> Path:
        return self._abs(self.ops_db_path)


class RetrievalConfig(BaseModel):
    top_k: int = 5
    min_score: float = 0.72
    rerank: str = "none"  # "none" or "lexical"
    candidates: int = 15  # how many hits to fetch before reranking down to top_k
    rerank_weight: float = 0.1


class ContextConfig(BaseModel):
    max_chars_per_source: int = 2500
    max_total_chars: int = 12000
    max_chunks: int = 5
    score_margin: float = 0.0  # drop sources scoring more than this below the best one (0 = off)


class GuardrailsConfig(BaseModel):
    max_question_chars: int = 1000
    blocked_patterns: list[str] = []
    require_citation: bool = True
    # Strict mode: an answer without a valid citation to a retrieved source is withheld and replaced
    # by the "unknown" message. The answer is buffered first, so it is not streamed token by token.
    enforce_grounding: bool = False
    # Ids cited by the model; group 1 is the id. Covers "ABC-123" and PDF ids like "MANUAL-VACACIONES-P12-3".
    citation_pattern: str = r"\[([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)\]"


class PromotionConfig(BaseModel):
    min_recall: float = 0.9


class UIConfig(BaseModel):
    title: str = "Assistant"
    disclaimer: str = ""


class KBConfig(BaseModel):
    id: str
    version: str
    audience: str
    department: str | None = None  # informational: which team owns the documents
    source_language: str = "es"
    languages: list[str] = ["es"]
    source_type: str = "md"  # "md" (one structured Markdown file) or "pdf" (a folder of PDFs)
    source_file: str = ""
    source_dir: str = ""
    chunk_chars: int = 1200  # PDF chunking: target characters per chunk
    chunk_overlap: int = 150
    golden_file: str = ""  # YAML of golden questions for KBs without a Block H (used by the promotion gate)
    system_prompt: str | None = None
    base_prompt: str | None = None  # shared rules prepended to every KB's own prompt
    retrieval: RetrievalConfig = RetrievalConfig()
    context: ContextConfig = ContextConfig()
    guardrails: GuardrailsConfig = GuardrailsConfig()
    promotion: PromotionConfig = PromotionConfig()
    ui: UIConfig = UIConfig()
    messages: dict[str, dict[str, str]] = {}

    def resolve(self, relative: str) -> Path:
        return ROOT / relative

    def prompt_text(self) -> str:
        parts = []
        if self.base_prompt:
            parts.append(self.resolve(self.base_prompt).read_text(encoding="utf-8").strip())
        if self.system_prompt:
            parts.append(self.resolve(self.system_prompt).read_text(encoding="utf-8").strip())
        return "\n\n".join(parts) + "\n"

    def message(self, key: str, lang: str) -> str:
        by_lang = self.messages.get(key, {})
        return by_lang.get(lang) or by_lang.get(self.source_language, "")


class LLMConfig(BaseModel):
    """Which model serves a profile. Empty fields fall back to the global settings."""

    provider: str = ""  # "ollama_cloud", "ollama_local" or "litellm"
    model: str = ""
    host: str = ""
    max_concurrent: int = 0  # simultaneous generations (0 = unlimited); set for a local model
    queue_wait_s: float = 5.0  # how long a request waits for a free slot before the "busy" message
    request_timeout_s: float = 60.0
    fallback: "LLMConfig | None" = None  # tried only if the primary fails before answering; off by default


class LimitsConfig(BaseModel):
    """Per-profile token limits; None keeps the global setting."""

    max_output_tokens: int | None = None
    max_request_tokens: int | None = None
    daily_tokens_per_user: int | None = None


class ProfileConfig(BaseModel):
    """An audience profile: how it behaves, which KB audiences it may search and which model serves it."""

    name: str = ""
    base: str  # KB yaml providing the behaviour (prompt, guardrails, retrieval, UI, messages)
    search_audiences: list[str]
    overrides: dict = {}
    llm: LLMConfig = LLMConfig()
    limits: LimitsConfig = LimitsConfig()


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge dicts; the KB overrides scalars, lists are concatenated (shared patterns + KB-specific ones)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        elif isinstance(value, list) and isinstance(out.get(key), list):
            out[key] = out[key] + [v for v in value if v not in out[key]]
        else:
            out[key] = value
    return out


def _abs(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_kb_config(path: str | Path | None = None, overrides: dict | None = None) -> KBConfig:
    settings = get_settings()
    p = _abs(path or settings.kb_config)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    defaults = _abs(settings.kb_defaults)
    if defaults.exists():
        data = _merge(yaml.safe_load(defaults.read_text(encoding="utf-8")) or {}, data)
    if overrides:
        data = _merge(data, overrides)
    return KBConfig(**data)


def load_catalog(directory: str | Path | None = None) -> list[KBConfig]:
    """Every KB declared in the catalog folder (yaml files not starting with an underscore)."""
    folder = _abs(directory or get_settings().kb_catalog_dir)
    return [load_kb_config(f) for f in sorted(folder.glob("*.yaml")) if not f.name.startswith("_")]


def load_profiles(path: str | Path | None = None) -> dict[str, ProfileConfig]:
    p = _abs(path or get_settings().profiles_config)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {name: ProfileConfig(name=name, **body) for name, body in raw.items()}


def load_profile_behavior(profile: ProfileConfig) -> KBConfig:
    return load_kb_config(profile.base, profile.overrides)
