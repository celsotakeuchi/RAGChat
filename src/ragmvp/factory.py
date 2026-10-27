"""Composition root: wires adapters into the core. The only place that knows concrete classes."""
from __future__ import annotations

from ragmvp.config import (
    KBConfig,
    LLMConfig,
    ProfileConfig,
    Settings,
    get_settings,
    load_kb_config,
    load_profile_behavior,
    load_profiles,
)
from ragmvp.core.budget import BudgetLimits, TokenBudget
from ragmvp.core.orchestrator import Orchestrator, Profile
from ragmvp.core.prompting import PromptEngine
from ragmvp.llm.gateway import LiteLLMGateway
from ragmvp.ops.store import OpsStore
from ragmvp.retrieval.backends import RetrievalBackend, build_backend
from ragmvp.retrieval.rerank import wrap_with_reranker


def build_llm(settings: Settings, cfg: LLMConfig | None = None):
    """The model for a profile: its own settings if given, otherwise the global ones."""
    cfg = cfg or LLMConfig()
    provider = cfg.provider or settings.llm_provider
    if provider in ("ollama_cloud", "ollama_local", "ollama"):
        from ragmvp.llm.ollama_cloud import OllamaGateway

        local = provider == "ollama_local"
        host = cfg.host or ("http://localhost:11434" if local else settings.ollama_host)
        key = "" if local else settings.ollama_api_key
        llm = OllamaGateway(cfg.model or settings.ollama_model, key, host, timeout=cfg.request_timeout_s)
    else:
        llm = LiteLLMGateway(cfg.model or settings.chat_model)
    if cfg.max_concurrent:
        from ragmvp.llm.guards import LimitedLLM

        llm = LimitedLLM(llm, cfg.max_concurrent, cfg.queue_wait_s)
    if cfg.fallback:
        from ragmvp.llm.guards import FallbackLLM

        llm = FallbackLLM(llm, build_llm(settings, cfg.fallback))
    return llm


def build_retrieval(settings: Settings, kb: KBConfig):
    """The configured backend plus the retriever the core uses (with reranking if the KB asks for it)."""
    backend: RetrievalBackend = build_backend(settings)
    r = kb.retrieval
    return backend, wrap_with_reranker(backend.retriever, r.rerank, r.candidates, r.rerank_weight)


def _limits(settings: Settings, profile: ProfileConfig | None = None) -> BudgetLimits:
    own = profile.limits if profile else None

    def pick(value, default):
        return default if value is None else value

    return BudgetLimits(
        max_output_tokens=pick(own and own.max_output_tokens, settings.max_output_tokens),
        max_request_tokens=pick(own and own.max_request_tokens, settings.max_request_tokens),
        daily_tokens_per_user=pick(own and own.daily_tokens_per_user, settings.daily_tokens_per_user),
        monthly_token_budget=settings.monthly_token_budget,
    )


def build_profile(settings: Settings, ops: OpsStore, config: ProfileConfig) -> Profile:
    behavior = load_profile_behavior(config)
    return Profile(
        name=config.name,
        kb=behavior,
        prompts=PromptEngine(behavior),
        llm=build_llm(settings, config.llm),
        budget=TokenBudget(ops, _limits(settings, config)),
        search_audiences=config.search_audiences,
    )


def build_orchestrator(
    settings: Settings | None = None, kb: KBConfig | None = None, ops: OpsStore | None = None
) -> Orchestrator:
    settings = settings or get_settings()
    kb = kb or load_kb_config()
    ops = ops or OpsStore(settings.ops_db_file)
    _, retriever = build_retrieval(settings, kb)
    profiles = {name: build_profile(settings, ops, cfg) for name, cfg in load_profiles().items()}
    return Orchestrator(
        kb=kb,
        retriever=retriever,
        llm=build_llm(settings),
        ops=ops,
        budget=TokenBudget(ops, _limits(settings)),
        profiles=profiles,
    )
