"""FastAPI service: a thin, channel-neutral HTTP adapter over the core.

Contract: POST /v1/auth/login, POST /v1/auth/logout, GET /v1/me, POST /v1/public/session (anonymous, off by
default), POST /v1/chat (SSE), POST /v1/feedback, GET /v1/config, GET /v1/health.
Any front end (web UI, a voice gateway that sends transcribed text, another system) uses the same calls.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Iterator

from fastapi import Depends, FastAPI, HTTPException, Request as HttpRequest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from ragmvp.api.deps import Principal, auth_http_error, get_principal
from ragmvp.api.middleware import apply_input_hooks, install_http_middleware
from ragmvp.api.schemas import (
    ChatRequest,
    ConfigResponse,
    FeedbackRequest,
    LoginRequest,
    MeResponse,
    TokenResponse,
)
from ragmvp.auth.service import AuthError, AuthService
from ragmvp.config import get_settings
from ragmvp.core.models import STATUS_BUDGET, Request, Response
from ragmvp.core.orchestrator import Orchestrator

log = logging.getLogger("ragmvp")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _client(request: HttpRequest) -> tuple[str, str]:
    # The direct peer address. Behind a reverse proxy, configure the server to pass the real client address.
    return (request.client.host if request.client else ""), request.headers.get("user-agent", "")


def create_app(orchestrator: Orchestrator | None = None, auth: AuthService | None = None) -> FastAPI:
    settings = get_settings()
    state: dict = {}
    if orchestrator is not None:
        state["core"] = orchestrator

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if orchestrator is None:
            from ragmvp.factory import build_orchestrator

            state["core"] = build_orchestrator(settings)
            if settings.auth_enabled:
                app.state.auth = AuthService(state["core"].ops, settings)
        yield

    app = FastAPI(title="RAG API", version="2", lifespan=lifespan)
    app.state.auth = auth
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[*settings.cors_origins, *settings.public_cors_origins],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"],
    )
    install_http_middleware(app)

    def service() -> AuthService:
        svc = app.state.auth
        if svc is None:
            raise HTTPException(status_code=404, detail="disabled")
        return svc

    @app.get("/v1/health")
    def health():
        return {"status": "ok"}

    @app.post("/v1/auth/login", response_model=TokenResponse)
    def login(body: LoginRequest, request: HttpRequest):
        ip, ua = _client(request)
        try:
            return TokenResponse(access_token=service().login(body.username, body.password, ip, ua))
        except AuthError as exc:
            raise auth_http_error(exc) from None

    @app.post("/v1/auth/logout", status_code=204)
    def logout(principal: Principal = Depends(get_principal)):
        if app.state.auth is not None:
            app.state.auth.logout(principal.session_id)

    @app.get("/v1/me", response_model=MeResponse)
    def me(principal: Principal = Depends(get_principal)):
        return MeResponse(
            user_id=principal.user_id,
            department=principal.department,
            audience=principal.audience,
            anonymous=principal.anonymous,
            session_id=principal.session_id,
        )

    @app.post("/v1/public/session", response_model=TokenResponse)
    def public_session(request: HttpRequest):
        """Anonymous visitor session for the public page. 404 unless PUBLIC_ACCESS_ENABLED."""
        ip, ua = _client(request)
        try:
            return TokenResponse(access_token=service().create_anonymous_session(ip, ua))
        except AuthError as exc:
            raise auth_http_error(exc) from None

    @app.get("/v1/config", response_model=ConfigResponse)
    def get_config(principal: Principal = Depends(get_principal)):
        kb = state["core"].profile_for(principal.audience).kb
        return ConfigResponse(
            title=kb.ui.title,
            disclaimer=kb.ui.disclaimer,
            languages=kb.languages,
            kb_id=kb.id,
            kb_version=kb.version,
        )

    @app.post("/v1/chat")
    def chat(req: ChatRequest, principal: Principal = Depends(get_principal)):
        core: Orchestrator = state["core"]
        profile = core.profile_for(principal.audience)
        lang = req.lang or profile.kb.source_language

        def notice(status: str, key: str):
            text = profile.kb.message(key, lang)
            yield sse("token", {"text": text})
            yield sse("done", {"message_id": "", "status": status, "sources": [], "flags": [key], "invalid_citations": []})

        if principal.anonymous and app.state.auth is not None:
            if not app.state.auth.allow_chat(principal.session_id):
                return StreamingResponse(notice("rate_limited", "rate_limited"), media_type="text/event-stream")
            if not app.state.auth.public_budget_left():
                return StreamingResponse(notice(STATUS_BUDGET, "budget"), media_type="text/event-stream")

        request = Request(
            user_id=principal.user_id,
            session_id=principal.session_id or req.session_id or principal.user_id,
            text=apply_input_hooks(req.message, principal),
            audience=principal.audience,
            lang=req.lang,
            department=principal.department,
        )

        def events() -> Iterator[str]:
            try:
                for item in core.handle(request):
                    if isinstance(item, Response):
                        yield sse(
                            "done",
                            {
                                "message_id": item.message_id,
                                "status": item.status,
                                "sources": item.sources,
                                "flags": item.flags,
                                "invalid_citations": item.invalid_citations,
                            },
                        )
                    else:
                        yield sse("token", {"text": item})
            except Exception as exc:  # the client gets a generic error; details stay in server logs
                log.exception("chat failed")
                yield sse("error", {"message": "No se pudo generar la respuesta.", "type": type(exc).__name__})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.post("/v1/feedback", status_code=204)
    def feedback(req: FeedbackRequest, principal: Principal = Depends(get_principal)):
        ops = state["core"].ops
        if req.rating not in (1, -1):
            raise HTTPException(status_code=422, detail="rating must be 1 or -1")
        if not ops.interaction_exists(req.message_id):
            raise HTTPException(status_code=404, detail="unknown message_id")
        if principal.session_id and ops.interaction_session(req.message_id) != principal.session_id:
            raise HTTPException(status_code=404, detail="unknown message_id")
        ops.add_feedback(req.message_id, principal.user_id, req.rating, req.comment)

    return app


app = create_app()
