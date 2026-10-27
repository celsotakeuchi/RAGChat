"""Identity seam.

Every endpoint depends on `get_principal`. With authentication enabled (the default) it validates the
bearer token, re-checks the server-side session and returns the `Principal` stored for that session;
`audience` and `department` therefore never come from the client. Without an `AuthService`
(tests / local experiments, AUTH_ENABLED=false) it returns a fixed development principal.
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from ragmvp.auth.principal import Principal
from ragmvp.auth.service import AuthError

__all__ = ["Principal", "DEV_PRINCIPAL", "get_principal"]

DEV_PRINCIPAL = Principal(user_id="dev", roles=("agent",), audience="internal")

STATUS_BY_CODE = {"disabled": 404, "rate_limited": 429, "locked": 429}


def auth_http_error(exc: AuthError) -> HTTPException:
    return HTTPException(
        status_code=STATUS_BY_CODE.get(exc.code, 401),
        detail=exc.code,
        headers={"WWW-Authenticate": "Bearer"} if exc.code not in STATUS_BY_CODE else None,
    )


def get_principal(request: Request) -> Principal:
    auth = getattr(request.app.state, "auth", None)
    if auth is None:
        return DEV_PRINCIPAL
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="missing_token", headers={"WWW-Authenticate": "Bearer"})
    try:
        return auth.authenticate(token.strip())
    except AuthError as exc:
        raise auth_http_error(exc) from None
