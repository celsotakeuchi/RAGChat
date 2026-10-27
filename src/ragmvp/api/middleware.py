"""Ordered hook slots for protections.

- Input hooks run on the user message before retrieval and before it is logged. The default hook masks
  personal data typed by anonymous public visitors.
- `install_http_middleware` adds basic security headers.
"""
from __future__ import annotations

import re
from typing import Callable

from fastapi import FastAPI, Request

from ragmvp.api.deps import Principal

InputHook = Callable[[str, Principal], str]

PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[EMAIL]"),
    (re.compile(r"\b[A-Za-z]{2}\d{2}(?:[ ]?\d{4}){4,5}\b"), "[IBAN]"),
    (re.compile(r"\b[XYZxyz]?\d{7,8}[- ]?[A-Za-z]\b"), "[DNI]"),
    (re.compile(r"\b\d{4}[- ]?[BCDFGHJKLMNPRSTVWXYZbcdfghjklmnpqrstvwxyz]{3}\b"), "[MATRICULA]"),
    (re.compile(r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\d[ .-]?){8,}\d(?!\w)"), "[TELEFONO]"),
]


def mask_pii(text: str) -> str:
    for pattern, label in PII_PATTERNS:
        text = pattern.sub(label, text)
    return text


def mask_pii_for_public(text: str, principal: Principal) -> str:
    return mask_pii(text) if principal.anonymous else text


INPUT_HOOKS: list[InputHook] = [mask_pii_for_public]


def apply_input_hooks(text: str, principal: Principal) -> str:
    for hook in INPUT_HOOKS:
        text = hook(text, principal)
    return text


def install_http_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Cache-Control", "no-store")
        return response
