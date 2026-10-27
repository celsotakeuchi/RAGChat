from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    # Ignored when authentication is on: the server uses the session id of the token.
    session_id: str | None = Field(default=None, max_length=100)
    message: str = Field(min_length=1, max_length=4000)
    lang: str | None = Field(default=None, max_length=10)


class FeedbackRequest(BaseModel):
    message_id: str
    rating: int = Field(description="1 = helpful, -1 = not helpful")
    comment: str | None = Field(default=None, max_length=2000)


class ConfigResponse(BaseModel):
    title: str
    disclaimer: str
    languages: list[str]
    kb_id: str
    kb_version: str


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class MeResponse(BaseModel):
    user_id: str
    department: str
    audience: str
    anonymous: bool
    session_id: str
