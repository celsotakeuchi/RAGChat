"""Users, login, server-issued sessions and anonymous public sessions.

A session row is created on the server at login (or when an anonymous visitor opens the public page)
and the JWT carries only its id. Every request re-checks the row, so logout, expiry and disabling a user
take effect immediately. Audience and department always come from the database, never from the client.
"""
from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import jwt

from ragmvp.auth.passwords import DUMMY_HASH, hash_password, verify_password
from ragmvp.auth.principal import Principal
from ragmvp.config import Settings
from ragmvp.ops.store import OpsStore

ALGORITHM = "HS256"
INTERNAL = "internal"
PUBLIC = "public"


class AuthError(Exception):
    """`code` is safe to return to the client; it never says whether a user exists."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _fmt(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


class AuthService:
    def __init__(self, ops: OpsStore, settings: Settings):
        if not settings.jwt_secret or len(settings.jwt_secret) < 16:
            raise ValueError("JWT_SECRET must be set (at least 16 characters) when authentication is enabled")
        self._ops = ops
        self._s = settings
        self._rate: dict[str, deque[float]] = defaultdict(deque)
        self._rate_lock = threading.Lock()

    # --- users (admin) -------------------------------------------------------------------------
    def create_user(self, username: str, password: str, department: str, role: str = "agent") -> None:
        username = username.strip().lower()
        if not username or not department.strip():
            raise ValueError("username and department are required")
        if self._ops.rows("SELECT 1 FROM users WHERE username=?", (username,)):
            raise ValueError(f"user '{username}' already exists")
        self._ops.run(
            "INSERT INTO users (username, password_hash, department, role) VALUES (?,?,?,?)",
            (username, hash_password(password), department.strip(), role),
        )

    def set_password(self, username: str, password: str) -> None:
        changed = self._ops.run(
            "UPDATE users SET password_hash=?, failed_attempts=0, locked_until=NULL WHERE username=?",
            (hash_password(password), username.strip().lower()),
        )
        if not changed:
            raise ValueError(f"unknown user '{username}'")

    def set_active(self, username: str, active: bool) -> None:
        username = username.strip().lower()
        if not self._ops.run("UPDATE users SET active=? WHERE username=?", (int(active), username)):
            raise ValueError(f"unknown user '{username}'")
        if not active:
            self._ops.run(
                "UPDATE sessions SET ended_at=? WHERE user_id=? AND ended_at IS NULL", (_fmt(_now()), username)
            )

    def list_users(self) -> list[dict]:
        rows = self._ops.rows(
            "SELECT username, department, role, active, failed_attempts, locked_until, created_at "
            "FROM users ORDER BY username"
        )
        return [dict(r) for r in rows]

    # --- login / sessions ----------------------------------------------------------------------
    def _issue(self, kind: str, user_id: str, department: str, audience: str, hours: int, ip: str, ua: str) -> str:
        now = _now()
        expires = now + timedelta(hours=hours)
        sid = uuid.uuid4().hex
        self._ops.run(
            "INSERT INTO sessions (session_id, kind, user_id, department, audience, created_at, expires_at, "
            "last_seen_at, ip_hash, user_agent) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (sid, kind, user_id, department, audience, _fmt(now), _fmt(expires), _fmt(now), self._hash_ip(ip), ua[:200]),
        )
        claims = {"sub": user_id, "sid": sid, "iat": int(now.timestamp()), "exp": int(expires.timestamp())}
        return jwt.encode(claims, self._s.jwt_secret, algorithm=ALGORITHM)

    def _hash_ip(self, ip: str) -> str:
        salt = self._s.ip_hash_salt or self._s.jwt_secret
        return hashlib.sha256(f"{salt}:{ip}".encode()).hexdigest()[:32]

    def login(self, username: str, password: str, ip: str = "", user_agent: str = "") -> str:
        username = username.strip().lower()
        rows = self._ops.rows("SELECT * FROM users WHERE username=?", (username,))
        user = rows[0] if rows else None
        now = _now()
        if user is None:
            verify_password(password, DUMMY_HASH)  # keep timing similar for unknown users
            raise AuthError("invalid_credentials")
        if user["locked_until"] and user["locked_until"] > _fmt(now):
            raise AuthError("locked")
        if not verify_password(password, user["password_hash"]) or not user["active"]:
            attempts = user["failed_attempts"] + 1
            locked = (
                _fmt(now + timedelta(minutes=self._s.lockout_minutes))
                if attempts >= self._s.max_failed_logins
                else None
            )
            self._ops.run(
                "UPDATE users SET failed_attempts=?, locked_until=? WHERE username=?",
                (0 if locked else attempts, locked, username),
            )
            raise AuthError("invalid_credentials")
        self._ops.run("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE username=?", (username,))
        return self._issue("user", username, user["department"], INTERNAL, self._s.session_hours, ip, user_agent)

    def create_anonymous_session(self, ip: str = "", user_agent: str = "") -> str:
        if not self._s.public_access_enabled:
            raise AuthError("disabled")
        since = _fmt(_now() - timedelta(hours=1))
        recent = self._ops.rows(
            "SELECT COUNT(*) c FROM sessions WHERE kind='anonymous' AND ip_hash=? AND created_at>=?",
            (self._hash_ip(ip), since),
        )[0]["c"]
        if recent >= self._s.public_ip_sessions_per_hour:
            raise AuthError("rate_limited")
        user_id = "anon:" + uuid.uuid4().hex[:12]
        return self._issue("anonymous", user_id, "", PUBLIC, self._s.public_session_hours, ip, user_agent)

    def authenticate(self, token: str) -> Principal:
        try:
            claims = jwt.decode(token, self._s.jwt_secret, algorithms=[ALGORITHM])
        except jwt.PyJWTError:
            raise AuthError("invalid_token") from None
        rows = self._ops.rows("SELECT * FROM sessions WHERE session_id=?", (claims.get("sid", ""),))
        if not rows:
            raise AuthError("invalid_token")
        s = rows[0]
        if s["ended_at"] or s["expires_at"] <= _fmt(_now()) or s["user_id"] != claims.get("sub"):
            raise AuthError("session_ended")
        if s["kind"] == "anonymous":
            if not self._s.public_access_enabled:
                raise AuthError("disabled")
        else:
            user = self._ops.rows("SELECT active FROM users WHERE username=?", (s["user_id"],))
            if not user or not user[0]["active"]:
                raise AuthError("session_ended")
        self._ops.run("UPDATE sessions SET last_seen_at=? WHERE session_id=?", (_fmt(_now()), s["session_id"]))
        anonymous = s["kind"] == "anonymous"
        return Principal(
            user_id=s["user_id"],
            roles=("visitor",) if anonymous else tuple(self._role(s["user_id"])),
            audience=s["audience"],
            department=s["department"],
            session_id=s["session_id"],
            anonymous=anonymous,
        )

    def _role(self, username: str) -> list[str]:
        rows = self._ops.rows("SELECT role FROM users WHERE username=?", (username,))
        return [rows[0]["role"]] if rows else []

    def logout(self, session_id: str) -> None:
        self._ops.run(
            "UPDATE sessions SET ended_at=? WHERE session_id=? AND ended_at IS NULL", (_fmt(_now()), session_id)
        )

    # --- abuse protection (public) -------------------------------------------------------------
    def allow_chat(self, session_id: str) -> bool:
        """Sliding one-minute window per anonymous session (single API process)."""
        limit = self._s.public_rate_per_minute
        if not limit:
            return True
        now = time.monotonic()
        with self._rate_lock:
            window = self._rate[session_id]
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= limit:
                return False
            window.append(now)
            return True

    def public_budget_left(self) -> bool:
        cap = self._s.public_daily_token_budget
        return not cap or self._ops.tokens_used_today_by_prefix("anon:") < cap

    # --- monitoring / retention ----------------------------------------------------------------
    def list_sessions(self, kind: str | None = None, limit: int = 50) -> list[dict]:
        where, params = ("WHERE s.kind=?", (kind,)) if kind else ("", ())
        rows = self._ops.rows(
            "SELECT s.session_id, s.kind, s.user_id, s.department, s.audience, s.created_at, s.expires_at, "
            "s.ended_at, s.last_seen_at, COUNT(i.message_id) AS questions, "
            "COALESCE(SUM(i.prompt_tokens + i.completion_tokens), 0) AS tokens, "
            "COALESCE(SUM(i.status IN ('budget_exceeded','busy')), 0) AS limited "
            f"FROM sessions s LEFT JOIN interactions i ON i.session_id = s.session_id {where} "
            "GROUP BY s.session_id ORDER BY s.created_at DESC LIMIT ?",
            (*params, limit),
        )
        return [dict(r) for r in rows]

    def end_session(self, session_id: str) -> None:
        self.logout(session_id)

    def purge_anonymous(self, older_than_days: int | None = None) -> int:
        days = self._s.public_retention_days if older_than_days is None else older_than_days
        cutoff = _fmt(_now() - timedelta(days=days))
        old = "(SELECT session_id FROM sessions WHERE kind='anonymous' AND created_at < ?)"
        self._ops.run(
            f"DELETE FROM feedback WHERE message_id IN (SELECT message_id FROM interactions WHERE session_id IN {old})",
            (cutoff,),
        )
        self._ops.run(f"DELETE FROM interactions WHERE session_id IN {old}", (cutoff,))
        self._ops.run(
            "DELETE FROM usage WHERE user_id IN (SELECT user_id FROM sessions WHERE kind='anonymous' AND created_at < ?)",
            (cutoff,),
        )
        return self._ops.run("DELETE FROM sessions WHERE kind='anonymous' AND created_at < ?", (cutoff,))
