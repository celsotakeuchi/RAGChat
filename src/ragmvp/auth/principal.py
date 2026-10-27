"""Who is calling. Built only on the server from a validated session, never from client input."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Principal:
    user_id: str
    roles: tuple[str, ...] = field(default_factory=tuple)
    audience: str = "internal"  # profile name: "internal" or "public"
    department: str = ""
    session_id: str = ""
    anonymous: bool = False
