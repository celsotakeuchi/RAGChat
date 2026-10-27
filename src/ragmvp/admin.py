"""Administration CLI: users, sessions and retention.

  python -m ragmvp.admin user-add USERNAME --department DEPT [--role agent|admin]
  python -m ragmvp.admin user-list
  python -m ragmvp.admin user-passwd USERNAME
  python -m ragmvp.admin user-disable USERNAME | user-enable USERNAME
  python -m ragmvp.admin sessions [--public | --internal] [--limit 50]
  python -m ragmvp.admin end-session SESSION_ID
  python -m ragmvp.admin purge-public [--days N]
"""
from __future__ import annotations

import argparse
import getpass
import sys

from ragmvp.auth.service import AuthService
from ragmvp.config import get_settings
from ragmvp.ops.store import OpsStore


def _password(args) -> str:
    if getattr(args, "password", None):
        return args.password
    first = getpass.getpass("Password: ")
    if first != getpass.getpass("Repeat password: "):
        raise ValueError("passwords do not match")
    return first


def main(argv: list[str] | None = None, auth: AuthService | None = None) -> int:
    parser = argparse.ArgumentParser(prog="admin")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("user-add")
    p.add_argument("username")
    p.add_argument("--department", required=True)
    p.add_argument("--role", default="agent")
    p.add_argument("--password", default=None, help="omit to be prompted (recommended)")
    sub.add_parser("user-list")
    p = sub.add_parser("user-passwd")
    p.add_argument("username")
    p.add_argument("--password", default=None)
    for name in ("user-disable", "user-enable", "end-session"):
        sub.add_parser(name).add_argument("target")
    p = sub.add_parser("sessions")
    p.add_argument("--public", action="store_true")
    p.add_argument("--internal", action="store_true")
    p.add_argument("--limit", type=int, default=50)
    p = sub.add_parser("purge-public")
    p.add_argument("--days", type=int, default=None)
    args = parser.parse_args(argv)

    if auth is None:
        settings = get_settings()
        auth = AuthService(OpsStore(settings.ops_db_file), settings)
    try:
        if args.cmd == "user-add":
            auth.create_user(args.username, _password(args), args.department, args.role)
            print(f"User {args.username.lower()} created in department {args.department}")
        elif args.cmd == "user-list":
            for u in auth.list_users():
                state = "active" if u["active"] else "DISABLED"
                lock = f" locked until {u['locked_until']}" if u["locked_until"] else ""
                print(f"{u['username']:<20} {u['department']:<15} {u['role']:<8} {state}{lock}")
        elif args.cmd == "user-passwd":
            auth.set_password(args.username, _password(args))
            print("Password updated")
        elif args.cmd in ("user-disable", "user-enable"):
            auth.set_active(args.target, args.cmd == "user-enable")
            print(f"{args.target}: {'enabled' if args.cmd == 'user-enable' else 'disabled, sessions ended'}")
        elif args.cmd == "end-session":
            auth.end_session(args.target)
            print("Session ended")
        elif args.cmd == "sessions":
            kind = "anonymous" if args.public else "user" if args.internal else None
            for s in auth.list_sessions(kind, args.limit):
                print(
                    f"{s['session_id'][:12]} {s['kind']:<9} {s['user_id']:<20} {s['department']:<12} "
                    f"{s['created_at']} q={s['questions']} tokens={s['tokens']} limited={s['limited']}"
                    f"{' ended' if s['ended_at'] else ''}"
                )
        else:
            print(f"Removed {auth.purge_anonymous(args.days)} anonymous sessions (with their questions and answers)")
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
