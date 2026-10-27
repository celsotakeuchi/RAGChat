"""Agent UI. A pure HTTP client of the /v1 API: no backend imports, no database, no LLM calls.

Users log in with the API's accounts (created with `python -m ragmvp.admin user-add`); the API issues the
session and token. Chainlit needs CHAINLIT_AUTH_SECRET set (any long random string) to enable login.

Run:  chainlit run ui/chainlit_app.py --port 8001
"""
from __future__ import annotations

import json
import os

import chainlit as cl
import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")


def _headers() -> dict[str, str]:
    user = cl.user_session.get("user")
    token = (user.metadata or {}).get("token", "") if user else ""
    return {"Authorization": f"Bearer {token}"} if token else {}


def _format_sources(sources: list[dict]) -> str:
    """The document (and page) is the source; the entry's web links are shown as its origin."""
    lines = []
    for s in sources:
        where = s.get("document", "")
        if s.get("page"):
            where += f", p. {s['page']}"
        detail = "; ".join(
            part
            for part in (
                f"fiabilidad {s['reliability']}" if s.get("reliability") else "",
                f"revisar antes de {s['review_before']}" if s.get("review_before") else "",
            )
            if part
        )
        line = f"- **[{s['id']}] {s['title']}** — _{where}_" + (f" ({detail})" if detail else "")
        refs = s.get("references", [])[:2]
        if refs:
            line += "\n  Origen: " + " · ".join(refs)
        lines.append(line)
    return "\n".join(lines)


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{API_URL}/v1/auth/login", json={"username": username, "password": password})
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    return cl.User(identifier=username, metadata={"token": resp.json()["access_token"]})


@cl.on_chat_start
async def start():
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{API_URL}/v1/config", headers=_headers())
            me = await client.get(f"{API_URL}/v1/me", headers=_headers())
        if resp.status_code == 401:
            await cl.Message(content="La sesión ha caducado. Vuelve a iniciar sesión.").send()
            return
        cfg = resp.json()
        who = f" · {me.json()['user_id']} ({me.json()['department']})" if me.status_code == 200 else ""
        await cl.Message(content=f"**{cfg['title']}**{who}\n\n_{cfg['disclaimer']}_").send()
    except httpx.HTTPError:
        await cl.Message(content=f"No se puede conectar con la API en {API_URL}.").send()


@cl.on_message
async def on_message(message: cl.Message):
    reply = cl.Message(content="")
    done: dict = {}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=10)) as client:
            async with client.stream(
                "POST",
                f"{API_URL}/v1/chat",
                json={"message": message.content},  # the session id is issued and tracked by the server
                headers=_headers(),
            ) as resp:
                if resp.status_code == 401:
                    await cl.Message(content="La sesión ha caducado. Cierra sesión y vuelve a entrar.").send()
                    return
                resp.raise_for_status()
                event = ""
                async for line in resp.aiter_lines():
                    if line.startswith("event: "):
                        event = line.removeprefix("event: ")
                    elif line.startswith("data: "):
                        data = json.loads(line.removeprefix("data: "))
                        if event == "token":
                            await reply.stream_token(data["text"])
                        elif event == "done":
                            done = data
                        elif event == "error":
                            await reply.stream_token(f"\n\n⚠️ {data['message']}")
    except httpx.HTTPError as exc:
        await reply.stream_token(f"⚠️ Error de conexión con la API ({type(exc).__name__}).")
        await reply.send()
        return

    if done.get("sources"):
        reply.content += "\n\n**Fuentes**\n" + _format_sources(done["sources"])
    if done.get("message_id"):
        reply.actions = [
            cl.Action(name="feedback", label="👍", payload={"message_id": done["message_id"], "rating": 1}),
            cl.Action(name="feedback", label="👎", payload={"message_id": done["message_id"], "rating": -1}),
        ]
    await reply.send()


@cl.action_callback("feedback")
async def on_feedback(action: cl.Action):
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(f"{API_URL}/v1/feedback", json=action.payload, headers=_headers())
    await cl.Message(content="Gracias por tu valoración." if resp.status_code == 204 else "No se pudo guardar la valoración.").send()
