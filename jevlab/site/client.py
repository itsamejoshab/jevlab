"""Trick Jev site client: TanStack Start server functions over the TSS framed protocol.

Every call takes `edition` ("jev" or "kev"), the attempt comes from mode-run,
the player from viewer, and `$TSR/Error` responses raise instead of passing through.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

import httpx

from .. import netlog
from ..config import EDITION, ROOT, SITE_BASE

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
SESSION_PATH = ROOT / "session.json"

FN = {
    "menu": "21e5ff2a5575cc25835a6be236fe07a907bd73259540607b4c153abcc8937c15",
    "viewer": "c9e14d2d213879054bf83931aea7abca20a36b3907b95a4ee71c8436a2752854",
    "board": "0f083c76e9ebaeb9f82b571245913fe8bc7d08790dc02e14545fb03786698129",
    "run": "4a3e733a8051282b1f805d385c0c2fa7799e5e4bc87375d17d72d0368b72e08a",
    "words": "bb26c4213735d03f2301379db55e8978972b5abaff5b558bf4c75eb528ebd601",
    "start": "c49e789e02bdbad439759ffc112373aa52ff433f93e3db4e44ffd90621600f1f",
    "turn": "9cb8e4106d2172ba1a5d3c168b2e81a292e5eb9ba1f8a5fe214f8ebd9f0f833f",
    "jevers": "f63c00cea86972d48cc25ae9e1fbebf38c4c30f4204a58a74a633feb4ad205e5",
    "round": "1bb4ef8e22e1aa671dedcd14bb46a1e14b0a8a1dc8b9679c4a22974c1d5aad42",
}

MODES = ("strict_chain", "word_chain", "golf", "emoji")


class SiteError(RuntimeError):
    """A failed site call. str() names the host and server function; the checks below read only the message."""

    def __init__(self, message: str, code: str = "", where: str = "", status: int | str = ""):
        super().__init__(f"site {where}: {message}" if where else message)
        self.message = message
        self.code = code
        self.where = where
        self.status = status

    @property
    def server_down(self) -> bool:
        """5xx or no response at all: the site itself is struggling, worth a patient retry."""
        return (isinstance(self.status, int) and self.status >= 500) or self.status in ("timeout", "network")

    @property
    def busy(self) -> bool:
        text = f"{self.code} {self.message}".casefold()
        return any(p in text for p in ("already being scored", "phrase change", "reloading"))

    @property
    def rate_limited(self) -> bool:
        text = f"{self.code} {self.message}".casefold()
        return any(p in text for p in ("rate", "limit", "often", "slow down", "429", "too many"))


def _encode(value, ident: list[int]):
    if value is None:
        return {"t": 2, "s": 0}
    if isinstance(value, bool):
        return {"t": 2, "s": 2 if value else 3}
    if isinstance(value, (int, float)):
        return {"t": 0, "s": value}
    if isinstance(value, str):
        return {"t": 1, "s": value}
    if isinstance(value, dict):
        node_id = ident[0]
        ident[0] += 1
        return {
            "t": 10,
            "i": node_id,
            "p": {"k": list(value), "v": [_encode(item, ident) for item in value.values()]},
            "o": 0,
        }
    if isinstance(value, list):
        node_id = ident[0]
        ident[0] += 1
        return {"t": 9, "i": node_id, "a": [_encode(item, ident) for item in value], "o": 0}
    raise TypeError(type(value))


def _decode(node, refs: dict):
    if node == 0 or node is None:
        return None
    if not isinstance(node, dict) or "t" not in node:
        if isinstance(node, dict):
            return {key: _decode(child, refs) for key, child in node.items()}
        return node
    kind = node["t"]
    if kind == 4:
        return refs.get(node.get("i"))
    if kind in (0, 1, 5):
        return node.get("s")
    if kind == 2:
        return {0: None, 1: None, 2: True, 3: False}.get(node.get("s"))
    if kind == 9:
        items: list = []
        if "i" in node:
            refs[node["i"]] = items
        items.extend(None if child == 0 else _decode(child, refs) for child in node.get("a") or [])
        return items
    if kind in (10, 11):
        obj: dict = {}
        if "i" in node:
            refs[node["i"]] = obj
        props = node.get("p") or {"k": [], "v": []}
        for key, child in zip(props.get("k") or [], props.get("v") or []):
            obj[key] = _decode(child, refs)
        return obj
    if kind == 25:
        payload = node.get("s")
        return {"plugin": node.get("c"), "value": _decode(payload, refs) if isinstance(payload, dict) else payload}
    return {"_t": kind}


def parse_cookie_header(header: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in header.split(";"):
        part = part.strip()
        if part and "=" in part:
            name, value = part.split("=", 1)
            cookies[name.strip()] = value.strip()
    return cookies


def decode_jev_request(raw: str | None) -> dict | None:
    """`jevRequest` is JSON whose quotes are backslash-escaped once more."""
    if not raw:
        return None
    candidates = [raw]
    try:
        # Unescapes exactly one level, so quotes inside the question text keep their own escaping.
        candidates.append(json.loads(f'"{raw}"'))
    except json.JSONDecodeError:
        pass
    candidates.append(raw.replace('\\"', '"'))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, str):
            value = json.loads(value)
        if isinstance(value, dict):
            return value
    return None


class SiteClient:
    def __init__(self, edition: str = EDITION, delay: float = 0.01, session_path: Path = SESSION_PATH):
        self.edition = edition
        self.delay = delay
        self.session_path = session_path
        self._next_ok = 0.0
        self.cookies: dict[str, str] = {}
        if session_path.exists():
            self.cookies.update(json.loads(session_path.read_text()).get("cookies") or {})
        from_env = os.environ.get("JEV_COOKIE", "").strip()
        if from_env:
            self.cookies.update(parse_cookie_header(from_env))
        self.http = httpx.Client(timeout=60, headers={"user-agent": UA})

    def save_session(self) -> None:
        self.session_path.write_text(json.dumps({"cookies": self.cookies}, indent=2))

    def _headers(self) -> dict:
        headers = {
            "accept": "application/x-tss-framed, application/x-ndjson, application/json",
            "x-tsr-serverFn": "true",
            "origin": SITE_BASE,
            "referer": SITE_BASE + ("/kev/" if self.edition == "kev" else "/"),
        }
        if self.cookies:
            headers["cookie"] = "; ".join(f"{k}={v}" for k, v in self.cookies.items())
        return headers

    def call(self, name: str, data: dict, method: str = "GET"):
        try:
            return self._call(name, data, method)
        except SiteError as error:
            if not error.where:
                error.where = f"{netlog.host_of(SITE_BASE)} {name}"
                error.args = (f"site {error.where}: {error.message}",)
            netlog.record("site", SITE_BASE, f" {name}", error.status or "app error",
                          f"{error.code} {error.message}".strip())
            raise

    def _call(self, name: str, data: dict, method: str = "GET"):
        function_id = FN.get(name, name)
        payload = json.dumps({"t": _encode({"data": data}, [0]), "f": 127, "m": []}, separators=(",", ":"))
        url = f"{SITE_BASE}/_serverFn/{function_id}"
        try:
            if method == "GET":
                response = self.http.get(url, params={"payload": payload}, headers=self._headers())
            else:
                headers = self._headers() | {"content-type": "application/json"}
                response = self.http.post(url, content=payload, headers=headers)
        except httpx.TimeoutException as error:
            raise SiteError(f"no reply ({error!r})", status="timeout") from error
        except httpx.TransportError as error:
            raise SiteError(f"connection failed ({error!r})", status="network") from error
        for name_, value in response.cookies.items():
            self.cookies[name_] = value
        try:
            decoded = _decode(response.json(), {})
        except json.JSONDecodeError as error:
            raise SiteError(f"HTTP {response.status_code}: {response.text[:200]!r}",
                            status=response.status_code) from error
        if not isinstance(decoded, dict):
            raise SiteError(f"unexpected response HTTP {response.status_code}", status=response.status_code)
        error = decoded.get("error")
        if isinstance(error, dict):
            value = error.get("value") or {}
            message = value.get("message") if isinstance(value, dict) else value
            raise SiteError(f"{error.get('plugin') or 'error'}: {message}",
                            status=response.status_code if response.status_code >= 400 else "")
        result = decoded.get("result")
        if isinstance(result, dict) and result.get("ok") is False:
            err = result.get("error") or {}
            raise SiteError(err.get("message") or "request failed", err.get("code") or "")
        if response.status_code >= 400:
            raise SiteError(f"HTTP {response.status_code}: {response.text[:240]}", status=response.status_code)
        if isinstance(result, dict) and "data" in result:
            return result["data"]
        return result

    # Reads (no pacing needed).

    def menu(self) -> dict:
        return self.call("menu", {"edition": self.edition})

    def viewer(self) -> dict:
        return self.call("viewer", {"edition": self.edition})

    def leaderboards(self, revision_id: str, mode: str) -> dict:
        return self.call("board", {"revisionId": revision_id, "playMode": mode}) or {}

    def attempt(self, revision_id: str, mode: str) -> dict | None:
        return self.call("run", {"edition": self.edition, "revisionId": revision_id, "playMode": mode})

    def words(self, revision_id: str, mode: str) -> list[dict]:
        return list(self.call("words", {"revisionId": revision_id, "playMode": mode}) or [])

    def jevers(self) -> dict:
        return self.call("jevers", {"edition": self.edition})

    # Writes (paced).

    def _pace(self) -> None:
        wait = self._next_ok - time.monotonic()
        if wait > 0:
            time.sleep(wait)

    def start(self, revision_id: str, mode: str) -> dict:
        self._pace()
        try:
            return self.call("start", {"revisionId": revision_id, "playMode": mode}, "POST")
        finally:
            self._next_ok = time.monotonic() + self.delay

    def turn(self, attempt: dict, operation: dict) -> tuple[dict, dict]:
        self._pace()
        try:
            data = self.call(
                "turn",
                {
                    "attemptId": attempt["id"],
                    "clientRequestId": str(uuid.uuid4()),
                    "expectedTurn": attempt["nextTurn"],
                    "operation": operation,
                },
                "POST",
            )
        finally:
            self._next_ok = time.monotonic() + self.delay
        updated = data["attempt"]
        turn = data["turn"]
        turns = list(updated.get("turns") or [])
        if not any(item.get("id") == turn.get("id") for item in turns):
            turns.append(turn)
        updated["turns"] = turns
        return updated, turn


def active_words(attempt: dict | None) -> list[str]:
    if not attempt:
        return []
    return [t.get("text") or "" for t in attempt.get("tokens") or [] if t.get("active", True)]
