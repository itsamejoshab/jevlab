"""Every failed HTTP call, named by service and host, so a 503 says whether it came from OpenRouter or the site.

Three services talk to the network:
  jev-oracle  api.typesafe.ai or openrouter.ai /systemone   Jev scoring phrases (search, publish checks)
  llm         openrouter.ai /chat/completions  idea writers and the planner
  site        i-wanna-date-jev ... /_serverFn  the game site (snapshot, boards, publishing turns)

Failures are appended to data/net_errors.log (one JSON line each, shared by the lab, TUI and publisher);
`jevlab errors` summarizes it.
"""

from __future__ import annotations

import json
import time
from collections import Counter, deque
from urllib.parse import urlparse

from .config import DATA

PATH = DATA / "net_errors.log"
counts: Counter[tuple[str, str, str]] = Counter()
# The search bar reads this, not `counts`: a lifetime total paints the bar for the rest of the run after one blip.
LIVE_WINDOW = 60.0
LIVE_MINIMUM = 3
_recent: deque[tuple[float, str, str, str]] = deque()


def host_of(url: str) -> str:
    return urlparse(url).hostname or url


def short_host(host: str) -> str:
    """openrouter.ai -> openrouter, api.typesafe.ai -> typesafe."""
    labels = [part for part in host.split(".") if part]
    if labels and labels[0] in {"api", "www"}:
        labels = labels[1:]
    return labels[0] if labels else host


def describe(service: str, host: str, endpoint: str, status: str) -> str:
    return f"{service} {status} from {host}{endpoint}"


def _prune(now: float) -> None:
    cutoff = now - LIVE_WINDOW
    while _recent and _recent[0][0] < cutoff:
        _recent.popleft()


def record(service: str, url: str, endpoint: str, status: int | str, detail: str = "") -> str:
    """Log one failure; returns a one-line description naming the host."""
    host = host_of(url)
    status = f"HTTP {status}" if isinstance(status, int) else str(status)
    counts[(service, f"{host}{endpoint}", status)] += 1
    now = time.monotonic()
    _recent.append((now, service, short_host(host), status))
    _prune(now)
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with PATH.open("a") as f:
            f.write(json.dumps({"at": time.time(), "service": service, "host": host, "endpoint": endpoint,
                                "status": status, "detail": " ".join(detail.split())[:300]}) + "\n")
    except OSError:
        pass
    text = describe(service, host, endpoint, status)
    return f"{text}: {' '.join(detail.split())[:160]}" if detail else text


def recent_counts(window: float = LIVE_WINDOW) -> list[tuple[str, str, str, int]]:
    """Failures inside `window` seconds: (service, short host, status, count), most common first."""
    now = time.monotonic()
    _prune(now)
    cutoff = now - window
    grouped: Counter[tuple[str, str, str]] = Counter()
    for at, service, host, status in _recent:
        if at >= cutoff:
            grouped[(service, host, status)] += 1
    return [(service, host, status, n) for (service, host, status), n in grouped.most_common()]


def live_line(concurrency: tuple[int, int] | None = None, *, minimum: int = LIVE_MINIMUM) -> str:
    """Compact search-bar chip for trouble in the last minute. Empty below `minimum` failures.

    A lone timeout never appears, and a burst leaves the bar a minute after it stops.
    `concurrency` is (current limit, cap). The cap is mentioned only when the limiter is below it.
    """
    rows = recent_counts()
    if sum(n for *_, n in rows) < minimum:
        return ""
    grouped: dict[tuple[str, str], list[tuple[str, int]]] = {}
    order: list[tuple[str, str]] = []
    for service, host, status, n in rows:
        key = (service, host)
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append((status.removeprefix("HTTP "), n))
    parts = []
    for service, host in order:
        kinds = " ".join(f"{status}×{n}" for status, n in grouped[(service, host)])
        who = host if service == "jev-oracle" else f"{service} {host}"
        parts.append(f"{who} {kinds}")
    text = "net/min " + " · ".join(parts)
    if concurrency is not None:
        limit, cap = concurrency
        if cap and limit < cap:
            text += f" · conc {limit}/{cap}"
    return text


def summary(service: str | None = None) -> str:
    """This process's failures so far, e.g. 'jev-oracle HTTP 503 from openrouter.ai/systemone x12'."""
    parts = [f"{svc} {status} from {where} x{n}" for (svc, where, status), n in counts.most_common()
             if service is None or svc == service]
    return "; ".join(parts)


def recent(minutes: float = 60.0) -> list[dict]:
    if not PATH.exists():
        return []
    since = time.time() - minutes * 60
    out = []
    for line in PATH.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("at", 0) >= since:
            out.append(row)
    return out
