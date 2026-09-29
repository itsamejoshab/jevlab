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
from collections import Counter
from urllib.parse import urlparse

from .config import DATA

PATH = DATA / "net_errors.log"
counts: Counter[tuple[str, str, str]] = Counter()


def host_of(url: str) -> str:
    return urlparse(url).hostname or url


def describe(service: str, host: str, endpoint: str, status: str) -> str:
    return f"{service} {status} from {host}{endpoint}"


def record(service: str, url: str, endpoint: str, status: int | str, detail: str = "") -> str:
    """Log one failure; returns a one-line description naming the host."""
    host = host_of(url)
    status = f"HTTP {status}" if isinstance(status, int) else str(status)
    counts[(service, f"{host}{endpoint}", status)] += 1
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True)
        with PATH.open("a") as f:
            f.write(json.dumps({"at": time.time(), "service": service, "host": host, "endpoint": endpoint,
                                "status": status, "detail": " ".join(detail.split())[:300]}) + "\n")
    except OSError:
        pass
    text = describe(service, host, endpoint, status)
    return f"{text}: {' '.join(detail.split())[:160]}" if detail else text


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
