"""Check the OpenRouter oracle against probabilities the site actually recorded."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import threading
from collections import defaultdict

from .config import DATA
from .db import DB
from .objective import goal_p, site_round
from .oracle import Oracle

CALIBRATION_PATH = DATA / "calibration.json"
_lock = threading.Lock()


def _comparable(db: DB, slugs: list[str] | None = None) -> dict[str, list]:
    """Site scores a calibration can replay: noul questions that still carry a request body."""
    rows = db.all(
        "SELECT slug, mode, state, probability, source FROM site_scores WHERE source = 'turn' OR source LIKE 'board:%'"
    )
    by_slug: dict[str, list] = defaultdict(list)
    for row in rows:
        if slugs and row["slug"] not in slugs:
            continue
        question = db.question(row["slug"])
        if not question or question["kind"] != "noul" or not question["jev_request"]:
            continue
        by_slug[row["slug"]].append(row)
    return by_slug


def score_fingerprint(db: DB) -> str | None:
    """Identity of the comparable scores. Unchanged probabilities do not ask for another run."""
    by_slug = _comparable(db)
    if not by_slug:
        return None
    lines = []
    for slug, items in sorted(by_slug.items()):
        ordered = sorted(items, key=lambda row: (row["mode"], row["state"], row["source"]))
        for row in ordered:
            lines.append(f"{slug}\t{row['mode']}\t{row['state']}\t{float(row['probability']):.6f}\t{row['source']}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def same_model_as_site() -> bool:
    """True when this edition's oracle is the model that wrote the site scores."""
    from .config import EDITION, ESTIMATE_EDITIONS

    return EDITION not in ESTIMATE_EDITIONS


def calibration_due(db: DB) -> bool:
    if same_model_as_site():
        return False
    fingerprint = score_fingerprint(db)
    if fingerprint is None or not CALIBRATION_PATH.exists():
        return fingerprint is not None
    try:
        saved = json.loads(CALIBRATION_PATH.read_text()).get("fingerprint")
    except (OSError, json.JSONDecodeError):
        return True
    return saved != fingerprint


def ensure_calibration(db: DB, log=print) -> dict:
    """Replay site scores through the oracle when the saved report does not cover them."""
    with _lock:
        if not calibration_due(db):
            return {}
        log("calibrating oracle against site scores")
        return asyncio.run(calibrate(db, log=log))


async def calibrate(db: DB, slugs: list[str] | None = None, per_question: int = 12, n: int = 3, log=print) -> dict:
    by_slug = _comparable(db, slugs)

    pairs = []
    rng = random.Random(7)
    for slug, items in sorted(by_slug.items()):
        question = db.question(slug)
        if not question:
            continue
        unique = {r["state"]: r for r in items}
        chosen = rng.sample(list(unique.values()), min(per_question, len(unique)))
        oracle = Oracle(db, question["jev_request"])
        try:
            scores = await oracle.score([r["state"] for r in chosen], n=n)
        finally:
            await oracle.close()
        for r in chosen:
            s = scores[r["state"]]
            est = goal_p(s.mean, question["goal"])
            pairs.append(
                {
                    "slug": slug,
                    "state": r["state"],
                    "site": float(r["probability"]),
                    "oracle": est,
                    "spread": s.spread,
                    "source": r["source"],
                }
            )
        log(f"  {slug}: {len(chosen)} phrases")

    if not pairs:
        log("no site-scored noul phrases to compare; run `jevlab snapshot` first")
        return {}
    diffs = [p["oracle"] - p["site"] for p in pairs]
    abs_diffs = [abs(d) for d in diffs]
    exact = sum(1 for p in pairs if abs(site_round(p["oracle"]) - site_round(p["site"])) < 0.005)
    within2 = sum(1 for d in abs_diffs if d <= 0.02 + 1e-9)
    xs = [p["site"] for p in pairs]
    ys = [p["oracle"] for p in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) or 1
    report = {
        "pairs": len(pairs),
        "bias": sum(diffs) / len(diffs),
        "mae": sum(abs_diffs) / len(abs_diffs),
        "max_abs": max(abs_diffs),
        "exact_rounded": exact / len(pairs),
        "within_0.02": within2 / len(pairs),
        "pearson": cov / var,
        "noise_sd": sum(p["spread"] for p in pairs) / len(pairs),
        "worst": sorted(pairs, key=lambda p: -abs(p["oracle"] - p["site"]))[:8],
    }
    if not slugs:
        report["fingerprint"] = score_fingerprint(db)
    CALIBRATION_PATH.write_text(json.dumps(report, indent=1))
    log(
        f"calibration over {report['pairs']} phrases: bias {report['bias']:+.4f}, MAE {report['mae']:.4f}, "
        f"exact {report['exact_rounded']:.0%}, within 2pts {report['within_0.02']:.0%}, "
        f"r={report['pearson']:.3f}, per-call noise sd {report['noise_sd']:.4f}"
    )
    for p in report["worst"][:5]:
        log(f"  worst: site {p['site']:.2f} oracle {p['oracle']:.3f} [{p['slug']}] {p['state'][:70]}")
    return report


def calibration_status() -> str:
    if same_model_as_site():
        return "n/a"
    if not CALIBRATION_PATH.exists():
        return "uncalibrated"
    r = json.loads(CALIBRATION_PATH.read_text())
    return f"cal MAE {r['mae']:.3f} bias {r['bias']:+.3f} n={r['pairs']}"
