#!/usr/bin/env python3
"""One-shot: drop clef-flash-era local scores from the clef edition.

Clef now scores through OpenRouter (`cloudflare/clef`). Every oracle sample and
vault candidate under data/clef/ was measured with local clef-flash. This script:

- Deletes flash search cache in data/clef/jev.db (oracle samples, phrase stats, lab_*).
- Deletes embeddings.db, models/global.pkl, and calibration.json under data/clef/.
- Rewrites the vault: drops `candidate` entries and `queued` entries with no site
  score; keeps published/failed/(queued with server_p); sets p_mean/p_lcb from
  server_p when present so local fields match the real site.

Keeps questions, boards, site_scores, snapshots, publishes, attempts, boosters,
defense words, and word impacts.

Run from the repo root:

    uv run python scripts/wipe_clef_flash_scores.py
    uv run python scripts/wipe_clef_flash_scores.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEF = ROOT / "jevlab" / "data" / "clef"

SEARCH_TABLES = (
    "oracle_samples",
    "oracle_refusals",
    "phrase_stats",
    "lab_candidates",
    "lab_memory",
    "lab_meta",
)

KEEP_STATUSES = frozenset({"published", "failed", "queued"})


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, isolation_level=None, timeout=600)
    conn.execute("PRAGMA busy_timeout=600000")
    return conn


def wipe_db(db_path: Path, *, dry_run: bool) -> dict[str, int]:
    if not db_path.exists():
        return {}
    conn = _connect(db_path)
    counts: dict[str, int] = {}
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for name in SEARCH_TABLES:
            if name not in tables:
                continue
            counts[name] = conn.execute(f"SELECT COUNT(*) FROM [{name}]").fetchone()[0]
        models = [
            row[0]
            for row in conn.execute("SELECT DISTINCT model FROM oracle_samples")
        ] if "oracle_samples" in tables else []
        if dry_run:
            counts["oracle_models"] = len(models)
            return counts
        conn.execute("BEGIN IMMEDIATE")
        try:
            for name in SEARCH_TABLES:
                if name in tables:
                    conn.execute(f"DELETE FROM [{name}]")
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        # Reclaim space from ~800MB of samples.
        conn.execute("VACUUM")
    finally:
        conn.close()
    counts["vacuum"] = 1
    return counts


def wipe_files(root: Path, *, dry_run: bool) -> list[str]:
    removed: list[str] = []
    for rel in ("embeddings.db", "embeddings.db-wal", "embeddings.db-shm", "models/global.pkl", "calibration.json"):
        path = root / rel
        if not path.exists():
            continue
        removed.append(f"{rel} ({path.stat().st_size} bytes)")
        if not dry_run:
            path.unlink()
    return removed


def _align_site_score(entry: dict) -> dict:
    server = entry.get("server_p")
    if server is None:
        return entry
    out = dict(entry)
    score = round(float(server), 4)
    out["p_mean"] = score
    out["p_lcb"] = score
    out["spread"] = 0.0
    out.pop("estimated_from", None)
    return out


def wipe_vault(vault: Path, *, dry_run: bool) -> dict[str, int]:
    stats = {
        "boards": 0,
        "kept": 0,
        "dropped_candidate": 0,
        "dropped_queued_no_site": 0,
        "dropped_other": 0,
        "aligned_to_server_p": 0,
        "kept_without_server_p": 0,
        "empty_boards_removed": 0,
    }
    if not vault.is_dir():
        return stats
    for path in sorted(vault.rglob("*.json")):
        data = json.loads(path.read_text())
        stats["boards"] += 1
        kept: list[dict] = []
        for entry in data.get("entries") or []:
            status = entry.get("status")
            if status == "candidate":
                stats["dropped_candidate"] += 1
                continue
            if status == "queued" and entry.get("server_p") is None:
                stats["dropped_queued_no_site"] += 1
                continue
            if status not in KEEP_STATUSES:
                stats["dropped_other"] += 1
                continue
            if entry.get("server_p") is not None:
                entry = _align_site_score(entry)
                stats["aligned_to_server_p"] += 1
            else:
                stats["kept_without_server_p"] += 1
            kept.append(entry)
            stats["kept"] += 1
        if dry_run:
            continue
        if not kept:
            path.unlink()
            stats["empty_boards_removed"] += 1
            continue
        data["entries"] = kept
        path.write_text(json.dumps(data, indent=1) + "\n")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print counts without changing files")
    parser.add_argument(
        "--root",
        type=Path,
        default=CLEF,
        help=f"Clef data directory (default: {CLEF})",
    )
    args = parser.parse_args()
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"missing clef data dir: {root}")

    print(f"{'DRY RUN ' if args.dry_run else ''}wiping flash-era scores under {root}")
    db_counts = wipe_db(root / "jev.db", dry_run=args.dry_run)
    print("database:")
    for key, value in db_counts.items():
        print(f"  {key}: {value}")
    files = wipe_files(root, dry_run=args.dry_run)
    print("files:")
    if files:
        for line in files:
            print(f"  {'would remove' if args.dry_run else 'removed'} {line}")
    else:
        print("  (none)")
    vault_stats = wipe_vault(root / "vault", dry_run=args.dry_run)
    print("vault:")
    for key, value in vault_stats.items():
        print(f"  {key}: {value}")
    if not args.dry_run:
        print("done. Clef search will score fresh with cloudflare/clef.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
