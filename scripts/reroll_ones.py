"""One-shot: re-roll every live 1-word row of ours, up to 15 times each.

The site keeps a player's best-ever phrase on each board, so a noisy score can
only improve. Same move as publish's re-roll (remove the last word, append it
again); Golf starts a fresh attempt each try. After the original casing, the
same word is tried in ALL CAPS for another 15 re-rolls. Not wired into the CLI.

  uv run python scripts/reroll_ones.py --dry-run
  uv run python scripts/reroll_ones.py
  uv run python scripts/reroll_ones.py --edition kev --q some-slug
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys

REROLLS = 15
PLAY_MODES = ("strict_chain", "golf", "word_chain")
BOARD_KEYS = ("highScores", "shortestYes", "champions")
CEILING = 0.99


def pop_edition(argv: list[str]) -> tuple[list[str], str | None]:
    out, edition, items = [], None, iter(argv)
    for item in items:
        if item == "--edition":
            edition = next(items, None)
        elif item.startswith("--edition="):
            edition = item.split("=", 1)[1]
        else:
            out.append(item)
    return out, edition


def one_word(row: dict) -> str | None:
    """The single word on this board row, or None when it is not a 1-word line."""
    phrase = (row.get("phrase") or "").strip()
    if phrase:
        words = phrase.split()
        return words[0] if len(words) == 1 else None
    if int(row.get("wordCount") or 0) == 1:
        return ""
    return None


def casings(word: str) -> list[str]:
    """Original casing, then ALL CAPS when that is a different string."""
    variants = [word]
    caps = word.upper()
    if caps != word and any(c.isalpha() for c in word):
        variants.append(caps)
    return variants


def collect(client, me: str, slugs: set[str] | None, modes: tuple[str, ...], log) -> list[dict]:
    """Live (slug, mode, word) rows of ours that are one word, one entry per attempt."""
    from concurrent.futures import ThreadPoolExecutor

    from jevlab.modes import GOLF
    from jevlab.site import SiteError

    questions = [q for q in client.menu().get("questions") or [] if not slugs or q["slug"] in slugs]
    jobs = [(q, mode) for q in questions for mode in modes]
    log(f"scanning {len(questions)} question(s) x {len(modes)} mode(s)")
    grouped: dict[tuple[str, str, str], dict] = {}
    errors = 0

    def fetch(job):
        question, mode = job
        try:
            return question, mode, client.leaderboards(question["revisionId"], mode).get("leaderboards") or {}, None
        except SiteError as error:
            return question, mode, {}, error

    with ThreadPoolExecutor(6) as pool:
        for i, (question, mode, boards, error) in enumerate(pool.map(fetch, jobs), 1):
            if error:
                errors += 1
                log(f"  skip {question['slug']} {mode}: {error}")
            else:
                for board in BOARD_KEYS:
                    for row in boards.get(board) or []:
                        if row.get("userId") != me:
                            continue
                        word = one_word(row)
                        if word is None:
                            continue
                        key = (question["slug"], mode, word)
                        p = float(row.get("probability") or 0)
                        hit = grouped.get(key)
                        if hit is None:
                            grouped[key] = {
                                "slug": question["slug"],
                                "title": question.get("title") or question["slug"],
                                "live": question,
                                "mode": mode,
                                "word": word,
                                "golf": mode == GOLF,
                                "p": p,
                                "boards": [(board, p, row.get("choice") or "")],
                            }
                        else:
                            hit["p"] = max(hit["p"], p)
                            hit["boards"].append((board, p, row.get("choice") or ""))
            if i % 40 == 0:
                log(f"  {i}/{len(jobs)} boards")
    missing = [t for t in grouped.values() if not t["word"]]
    if missing:
        log(f"{len(missing)} 1-word row(s) have no phrase on the board; will try the live attempt")
    log(f"found {len(grouped)} 1-word attempt(s)" + (f", {errors} board fetch error(s)" if errors else ""))
    return sorted(grouped.values(), key=lambda t: (t["slug"], t["mode"], t["word"]))


def resolve_word(client, target: dict, log) -> str | None:
    """Fill in a 1-word row whose board payload omitted the phrase."""
    from jevlab.site import active_words

    if target["word"]:
        return target["word"]
    attempt = client.attempt(target["live"]["revisionId"], target["mode"])
    words = active_words(attempt)
    if len(words) == 1:
        target["word"] = words[0]
        log(f"    phrase from attempt: {words[0]}")
        return words[0]
    log(f"    no phrase on the board and attempt is {len(words)} word(s); skipping")
    return None


def reroll_chain(client, target: dict, word: str, n: int, log) -> tuple[float, list[float]]:
    from jevlab.objective import site_round
    from jevlab.publish import build_chain, reroll
    from jevlab.site import active_words

    live, mode = target["live"], target["mode"]
    attempt = client.attempt(live["revisionId"], mode) or client.start(live["revisionId"], mode)
    attempt, p = build_chain(client, attempt, [word], log)
    if active_words(attempt) != [word]:
        raise RuntimeError(f"chain is {active_words(attempt)!r}, want {[word]!r}")
    best = float(p or 0.0)
    rolls = [best]
    log(f"    on chain {best:.2f}")
    for i in range(n):
        if site_round(best) >= CEILING:
            log(f"    ceiling {best:.2f}; stopping")
            break
        attempt, p = reroll(client, attempt, log)
        rolls.append(p)
        best = max(best, p)
        log(f"    re-roll {i + 1}/{n}: {p:.2f} (best {best:.2f})")
    return best, rolls


def reroll_golf(client, target: dict, word: str, n: int, start: float, log) -> tuple[float, list[float]]:
    from jevlab.objective import site_round
    from jevlab.publish import golf_try

    best = start
    rolls = [best] if start else []
    if start:
        log(f"    board {best:.2f}")
    for i in range(n):
        if site_round(best) >= CEILING:
            log(f"    ceiling {best:.2f}; stopping")
            break
        p = golf_try(client, target["live"], word, log)
        rolls.append(p)
        best = max(best, p)
        log(f"    re-roll {i + 1}/{n}: {p:.2f} (best {best:.2f})")
    return best, rolls


def record(db, target: dict, word: str, best: float) -> None:
    from jevlab.publish import now_iso

    db.execute(
        "INSERT OR REPLACE INTO site_scores VALUES (?, ?, ?, ?, ?, ?)",
        (target["slug"], target["mode"], word, best, "reroll_ones", now_iso()),
    )


def run(args) -> int:
    import time

    from jevlab.config import EDITION
    from jevlab.db import DB
    from jevlab.objective import site_round
    from jevlab.publish import GOLF_BOARD_LAG, LOCK_PATH, WordRejected, refresh_board
    from jevlab.rules.banned import BannedPhrase
    from jevlab.site import SiteClient, SiteError

    def log(message: str) -> None:
        print(message, flush=True)

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another publisher is running")
            return 1

        client = SiteClient(delay=args.delay)
        me = ((client.viewer().get("player") or {}).get("user") or {}).get("id") or ""
        if not me:
            log("not signed in: put a session cookie in session.json or JEV_COOKIE")
            return 1
        log(f"edition {EDITION} me={me}")
        targets = collect(client, me, set(args.q) if args.q else None, tuple(args.modes), log)
        if args.limit:
            targets = targets[: args.limit]
        if not targets:
            log("no 1-word rows of ours on the live boards")
            return 0
        for target in targets:
            boards = ", ".join(
                f"{b} {p:.2f}" + (f"/{c}" if c else "") for b, p, c in target["boards"]
            )
            word = target["word"] or "?"
            extra = ""
            if target["word"]:
                more = casings(target["word"])[1:]
                if more:
                    extra = f" + {' '.join(more)}"
            log(f"  {target['p']:.2f} {target['mode']:<12} {target['slug']} [{boards}] {word}{extra}")
        if args.dry_run:
            n = sum(len(casings(t["word"])) if t["word"] else 2 for t in targets)
            log(
                f"dry run: would re-roll {len(targets)} 1-word row(s) "
                f"({n} casing(s)) up to {args.rerolls} times each"
            )
            return 0

        db = DB()
        improved = skipped = failed = 0
        gains: list[tuple[float, str, str]] = []
        done: set[tuple[str, str, str]] = set()
        try:
            for i, target in enumerate(targets, 1):
                log(f"\n== {i}/{len(targets)} {target['slug']} {target['mode']} {target['p']:.2f}")
                word = resolve_word(client, target, log)
                if not word:
                    skipped += 1
                    continue
                start = target["p"]
                if site_round(start) >= CEILING:
                    log(f"  already {start:.2f}; skipping")
                    skipped += 1
                    continue
                best_board = start
                rolled = False
                for variant in casings(word):
                    key = (target["slug"], target["mode"], variant)
                    if key in done:
                        continue
                    done.add(key)
                    if site_round(best_board) >= CEILING:
                        log(f"  ceiling {best_board:.2f}; skipping {variant}")
                        break
                    if variant != word:
                        log(f"  ALL CAPS {variant}")
                    try:
                        if target["golf"]:
                            golf_start = start if variant == word else 0.0
                            best, rolls = reroll_golf(client, target, variant, args.rerolls, golf_start, log)
                        else:
                            best, rolls = reroll_chain(client, target, variant, args.rerolls, log)
                    except BannedPhrase as error:
                        failed += 1
                        log(f"  banned {variant}: {error}")
                        continue
                    except WordRejected as error:
                        failed += 1
                        log(f"  rejected {variant}: {error}")
                        continue
                    except (SiteError, RuntimeError) as error:
                        failed += 1
                        log(f"  failed {variant}: {error}")
                        continue
                    rolled = True
                    best_board = max(best_board, best)
                    record(db, target, variant, best)
                    log(
                        f"  -> {variant} best {best:.2f}  rolls {[round(r, 2) for r in rolls]}"
                    )
                if not rolled:
                    continue
                if target["golf"]:
                    time.sleep(GOLF_BOARD_LAG)
                refresh_board(db, client, target["live"], target["mode"])
                delta = best_board - start
                if site_round(best_board) > site_round(start):
                    improved += 1
                    shown = " / ".join(casings(word))
                    gains.append((delta, target["slug"], shown))
                    log(f"  -> improved {start:.2f} -> {best_board:.2f}")
                else:
                    log(f"  -> held {start:.2f} (best {best_board:.2f})")
        except KeyboardInterrupt:
            log("\ninterrupted")
        log(
            f"\ndone: {len(targets)} target(s), {improved} improved, {skipped} skipped, {failed} failed"
        )
        for delta, slug, word in sorted(gains, reverse=True):
            log(f"  +{delta:.2f}  {slug}  {word}")
        client.save_session()
        return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, edition = pop_edition(argv)
    if edition is not None:
        os.environ["JEV_EDITION"] = edition
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="list 1-word rows, send nothing")
    parser.add_argument("--rerolls", type=int, default=REROLLS, help=f"remove-and-reappend this many times (default {REROLLS})")
    parser.add_argument("--delay", type=float, default=0.01, help="seconds between site turns")
    parser.add_argument("--q", nargs="*", default=[], help="limit to these slugs")
    parser.add_argument("--limit", type=int, default=0, help="cap how many attempts to re-roll (0 = all)")
    parser.add_argument("--modes", nargs="*", default=list(PLAY_MODES), help="play modes to scan")
    parser.add_argument("--edition", choices=["jev", "kev", "laya", "clef"],
                        help="game edition (also accepted anywhere on the line)")
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
