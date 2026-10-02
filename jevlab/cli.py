"""jevlab command line."""

from __future__ import annotations

import argparse
import os
import sys
import urllib.parse


def question_slug(value: str) -> str:
    text = value.strip()
    if "://" in text or text.startswith(("?", "/")):
        parsed = urllib.parse.urlparse(text if "://" in text else "https://x" + ("" if text.startswith("/") else "/") + text)
        found = urllib.parse.parse_qs(parsed.query).get("q", [""])[0].strip()
        if found:
            return found
    return text


MODE_NAMES = {"strict": "strict_chain", "casual": "word_chain", "golf": "golf", "emoji": "emoji"}


def mode_name(value: str) -> str:
    return MODE_NAMES.get(value, value)


def board_of(args) -> str | None:
    """Site board key for --board, or None when the flag was left out (commands that span every board)."""
    from .modes import from_name

    value = getattr(args, "board", None)
    return from_name(value).board if value else None


def cmd_install_laya(_args) -> int:
    from .laya_local import LayaError, cache_dir, install

    try:
        path = install()
    except LayaError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Laya weights ready at {path}")
    print(f"cache {cache_dir()}")
    return 0


def cmd_snapshot(args) -> int:
    from .db import DB
    from .site.snapshot import take_snapshot

    slugs = [question_slug(s) for s in args.q] if args.q else None
    modes = tuple(mode_name(m) for m in args.modes) if args.modes else None
    kwargs = {"modes": modes} if modes else {}
    take_snapshot(DB(), slugs=slugs, workers=args.workers, **kwargs)
    return 0


def cmd_calibrate(args) -> int:
    import asyncio

    from .calibrate import calibrate
    from .db import DB

    slugs = [question_slug(s) for s in args.q] if args.q else None
    report = asyncio.run(calibrate(DB(), slugs, per_question=args.per_question, n=args.n))
    return 0 if report else 1


def cmd_search(args) -> int:
    """Headless lab run: same engine as the TUI, prints progress and the final top lines."""
    import asyncio

    from .db import DB
    from .search.engine import LEVELS, Engine

    from . import activity

    def on_event(event):
        d = event.data
        if event.kind == "log":
            print(d["message"], flush=True)
            activity.log("lab", d["message"])
        elif event.kind == "best":
            print(f"  ** best {d['p']:.3f} {d['units']}w n={d['n']} [{d['origin']}] {d['phrase']}", flush=True)
            activity.log("best", f"{d['p']:.3f} {d['units']}w n={d['n']} [{d['origin']}] {d['phrase']}")
        elif event.kind == "round":
            print(f"-- round {d['round']}: {d['strategy']}", flush=True)
        elif event.kind in ("vault", "plan", "surrogate"):
            print(f"  [{event.kind}] {d}", flush=True)

    db = DB()
    slug = question_slug(args.q)
    for target in search_targets(db, slug, args):
        if target:
            print(f"\n==== {slug} -> {target}", flush=True)
        engine = Engine(db, slug, budget=args.budget, use_llm=not args.no_llm,
                        seed=args.seed, on_event=on_event, max_stall=args.plateau,
                        escalate=not args.no_escalate, max_level=args.max_level, board=board_of(args), target=target,
                        win_extra=args.win_after, **({"triage_k": args.transfer_k} if args.transfer_k is not None
                                                     else {}))
        if args.force:
            engine.force(args.force)
        asyncio.run(engine.run())
        print("\n".join(engine.summary(15)))
        print(f"{engine.oracle.calls} oracle calls, {engine.llm.calls if engine.llm else 0} LLM calls, "
              f"ended at L{engine.level} {LEVELS[engine.level][0]}")
    return 0


def search_targets(db, slug: str, args) -> list[str]:
    """--target runs one answer, --all-targets every answer of a choice question; "" is the whole question."""
    from .rules import option_names

    if args.target:
        return [args.target]
    if not args.all_targets:
        return [""]
    question = db.question(slug)
    answers = option_names((question or {}).get("raw", {}).get("choices"))
    if not answers:
        raise SystemExit(f"{slug} has no answers to target")
    return answers


def cmd_score(args) -> int:
    import asyncio

    from .db import DB
    from .objective import Objective
    from .oracle import Oracle
    from .rules import normalize

    db = DB()
    question = db.question(question_slug(args.q))
    if not question:
        print("question not in snapshot; run `jevlab snapshot`", file=sys.stderr)
        return 1
    objective = Objective(question.get("goal") or "yes", question["kind"])

    async def go():
        oracle = Oracle(db, question["jev_request"])
        try:
            phrases = [normalize(p) if args.strict else p for p in args.phrases]
            scores = await oracle.score(phrases, n=args.n)
            for phrase in phrases:
                s = scores[phrase]
                print(f"{objective.p(s):.3f} ±{s.spread:.3f} n={s.n} {objective.units(phrase)}w  {phrase}")
        finally:
            await oracle.close()

    asyncio.run(go())
    return 0


def cmd_triage(args) -> int:
    import asyncio

    from .db import DB
    from .modes import is_searchable
    from .search.triage import run_triage

    db = DB()
    if args.q:
        slugs = [question_slug(s) for s in args.q]
    else:
        slugs = [r["slug"] for r in db.all("SELECT slug, kind, json_extract(raw, '$.ranked') AS ranked "
                                            "FROM questions ORDER BY slug")
                 if is_searchable(r["kind"], bool(r["ranked"]))]
    if not slugs:
        print("no questions; run `jevlab snapshot` first", file=sys.stderr)
        return 1
    for slug in slugs:
        asyncio.run(run_triage(db, slug, board_of(args), k=args.k, n=args.n, confirm_n=args.confirm_n,
                               target=args.target or "", queue=args.queue))
    return 0


def cmd_tui(_args) -> int:
    from .tui.app import run_home

    return run_home()


def cmd_lab(args) -> int:
    from .tui.app import run_lab

    from .boards import TARGET_SEP

    slug = question_slug(args.q)
    key = f"{slug}{TARGET_SEP}{args.target}" if args.target else slug
    return run_lab(key, budget=args.budget, use_llm=not args.no_llm, seed=args.seed,
                   escalate=not args.no_escalate, max_level=args.max_level, board=board_of(args))


def cmd_vault(args) -> int:
    from . import vault
    from .boards import has_rejected_word
    from .modes import from_board
    from .rules.banned import contains as phrase_banned

    board = board_of(args)
    if args.action == "import-jev":
        from .db import DB
        from .transfer import import_from_jev

        try:
            import_from_jev(DB(), [question_slug(args.q)] if args.q else None, board,
                            mode_name(args.mode) if args.mode else None, queue=args.queue)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            return 1
        return 0
    entries = vault.all_entries(board)
    if args.mode:
        entries = [e for e in entries if e["mode"] == mode_name(args.mode)]
    elif args.action != "list":
        entries = [e for e in entries if e["mode"] == "strict_chain"]
    if args.q:
        slug = question_slug(args.q)
        entries = [e for e in entries if e["slug"] == slug]
    if args.target is not None:
        entries = [e for e in entries if (e.get("target") or "") == args.target]
    elif args.action != "list":
        entries = [e for e in entries if not e.get("target")]
    if args.action == "list":
        if args.status:
            entries = [e for e in entries if e["status"] == args.status]
        if not entries:
            print("no matching vault entries")
        for e in entries:
            game_mode = from_board(e["board"], e["mode"])
            u = game_mode.unit_abbr
            lead = e.get("leader") or {}
            need = f"{lead.get('p', 0):.2f}/{lead.get('units', 0)}{u}" if lead else "-"
            server = f" server {e['server_p']:.2f}" if e.get("server_p") is not None else ""
            estimated = f"est({e['estimated_from']}) " if e.get("estimated_from") else ""
            print(f"{game_mode.name:<13} {e['status']:<9} {estimated}{e['p_mean']:.3f} lcb {e['p_lcb']:.3f} "
                  f"n={e['n']:<2} {e['units']:2}{u} vs {need:<9} {'WIN ' if e['beats'] else '    '}"
                  f"{e['slug'][:40]:<40} {'-> ' + e['target'] + ' | ' if e.get('target') else ''}"
                  f"{e['phrase']}{server}")
        return 0
    if not args.q:
        print("--q is required for queue/unqueue/drop", file=sys.stderr)
        return 1
    if board is None:
        board = from_board(None).board
        entries = [e for e in entries if e["board"] == board]
    if args.phrase:
        targets = [e for e in entries if e["phrase"] == args.phrase]
    elif args.action == "unqueue":
        targets = [e for e in entries if e["status"] == "queued"]
    elif args.all:
        targets = [e for e in entries if e["beats"] and e["status"] == "candidate"]
    else:
        pool = [
            e
            for e in entries
            if e["status"] in ("candidate", "failed")
            and not has_rejected_word(e["phrase"])
            and not phrase_banned(e["phrase"])
        ]
        if from_board(board).shortest:
            pool.sort(key=lambda e: (e["units"], -round(e["p_mean"], 2), -e["p_lcb"]))
        else:
            pool.sort(key=lambda e: (-round(e["p_mean"], 2), e["units"], -e["p_lcb"]))
        targets = pool[:1]
    if not targets:
        print("no matching vault entry")
        return 1
    status = {"queue": "queued", "unqueue": "candidate", "drop": "dropped"}[args.action]
    for e in targets:
        vault.set_status(e["slug"], e["mode"], e["phrase"], status, e["board"], target=e.get("target") or "")
        aimed = f" -> {e['target']}" if e.get("target") else ""
        print(f"{status}: {e['slug']}{aimed} {e['p_mean']:.3f}/{e['units']}w {e['phrase']}")
    return 0


def cmd_publish(args) -> int:
    from .publish import publish

    slugs = [question_slug(s) for s in args.q] if args.q else None
    mode = mode_name(args.mode) if args.mode else None
    from . import activity

    def log(message: str) -> None:
        print(message, flush=True)
        for line in str(message).splitlines():
            if line.strip():
                activity.log("publish", line)

    return publish(dry_run=args.dry_run, rerolls=args.rerolls, verify=not args.no_verify, slugs=slugs,
                   delay=args.delay, board=board_of(args), mode=mode, crosspost=not args.no_crosspost, log=log)


def cmd_errors(args) -> int:
    import time
    from collections import Counter

    from . import netlog

    rows = netlog.recent(args.minutes)
    if not rows:
        print(f"no failed calls in the last {args.minutes:g} minutes ({netlog.PATH})")
        return 0
    tally = Counter((r["service"], f"{r['host']}{r['endpoint']}", r["status"]) for r in rows)
    last = {}
    for r in rows:
        last[(r["service"], f"{r['host']}{r['endpoint']}", r["status"])] = r["at"]
    print(f"{len(rows)} failed calls in the last {args.minutes:g} minutes:")
    for (service, where, status), n in tally.most_common():
        ago = int(time.time() - last[(service, where, status)])
        print(f"  {n:>5}x  {service:<10} {status:<16} {where}   (last {ago // 60}m{ago % 60:02d}s ago)")
    for r in rows[-args.tail:]:
        stamp = time.strftime("%H:%M:%S", time.localtime(r["at"]))
        print(f"  {stamp} {r['service']:<10} {r['status']:<14} {r['host']}{r['endpoint']}  {r['detail'][:120]}")
    return 0


def cmd_crosspost(args) -> int:
    from .crosspost import picks, queue
    from .db import DB
    from .modes import GOLF_HIGHEST, GOLF_SHORTEST

    db = DB()
    modes = {"highest": (GOLF_HIGHEST,), "shortest": (GOLF_SHORTEST,)}.get(args.board, (GOLF_HIGHEST, GOLF_SHORTEST))
    slugs = [question_slug(s) for s in args.q] if args.q else None
    found = picks(db, modes, slugs)
    if not found:
        print("no Strict vault line beats a Golf leader (snapshot golf first: `jevlab snapshot --modes golf`)")
        return 0
    for pick in found:
        e, lead = pick.entry, pick.leader
        need = f"{lead.probability:.2f}/{lead.units}c {lead.name}" if lead else "empty board"
        aimed = f"-> {pick.target} | " if pick.target else ""
        print(f"{pick.mode.name:<13} {e['p_mean']:.3f} lcb {e['p_lcb']:.3f} {pick.units:>3}c vs {need:<28} "
              f"{pick.slug[:40]:<40} {aimed}{e['phrase']}")
    if args.dry_run:
        print(f"\ndry run: {len(found)} line(s) would be queued")
        return 0
    queue(db, found)
    print(f"\nqueued {len(found)} line(s); `jevlab publish --mode golf` sends them")
    return 0


def cmd_boosters(args) -> int:
    import asyncio

    from .db import DB
    from .search import boosters

    db = DB()
    if args.action == "mine":
        boosters.mine(db)
    elif args.action == "optimise":
        for goal in args.goal or ["yes", "no"]:
            asyncio.run(boosters.optimise(db, goal, k=args.k, questions=args.questions, budget=args.budget))
    for goal in args.goal or ["yes", "no"]:
        print(f"\n[{goal}] boosters (source, lift, questions, wins/tries):")
        for row in boosters.library(db, goal, args.top):
            print(f"  {row['source']:<9} {row['lift']:+.3f} x{row['questions']:<3} {row['wins']}/{row['tries']:<4} "
                  f"{row['text']}")
    return 0


def cmd_bench_embed(args) -> int:
    import asyncio

    from .bench import bench_embed

    return asyncio.run(bench_embed(models=args.models, phrases=args.phrases))


def cmd_bench_proxy(args) -> int:
    import asyncio

    from .bench import bench_proxy

    return asyncio.run(bench_proxy(models=args.models, phrases=args.phrases))


def cmd_train_global(args) -> int:
    from .db import DB
    from .search.global_model import train_global

    train_global(DB(), limit=args.limit)
    return 0


def pop_edition(argv: list[str]) -> tuple[list[str], str | None]:
    """Strip `--edition X` / `--edition=X` from anywhere on the command line."""
    out, edition = [], None
    items = iter(argv)
    for item in items:
        if item == "--edition":
            edition = next(items, None)
        elif item.startswith("--edition="):
            edition = item.split("=", 1)[1]
        else:
            out.append(item)
    return out, edition


def main(argv: list[str] | None = None) -> int:
    # Every path and oracle setting is fixed when jevlab.config is first imported, so the edition goes into the
    # environment before anything imports it.
    argv, edition = pop_edition(sys.argv[1:] if argv is None else list(argv))
    if edition is not None:
        os.environ["JEV_EDITION"] = edition
    from .config import EDITION, EDITIONS

    parser = argparse.ArgumentParser(prog="jevlab", description="Offline Jev search lab")
    parser.add_argument("--edition", choices=EDITIONS, default=EDITION,
                        help="game edition: jev (data/), kev (data/kev/), or laya (data/laya/); "
                             "read before anything else loads")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("tui", help="home screen: choose Search or Publish (default when no command is given)")
    p.set_defaults(func=cmd_tui)

    p = sub.add_parser(
        "install-laya",
        help="download Laya weights from Hugging Face into ~/.cache/jevlab (outside the repo)",
    )
    p.set_defaults(func=cmd_install_laya)

    p = sub.add_parser("snapshot", help="pull questions, boards, word impacts, and our attempts into data/jev.db")
    p.add_argument("--q", nargs="*", help="limit to these slugs or play URLs")
    p.add_argument("--modes", nargs="*", help="strict casual golf emoji (default all)")
    p.add_argument("--workers", type=int, default=6)
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("calibrate", help="replay site-scored phrases through the oracle and report parity")
    p.add_argument("--q", nargs="*", help="limit to these slugs")
    p.add_argument("--per-question", type=int, default=12)
    p.add_argument("--n", type=int, default=3, help="oracle samples per phrase")
    p.set_defaults(func=cmd_calibrate)

    board_names = ["highest", "shortest"]
    for name, func, text in (
        ("lab", cmd_lab, "interactive TUI search for one yes/no or choice question (strict highest or shortest yes)"),
        ("search", cmd_search, "headless search for one yes/no or choice question (strict highest or shortest yes)"),
    ):
        p = sub.add_parser(name, help=text)
        p.add_argument("--q", required=True, help="question slug or play URL")
        p.add_argument("--mode", default="strict", choices=["strict"], help="v1 supports strict only")
        p.add_argument("--board", default="highest", choices=board_names, help="game mode board to compete on")
        p.add_argument("--budget", type=int, default=20000, help="max oracle calls this session")
        p.add_argument("--no-llm", action="store_true", help="skip LLM generators and planner")
        p.add_argument("--seed", type=int, default=None)
        p.add_argument("--no-escalate", action="store_true", help="move on at a plateau instead of escalating")
        p.add_argument("--max-level", type=int, default=4, help="highest escalation level (0-4)")
        p.add_argument("--target", help="choice questions: aim at this answer (score = P(answer))")
        if name == "search":
            p.add_argument("--all-targets", action="store_true",
                           help="choice questions: run once per answer, one after another")
            p.add_argument("--win-after", type=int, default=0, metavar="N",
                           help="win mode: once a line beats the leader, stop after N more oracle calls")
        if name == "search":
            p.add_argument("--transfer-k", type=int, default=None, metavar="K",
                           help="score K existing lines (Jev's best, estimated vault lines) before generating; "
                                "default JEV_TRIAGE_K (40 on Kev and Laya, 0 on Jev)")
            p.add_argument("--force", help="run only this strategy")
            p.add_argument("--plateau", type=int, default=0, help="flat rounds per level (0 = run to budget)")
        p.set_defaults(func=func)

    p = sub.add_parser("triage", help="score the most promising existing lines (Jev's vault and history, "
                                      "estimated vault lines) with this edition's oracle, without searching")
    p.add_argument("--q", nargs="*", help="question slugs or play URLs (default: every searchable question)")
    p.add_argument("--board", default="highest", choices=board_names, help="board to judge the lines on")
    p.add_argument("--target", help="choice questions: lines aimed at this answer")
    p.add_argument("--k", type=int, default=20, help="lines per question")
    p.add_argument("--n", type=int, default=1, help="oracle samples per line on the first pass")
    p.add_argument("--confirm-n", type=int, default=3, help="samples for lines whose first pass comes close")
    p.add_argument("--queue", action="store_true", help="queue the best winner per board (if none is queued)")
    p.set_defaults(func=cmd_triage)

    p = sub.add_parser("score", help="score phrases with the oracle")
    p.add_argument("--q", required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--strict", action="store_true", help="normalize to strict-chain words first")
    p.add_argument("phrases", nargs="+")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("vault", help="list saved winners, queue them for publishing, or drop them; under "
                                     "--edition kev or laya, import-jev copies Jev's lines in as estimates")
    p.add_argument("action", nargs="?", default="list", choices=["list", "queue", "unqueue", "drop", "import-jev"])
    p.add_argument("--queue", action="store_true",
                   help="import-jev: queue the best estimated line on each board where Jev beats the leader")
    p.add_argument("--q", help="question slug or play URL")
    p.add_argument("--phrase", help="exact phrase to queue/drop (default: the best candidate)")
    p.add_argument("--all", action="store_true", help="queue every winning candidate for the question")
    p.add_argument("--status", choices=["candidate", "queued", "published", "failed", "rejected", "dropped"])
    p.add_argument("--target", help="choice questions: only lines aimed at this answer "
                                    "(queue/unqueue/drop default: whole-question lines)")
    p.add_argument("--board", choices=board_names,
                   help="game mode board (list: default every board; queue/unqueue/drop: default highest)")
    p.add_argument("--mode", choices=["strict", "golf"],
                   help="play mode (list: default every mode; queue/unqueue/drop: default strict)")
    p.set_defaults(func=cmd_vault)

    p = sub.add_parser("publish", help="submit queued vault entries to the live site (separate batch loop)")
    p.add_argument("--q", nargs="*", help="only these slugs")
    p.add_argument("--dry-run", action="store_true", help="verify boards and oracle, but send no turns")
    p.add_argument("--rerolls", type=int, default=3,
                   help="re-score the final word N times even after a win; the board keeps the best")
    p.add_argument("--no-verify", action="store_true", help="skip the oracle re-check")
    p.add_argument("--delay", type=float, default=0.01, help="seconds between site turns")
    p.add_argument("--board", choices=board_names, help="only this game mode board (default every board)")
    p.add_argument("--mode", choices=["strict", "golf"], help="only this play mode (default every mode)")
    p.add_argument("--no-crosspost", action="store_true",
                   help="do not follow each landed Strict line with its Golf cross-posts")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("crosspost", help="queue Strict vault lines that beat a Golf leader into the golf vault")
    p.add_argument("--q", nargs="*", help="only these slugs")
    p.add_argument("--board", choices=board_names, help="only this Golf board (default both)")
    p.add_argument("--dry-run", action="store_true", help="list the picks without queueing them")
    p.set_defaults(func=cmd_crosspost)

    p = sub.add_parser("errors", help="failed network calls by service and host (OpenRouter vs the Jev site)")
    p.add_argument("--minutes", type=float, default=60, help="look back this far (default 60)")
    p.add_argument("--tail", type=int, default=8, help="also print the last N failures")
    p.set_defaults(func=cmd_errors)

    p = sub.add_parser("boosters", help="universal boosters: mine from samples, optimise across questions, list")
    p.add_argument("action", nargs="?", default="list", choices=["list", "mine", "optimise"])
    p.add_argument("--goal", nargs="*", choices=["yes", "no"])
    p.add_argument("--k", type=int, default=3, help="trigger length in words (optimise)")
    p.add_argument("--questions", type=int, default=12, help="questions to optimise over")
    p.add_argument("--budget", type=int, default=20000, help="oracle calls per goal (optimise)")
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(func=cmd_boosters)

    p = sub.add_parser("bench-embed", help="compare embedding models for the phrase predictor (holdout rho)")
    p.add_argument("--models", nargs="*", help="OpenRouter embedding model ids (default: a shortlist)")
    p.add_argument("--phrases", type=int, default=3000)
    p.set_defaults(func=cmd_bench_embed)

    p = sub.add_parser("bench-proxy", help="test logprob chat models as stand-ins for Jev (rho vs oracle)")
    p.add_argument("--models", nargs="*", help="OpenRouter chat model ids (default: a shortlist)")
    p.add_argument("--phrases", type=int, default=400)
    p.set_defaults(func=cmd_bench_proxy)

    p = sub.add_parser("train-global", help="train the cross-question predictor on every scored phrase")
    p.add_argument("--limit", type=int, default=0, help="cap on training rows (0 = all)")
    p.set_defaults(func=cmd_train_global)

    args = parser.parse_args((["--edition", edition] if edition is not None else []) + argv)
    if args.edition != EDITION:
        parser.error(f"--edition {args.edition} came too late: the config was already loaded as {EDITION}")
    if not getattr(args, "func", None):
        return cmd_tui(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
