"""Emoji search: project, singles, climb, compact. A capped oracle budget, then a vaulted winner."""

from __future__ import annotations

import asyncio
import itertools
import json
import os
from pathlib import Path

from ..config import DATA
from ..db import DB
from ..modes import EMOJI, HIGH_SCORES
from ..objective import Leader, board_leader, logit, objective_for, site_round
from ..oracle import Oracle, Score
from .learn import Model, _local_embed, atom_vectors, load_model, observe, question_vector
from .lexicon import Index
from .project import aim_phrases, bearing, glyph_for, projections, strict_impacts, strict_lines
from .rank import Atom, shortlist
from .space import beam_phrases, cached_table, mean_vector, nearest_glyphs, residual_glyph, round_robin
from .rules import ZWJ, RuleError, check_phrase, check_word

DEFAULT_BUDGET = 40
RANK_K = 24
SINGLES_CAP = 12
PROJECT_CAP = 3
COMPACT_CAP = 12
CLIMB_TRIALS = 8
CLIMB_WORDS = 12
ORDER_CAP = 8
CATALOG = Path(__file__).with_name("catalog.json")
_SPACE_LOGGED = False


def load_catalog(path: Path | None = None) -> list[Atom]:
    raw = json.loads((path or CATALOG).read_text())
    return [Atom(row["g"], row["n"]) for row in raw]


def orderings(phrase: str) -> list[str]:
    """Other orders of one line. Two or three glyphs: every order. Longer: the reversal, adjacent
    swaps, and moving a glyph to either end, capped so the oracle does not see every permutation.
    """
    words = phrase.split()
    if len(words) < 2:
        return []
    original = _canon(words)
    if len(words) <= 3:
        return [canon for perm in itertools.permutations(words) if (canon := _canon(perm)) and canon != original]
    seen = {original}
    out: list[str] = []

    def add(parts: list[str]) -> None:
        if len(out) >= ORDER_CAP:
            return
        canon = _canon(parts)
        if not canon or canon in seen:
            return
        seen.add(canon)
        out.append(canon)

    add(list(reversed(words)))
    for index in range(len(words) - 1):
        swapped = words[:]
        swapped[index], swapped[index + 1] = swapped[index + 1], swapped[index]
        add(swapped)
    add(words[1:] + words[:1])
    add(words[-1:] + words[:-1])
    for index, word in enumerate(words):
        rest = words[:index] + words[index + 1 :]
        add([word] + rest)
        add(rest + [word])
    return out


def _canon(parts) -> str:
    try:
        return check_phrase(" ".join(parts))
    except RuleError:
        return ""


def _impacts(db: DB, slug: str) -> dict[str, float]:
    found = {}
    for row in db.word_impacts(slug, EMOJI):
        word = row.get("word") or ""
        if word:
            found[word] = float(row.get("averageImpact") or 0.0)
    return found


def dead_path() -> Path:
    return DATA / "models" / "emoji_dead.json"


def load_dead() -> dict:
    """Glyphs that have been single-scored often and never moved a probability. A bad file is empty."""
    try:
        raw = json.loads(dead_path().read_text())
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def is_dead(glyph: str, table: dict) -> bool:
    row = table.get(glyph)
    if not isinstance(row, dict):
        return False
    try:
        count = int(row.get("n") or 0)
        max_p = float(row.get("max_p") or 0.0)
    except (TypeError, ValueError):
        return False
    return count >= 3 and max_p < 0.20


def record_singles(scored: dict[str, float]) -> None:
    """Remember single-glyph scores. Tests skip the write so a run cannot mark the real catalog dead."""
    if not scored or os.environ.get("PYTEST_CURRENT_TEST"):
        return
    table = load_dead()
    for glyph, p in scored.items():
        row = table.get(glyph) if isinstance(table.get(glyph), dict) else {}
        try:
            count = int(row.get("n") or 0)
            max_p = float(row.get("max_p") or 0.0)
        except (TypeError, ValueError):
            count, max_p = 0, 0.0
        table[glyph] = {"n": count + 1, "max_p": max(max_p, float(p))}
    path = dead_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(table))
    except OSError:
        return


def collect_usage(phrase: str, index: Index, embed: bool, usage, log) -> list[Atom]:
    """Injected probe only. The joint space supplies glyphs when no probe is passed."""
    del embed, log
    if usage is None:
        return []
    try:
        return [atom for atom in usage(phrase, index) if isinstance(atom, Atom)]
    except Exception:
        return []


async def search(
    db: DB,
    slug: str,
    *,
    board: str = HIGH_SCORES,
    budget: int = DEFAULT_BUDGET,
    log=print,
    score=None,
    catalog: list[Atom] | None = None,
    embed: bool = True,
    halt=None,
    progress=None,
    win_mode: bool = False,
    usage=None,
    patience: int = 6,
    space=None,
) -> str | None:
    """Project, then singles, climb, and compact. Returns the vaulted phrase, or None.

    `score(phrase, n)` is an async callable returning an oracle Score. The live oracle is used when it is omitted.
    `embed=False` keeps the run on name overlap and snapshot impacts, with no local embedding.
    `space` is the joint glyph and word table. When it is omitted and `embed` is on, the cached table is used.
    `usage(phrase, index)` supplies extra glyphs. Nothing is loaded when it is omitted.
    `progress` is called with the question's running best, the current arm, and one event per scored batch.
    `win_mode` saves a WIN for publish and moves on instead of spending the rest of the budget.
    `patience` counts climb passes that do not raise the score. The question still runs until the budget is spent, or a one-glyph line rounds to 1.00.
    """
    question = db.question(slug)

    def tell(state: str) -> None:
        if progress is not None:
            progress({"slug": slug, "state": state, "best": None, "win": False, "phrase": None})

    if question is None:
        log(f"{slug}: not in the snapshot")
        tell("error")
        return None
    catalog = catalog if catalog is not None else load_catalog()
    if score is None:
        request = question.get("jev_request")
        if not request:
            log(f"{slug}: no jev request in the snapshot")
            tell("error")
            return None
        oracle = Oracle(db, request)
        try:

            async def live(phrase: str, n: int = 1) -> Score:
                return await oracle.score_one(phrase, n)

            return await _search(
                db,
                question,
                board,
                budget,
                log,
                live,
                catalog,
                embed,
                halt,
                progress,
                win_mode,
                usage,
                patience,
                space,
            )
        finally:
            await oracle.close()
    return await _search(
        db,
        question,
        board,
        budget,
        log,
        score,
        catalog,
        embed,
        halt,
        progress,
        win_mode,
        usage,
        patience,
        space,
    )


async def _search(
    db,
    question,
    board,
    budget,
    log,
    score,
    catalog,
    embed,
    halt=None,
    progress=None,
    win_mode: bool = False,
    usage=None,
    patience: int = 6,
    space=None,
) -> str | None:
    from .. import vault

    slug = question["slug"]
    objective = objective_for(question, unit="word", board=board)
    leader = board_leader(db.board(slug, EMOJI, board), db.me(), board=board)
    bar = leader
    title = question.get("title") or slug
    kind = question.get("kind") or "noul"
    goal = question.get("goal") or "yes"
    best_phrase: str | None = None
    best: Score | None = None
    arm = "project"
    single_ps: dict[str, float] = {}
    used = 0
    stalls = 0
    patience = max(1, int(patience))

    def report(state: str) -> None:
        if progress is None:
            return
        info = {
            "slug": slug,
            "state": state,
            "best": None,
            "win": False,
            "phrase": None,
            "arm": arm,
            "calls": used,
            "budget": budget,
            "stalls": stalls,
            "patience": patience,
        }
        if best_phrase is not None and best is not None:
            p = objective.p(best)
            units = objective.units(best_phrase)
            info["best"] = f"{p:.3f}/{units}w"
            info["win"] = objective.wins(best, units, leader)
            info["phrase"] = best_phrase
        progress(info)

    if leader and leader.units <= 1 and objective.at_ceiling(leader.probability):
        log(f"  {slug}: leader already holds the ceiling in one emoji")
        report("skipped")
        return None
    if not catalog:
        log(f"  {slug}: empty catalog")
        report("skipped")
        return None

    qvec = None
    name_vecs = None
    model: Model | None = None
    if embed:
        qvec = question_vector(title, question.get("goal") or "yes")
        name_vecs = atom_vectors([(atom.text, atom.name) for atom in catalog])
        model = load_model()
    if embed and space is None:
        global _SPACE_LOGGED
        space = cached_table()
        if space is None and not _SPACE_LOGGED:
            log("emoji space: model not cached")
            _SPACE_LOGGED = True
    if halt and halt():
        log(f"  {slug}: stopped")
        report("stopped")
        return None

    index = Index.build(catalog)
    impacts = strict_impacts(db, slug)
    dead = load_dead()
    vec_cache: dict[str, object] = {}

    def word_vec(word: str):
        if not embed:
            return None
        if word not in vec_cache:
            try:
                vec_cache[word] = _local_embed([word])[0]
            except Exception:
                vec_cache[word] = None
        return vec_cache[word]

    lines = strict_lines(slug, kind=kind, goal=goal)
    source_phrases = [str(line.get("phrase") or "") for line in lines if line.get("phrase")]
    phrases = aim_phrases(lines, impacts, None if leader is None else leader.probability)
    anchor = phrases[0] if phrases else (source_phrases[0] if source_phrases else title)
    targets = []
    if space is not None:
        for phrase in phrases:
            vec = mean_vector(space, bearing(phrase, impacts))
            if vec is not None:
                targets.append(vec)
    target = targets[0] if targets else None
    aimed = round_robin(space, targets, catalog, SINGLES_CAP)
    usage_phrase = anchor if phrases else ""
    usage_atoms = collect_usage(usage_phrase, index, embed, usage, log) if usage_phrase or usage is not None else []
    glyph_lines = projections(
        [{"phrase": phrase} for phrase in phrases],
        impacts,
        index,
        name_vecs if embed else None,
        word_vec if embed else None,
    )
    ranked = shortlist(title, catalog, _impacts(db, slug), qvec, name_vecs, model, k=RANK_K)
    confirm_n = min(3, budget // 10)
    halted = False
    scored: dict[str, Score] = {}
    batch: list[dict] = []

    def stopping() -> bool:
        nonlocal halted
        if halted:
            return True
        if halt and halt():
            halted = True
        return halted

    async def sample(phrase: str, n: int, *, reserve: bool = True) -> Score | None:
        nonlocal used
        cap = budget - confirm_n if reserve else budget
        # Leave the compact arm a slice once the best line has something to cut.
        if reserve and arm != "compact" and best_phrase is not None and objective.units(best_phrase) >= 2:
            cap -= COMPACT_CAP
        if stopping() or used + n > cap or used + n > budget:
            return None
        used += n
        got = await score(phrase, n)
        if qvec is not None:
            observe(qvec, phrase.split(), logit(objective.p(got)), name_vecs)
        return got

    advanced = False

    def consider(phrase: str, got: Score) -> None:
        nonlocal best_phrase, best, stalls, advanced
        units = objective.units(phrase)
        if best is None or best_phrase is None:
            best_phrase, best = phrase, got
            report("running")
            return
        old_units = objective.units(best_phrase)
        old_key = objective.score_key(best, old_units)
        new_key = objective.score_key(got, units)
        higher = units == old_units and objective.p(got) > objective.p(best)
        if new_key > old_key or higher:
            best_phrase, best = phrase, got
            if arm == "climb":
                stalls = 0
                advanced = True
            report("running")

    def claim(phrase: str, got: Score) -> str | None:
        """A WIN is saved for publish and ends the question. The search then moves on."""
        nonlocal best_phrase, best
        units = objective.units(phrase)
        if not win_mode or not objective.wins(got, units, leader):
            return None
        best_phrase, best = phrase, got
        p = objective.p(got)
        log(f"  WIN {p:.3f}/{units} {phrase}; next question")
        vault.save(
            slug,
            EMOJI,
            phrase,
            p_mean=p,
            p_lcb=objective.p_lcb(got),
            spread=got.spread,
            n=got.n,
            units=units,
            leader=leader,
            beats=True,
            title=title,
            origin="emoji",
            board=board,
        )
        return phrase

    def flush() -> None:
        if progress is None or not batch:
            batch.clear()
            return
        info = {
            "slug": slug,
            "state": "batch",
            "best": None,
            "win": False,
            "phrase": None,
            "arm": arm,
            "calls": used,
            "budget": budget,
            "stalls": stalls,
            "patience": patience,
            "scored": list(batch),
        }
        if best_phrase is not None and best is not None:
            p = objective.p(best)
            units = objective.units(best_phrase)
            info["best"] = f"{p:.3f}/{units}w"
            info["win"] = objective.wins(best, units, leader)
            info["phrase"] = best_phrase
        progress(info)
        batch.clear()

    async def score_line(text: str, *, as_single: bool = False) -> tuple[Score | None, str | None]:
        nonlocal bar
        try:
            phrase = check_phrase(text)
        except RuleError:
            return None, None
        cached = phrase in scored
        if cached:
            got = scored[phrase]
        else:
            got = await sample(phrase, 1)
            if got is None:
                return None, None
            scored[phrase] = got
        p = objective.p(got)
        units = objective.units(phrase)
        if as_single and units == 1:
            single_ps[phrase] = p
        if not cached:
            took_lead = objective.beats(p, units, bar)
            if took_lead:
                bar = Leader(p, units)
            batch.append(
                {
                    "phrase": phrase,
                    "p": p,
                    "units": units,
                    "arm": arm,
                    "win": objective.wins(got, units, leader),
                    "lead": took_lead,
                }
            )
            log(f"  {arm} {p:.3f} {phrase}")
        consider(phrase, got)
        return got, claim(phrase, got)

    def finish_win(saved: str) -> str:
        flush()
        report("running")
        report("done")
        return saved

    try:
        log(f"{slug}: project, singles, climb, compact, budget {budget}")
        report("running")

        arm = "project"
        for text in glyph_lines[:PROJECT_CAP]:
            got, saved = await score_line(text)
            if saved:
                return finish_win(saved)
            if got is None:
                break
        flush()
        if halted:
            log(f"  {slug}: stopped")
            report("stopped")
            return None

        arm = "singles"
        single_count = 0
        single_queue: list[Atom] = []
        seen_single: set[str] = set()
        for atom in list(aimed) + list(ranked) + list(usage_atoms):
            if atom.text in seen_single or is_dead(atom.text, dead):
                continue
            seen_single.add(atom.text)
            single_queue.append(atom)
        for atom in single_queue:
            if single_count >= SINGLES_CAP:
                break
            if atom.text in scored:
                continue
            got, saved = await score_line(atom.text, as_single=True)
            if saved:
                return finish_win(saved)
            if got is None:
                break
            single_count += 1
            if (
                best_phrase is not None
                and best is not None
                and objective.units(best_phrase) == 1
                and site_round(objective.p(best)) >= 1.0 - 1e-9
            ):
                log(f"  1.00/1w {best_phrase}")
                break
        flush()
        if halted:
            log(f"  {slug}: stopped")
            report("stopped")
            return None

        arm = "climb"
        cursor = best_phrase
        cursor_score = best
        repeat_used = False

        def candidate_atoms(current: str) -> list[Atom]:
            present = set(current.split())
            ordered: list[Atom] = []
            seen = set(present)

            def add(atom: Atom | None) -> None:
                if atom is None or atom.text in seen or is_dead(atom.text, dead):
                    return
                seen.add(atom.text)
                ordered.append(atom)

            for english in source_phrases:
                for word in bearing(english, impacts):
                    add(glyph_for(word, index, name_vecs if embed else None, word_vec(word) if embed else None))
            for atom in usage_atoms:
                add(atom)
            for atom in ranked:
                add(atom)
            # Win mode off spends the budget. The shortlist is only the first handful of glyphs.
            if not win_mode:
                for atom in catalog:
                    add(atom)
            return ordered

        def climb_trials(current: str, pool: list[Atom], allow_repeat: bool, skip: set[str]) -> list[str]:
            words = current.split()
            trials: list[str] = []
            seen = {current}

            def add(text: str) -> bool:
                if len(trials) >= CLIMB_TRIALS:
                    return False
                try:
                    phrase = check_phrase(text)
                except RuleError:
                    return False
                if phrase in seen or phrase in skip:
                    return False
                seen.add(phrase)
                trials.append(phrase)
                return True

            chosen = residual_glyph(
                space,
                target,
                current,
                [atom for atom in catalog if not is_dead(atom.text, dead)],
            )
            if chosen is not None and len(words) < CLIMB_WORDS:
                add(" ".join(words + [chosen.text]))

            if allow_repeat and single_ps and len(words) < CLIMB_WORDS:
                glyph = max(single_ps, key=lambda item: single_ps[item])
                if words.count(glyph) < 2:
                    add(" ".join(words + [glyph]))
            if len(words) < CLIMB_WORDS:
                added = 0
                for atom in pool:
                    if added >= 4 or len(trials) >= CLIMB_TRIALS:
                        break
                    if add(" ".join(words + [atom.text])):
                        added += 1
            for index_at in range(len(words) - 1, -1, -1):
                for atom in pool:
                    if len(trials) >= CLIMB_TRIALS:
                        return trials
                    new = words[:]
                    new[index_at] = atom.text
                    add(" ".join(new))
            return trials

        def is_perfect() -> bool:
            return (
                best_phrase is not None
                and best is not None
                and objective.units(best_phrase) == 1
                and site_round(objective.p(best)) >= 1.0 - 1e-9
            )

        if is_perfect():
            cursor = None

        skip: set[str] = set()
        beamed = False
        won: list[str] = []
        exhausted: set[str] = set()
        stopped = False

        def has_extension(phrase: str) -> bool:
            words = phrase.split()
            if len(words) >= CLIMB_WORDS:
                return False
            for atom in catalog:
                if is_dead(atom.text, dead):
                    continue
                try:
                    longer = check_phrase(" ".join(words + [atom.text]))
                except RuleError:
                    continue
                if longer not in scored and longer not in skip:
                    return True
            return False

        def adopt() -> bool:
            """Climb from another scored line that still has an unscored extension."""
            nonlocal cursor, cursor_score, repeat_used
            parents = sorted(
                (phrase for phrase in scored if phrase not in exhausted and has_extension(phrase)),
                key=lambda phrase: objective.units(phrase),
                reverse=True,
            )
            if not parents:
                return False
            cursor = parents[0]
            cursor_score = scored[cursor]
            repeat_used = False
            return True

        async def lengthen() -> bool:
            """Score one longer unscored line and climb from it. False when nothing legal is left."""
            nonlocal cursor, cursor_score, repeat_used, stalls, stopped
            if cursor is None:
                return False
            words = cursor.split()
            if len(words) >= CLIMB_WORDS:
                return False
            for atom in catalog:
                if is_dead(atom.text, dead):
                    continue
                try:
                    phrase = check_phrase(" ".join(words + [atom.text]))
                except RuleError:
                    continue
                if phrase in scored or phrase in skip:
                    continue
                got, saved = await score_line(phrase)
                if saved:
                    won.append(saved)
                    return False
                if got is None:
                    stopped = True
                    return False
                cursor, cursor_score = phrase, got
                repeat_used = False
                stalls = 0
                return True
            return False

        ordered_for: set[str] = set()

        async def run_order() -> bool:
            """Reorder the current best once. A better order takes the lead and climb resumes."""
            nonlocal arm, stalls, cursor, cursor_score, repeat_used, stopped
            if halted or stopped or best_phrase is None or best is None:
                return False
            if objective.units(best_phrase) < 2 or best_phrase in ordered_for:
                return False
            ordered_for.add(best_phrase)
            arm = "order"
            log("  order")
            report("running")
            key_before = objective.score_key(best, objective.units(best_phrase))
            raised = False
            for text in orderings(best_phrase):
                if text in scored:
                    continue
                got, saved = await score_line(text)
                if saved:
                    won.append(saved)
                    return False
                if got is None:
                    stopped = True
                    break
                if (
                    best is not None
                    and best_phrase is not None
                    and objective.score_key(best, objective.units(best_phrase)) > key_before
                ):
                    raised = True
            flush()
            arm = "climb"
            if not raised or best_phrase is None or best is None:
                return False
            stalls = 0
            log("  patience 0")
            report("running")
            cursor, cursor_score = best_phrase, best
            exhausted.discard(best_phrase)
            skip.clear()
            repeat_used = False
            return True

        while cursor and cursor_score is not None and not halted:
            if is_perfect():
                log(f"  1.00/1w {best_phrase}")
                break
            trials = climb_trials(cursor, candidate_atoms(cursor), not repeat_used, skip)
            repeat_used = True
            if not trials:
                if cursor is not None:
                    exhausted.add(cursor)
                grew = await lengthen()
                if stopped or halted or won:
                    break
                if grew:
                    continue
                if not adopt():
                    break
                continue
            skip.update(trials)
            advanced = False
            stopped = False
            fresh = 0
            for trial in trials:
                before = used
                got, saved = await score_line(trial)
                if used > before:
                    fresh += 1
                if saved:
                    return finish_win(saved)
                if got is None:
                    stopped = True
                    break
            flush()
            if stopped or halted:
                break
            if is_perfect():
                log(f"  1.00/1w {best_phrase}")
                break
            if advanced and best_phrase is not None and best is not None:
                log("  patience 0")
                report("running")
                cursor, cursor_score = best_phrase, best
                exhausted.discard(best_phrase)
                skip.clear()
                continue
            if fresh == 0:
                continue
            stalls += 1
            report("running")
            if stalls < patience:
                continue
            log(f"  patience {stalls}/{patience}")
            stalls = 0
            report("running")
            raised = await run_order()
            if won or stopped or halted:
                break
            if raised:
                continue
            if beamed or space is None or target is None or best is None or best_phrase is None:
                continue
            beamed = True
            arm = "beam"
            log("  beam")
            report("running")
            base = objective.units(best_phrase)
            key_before = objective.score_key(best, base)
            pool = nearest_glyphs(
                space,
                target,
                [atom for atom in catalog if not is_dead(atom.text, dead)],
                RANK_K,
            )
            raised = False
            budget_done = False
            for text in beam_phrases(
                space,
                target,
                pool,
                range(base, min(base + 3, CLIMB_WORDS + 1)),
                8,
                8,
            ):
                try:
                    phrase = check_phrase(text)
                except RuleError:
                    continue
                if phrase in scored:
                    continue
                got, saved = await score_line(phrase)
                if saved:
                    return finish_win(saved)
                if got is None:
                    budget_done = True
                    break
                if (
                    best is not None
                    and best_phrase is not None
                    and objective.score_key(best, objective.units(best_phrase)) > key_before
                ):
                    raised = True
            flush()
            if halted or budget_done:
                break
            arm = "climb"
            if raised and best is not None and best_phrase is not None:
                stalls = 0
                log("  patience 0")
                report("running")
                cursor, cursor_score = best_phrase, best
                skip.clear()
                repeat_used = False
        if won:
            return finish_win(won[0])
        if not halted and not stopped:
            await run_order()
        if won:
            return finish_win(won[0])
        if halted:
            log(f"  {slug}: stopped")
            report("stopped")
            return None

        arm = "compact"

        def worth_cutting() -> bool:
            return best is not None and best_phrase is not None and objective.units(best_phrase) >= 2

        def holds(got: Score, root_round: float) -> bool:
            if objective.shortest:
                return objective.qualifies(objective.p(got)) or objective.reachable(got)
            return site_round(objective.p(got)) >= root_round - 1e-9

        if worth_cutting() and best_phrase is not None and best is not None:
            log("  compact")
            root = best_phrase
            root_round = site_round(objective.p(best))
            compact_calls = 0
            while compact_calls < COMPACT_CAP and objective.units(root) >= 2 and not halted:
                words = root.split()
                held: tuple[str, Score] | None = None

                async def try_cut(text: str) -> str | None:
                    """None keeps compacting. A phrase is a win. '' means the budget or a halt stopped the arm."""
                    nonlocal compact_calls, held
                    if compact_calls >= COMPACT_CAP or not text.strip():
                        return None
                    try:
                        canon = check_phrase(text)
                    except RuleError:
                        return None
                    fresh = canon not in scored
                    got, saved = await score_line(canon)
                    if saved:
                        return saved
                    if got is None:
                        return ""
                    if fresh:
                        compact_calls += 1
                    if holds(got, root_round) and (
                        held is None
                        or objective.units(canon) < objective.units(held[0])
                        or (
                            objective.units(canon) == objective.units(held[0])
                            and objective.p(got) > objective.p(held[1])
                        )
                    ):
                        held = (canon, got)
                    return None

                winner: str | None = None
                for drop_at in range(len(words)):
                    winner = await try_cut(" ".join(words[:drop_at] + words[drop_at + 1 :]))
                    if winner is not None:
                        break
                if winner is not None:
                    if winner:
                        return finish_win(winner)
                    break
                if held is None and len(words) >= 3:
                    for drop_at in range(len(words) - 1):
                        winner = await try_cut(" ".join(words[:drop_at] + words[drop_at + 2 :]))
                        if winner is not None:
                            break
                if winner is not None:
                    if winner:
                        return finish_win(winner)
                    break
                if held is None:
                    break
                root = held[0]
            flush()
            if (
                not halted
                and best_phrase is not None
                and 2 <= objective.units(best_phrase) <= 4
                and compact_calls < COMPACT_CAP
                and best is not None
                and holds(best, root_round)
            ):
                try:
                    joined = check_word(ZWJ.join(best_phrase.split()))
                except RuleError:
                    joined = ""
                if joined:
                    got, saved = await score_line(joined)
                    if saved:
                        return finish_win(saved)
                    if got is not None:
                        compact_calls += 1
            flush()
        elif best_phrase is not None:
            log("  compact skipped, one glyph")

        if halted:
            log(f"  {slug}: stopped")
            report("stopped")
            return None
        if best_phrase is None or best is None:
            log(f"  {slug}: budget {budget} produced no score")
            report("done")
            return None
        if confirm_n and used + confirm_n <= budget:
            confirmed = await sample(best_phrase, confirm_n, reserve=False)
            if confirmed is not None:
                best = confirmed
        if halted:
            log(f"  {slug}: stopped")
            report("stopped")
            return None
        phrase = best_phrase
        units = objective.units(phrase)
        p = objective.p(best)
        if not objective.beats(p, units, leader):
            log(f"  {slug}: {p:.3f}/{units} does not beat the leader")
            report("done")
            return None
        vault.save(
            slug,
            EMOJI,
            phrase,
            p_mean=p,
            p_lcb=objective.p_lcb(best),
            spread=best.spread,
            n=best.n,
            units=units,
            leader=leader,
            beats=True,
            title=title,
            origin="emoji",
            board=board,
        )
        log(f"  vault {p:.3f}/{units} {phrase}")
        report("done")
        return phrase
    finally:
        record_singles(single_ps)


def run_slugs(
    db: DB,
    slugs: list[str],
    *,
    board: str = HIGH_SCORES,
    budget: int = DEFAULT_BUDGET,
    log=print,
    halt=None,
) -> list[str]:
    """Search each slug. One failure does not stop the rest. `halt` true stops the rest."""

    async def _all() -> list[str]:
        saved = []
        for slug in slugs:
            if halt and halt():
                log("stopped")
                break
            try:
                phrase = await search(db, slug, board=board, budget=budget, log=log, halt=halt)
            except Exception as error:
                log(f"{slug}: {error}")
                continue
            if phrase:
                saved.append(phrase)
        return saved

    return asyncio.run(_all())
