"""Emoji search: rank locally, spend a capped oracle budget, vault a winner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from ..db import DB
from ..modes import EMOJI, HIGH_SCORES
from ..objective import board_leader, logit, objective_for, site_round
from ..oracle import Oracle, Score
from .learn import Model, atom_vectors, load_model, observe, question_vector
from .rank import Atom, shortlist
from .rules import ZWJ, RuleError, check_word

DEFAULT_BUDGET = 40
SHORTLIST = 24
MAX_CHAIN = 6
CONFIRM = 3
CATALOG = Path(__file__).with_name("catalog.json")


def load_catalog(path: Path | None = None) -> list[Atom]:
    raw = json.loads((path or CATALOG).read_text())
    return [Atom(row["g"], row["n"]) for row in raw]


def _impacts(db: DB, slug: str) -> dict[str, float]:
    found = {}
    for row in db.word_impacts(slug, EMOJI):
        word = row.get("word") or ""
        if word:
            found[word] = float(row.get("averageImpact") or 0.0)
    return found


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
) -> str | None:
    """Score a short ZWJ chain for one question. Returns the vaulted phrase, or None.

    `score(phrase, n)` is an async callable returning an oracle Score. The live oracle is used when it is omitted.
    `embed=False` keeps the run on name overlap and snapshot impacts, with no local embedding.
    `progress` is called with the question's running best: score, phrase, and whether it wins.
    `win_mode` saves a WIN for publish and moves on instead of spending the rest of the budget.
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

            return await _search(db, question, board, budget, log, live, catalog, embed, halt, progress, win_mode)
        finally:
            await oracle.close()
    return await _search(db, question, board, budget, log, score, catalog, embed, halt, progress, win_mode)


async def _search(
    db, question, board, budget, log, score, catalog, embed, halt=None, progress=None, win_mode: bool = False
) -> str | None:
    from .. import vault

    slug = question["slug"]
    objective = objective_for(question, unit="word", board=board)
    leader = board_leader(db.board(slug, EMOJI, board), db.me(), board=board)
    title = question.get("title") or slug
    best_phrase: str | None = None
    best: Score | None = None

    def report(state: str) -> None:
        if progress is None:
            return
        info = {"slug": slug, "state": state, "best": None, "win": False, "phrase": None}
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
    qvec = None
    name_vecs = None
    model: Model | None = None
    if embed:
        qvec = question_vector(title, question.get("goal") or "yes")
        name_vecs = atom_vectors([(atom.text, atom.name) for atom in catalog])
        model = load_model()
    if halt and halt():
        log(f"  {slug}: stopped")
        report("stopped")
        return None
    picks = shortlist(title, catalog, _impacts(db, slug), qvec, name_vecs, model, k=len(catalog))
    if not picks:
        log(f"  {slug}: empty catalog")
        report("skipped")
        return None
    used = 0
    halted = False

    def stopping() -> bool:
        nonlocal halted
        if halted:
            return True
        if halt and halt():
            halted = True
        return halted

    async def sample(phrase: str, n: int) -> Score | None:
        nonlocal used
        if stopping() or used + n > budget:
            return None
        used += n
        got = await score(phrase, n)
        if qvec is not None:
            pieces = phrase.split(ZWJ)
            observe(qvec, pieces, logit(objective.p(got)), name_vecs)
        return got

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
        report("running")
        report("done")
        return phrase

    ranked = picks
    pool: list[Atom] = []
    cursor = 0
    singles: dict[str, Score] = {}
    grown: set[str] = set()
    scored: dict[str, Score] = {}

    async def pull(count: int) -> str | None:
        """Score the next ranked emoji. A returned phrase is a win-mode save."""
        nonlocal cursor, best_phrase, best
        batch = ranked[cursor : cursor + count]
        for atom in batch:
            got = await sample(atom.text, 1)
            if got is None:
                return None
            cursor += 1
            pool.append(atom)
            singles[atom.text] = got
            scored[atom.text] = got
            if best is None or objective.p(got) > objective.p(best):
                best_phrase, best = atom.text, got
                report("running")
            log(f"  {objective.p(got):.3f} {atom.text} {atom.name}")
            saved = claim(atom.text, got)
            if saved is not None:
                return saved
        return None

    async def grow(root: str) -> str | None:
        """Extend one emoji while the rounded score rises. Highest keeps the global best."""
        nonlocal best_phrase, best
        phrase = root
        while len(phrase.split(ZWJ)) < MAX_CHAIN:
            current = scored.get(phrase)
            if current is None:
                break
            base = site_round(objective.p(current))
            improved: str | None = None
            improved_score: Score | None = None
            present = set(phrase.split(ZWJ))
            for atom in pool:
                if atom.text in present:
                    continue
                try:
                    trial = check_word(phrase + ZWJ + atom.text)
                except RuleError:
                    continue
                got = scored.get(trial)
                if got is None:
                    got = await sample(trial, 1)
                    if got is None:
                        return None
                    scored[trial] = got
                    log(f"  {objective.p(got):.3f} {trial}")
                    saved = claim(trial, got)
                    if saved is not None:
                        return saved
                if site_round(objective.p(got)) > base and (
                    improved_score is None or objective.p(got) > objective.p(improved_score)
                ):
                    improved, improved_score = trial, got
            if improved is None or improved_score is None:
                break
            phrase = improved
            if best is None or objective.p(improved_score) > objective.p(best):
                best_phrase, best = improved, improved_score
                report("running")
        return None

    log(f"{slug}: {min(SHORTLIST, len(ranked))} emoji to start, budget {budget}")
    report("running")
    saved = await pull(SHORTLIST)
    if saved is not None:
        return saved
    if best_phrase and not halted and used < budget:
        grown.add(best_phrase)
        saved = await grow(best_phrase)
        if saved is not None:
            return saved

    # Highest keeps a higher score even when the first chain stalls. The budget is the limit.
    # A rounded 1.00 in one word cannot be beaten, so the climb stops there.
    while (
        not objective.shortest
        and not halted
        and used < budget
        and not (best is not None and site_round(objective.p(best)) >= 1.0)
    ):
        pending = [text for text in singles if text not in grown]
        pending.sort(key=lambda text: objective.p(singles[text]), reverse=True)
        if pending:
            root = pending[0]
            grown.add(root)
            held = objective.p(best) if best is not None else 0.0
            log(f"  {slug}: {held:.3f} held, growing {root}")
            saved = await grow(root)
            if saved is not None:
                return saved
            continue
        if cursor >= len(ranked):
            break
        held = objective.p(best) if best is not None else 0.0
        log(f"  {slug}: {held:.3f} held, scoring the next emoji")
        before = cursor
        saved = await pull(SHORTLIST)
        if saved is not None:
            return saved
        if cursor == before:
            break
        grown.clear()

    if halted:
        log(f"  {slug}: stopped")
        report("stopped")
        return None
    if best_phrase is None or best is None:
        log(f"  {slug}: budget {budget} produced no score")
        report("done")
        return None
    if used + CONFIRM <= budget:
        confirmed = await sample(best_phrase, CONFIRM)
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
