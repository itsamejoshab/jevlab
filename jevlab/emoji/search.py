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
) -> str | None:
    """Score a short ZWJ chain for one question. Returns the vaulted phrase, or None.

    `score(phrase, n)` is an async callable returning an oracle Score. The live oracle is used when it is omitted.
    `embed=False` keeps the run on name overlap and snapshot impacts, with no local embedding.
    """
    question = db.question(slug)
    if question is None:
        log(f"{slug}: not in the snapshot")
        return None
    catalog = catalog if catalog is not None else load_catalog()
    if score is None:
        request = question.get("jev_request")
        if not request:
            log(f"{slug}: no jev request in the snapshot")
            return None
        oracle = Oracle(db, request)
        try:

            async def live(phrase: str, n: int = 1) -> Score:
                return await oracle.score_one(phrase, n)

            return await _search(db, question, board, budget, log, live, catalog, embed)
        finally:
            await oracle.close()
    return await _search(db, question, board, budget, log, score, catalog, embed)


async def _search(db, question, board, budget, log, score, catalog, embed) -> str | None:
    from .. import vault

    slug = question["slug"]
    objective = objective_for(question, unit="word", board=board)
    leader = board_leader(db.board(slug, EMOJI, board), db.me(), board=board)
    if leader and leader.units <= 1 and objective.at_ceiling(leader.probability):
        log(f"  {slug}: leader already holds the ceiling in one emoji")
        return None
    title = question.get("title") or slug
    qvec = None
    name_vecs = None
    model: Model | None = None
    if embed:
        qvec = question_vector(title, question.get("goal") or "yes")
        name_vecs = atom_vectors([(atom.text, atom.name) for atom in catalog])
        model = load_model()
    picks = shortlist(title, catalog, _impacts(db, slug), qvec, name_vecs, model, k=SHORTLIST)
    if not picks:
        log(f"  {slug}: empty catalog")
        return None
    used = 0
    best_phrase: str | None = None
    best: Score | None = None

    async def sample(phrase: str, n: int) -> Score | None:
        nonlocal used
        if used + n > budget:
            return None
        used += n
        got = await score(phrase, n)
        if qvec is not None:
            pieces = phrase.split(ZWJ)
            observe(qvec, pieces, logit(objective.p(got)), name_vecs)
        return got

    log(f"{slug}: {len(picks)} emoji shortlisted, budget {budget}")
    for atom in picks:
        got = await sample(atom.text, 1)
        if got is None:
            break
        if best is None or objective.p(got) > objective.p(best):
            best_phrase, best = atom.text, got
        log(f"  {objective.p(got):.3f} {atom.text} {atom.name}")

    while best_phrase and best is not None and len(best_phrase.split(ZWJ)) < MAX_CHAIN:
        base = site_round(objective.p(best))
        improved: str | None = None
        improved_score: Score | None = None
        present = set(best_phrase.split(ZWJ))
        stopped = False
        for atom in picks:
            if atom.text in present:
                continue
            try:
                trial = check_word(best_phrase + ZWJ + atom.text)
            except RuleError:
                continue
            got = await sample(trial, 1)
            if got is None:
                stopped = True
                break
            log(f"  {objective.p(got):.3f} {trial}")
            if site_round(objective.p(got)) > base and (
                improved_score is None or objective.p(got) > objective.p(improved_score)
            ):
                improved, improved_score = trial, got
        if improved is not None and improved_score is not None:
            best_phrase, best = improved, improved_score
        if stopped or improved is None:
            break

    if best_phrase is None or best is None:
        log(f"  {slug}: budget {budget} produced no score")
        return None
    if used + CONFIRM <= budget:
        confirmed = await sample(best_phrase, CONFIRM)
        if confirmed is not None:
            best = confirmed
    phrase = best_phrase
    units = objective.units(phrase)
    p = objective.p(best)
    if not objective.beats(p, units, leader):
        log(f"  {slug}: {p:.3f}/{units} does not beat the leader")
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
    return phrase


def run_slugs(
    db: DB, slugs: list[str], *, board: str = HIGH_SCORES, budget: int = DEFAULT_BUDGET, log=print
) -> list[str]:
    """Search each slug. One failure does not stop the rest."""

    async def _all() -> list[str]:
        saved = []
        for slug in slugs:
            try:
                phrase = await search(db, slug, board=board, budget=budget, log=log)
            except Exception as error:
                log(f"{slug}: {error}")
                continue
            if phrase:
                saved.append(phrase)
        return saved

    return asyncio.run(_all())
