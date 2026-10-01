"""Triage: which lines we already have should a mirror edition score next, before any new ones are generated.

Kev and Laya start from lines Jev already rates well: estimated lines in this edition's vault, Jev's vault, and
the top of Jev's oracle history. They are ranked by Jev's score for this board, lines already vaulted first on
ties, near-duplicates dropped, and anything this edition has already measured skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import vault
from ..config import EDITION, JEV_SOURCE_MODEL, JEV_VAULT
from ..db import DB
from ..modes import STRICT
from ..objective import Leader, Objective, goal_p, objective_for
from ..oracle import Oracle, Score, history, question_key
from ..rules import RuleError, check_phrase, normalize, option_names
from ..rules.banned import contains as phrase_banned
from ..rules.strict import MAX_WORDS

JEV_HISTORY_TOP = 200
# A line is resampled to `confirm_n` when its first roll, plus this, would beat the leader.
CLOSE_MARGIN = 0.05
DUPLICATE_JACCARD = 0.8
# Vaulted lines count as this much more Jev score when ranking: a winner once is worth a look before a
# slightly higher unvaulted history line.
VAULT_BONUS = 0.02


@dataclass
class TriageItem:
    phrase: str
    source: str  # {edition}-vault | jev-vault | jev-history
    jev_p: float
    units: int
    vaulted: bool  # in this edition's vault or Jev's
    kev_entry: bool  # already an (estimated) line in this edition's vault


def jev_history(jev: DB, slug: str, target: str = "", limit: int = JEV_HISTORY_TOP) -> dict[str, float]:
    """Jev's best-scored phrases for a question (mean P(goal), or P(target)), at least 2 samples each."""
    question = jev.question(slug)
    if not question or not question.get("jev_request"):
        return {}
    goal, kind = question.get("goal") or "yes", question.get("kind") or "noul"
    qkey = question_key(question["jev_request"], JEV_SOURCE_MODEL)
    if target:
        value, params = "json_extract(answer, ?)", [f'$.probabilities."{target}"', qkey]
    else:
        value, params = "noul", [qkey]
    order = "ASC" if not target and kind == "noul" and goal == "no" else "DESC"
    rows = jev.all(f"SELECT state, AVG(v) AS m FROM (SELECT state, {value} AS v FROM oracle_samples WHERE qkey = ?) "
                   f"WHERE v IS NOT NULL GROUP BY state HAVING COUNT(*) >= 2 ORDER BY m {order} LIMIT ?",
                   params + [limit])
    return {r["state"]: float(r["m"]) if target else goal_p(float(r["m"]), goal, kind) for r in rows}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def rank_candidates(db: DB, slug: str, objective: Objective, leader: Leader | None, k: int, target: str = "",
                    jev: DB | None = None, exclude: set[str] | None = None,
                    length_cap: int = MAX_WORDS) -> list[TriageItem]:
    """The k lines this edition should score first on this board. `exclude` adds phrases already scored elsewhere."""
    question = db.question(slug)
    if not question or not question.get("jev_request"):
        return []
    choices = option_names((question.get("raw") or {}).get("choices"))
    measured = set(history(db, question["jev_request"], target)) | (exclude or set())
    pool: dict[str, TriageItem] = {}

    def offer(text: str, source: str, p: float, vaulted: bool, kev_entry: bool = False) -> None:
        if phrase_banned(text):
            return
        try:
            phrase = check_phrase(normalize(text), choices, length_cap)
        except RuleError:
            return
        if phrase in measured:
            return
        prior = pool.get(phrase)
        if prior is None:
            pool[phrase] = TriageItem(phrase, source, p, len(phrase.split()), vaulted, kev_entry)
        else:
            prior.jev_p = max(prior.jev_p, p)
            prior.vaulted |= vaulted
            prior.kev_entry |= kev_entry

    for e in vault.load(slug, STRICT, objective.board, target)["entries"]:
        if e.get("estimated_from") and e.get("status") not in ("rejected", "dropped"):
            offer(e["phrase"], f"{EDITION}-vault", float(e.get("jev_p") or e["p_mean"]), True, True)
    for e in vault.all_entries(root=JEV_VAULT):
        if e["slug"] == slug and e["mode"] == STRICT and (e.get("target") or "") == target \
                and e.get("status") not in ("rejected", "dropped"):
            offer(e["phrase"], "jev-vault", float(e["p_mean"]), True)
    if jev is not None:
        for state, p in jev_history(jev, slug, target).items():
            offer(state, "jev-history", p, False)

    items = list(pool.values())
    if objective.shortest:
        # Only lines shorter than the leader that Jev (nearly) took to yes can win here.
        items = [i for i in items if objective.qualifies(i.jev_p + 0.1) and (leader is None or i.units < leader.units)]
        items.sort(key=lambda i: (i.units, -(i.jev_p + VAULT_BONUS * i.vaulted)))
    else:
        items.sort(key=lambda i: (-round(i.jev_p + VAULT_BONUS * i.vaulted, 2), i.units, -i.jev_p))
    picked: list[TriageItem] = []
    for item in items:
        words = {w.casefold() for w in item.phrase.split()}
        if any(jaccard(words, {w.casefold() for w in p.phrase.split()}) > DUPLICATE_JACCARD for p in picked):
            continue
        picked.append(item)
        if len(picked) >= k:
            break
    return picked


def record(slug: str, objective: Objective, leader: Leader | None, item: TriageItem, score: Score, title: str,
           target: str = "") -> dict | None:
    """Write a measurement to this edition's vault: always for lines already there (the estimate becomes measured),
    otherwise only when it wins."""
    wins = objective.wins(score, item.units, leader)
    if not item.kev_entry and not wins:
        return None
    extra = {}
    if objective.gamble(score, item.units, leader):
        extra = {"gamble": True, "p_reach": round(max(objective.goal_samples(score)), 4),
                 "hits": f"{objective.hits(score)}/{score.n}"}
    return vault.save(slug, STRICT, item.phrase, p_mean=objective.p(score), p_lcb=objective.p_lcb(score),
                      spread=score.spread, n=score.n, units=item.units, leader=leader, beats=wins, title=title,
                      origin=f"triage:{item.source}", note="triage", board=objective.board, extra=extra,
                      target=target, jev_p=item.jev_p)


async def run_triage(db: DB, slug: str, board: str, k: int = 20, n: int = 1, confirm_n: int = 3, target: str = "",
                     queue: bool = False, log=print) -> list[tuple[TriageItem, Score]]:
    """Score the top-k existing lines for one board with the edition's oracle, resample the close ones, and
    record the measurements in the vault. With `queue`, queue the best winner unless the board has one queued."""
    from ..transfer import jev_db, kev_leader

    question = db.question(slug)
    if not question or not question.get("jev_request"):
        log(f"{slug}: not in the snapshot")
        return []
    objective = objective_for(question, board=board)
    leader = kev_leader(db, slug, STRICT, board, target)
    jev = jev_db()
    try:
        items = rank_candidates(db, slug, objective, leader, k, target, jev)
    finally:
        if jev is not None:
            jev.conn.close()
    aimed = f" -> {target}" if target else ""
    lead = f"{leader.probability:.2f}/{leader.units}w {leader.name}" if leader else "empty board"
    if not items:
        log(f"{slug}{aimed} [{board}]: nothing left to triage (leader {lead})")
        return []
    log(f"{slug}{aimed} [{board}]: scoring {len(items)} existing line(s) at n={n} (leader {lead})")
    oracle = Oracle(db, question["jev_request"], target=target)
    try:
        scores = await oracle.score([i.phrase for i in items], n=n)
        close = [i.phrase for i in items
                 if objective.beats(objective.p(scores[i.phrase]) + CLOSE_MARGIN, i.units, leader)]
        if close and confirm_n > n:
            log(f"  {len(close)} close to the leader; resampling to n={confirm_n}")
            scores.update(await oracle.score(close, n=confirm_n))
    finally:
        await oracle.close()
    out = []
    for item in sorted(items, key=lambda i: objective.score_key(scores[i.phrase], i.units), reverse=True):
        score = scores[item.phrase]
        record(slug, objective, leader, item, score, question.get("title") or "", target)
        wins = objective.wins(score, item.units, leader)
        log(f"  {objective.p(score):.3f} n={score.n} (jev {item.jev_p:.3f}) {item.units:>2}w "
            f"{'WIN ' if wins else '    '}[{item.source}] {item.phrase}")
        out.append((item, score))
    log(f"  {oracle.calls} oracle calls, p50 {oracle.p50_ms:.0f}ms, ${oracle.cost:.4f}")
    if queue:
        winners = [(i, s) for i, s in out if objective.wins(s, i.units, leader)]
        current = vault.load(slug, STRICT, board, target)["entries"]
        if winners and not any(e.get("status") == "queued" for e in current):
            best, _ = max(winners, key=lambda w: (objective.score_key(w[1], w[0].units), objective.p_lcb(w[1])))
            vault.set_status(slug, STRICT, best.phrase, "queued", board, target=target, detail="triage")
            log(f"  queued: {best.phrase}")
    return out
