"""Cross-edition scores for phrases that lead a board.

A play is a question plus the answer it is aimed at. Yes/no questions use their goal
(`yes` or `no`). A choice question is one play per option, and only when it is not ranked
and names fewer than eight options. Anything else is left out.

Each edition that has a leader on its strict Highest board adds one row. The phrase is
that leader's submitted line. Vault lines that were never submitted, and submitted lines
that are not leading, are not rows.

The remaining columns are that phrase's score on every edition's model: the goal
probability for a yes/no question, or P(option) for a choice. The source column is the
probability already on the board. Other columns come from oracle samples when we have
them. Decider is down, so a Decider cell with no sample stays blank and is not retried.
A model that refuses the phrase stays blank too.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import EDITION_MODELS, EDITIONS, JEV_DATA, ROOT
from .db import DB
from .modes import HIGH_SCORES, STRICT
from .objective import goal_p, site_round
from .oracle import question_key
from .rules import option_names

# A choice question is in the table only when it names fewer options than this.
CHOICE_LIMIT = 8
DECIDER = "decider"
TABLE_NAME = "cross.json"
CSV_NAME = "cross.csv"
BASELINES_NAME = "baselines.json"
BASELINES_CSV = "baselines.csv"


def analysis_dir() -> Path:
    return ROOT / "analysis"


def table_path() -> Path:
    return analysis_dir() / TABLE_NAME


def csv_path() -> Path:
    return analysis_dir() / CSV_NAME


def baselines_path() -> Path:
    return analysis_dir() / BASELINES_NAME


def baselines_csv_path() -> Path:
    return analysis_dir() / BASELINES_CSV


def play_key(slug: str, answer: str) -> str:
    return f"{slug} | {answer}"


def plays_for(question: dict) -> list[str]:
    """Answers that make a play. Empty when this question is out of the table."""
    kind = question.get("kind") or ""
    if kind == "noul":
        return [question.get("goal") or "yes"]
    if kind != "choice":
        return []
    raw = question.get("raw") or {}
    if raw.get("ranked"):
        return []
    options = [name for name in option_names(raw.get("choices")) if name]
    if not options or len(options) >= CHOICE_LIMIT:
        return []
    return options


def leading_row(rows: list, answer: str, kind: str) -> dict | None:
    """The strict Highest leader for this answer. Our own row counts; a lower row does not."""
    if not isinstance(rows, list):
        return None
    best: dict | None = None
    best_key: tuple | None = None
    for row in rows:
        if not isinstance(row, dict) or row.get("probability") is None or not row.get("phrase"):
            continue
        filed = row.get("choice") or ""
        if kind == "choice":
            if filed != answer:
                continue
        elif filed:
            continue
        probability = float(row["probability"])
        words = int(row.get("wordCount") or 0)
        key = (site_round(probability), -words, probability)
        if best_key is None or key > best_key:
            best, best_key = row, key
    return best


def probability_of(answer_json: str | None, target: str) -> float | None:
    """P(target) from a stored choice answer. A missing option is unknown, not zero."""
    try:
        answer = json.loads(answer_json or "")
    except ValueError:
        return None
    probabilities = answer.get("probabilities") if isinstance(answer, dict) else None
    if not isinstance(probabilities, dict) or target not in probabilities:
        return None
    try:
        return float(probabilities[target])
    except (TypeError, ValueError):
        return None


def phrase_values(samples: list[tuple], kind: str, answer: str) -> list[float]:
    """Goal probability, or P(answer), from stored samples. Each item is (noul, answer json)."""
    values = []
    for noul, payload in samples:
        if kind == "choice":
            value = probability_of(payload, answer)
        elif noul is None:
            value = None
        else:
            value = goal_p(float(noul), answer, "noul")
        if value is not None and math.isfinite(value):
            values.append(value)
    return values


def shown_score(kind: str, answer: str, mean: float) -> float:
    """A fresh oracle mean is P(yes), or already P(answer) when the oracle was aimed."""
    if kind == "choice":
        return round(mean, 4)
    return round(goal_p(mean, answer, "noul"), 4)


def baseline_questions(table: dict) -> list[dict]:
    """One record per question in the cross table. Choice options share that one record.

    `goal` is the yes/no goal. A choice question has no single goal answer; its baseline is the
    model's top-option probability with an empty phrase. `phrases` lists the leader phrases so a
    phrase row can be matched back to this question.
    """
    grouped: dict[str, dict] = {}
    for row in table["rows"]:
        item = grouped.get(row["slug"])
        if item is None:
            item = {
                "slug": row["slug"],
                "title": row["title"],
                "kind": row["kind"],
                "goal": row["answer"] if row["kind"] == "noul" else "",
                "phrases": [],
            }
            grouped[row["slug"]] = item
        if row["phrase"] not in item["phrases"]:
            item["phrases"].append(row["phrase"])
    return [grouped[slug] for slug in sorted(grouped)]


def baseline_value(kind: str, goal: str, mean: float) -> float:
    """Empty-phrase score in the same units as the cross table.

    Yes/no is P(goal). Choice is the top-option probability from that single empty call.
    """
    return shown_score(kind, goal or "yes", mean)


def empty_row(edition: str, question: dict, kind: str, answer: str, leader: dict) -> dict:
    probability = float(leader["probability"])
    row = {
        "play": play_key(question["slug"], answer),
        "slug": question["slug"],
        "title": question.get("title") or question["slug"],
        "kind": kind,
        "answer": answer,
        "edition": edition,
        "phrase": leader["phrase"],
        "board_p": round(probability, 4),
        "words": int(leader.get("wordCount") or 0),
        "leader": leader.get("name") or "",
        "scores": {name: None for name in EDITIONS},
        "n": {},
        "from": {},
        "blank": {},
    }
    write_score(row, edition, probability, None, "board")
    return row


def write_score(row: dict, edition: str, value: float, n: int | None, source: str) -> None:
    row["scores"][edition] = round(float(value), 4)
    if n is None:
        row["n"].pop(edition, None)
    else:
        row["n"][edition] = int(n)
    row["from"][edition] = source
    row["blank"].pop(edition, None)


def write_blank(row: dict, edition: str, reason: str) -> None:
    row["scores"][edition] = None
    row["n"].pop(edition, None)
    row["from"].pop(edition, None)
    row["blank"][edition] = reason


@dataclass(frozen=True)
class Gap:
    row: int
    scorer: str
    slug: str
    kind: str
    answer: str
    phrase: str


def gaps(table: dict) -> list[Gap]:
    """Cells with no score and no recorded blank. Decider and refusals are not gaps."""
    found = []
    for index, row in enumerate(table["rows"]):
        blank = row.get("blank") or {}
        for edition in table["editions"]:
            if row["scores"].get(edition) is None and edition not in blank:
                found.append(Gap(index, edition, row["slug"], row["kind"], row["answer"], row["phrase"]))
    return found


def edition_db_path(name: str, data: Path | None = None) -> Path:
    root = data or JEV_DATA
    if name == "jev":
        return root / "jev.db"
    return root / name / "jev.db"


def open_edition_dbs() -> dict[str, DB]:
    opened = {}
    for name in EDITIONS:
        path = edition_db_path(name)
        if path.exists():
            opened[name] = DB(path, readonly=True)
    return opened


def _boards(db: DB) -> dict[str, list]:
    found = {}
    for row in db.all("SELECT slug, rows FROM boards WHERE mode = ? AND board = ?", (STRICT, HIGH_SCORES)):
        try:
            parsed = json.loads(row["rows"]) if row["rows"] else []
        except ValueError:
            parsed = []
        found[row["slug"]] = parsed if isinstance(parsed, list) else []
    return found


def _chunked(items: list[str], size: int = 80) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def lookup_phrases(db: DB, question: dict, model: str, phrases: set[str]) -> tuple[dict[str, list], set[str]] | None:
    """Stored samples and refusals for these phrases on this edition's current question.

    None when the edition has no request to score against.
    """
    request = question.get("jev_request")
    if not request:
        return None
    if not phrases:
        return {}, set()
    qkey = question_key(request, model)
    samples: dict[str, list] = {phrase: [] for phrase in phrases}
    refused: set[str] = set()
    ordered = list(phrases)
    for group in _chunked(ordered):
        marks = ",".join("?" * len(group))
        for row in db.all(
            f"SELECT state, noul, answer FROM oracle_samples WHERE qkey = ? AND state IN ({marks})",
            [qkey, *group],
        ):
            samples.setdefault(row["state"], []).append((row["noul"], row["answer"]))
        for row in db.all(
            f"SELECT state FROM oracle_refusals WHERE qkey = ? AND state IN ({marks})",
            [qkey, *group],
        ):
            refused.add(row["state"])
    return samples, refused


def _apply_known(row: dict, scorer: str, found: tuple[dict[str, list], set[str]] | None) -> None:
    if found is None:
        write_blank(row, scorer, "missing-question")
        return
    samples, refused = found
    values = phrase_values(samples.get(row["phrase"]) or [], row["kind"], row["answer"])
    if values:
        write_score(row, scorer, sum(values) / len(values), len(values), "oracle")
        return
    if row["phrase"] in refused:
        write_blank(row, scorer, "refused")
        return
    if scorer == DECIDER:
        write_blank(row, scorer, "down")


def assemble(editions: dict[str, DB], log=None) -> dict:
    """One row per leading phrase, with whatever cross-scores are already stored."""
    log = log or (lambda *_args: None)
    log(f"reading {len(editions)} edition databases")
    questions = {name: db.questions() for name, db in editions.items()}
    boards = {name: _boards(db) for name, db in editions.items()}
    rows = []
    for name in EDITIONS:
        if name not in editions:
            continue
        for question in questions[name].values():
            kind = question.get("kind") or ""
            for answer in plays_for(question):
                leader = leading_row(boards[name].get(question["slug"]) or [], answer, kind)
                if leader is not None:
                    rows.append(empty_row(name, question, kind, answer, leader))
    order = {name: index for index, name in enumerate(EDITIONS)}
    rows.sort(key=lambda row: (row["slug"], row["answer"], order.get(row["edition"], len(EDITIONS))))

    needed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        for scorer in editions:
            if scorer != row["edition"]:
                needed[(scorer, row["slug"])].add(row["phrase"])
    log(f"leaders {len(rows)}; matching stored scores across {len(needed)} edition questions")
    known: dict[tuple[str, str], tuple[dict[str, list], set[str]] | None] = {}
    for scorer in EDITIONS:
        slugs = [slug for other, slug in needed if other == scorer]
        if not slugs:
            continue
        log(f"  {scorer}: {len(slugs)} questions")
        for slug in slugs:
            phrases = needed[(scorer, slug)]
            question = questions[scorer].get(slug)
            if not question:
                known[(scorer, slug)] = None
                continue
            known[(scorer, slug)] = lookup_phrases(editions[scorer], question, EDITION_MODELS[scorer], phrases)

    for row in rows:
        for scorer in EDITIONS:
            if scorer == row["edition"]:
                continue
            if scorer not in editions:
                write_blank(row, scorer, "missing-edition")
                continue
            _apply_known(row, scorer, known.get((scorer, row["slug"])))
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "board": HIGH_SCORES,
        "mode": STRICT,
        "editions": list(EDITIONS),
        "decider": "down",
        "rows": rows,
    }


def summary(table: dict) -> str:
    rows = table["rows"]
    editions = table["editions"]
    filled = sum(1 for row in rows for edition in editions if row["scores"].get(edition) is not None)
    reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        for reason in (row.get("blank") or {}).values():
            reasons[reason] += 1
    pending = gaps(table)
    plays = len({row["play"] for row in rows})
    lines = [
        f"plays {plays}",
        f"rows {len(rows)}",
        f"cells {len(rows) * len(editions)}  filled {filled}  gaps {len(pending)}  blank {sum(reasons.values())}",
    ]
    if reasons:
        lines.append("blanks " + ", ".join(f"{reason} {count}" for reason, count in sorted(reasons.items())))
    return "\n".join(lines)


def save_table(table: dict, path: Path | None = None) -> Path:
    path = path or table_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(table, indent=1) + "\n")
    tmp.replace(path)
    _write_csv(table, path.with_name(CSV_NAME))
    return path


def load_table(path: Path | None = None) -> dict:
    path = path or table_path()
    return json.loads(path.read_text())


def _write_csv(table: dict, path: Path) -> None:
    fields = ["play", "slug", "answer", "kind", "edition", "phrase", "board_p", *table["editions"]]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in table["rows"]:
            flat = {field: row.get(field, "") for field in fields}
            for edition in table["editions"]:
                value = row["scores"].get(edition)
                flat[edition] = "" if value is None else value
            writer.writerow(flat)
    tmp.replace(path)


async def fill_missing(table: dict, *, limit: int | None = None, log=print) -> int:
    """Score gap cells. Decider is skipped. A refusal is stored as a blank, not a score."""
    from .config import activate
    from .oracle import Oracle, OracleError

    pending = gaps(table)
    if limit is not None and limit >= 0:
        pending = pending[:limit]
    if not pending:
        log("no gaps")
        return 0
    by_scorer: dict[str, list[Gap]] = defaultdict(list)
    for gap in pending:
        by_scorer[gap.scorer].append(gap)
    done = 0
    for scorer, items in by_scorer.items():
        log(f"scoring {len(items)} cell(s) on {scorer}")
        activate(scorer)
        db = DB()
        try:
            done += await _fill_edition(db, table, scorer, items, log, Oracle, OracleError)
        finally:
            db.conn.close()
        save_table(table)
    return done


async def _fill_edition(db: DB, table: dict, scorer: str, items: list[Gap], log, oracle_cls, error_cls) -> int:
    groups: dict[tuple[str, str, str], list[Gap]] = defaultdict(list)
    for gap in items:
        groups[(gap.slug, gap.answer, gap.kind)].append(gap)
    done = 0
    for (slug, answer, kind), group in groups.items():
        question = db.question(slug)
        if not question or not question.get("jev_request"):
            for gap in group:
                write_blank(table["rows"][gap.row], scorer, "missing-question")
            continue
        phrases = list(dict.fromkeys(gap.phrase for gap in group))
        target = answer if kind == "choice" else ""
        oracle = oracle_cls(db, question["jev_request"], target=target)
        try:
            try:
                scored = await oracle.score(phrases, n=1)
            except error_cls as error:
                log(f"  {slug} [{answer}]: {error}")
                continue
        finally:
            await oracle.close()
        for gap in group:
            score = scored.get(gap.phrase)
            if score is not None and score.n and math.isfinite(score.mean):
                write_score(table["rows"][gap.row], scorer, shown_score(kind, answer, score.mean), score.n, "oracle")
                done += 1
                continue
            refused = db.one("SELECT 1 FROM oracle_refusals WHERE qkey = ? AND state = ?", (oracle.qkey, gap.phrase))
            if refused:
                write_blank(table["rows"][gap.row], scorer, "refused")
                continue
            log(f"  no score {slug} [{answer}] {gap.phrase[:80]}")
    return done


def empty_baselines(table: dict) -> dict:
    """Question-level shell. Decider stays blank; the other models are filled by a call with no phrase."""
    rows = []
    for question in baseline_questions(table):
        row = {
            "slug": question["slug"],
            "title": question["title"],
            "kind": question["kind"],
            "goal": question["goal"],
            "phrases": question["phrases"],
            "baselines": {name: None for name in EDITIONS},
            "blank": {DECIDER: "down"},
        }
        rows.append(row)
    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "join": "slug",
        "editions": list(EDITIONS),
        "decider": "down",
        "rows": rows,
    }


def save_baselines(table: dict, path: Path | None = None) -> Path:
    path = path or baselines_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(table, indent=1) + "\n")
    tmp.replace(path)
    _write_baselines_csv(table, path.with_name(BASELINES_CSV))
    return path


def _write_baselines_csv(table: dict, path: Path) -> None:
    fields = ["slug", "kind", "goal", "title", "phrases", *table["editions"]]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in table["rows"]:
            flat = {
                "slug": row["slug"],
                "kind": row["kind"],
                "goal": row["goal"],
                "title": row["title"],
                "phrases": " || ".join(row["phrases"]),
            }
            for edition in table["editions"]:
                value = row["baselines"].get(edition)
                flat[edition] = "" if value is None else value
            writer.writerow(flat)
    tmp.replace(path)


def baseline_summary(table: dict) -> str:
    rows = table["rows"]
    editions = table["editions"]
    filled = sum(1 for row in rows for edition in editions if row["baselines"].get(edition) is not None)
    reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        for reason in (row.get("blank") or {}).values():
            reasons[reason] += 1
    gaps = sum(
        1
        for row in rows
        for edition in editions
        if row["baselines"].get(edition) is None and edition not in (row.get("blank") or {})
    )
    lines = [
        f"questions {len(rows)}",
        f"cells {len(rows) * len(editions)}  filled {filled}  gaps {gaps}  blank {sum(reasons.values())}",
    ]
    if reasons:
        lines.append("blanks " + ", ".join(f"{reason} {count}" for reason, count in sorted(reasons.items())))
    return "\n".join(lines)


async def collect_baselines(table: dict, *, log=print) -> dict:
    """Score each cross-table question once per model with an empty phrase. Decider is not called."""
    from .config import activate
    from .oracle import Oracle, OracleError

    baselines = empty_baselines(table)
    for scorer in EDITIONS:
        if scorer == DECIDER:
            continue
        pending = [row for row in baselines["rows"] if row["baselines"].get(scorer) is None and scorer not in row["blank"]]
        if not pending:
            continue
        log(f"baselines on {scorer}: {len(pending)} questions")
        activate(scorer)
        db = DB()
        try:
            done = 0
            for index, row in enumerate(pending, start=1):
                question = db.question(row["slug"])
                if not question or not question.get("jev_request"):
                    row["blank"][scorer] = "missing-question"
                    continue
                goal = row["goal"] or (question.get("goal") or "yes")
                oracle = Oracle(db, question["jev_request"])
                try:
                    try:
                        scored = await oracle.score([""], n=1)
                    except OracleError as error:
                        log(f"  {row['slug']}: {error}")
                        continue
                finally:
                    await oracle.close()
                score = scored.get("")
                if score is not None and score.n and math.isfinite(score.mean):
                    row["baselines"][scorer] = baseline_value(row["kind"], goal, score.mean)
                    row["blank"].pop(scorer, None)
                    done += 1
                elif db.one("SELECT 1 FROM oracle_refusals WHERE qkey = ? AND state = ?", (oracle.qkey, "")):
                    row["blank"][scorer] = "refused"
                else:
                    log(f"  no baseline {row['slug']}")
                if index % 40 == 0:
                    log(f"  {scorer} {index}/{len(pending)}")
            log(f"  {scorer} scored {done}")
        finally:
            db.conn.close()
        baselines["built_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_baselines(baselines)
    return baselines
