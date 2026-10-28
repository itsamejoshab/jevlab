"""Turn a Strict Highest line into a space-separated glyph string. No oracle."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..modes import HIGH_SCORES, STRICT
from .lexicon import CLOSED, STOPS, Index, tokens
from .rank import Atom
from .rules import RuleError, check_phrase

SIM_FLOOR = 0.45


def bearing(phrase: str, impacts: dict[str, float]) -> list[str]:
    """Load-bearing tokens in phrase order. Cap 12, or 8 content words when nothing else qualifies."""
    folded = {word.casefold(): value for word, value in impacts.items()}
    words = tokens(phrase)
    kept: list[str] = []
    for word in words:
        if word in CLOSED or folded.get(word, 0.0) > 0:
            kept.append(word)
        if len(kept) >= 12:
            break
    if kept:
        return kept
    fallback: list[str] = []
    for word in words:
        if word not in STOPS and len(word) > 3:
            fallback.append(word)
        if len(fallback) >= 8:
            break
    return fallback


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def glyph_for(
    word: str,
    index: Index,
    name_vecs: dict[str, np.ndarray] | None = None,
    word_vec: np.ndarray | None = None,
) -> Atom | None:
    """Closed-class glyph, else a name token, else the nearest name at cosine 0.45 or higher."""
    if word in CLOSED:
        for text in CLOSED[word]:
            atom = index.atom_by_text.get(text)
            if atom is not None:
                return atom
    hits = index.by_token.get(word)
    if hits:
        return hits[0]
    if word_vec is None or not name_vecs:
        return None
    query = _unit(word_vec)
    best: Atom | None = None
    best_cos = SIM_FLOOR
    for text, vec in name_vecs.items():
        score = float(np.dot(query, _unit(vec)))
        atom = index.atom_by_text.get(text)
        if atom is None or score < SIM_FLOOR:
            continue
        if best is None or score > best_cos:
            best = atom
            best_cos = score
    return best


def project_phrase(
    phrase: str,
    impacts: dict[str, float],
    index: Index,
    name_vecs: dict[str, np.ndarray] | None = None,
    word_vec: Callable[[str], np.ndarray] | None = None,
) -> str:
    """One glyph per bearing word, in order. Empty when nothing maps or the phrase is illegal."""
    glyphs: list[str] = []
    seen: set[str] = set()
    for word in bearing(phrase, impacts):
        vec = word_vec(word) if word_vec is not None else None
        atom = glyph_for(word, index, name_vecs, vec)
        if atom is None or atom.text in seen:
            continue
        seen.add(atom.text)
        glyphs.append(atom.text)
    if not glyphs:
        return ""
    try:
        return check_phrase(" ".join(glyphs))
    except RuleError:
        return ""


def strict_impacts(db, slug: str) -> dict[str, float]:
    """Strict word impacts for this question, casefolded. Emoji impacts are a different table."""
    found: dict[str, float] = {}
    for row in db.word_impacts(slug, STRICT):
        word = str(row.get("word") or "").casefold()
        if word:
            found[word] = float(row.get("averageImpact") or 0.0)
    return found


def strict_lines(slug: str, *, kind: str, goal: str) -> list[dict]:
    """This edition's measured Strict Highest lines, strongest first.

    A choice question prefers lines aimed at its goal, then lines with no target.
    """
    from ..config import EDITION
    from ..transfer import edition_vault
    from .. import vault

    rows = [
        entry
        for entry in vault.entries_for(slug, root=edition_vault(EDITION))
        if entry.get("mode") == STRICT
        and (entry.get("board") or HIGH_SCORES) == HIGH_SCORES
        and not entry.get("estimated_from")
    ]
    rows.sort(key=lambda entry: float(entry.get("p_mean") or 0.0), reverse=True)
    if kind == "choice":
        targeted = [entry for entry in rows if (entry.get("target") or "") == goal]
        if targeted:
            return targeted
        return [entry for entry in rows if not entry.get("target")]
    return rows


def projections(
    lines: list[dict],
    impacts: dict[str, float],
    index: Index,
    name_vecs: dict[str, np.ndarray] | None = None,
    word_vec: Callable[[str], np.ndarray] | None = None,
    limit: int = 3,
) -> list[str]:
    """Up to `limit` glyph strings. A later line whose bearing-token set was already used is skipped."""
    used: list[frozenset[str]] = []
    out: list[str] = []
    for line in lines:
        phrase = str(line.get("phrase") or "")
        key = frozenset(bearing(phrase, impacts))
        if key in used:
            continue
        text = project_phrase(phrase, impacts, index, name_vecs, word_vec)
        if not text:
            continue
        used.append(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out
