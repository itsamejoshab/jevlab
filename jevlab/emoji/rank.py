"""Local ranking of the pictograph catalog. No oracle and no generator."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .learn import TRUST_RHO

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Atom:
    text: str
    name: str


def tokens(text: str) -> set[str]:
    return {piece for piece in _TOKEN.findall(text.casefold()) if len(piece) > 2}


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def cold_score(title: str, name: str, impact: float, qvec: np.ndarray | None, name_vec: np.ndarray | None) -> float:
    """Name overlap, plus cosine when both vectors exist, plus a snapshot impact when we have one."""
    overlap = len(tokens(title) & tokens(name))
    cosine = 0.0
    if qvec is not None and name_vec is not None:
        cosine = float(np.dot(_unit(qvec), _unit(name_vec)))
    return overlap + max(cosine, 0.0) + float(impact or 0.0)


def _spread(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.0 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def shortlist(
    title: str,
    atoms: list[Atom],
    impacts: dict[str, float],
    qvec: np.ndarray | None = None,
    name_vecs: dict[str, np.ndarray] | None = None,
    model=None,
    k: int = 24,
) -> list[Atom]:
    """The atoms worth an oracle call. A model below the Spearman gate does not move this list."""
    name_vecs = name_vecs or {}
    ranked = []
    for atom in atoms:
        ranked.append(
            (
                atom,
                cold_score(title, atom.name, impacts.get(atom.text, 0.0), qvec, name_vecs.get(atom.text)),
            )
        )
    trusted = model is not None and qvec is not None and getattr(model, "rho", 0.0) >= TRUST_RHO
    if not trusted:
        ranked.sort(key=lambda row: row[1], reverse=True)
        return [atom for atom, _score in ranked[:k]]
    learned = [float(model.atom_score(qvec, atom.text)) for atom, _score in ranked]
    weight = min(float(model.rho), 1.0)
    cold = _spread([score for _atom, score in ranked])
    learned_n = _spread(learned)
    blended = [
        ((1.0 - weight) * cold_part + weight * learned_part, atom)
        for (atom, _score), cold_part, learned_part in zip(ranked, cold, learned_n)
    ]
    blended.sort(key=lambda row: row[0], reverse=True)
    return [atom for _score, atom in blended[:k]]
