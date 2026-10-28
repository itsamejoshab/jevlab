"""Glyphs and English words in one vector space. No oracle."""

from __future__ import annotations

import numpy as np

from .rank import Atom


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def mean_vector(space, words: list[str]) -> np.ndarray | None:
    """Average of the word vectors that exist. None when none of them do."""
    found: list[np.ndarray] = []
    for word in words:
        got = space.word_vector(word)
        if got is None:
            continue
        vec = np.asarray(got, dtype=np.float64)
        if vec.size == 0:
            continue
        found.append(vec)
    if not found:
        return None
    return np.mean(found, axis=0)


def residual_glyph(space, target: np.ndarray | None, current: str, catalog: list[Atom]) -> Atom | None:
    """Glyph nearest the phrase vector minus the vectors already in the line."""
    if space is None or target is None:
        return None
    total = np.zeros(np.asarray(target, dtype=np.float64).shape, dtype=np.float64)
    for glyph in current.split():
        got = space.glyph_vector(glyph)
        if got is None:
            continue
        vec = np.asarray(got, dtype=np.float64)
        if vec.shape != total.shape:
            continue
        total = total + vec
    present = set(current.split())
    ranked = nearest_glyphs(
        space,
        np.asarray(target, dtype=np.float64) - total,
        [atom for atom in catalog if atom.text not in present],
        1,
    )
    return ranked[0] if ranked else None


def gap_chain(space, target: np.ndarray | None, catalog: list[Atom], limit: int) -> list[str]:
    """Prefixes that walk toward `target`. Each step is the glyph nearest the residual.

    The walk stops when that glyph would not reduce the distance, or at `limit`.
    """
    if space is None or target is None or limit < 1:
        return []
    aim = np.asarray(target, dtype=np.float64)
    chosen: list[str] = []
    total = np.zeros(aim.shape, dtype=np.float64)
    dist = float(np.linalg.norm(aim))
    while len(chosen) < limit:
        atom = residual_glyph(space, aim, " ".join(chosen), catalog)
        if atom is None:
            break
        got = space.glyph_vector(atom.text)
        if got is None:
            break
        vec = np.asarray(got, dtype=np.float64)
        if vec.shape != total.shape:
            break
        nxt = total + vec
        nxt_dist = float(np.linalg.norm(aim - nxt))
        if nxt_dist >= dist - 1e-9:
            break
        chosen.append(atom.text)
        total = nxt
        dist = nxt_dist
    return [" ".join(chosen[:end]) for end in range(1, len(chosen) + 1)]


def alternate_phrases(
    space,
    target: np.ndarray | None,
    catalog: list[Atom],
    avoid: set[str],
    lengths,
    per_length: int,
) -> list[str]:
    """Glyph strings near `target` at each length, using glyphs outside `avoid`."""
    if per_length < 1:
        return []
    pool = nearest_glyphs(
        space,
        target,
        [atom for atom in catalog if atom.text not in avoid],
        len(catalog),
    )
    out: list[str] = []
    seen: set[str] = set()
    for length in lengths:
        for text in beam_phrases(space, target, pool, [length], 8, per_length):
            if text in seen or set(text.split()) & avoid:
                continue
            seen.add(text)
            out.append(text)
    return out


def beam_phrases(space, target: np.ndarray | None, pool: list[Atom], lengths, width: int, limit: int) -> list[str]:
    """Strings of the asked lengths whose glyph average sits nearest `target`. At most `limit`."""
    if space is None or target is None or width < 1 or limit < 1:
        return []
    query = _unit(np.asarray(target, dtype=np.float64))
    finished: list[tuple[float, str]] = []
    for length in lengths:
        if length < 1:
            continue
        beam: list[tuple[str, ...]] = [()]
        for _step in range(length):
            nxt: list[tuple[float, tuple[str, ...]]] = []
            for glyphs in beam:
                for atom in pool:
                    seq = glyphs + (atom.text,)
                    vecs: list[np.ndarray] = []
                    for glyph in seq:
                        got = space.glyph_vector(glyph)
                        if got is None:
                            vecs = []
                            break
                        vecs.append(np.asarray(got, dtype=np.float64))
                    if not vecs:
                        continue
                    cosine = float(np.dot(query, _unit(np.mean(vecs, axis=0))))
                    nxt.append((cosine, seq))
            nxt.sort(key=lambda item: item[0], reverse=True)
            seen: set[tuple[str, ...]] = set()
            beam = []
            for _cosine, seq in nxt:
                if seq in seen:
                    continue
                seen.add(seq)
                beam.append(seq)
                if len(beam) >= width:
                    break
        for seq in beam:
            if not seq:
                continue
            vecs = []
            for glyph in seq:
                got = space.glyph_vector(glyph)
                if got is None:
                    vecs = []
                    break
                vecs.append(np.asarray(got, dtype=np.float64))
            if not vecs:
                continue
            cosine = float(np.dot(query, _unit(np.mean(vecs, axis=0))))
            finished.append((cosine, " ".join(seq)))
    finished.sort(key=lambda item: item[0], reverse=True)
    out: list[str] = []
    seen_text: set[str] = set()
    for _cosine, text in finished:
        if text in seen_text:
            continue
        seen_text.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


class Table:
    """Word and glyph rows loaded from a local table."""

    def __init__(self, words: dict[str, np.ndarray], glyphs: dict[str, np.ndarray]):
        self._words = words
        self._glyphs = glyphs

    def word_vector(self, word: str) -> np.ndarray | None:
        return self._words.get(word.casefold())

    def glyph_vector(self, glyph: str) -> np.ndarray | None:
        return self._glyphs.get(glyph)


_TRIED = False
_TABLE: Table | None = None


def _table_path():
    """Edition models first, then the shared `data/models` table."""
    from ..config import DATA, JEV_DATA

    path = DATA / "models" / "emoji_space.npz"
    if path.exists():
        return path
    shared = JEV_DATA / "models" / "emoji_space.npz"
    if shared != path and shared.exists():
        return shared
    return None


def cached_table() -> Table | None:
    """The on-disk joint table, or None when no `emoji_space.npz` is present."""
    global _TRIED, _TABLE
    if _TRIED:
        return _TABLE
    _TRIED = True
    path = _table_path()
    if path is None:
        return None
    try:
        blob = np.load(path, allow_pickle=False)
        word_keys = [str(key) for key in blob["word_keys"]]
        glyph_keys = [str(key) for key in blob["glyph_keys"]]
        words = {key: blob["word_vecs"][index] for index, key in enumerate(word_keys)}
        glyphs = {key: blob["glyph_vecs"][index] for index, key in enumerate(glyph_keys)}
    except (OSError, ValueError, KeyError):
        return None
    _TABLE = Table(words, glyphs)
    return _TABLE


def nearest_glyphs(space, target: np.ndarray | None, catalog: list[Atom], limit: int) -> list[Atom]:
    """Catalog glyphs nearest `target`, highest cosine first. Ties keep catalog order."""
    if space is None or target is None or limit < 1:
        return []
    query = _unit(target)
    ranked: list[tuple[float, int, Atom]] = []
    for index, atom in enumerate(catalog):
        got = space.glyph_vector(atom.text)
        if got is None:
            continue
        ranked.append((float(np.dot(query, _unit(got))), -index, atom))
    ranked.sort(reverse=True)
    return [atom for _score, _index, atom in ranked[:limit]]


def round_robin(space, targets: list[np.ndarray], catalog: list[Atom], limit: int) -> list[Atom]:
    """One nearest glyph from each target, then the next, until `limit`."""
    if space is None or not targets or limit < 1:
        return []
    lists = [nearest_glyphs(space, target, catalog, limit) for target in targets]
    aimed: list[Atom] = []
    seen: set[str] = set()
    depth = 0
    while len(aimed) < limit:
        grew = False
        for glyphs in lists:
            if depth >= len(glyphs):
                continue
            atom = glyphs[depth]
            if atom.text in seen:
                continue
            seen.add(atom.text)
            aimed.append(atom)
            grew = True
            if len(aimed) >= limit:
                break
        if not grew:
            break
        depth += 1
    return aimed
