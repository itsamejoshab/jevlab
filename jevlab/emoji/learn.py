"""A small local model over the fixed emoji catalog.

Each atom's weight starts as the embedding of its name, so an untrained score
is the cosine with the question vector. Oracle scores we already pay for
shift those weights. The shortlist ignores the model until held-out Spearman
clears the gate.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from ..config import DATA
from ..search.surrogate import spearman

TRUST_RHO = 0.2
RETRAIN_AFTER = 50
_SHRINK = 10.0


def _unit(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def model_path() -> Path:
    return DATA / "models" / "emoji.pkl"


def rows_path() -> Path:
    return DATA / "models" / "emoji_rows.jsonl"


def vector_path() -> Path:
    return DATA / "models" / "emoji_vectors.npz"


class Model:
    """Per-atom score is cosine(question, name) plus a learned bias."""

    def __init__(self, names: dict[str, np.ndarray], bias: dict[str, float] | None = None, rho: float = 0.0):
        self.names = {key: _unit(value) for key, value in names.items()}
        self.bias = {key: 0.0 for key in self.names} if bias is None else dict(bias)
        self.rho = float(rho)

    def atom_score(self, qvec: np.ndarray, atom: str) -> float:
        name = self.names.get(atom)
        if name is None:
            return 0.0
        return float(np.dot(_unit(qvec), name)) + float(self.bias.get(atom, 0.0))

    def chain_score(self, qvec: np.ndarray, atoms: list[str]) -> float:
        return sum(self.atom_score(qvec, atom) for atom in atoms)

    @classmethod
    def untrained(cls, names: dict[str, np.ndarray]) -> "Model":
        return cls(names)

    def save(self, path: Path | None = None) -> None:
        path = path or model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))


def load_model(path: Path | None = None) -> Model | None:
    path = path or model_path()
    if not path.exists():
        return None
    try:
        model = pickle.loads(path.read_bytes())
    except (OSError, pickle.UnpicklingError, AttributeError):
        return None
    return model if isinstance(model, Model) else None


def _oof_rho(features: np.ndarray, target: np.ndarray, shrink: float) -> float:
    count = len(target)
    if count < 10:
        return 0.0
    predicted = np.zeros(count)
    width = features.shape[1]
    for fold in range(5):
        test = np.arange(count) % 5 == fold
        train = ~test
        if int(train.sum()) < 2 or int(test.sum()) == 0:
            return 0.0
        gram = features[train].T @ features[train] + shrink * np.eye(width)
        coef = np.linalg.solve(gram, features[train].T @ target[train])
        predicted[test] = features[test] @ coef
    return float(spearman(predicted, target))


def fit(
    rows: list[tuple[np.ndarray, list[str], float]], names: dict[str, np.ndarray], shrink: float = _SHRINK
) -> Model:
    """Ridge from scored chains. `rows` are (question vector, atoms in the chain, logit of the score)."""
    model = Model.untrained(names)
    atoms = list(model.names)
    if len(rows) < 8 or not atoms:
        return model
    index = {atom: pos for pos, atom in enumerate(atoms)}
    features = np.zeros((len(rows), len(atoms)))
    residual = np.zeros(len(rows))
    for row_index, (qvec, chain, target) in enumerate(rows):
        residual[row_index] = target - model.chain_score(qvec, chain)
        for atom in chain:
            pos = index.get(atom)
            if pos is not None:
                features[row_index, pos] = 1.0
    gram = features.T @ features + shrink * np.eye(len(atoms))
    coef = np.linalg.solve(gram, features.T @ residual)
    model.bias = {atom: float(value) for atom, value in zip(atoms, coef)}
    model.rho = _oof_rho(features, residual, shrink)
    return model


def _read_matrix(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    stored = np.load(path, allow_pickle=False)
    keys = [str(key) for key in stored["keys"]]
    return {key: np.asarray(vec, dtype=np.float64) for key, vec in zip(keys, stored["vecs"])}


def _write_matrix(path: Path, mapping: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(mapping)
    np.savez(path, keys=np.array(keys), vecs=np.stack([_unit(mapping[key]) for key in keys]))


def question_vector(title: str, goal: str, embed=None) -> np.ndarray | None:
    """Embed `title + goal` once and reuse the stored vector."""
    text = f"{title}\n{goal or 'yes'}"
    found = _read_matrix(vector_path()).get(text)
    if found is not None:
        return found
    if embed is None:
        embed = _local_embed
    try:
        vec = _unit(np.asarray(embed([text])[0], dtype=np.float64))
    except Exception:
        return None
    saved = _read_matrix(vector_path())
    saved[text] = vec
    _write_matrix(vector_path(), saved)
    return vec


def atom_vectors(atoms: list[tuple[str, str]], embed=None) -> dict[str, np.ndarray]:
    """One stored vector per catalog name. Missing names are embedded once."""
    path = vector_path()
    saved = _read_matrix(path)
    missing = [(text, name) for text, name in atoms if f"atom:{text}" not in saved]
    if missing and embed is not False:
        if embed is None:
            embed = _local_embed
        try:
            vectors = embed([name for _text, name in missing])
        except Exception:
            vectors = []
        for (text, _name), vec in zip(missing, vectors):
            saved[f"atom:{text}"] = _unit(np.asarray(vec, dtype=np.float64))
        if len(vectors):
            _write_matrix(path, saved)
    return {text: saved[f"atom:{text}"] for text, _name in atoms if f"atom:{text}" in saved}


def _local_embed(texts: list[str]) -> np.ndarray:
    from ..search.surrogate import Embedder

    return Embedder(model="local").embed(texts)


def _load_rows(path: Path | None = None) -> list[tuple[np.ndarray, list[str], float]]:
    path = path or rows_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        rows.append((np.asarray(item["q"], dtype=np.float64), list(item["a"]), float(item["y"])))
    return rows


def observe(
    qvec: np.ndarray, atoms: list[str], target: float, names: dict[str, np.ndarray] | None = None
) -> Model | None:
    """Append one scored chain. Retrain every RETRAIN_AFTER new rows."""
    path = rows_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps({"q": _unit(qvec).tolist(), "a": atoms, "y": target}) + "\n")
    count = sum(1 for line in path.read_text().splitlines() if line.strip())
    if count == 0 or count % RETRAIN_AFTER != 0:
        return None
    return retrain(names)


def retrain(names: dict[str, np.ndarray] | None = None) -> Model | None:
    rows = _load_rows()
    if names is None:
        names = {
            key.removeprefix("atom:"): vec
            for key, vec in _read_matrix(vector_path()).items()
            if key.startswith("atom:")
        }
    if not names or len(rows) < 8:
        return None
    model = fit(rows, names)
    model.save()
    return model
