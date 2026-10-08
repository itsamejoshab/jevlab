"""One predictor across every question: embedding of "Q: <title> goal <goal> || <phrase>" -> logit P(goal),
as an offset from that question's mean.

A brand-new question has no data for its own predictor, but what makes Jev lean (framing, renames, scenes)
transfers between questions. The engine blends this prior with the per-question predictor until the local
one is trusted. `ensure()` runs as each question starts: it retrains only when enough new phrases have been
scored, and embedding vectors are cached in SQLite, so a retrain embeds just the new phrases."""

from __future__ import annotations

import pickle
import random
import threading
import zlib

import numpy as np
from sklearn.linear_model import Ridge

from ..config import DATA
from ..db import DB
from ..objective import goal_p, logit
from ..oracle import history
from .surrogate import Embedder, spearman

MODEL_PATH = DATA / "models" / "global.pkl"
RETRAIN_AFTER = 500  # newly scored phrases (across all questions) before the next retrain
_lock = threading.Lock()
_train_lock = threading.Lock()


def framed(title: str, goal: str, phrase: str) -> str:
    return f"Q: {title} goal {goal} || {phrase}"


def _spearman(a, b) -> float:
    return spearman(np.asarray(a), np.asarray(b))


def scored_phrases(db: DB) -> int:
    return db.phrase_count()


class GlobalPredictor:
    def __init__(self, embedder_kind: str, model=None):
        self.embedder_kind = embedder_kind
        self.model = model
        self.rho = 0.0
        self.trained_on = 0
        self.phrase_count = 0  # scored_phrases() when trained, to tell when a retrain is due
        self.embedder = None  # attached at load time, never pickled

    def features(self, texts: list[str], units: list[int], bulk: bool = False) -> np.ndarray:
        emb = self.embedder.embed_bulk(texts) if bulk else self.embedder.embed(texts)
        # Trained on lines of at most 60 words; longer chains clip rather than extrapolate.
        return np.hstack([emb, np.array([[min(u / 60.0, 1.0)] for u in units], dtype=np.float32)])

    def predict(self, title: str, goal: str, phrases: list[str]) -> np.ndarray:
        if not phrases:
            return np.zeros(0)
        X = self.features([framed(title, goal, p) for p in phrases], [len(p.split()) for p in phrases])
        return self.model.predict(X)

    def save(self) -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        embedder, self.embedder = self.embedder, None
        try:
            with _lock, open(MODEL_PATH, "wb") as fh:
                pickle.dump(self, fh)
        finally:
            self.embedder = embedder


def load(embedder) -> GlobalPredictor | None:
    """The saved model if it was trained with the same embedding backend."""
    if not MODEL_PATH.exists():
        return None
    try:
        with _lock, open(MODEL_PATH, "rb") as fh:
            model: GlobalPredictor = pickle.load(fh)
    except Exception:
        return None
    if model.embedder_kind != embedder.kind:
        return None
    model.embedder = embedder
    return model


def ensure(embedder, log=print) -> GlobalPredictor | None:
    """The saved model, retrained first if it is missing, uses another embedding backend, or is RETRAIN_AFTER
    phrases behind. Only phrases that were never embedded cost an embedding request."""
    with _train_lock:  # a retrain left running by the previous question finishes first, then gets loaded
        return _ensure(embedder, log)


def _ensure(embedder, log) -> GlobalPredictor | None:
    db = DB()
    model = load(embedder)
    behind = scored_phrases(db) - (getattr(model, "phrase_count", 0) if model else 0)
    if model is not None and behind < RETRAIN_AFTER:
        return model
    why = "no saved model for this embedding" if model is None else f"{behind} new phrases since last training"
    log(f"cross-question prior: retraining ({why}); only unseen phrases get embedded")
    return train_global(db, log=log, embedder=embedder) or model


def train_global(db: DB, limit: int = 0, per_question: int = 1500, log=print, embedder=None) -> GlobalPredictor | None:
    count = scored_phrases(db)
    rows: list[tuple[str, str, int, float]] = []  # text, question slug, units, target
    for q in db.all("SELECT slug FROM questions WHERE kind = 'noul'"):
        question = db.question(q["slug"])
        if not question or not question.get("jev_request"):
            continue
        samples = history(db, question["jev_request"])
        goal = question.get("goal") or "yes"
        items = [(s, v) for s, v in samples.items() if v and s.strip()]
        # A stable per-phrase order: retrains keep the same sample, so their vectors come from the cache.
        items.sort(key=lambda item: zlib.crc32(item[0].encode()))
        for phrase, values in items[:per_question]:
            p = goal_p(sum(values) / len(values), goal)
            rows.append((framed(question["title"], goal, phrase), q["slug"], len(phrase.split()), logit(p, 0.005)))
    if limit:
        random.Random(3).shuffle(rows)
        rows = rows[:limit]
    slugs = sorted({r[1] for r in rows})
    if len(slugs) < 4:
        log(f"only {len(slugs)} questions with data; nothing to train")
        return None
    embedder = embedder or Embedder()
    log(f"training on {len(rows)} phrases from {len(slugs)} questions with {embedder.kind} embeddings")
    model = GlobalPredictor(embedder.kind)
    model.embedder = embedder
    X = model.features([r[0] for r in rows], [r[2] for r in rows], bulk=True)
    y = np.array([r[3] for r in rows])
    # Only the ranking within a question matters, so learn offsets from each question's mean.
    slug_of = np.array([r[1] for r in rows])
    for slug in slugs:
        mask = slug_of == slug
        y[mask] -= y[mask].mean()
    # Grouped 5-fold: every question is held out once, so rho doesn't swing with which few were picked.
    order = slugs[:]
    random.Random(3).shuffle(order)
    per_q = []
    for fold in range(5):
        held = order[fold::5]
        train = ~np.isin(slug_of, held)
        pred = Ridge(alpha=10.0).fit(X[train], y[train]).predict(X[~train])
        per_q += [_spearman(pred[m], y[~train][m]) for m in (slug_of[~train] == s for s in held) if m.sum() >= 20]
    model.rho = float(np.mean(per_q)) if per_q else 0.0
    log(
        f"held-out within-question rho {model.rho:.3f} (median {np.median(per_q):.2f}) over {len(per_q)} "
        f"questions, 5-fold by question; the engine blends it in when rho >= 0.2"
    )
    model.model = Ridge(alpha=10.0).fit(X, y)
    model.trained_on = len(rows)
    model.phrase_count = count
    model.save()
    log(f"saved {MODEL_PATH}")
    return model
