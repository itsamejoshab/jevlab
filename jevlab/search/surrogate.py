"""CPU surrogates of Jev trained on our own oracle samples.

PhraseSurrogate: sentence embedding -> BayesianRidge on logit(p). Used to
prefilter big candidate pools and for UCB acquisition.

WordSurrogate: per-word embeddings pooled two ways (mean and position-weighted)
-> small MLP. It is differentiable with respect to each word vector, which is
what the GCG/HotFlip proposer needs, since Jev itself exposes no gradients.

Both report a holdout Spearman rho; callers gate on it.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import threading
import time
import warnings

import numpy as np

# Imported here, not inside fit(): per-question training and the cross-question prior both start in worker
# threads, and two threads doing sklearn's first import at once can raise importlib's _DeadlockError.
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor

from ..objective import logit

warnings.filterwarnings("ignore", module="sklearn")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


class _EmbeddingStore:
    """The vector cache, in its own file. The hot database no longer carries these blobs."""

    def __init__(self):
        from ..config import embeddings_path

        path = embeddings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (model TEXT, text TEXT, vec BLOB, PRIMARY KEY (model, text))"
        )
        self.lock = threading.Lock()

    def all(self, sql: str, params: tuple | list = ()) -> list[sqlite3.Row]:
        with self.lock:
            return list(self.conn.execute(sql, params))

    def executemany(self, sql: str, rows: list) -> None:
        with self.lock:
            self.conn.execute("BEGIN")
            try:
                self.conn.executemany(sql, rows)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise


class OpenRouterEmbeddings:
    """POST /embeddings with batching; every vector is cached in SQLite so a text is embedded once."""

    def __init__(self, model: str, batch: int = 256):
        import httpx

        from ..config import OPENROUTER_BASE, OPENROUTER_KEY

        if not OPENROUTER_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self.model = model
        self.batch = batch
        self.db = _EmbeddingStore()
        self.http = httpx.Client(
            base_url=OPENROUTER_BASE, timeout=120, headers={"Authorization": f"Bearer {OPENROUTER_KEY}"}
        )
        self.cost = 0.0

    def _stored(self, texts: list[str]) -> dict[str, np.ndarray]:
        out = {}
        for start in range(0, len(texts), 500):
            chunk = texts[start : start + 500]
            marks = ",".join("?" * len(chunk))
            for row in self.db.all(
                f"SELECT text, vec FROM embeddings WHERE model = ? AND text IN ({marks})", (self.model, *chunk)
            ):
                out[row["text"]] = np.frombuffer(row["vec"], dtype=np.float32)
        return out

    def embed(self, texts: list[str], batch_size: int = 0):
        found = self._stored(texts)
        missing = [t for t in texts if t not in found]
        start = 0
        while start < len(missing):
            chunk = missing[start : start + self.batch]
            for attempt in range(3):
                response = self.http.post("/embeddings", json={"model": self.model, "input": chunk})
                if response.status_code == 200:
                    break
                limit = re.search(r"at most (\d+)", response.text)
                if response.status_code == 400 and limit and int(limit.group(1)) < len(chunk):
                    self.batch = int(limit.group(1))  # e.g. Gemini takes at most 100 texts per request
                    chunk = chunk[: self.batch]
                    continue
                time.sleep(1 + attempt * 2)
            response.raise_for_status()
            start += len(chunk)
            data = response.json()
            self.cost += float((data.get("usage") or {}).get("cost") or 0)
            rows = sorted(data["data"], key=lambda r: r.get("index", 0))
            vectors = [np.asarray(r["embedding"], dtype=np.float32) for r in rows]
            self.db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?)",
                [(self.model, t, v.tobytes()) for t, v in zip(chunk, vectors)],
            )
            found.update(zip(chunk, vectors))
        for text in texts:
            yield found[text]


class Embedder:
    """`model` None/"local": fastembed bge-small if available, else hashed character n-grams (offline).
    "openrouter:<id>": an OpenRouter embedding model, cached in SQLite."""

    def __init__(self, model: str | None = None, dim_fallback: int = 512, cache_limit: int = 500_000):
        from ..config import EMBED_BACKEND

        self.cache: dict[str, np.ndarray] = {}
        self.cache_limit = cache_limit  # OpenRouter vectors evicted here are re-read from SQLite, not re-requested
        self.lock = threading.Lock()
        self.kind = "hash"
        self.dim = dim_fallback
        self.model = None
        model = model if model is not None else EMBED_BACKEND
        if model and model.startswith("openrouter:"):
            try:
                self.model = OpenRouterEmbeddings(model.split(":", 1)[1])
                self.dim = len(next(iter(self.model.embed(["probe"]))))
                self.kind = model.split("/")[-1]
                return
            except Exception:
                self.model = None
        try:
            from fastembed import TextEmbedding

            self.model = TextEmbedding(EMBED_MODEL, threads=max(4, (os.cpu_count() or 8) - 4))
            self.dim = len(next(iter(self.model.embed(["probe"]))))
            self.kind = "bge-small"
        except Exception:
            self.model = None

    def _hash_embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        padded = f"  {text.casefold()}  "
        for n in (3, 4, 5):
            for i in range(len(padded) - n + 1):
                h = int.from_bytes(hashlib.blake2b(padded[i : i + n].encode(), digest_size=8).digest(), "little")
                vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        for word in text.casefold().split():
            h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 2.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self.lock:
            found = {t: self.cache[t] for t in dict.fromkeys(texts) if t in self.cache}
        missing = [t for t in dict.fromkeys(texts) if t not in found]
        if missing:
            if self.model is not None:
                vectors = list(self.model.embed(missing, batch_size=256))
            else:
                vectors = [self._hash_embed(t) for t in missing]
            fresh = {text: np.asarray(vec, dtype=np.float32) for text, vec in zip(missing, vectors)}
            found.update(fresh)
            with self.lock:
                self.cache.update(fresh)
                if len(self.cache) > self.cache_limit:
                    for text in list(self.cache)[: len(self.cache) - int(self.cache_limit * 0.9)]:
                        del self.cache[text]
        return np.stack([found[t] for t in texts])

    def embed_bulk(self, texts: list[str]) -> np.ndarray:
        """embed() for training sets: skips the in-memory cache. OpenRouter backends still read and fill the
        SQLite cache, so only texts never embedded before cost a request."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.model is None:
            return np.stack([self._hash_embed(t) for t in texts])
        return np.stack([np.asarray(v, dtype=np.float32) for v in self.model.embed(texts, batch_size=256)])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _target(p: float) -> float:
    return logit(p, eps=0.005)


class PhraseSurrogate:
    def __init__(self, embedder: Embedder, prior=None, question: tuple[str, str] = ("", "yes")):
        self.embedder = embedder
        self.model = None
        self.rho = 0.0
        self.trained_on = 0
        self.prior = prior  # GlobalPredictor trained across questions, or None
        self.question = question  # (title, goal) for the prior
        self.y_mean = 0.0
        self.length_scale = 60.0

    @property
    def prior_weight(self) -> float:
        """Share of the prediction taken from the cross-question prior: all of it until the local model is
        trusted, fading to none once local holdout rho reaches 0.6."""
        if self.prior is None or self.prior.rho < 0.2:
            return 0.0
        if self.model is None:
            return 1.0
        return float(np.clip(1.0 - self.rho / 0.6, 0.0, 1.0))

    def features(self, phrases: list[str]) -> np.ndarray:
        emb = self.embedder.embed(phrases)
        units = np.array([[len(p.split()) / self.length_scale] for p in phrases], dtype=np.float32)
        return np.hstack([emb, units])

    @property
    def ready(self) -> bool:
        return (self.model is not None and self.rho >= 0.2) or self.prior_weight > 0

    def fit(self, data: dict[str, float], seed: int = 0) -> float:
        phrases = list(data)
        if len(phrases) < 40:
            return 0.0
        X = self.features(phrases)
        y = np.array([_target(data[p]) for p in phrases])
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(phrases))
        cut = int(len(order) * 0.8)
        holdout = BayesianRidge().fit(X[order[:cut]], y[order[:cut]])
        self.rho = spearman(holdout.predict(X[order[cut:]]), y[order[cut:]])
        self.model = BayesianRidge().fit(X, y)
        self.y_mean = float(y.mean())
        self.trained_on = len(phrases)
        return self.rho

    def predict(self, phrases: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not phrases:
            return np.zeros(0), np.ones(0)
        w = self.prior_weight
        if self.model is None:
            if w > 0:
                return self.prior.predict(*self.question, phrases), np.ones(len(phrases))
            return np.zeros(len(phrases)), np.ones(len(phrases))
        mean, std = self.model.predict(self.features(phrases), return_std=True)
        if w > 0:
            # The prior predicts offsets from a question's mean, so shift it onto this question's scale.
            prior = self.prior.predict(*self.question, phrases) + self.y_mean
            mean = w * prior + (1 - w) * mean
        return mean, std

    def ucb(self, phrases: list[str], kappa: float = 1.0) -> np.ndarray:
        mean, std = self.predict(phrases)
        return mean + kappa * std


class WordSurrogate:
    """MLP over [mean word vector, position-weighted word vector, length]."""

    def __init__(self, embedder: Embedder, hidden: int = 64):
        self.embedder = embedder
        self.hidden = hidden
        self.model = None
        self.rho = 0.0
        self.trained_on = 0
        self.length_scale = 60.0

    def _pool(self, words: list[str]) -> np.ndarray:
        E = self.embedder.embed(words)
        weights = np.arange(1, len(words) + 1, dtype=np.float32)
        weights /= weights.sum()
        return np.concatenate([E.mean(axis=0), (E * weights[:, None]).sum(axis=0), [len(words) / self.length_scale]])

    def features(self, phrases: list[str]) -> np.ndarray:
        return np.stack([self._pool(p.split()) for p in phrases])

    @property
    def ready(self) -> bool:
        return self.model is not None and self.rho >= 0.3

    def fit(self, data: dict[str, float], seed: int = 0) -> float:
        phrases = [p for p in data if p.split()]
        if len(phrases) < 80:
            return 0.0
        words = sorted({w for p in phrases for w in p.split()})
        self.embedder.embed(words)
        X = self.features(phrases)
        y = np.array([_target(data[p]) for p in phrases])
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(phrases))
        cut = int(len(order) * 0.8)

        def make():
            return MLPRegressor(
                hidden_layer_sizes=(self.hidden,),
                activation="relu",
                alpha=1e-3,
                max_iter=400,
                early_stopping=True,
                random_state=seed,
            )

        holdout = make().fit(X[order[:cut]], y[order[:cut]])
        self.rho = spearman(holdout.predict(X[order[cut:]]), y[order[cut:]])
        self.model = make().fit(X, y)
        self.trained_on = len(phrases)
        return self.rho

    def predict(self, phrases: list[str]) -> np.ndarray:
        if self.model is None or not phrases:
            return np.zeros(len(phrases))
        return self.model.predict(self.features(phrases))

    def word_gradients(self, words: list[str]) -> np.ndarray:
        """d(prediction)/d(word vector) for each position, shape (L, dim)."""
        x = self._pool(words)
        W1, W2 = self.model.coefs_
        b1 = self.model.intercepts_[0]
        pre = x @ W1 + b1
        dx = W1 @ (W2[:, 0] * (pre > 0))
        dim = self.embedder.dim
        g_mean, g_pos = dx[:dim], dx[dim : 2 * dim]
        L = len(words)
        weights = np.arange(1, L + 1, dtype=np.float32)
        weights /= weights.sum()
        return np.stack([g_mean / L + g_pos * weights[i] for i in range(L)])

    def hotflip(self, words: list[str], vocab: list[str], topk: int = 16) -> list[tuple[int, str, float]]:
        """First-order estimate of the gain from swapping each position to each vocab word."""
        if not words or not vocab:
            return []
        G = self.word_gradients(words)
        E_words = self.embedder.embed(words)
        E_vocab = self.embedder.embed(vocab)
        gains = E_vocab @ G.T - np.sum(E_words * G, axis=1)[None, :]
        out: list[tuple[int, str, float]] = []
        for i in range(len(words)):
            column = gains[:, i]
            for j in np.argsort(-column)[:topk]:
                if vocab[j] != words[i] and math.isfinite(column[j]):
                    out.append((i, vocab[j], float(column[j])))
        return out
