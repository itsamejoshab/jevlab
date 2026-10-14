"""Local surrogate rho is one Spearman on pooled 5-fold out-of-fold predictions."""

import numpy as np

from jevlab.search.surrogate import PhraseSurrogate, WordSurrogate


class _Table:
    """Embeds each phrase as a row chosen by the test."""

    def __init__(self, rows: dict[str, np.ndarray]):
        self.rows = rows

    def embed(self, texts):
        return np.stack([self.rows[text] for text in texts]).astype(np.float32)


def _sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))


def _folds(n: int, seed: int = 0) -> np.ndarray:
    """Fold id of each row under the same stride the scorer uses."""
    order = np.random.default_rng(seed).permutation(n)
    position = np.empty(n, dtype=int)
    position[order] = np.arange(n)
    return position % 5


def _reversed_holdout(n: int, seed: int = 0) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Labels follow the embedding, except the 20% one 80/20 split with this seed would score."""
    xs = np.linspace(-3.5, 3.5, n)
    order = np.random.default_rng(seed).permutation(n)
    held = set(order[int(n * 0.8) :].tolist())
    data, rows = {}, {}
    for i, x in enumerate(xs):
        key = str(i)
        z = -float(x) if i in held else float(x)
        data[key] = _sigmoid(z)
        rows[key] = np.array([float(x)])
    return data, rows


def _simpson(n: int) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Within each stride fold the label fights the feature; across folds it follows it.

    Averaging the five within-fold Spearmans is about -1. One Spearman on the pooled
    out-of-fold predictions stays high, because the between-fold trend dominates.
    """
    fold = _folds(n)
    data, rows = {}, {}
    within = np.zeros(n, dtype=int)
    for k in range(5):
        members = np.flatnonzero(fold == k)
        within[members] = np.arange(len(members))
    for i in range(n):
        key = str(i)
        feature = float(fold[i]) * 1.5 + float(within[i]) * 0.15
        label = float(fold[i]) * 1.5 - float(within[i]) * 0.15
        data[key] = _sigmoid(label)
        rows[key] = np.array([feature])
    return data, rows


def _private_noise(n: int = 40, seed: int = 0) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Each row's only feature is private to that row, and the label is noise.

    A model scored on rows it just fit ranks them perfectly. Out-of-fold predictions cannot.
    """
    labels = np.random.default_rng(seed).normal(size=n)
    eye = np.eye(n)
    data = {str(i): _sigmoid(float(labels[i])) for i in range(n)}
    rows = {str(i): eye[i] for i in range(n)}
    return data, rows


def test_phrase_rho_is_pooled_out_of_fold():
    data, rows = _reversed_holdout(40)
    # The reversed slice alone would score -1. The other rows are still ranked out of fold.
    assert PhraseSurrogate(_Table(rows)).fit(data, seed=0) > 0.4

    data, rows = _simpson(40)
    # Mean of the five fold Spearmans is about -1. The pooled Spearman is not.
    assert PhraseSurrogate(_Table(rows)).fit(data, seed=0) > 0.5


def test_phrase_rho_does_not_score_rows_the_model_just_fit():
    data, rows = _private_noise()
    rho = PhraseSurrogate(_Table(rows)).fit(data, seed=0)
    assert abs(rho) < 0.4


def test_word_rho_is_pooled_out_of_fold():
    data, rows = _reversed_holdout(80)
    surrogate = WordSurrogate(_Table(rows))
    assert surrogate.fit(data, seed=0) > 0.4
    assert surrogate.trained_on == len(data)

    data, rows = _simpson(80)
    assert WordSurrogate(_Table(rows)).fit(data, seed=0) > 0.5
