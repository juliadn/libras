"""k-NN sign classifier.

Every saved example is a descriptor (see features.py):
  * static examples  -> compared with the held part of an attempt,
  * dynamic examples -> compared with the whole attempt resampled to 32 frames.

Distances are divided by the typical distance between two examples of the same
sign ("spread", measured on the library itself), so static and dynamic signs
end up on the same scale: a score of ~1 means "as close as two of your own
examples are to each other".

Per sign we average the k nearest scores (k-NN), then turn the scores into
probabilities. An extra "unknown" option at a fixed score soaks up probability
when nothing is close, which gives the "Not recognized, try again" case.

Adding a sign = adding examples. There is no training step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .features import describe
from .segmenter import Segment

K = 3                     # neighbours averaged per sign
UNKNOWN_SCORE = 2.6       # score of the "none of these" option
N_RETRIEVE = 150          # nearest example variants fetched from the index per query
# Fallback / minimum spreads (in feature units) when a sign has few examples.
# (Two different handshapes are ~2 apart; landmark jitter alone gives ~0.05.)
DEFAULT_SPREAD = {"static": 0.3, "dynamic": 0.45}
MIN_SPREAD = {"static": 0.15, "dynamic": 0.12}
PRIOR_WEIGHT = 1.5        # how many "virtual examples" pull a sign's spread to the average


@dataclass
class Example:
    sample_id: str
    label: str
    kind: str                       # "static" | "dynamic"
    desc: np.ndarray                # (V, d) variants; row 0 = uncut (see features.describe)
    desc_mirror: np.ndarray         # (V, d) same, as if signed with the other hand
    penalty: np.ndarray             # (V,) score cost of each variant


@dataclass
class Query:
    # each: (variants, mirrored variants, penalties) as returned by features.describe
    static: tuple | None = None
    dynamic: tuple | None = None

    def get(self, kind: str):
        return self.static if kind == "static" else self.dynamic


def query_from_segment(seg: Segment) -> Query:
    still = seg.still_frames
    return Query(static=describe(still, "static") if still else None,
                 dynamic=describe(seg.frames, "dynamic", role="query"))


def query_from_static_take(frames) -> Query:
    return Query(static=describe(frames, "static"))


@dataclass
class Prediction:
    label: str | None
    confidence: float
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    recognized: bool = False
    score: float | None = None

    def to_json(self) -> dict:
        return {
            "type": "prediction",
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "alternatives": [[lbl, round(p, 3)] for lbl, p in self.alternatives],
            "recognized": self.recognized,
        }


class _Group:
    """All examples of one kind, with a k-NN index and the measured spread."""

    def __init__(self, kind: str, examples: list[Example]):
        self.kind = kind
        self.examples = examples
        self.labels = np.array([e.label for e in examples], dtype=object)
        # uncut version of each example (spread, library checks)
        self.X = np.stack([e.desc[0] for e in examples])
        self.Xm = np.stack([e.desc_mirror[0] for e in examples])
        # every variant of every example (the k-NN index)
        self.rows = np.concatenate([e.desc for e in examples])
        self.row_ex = np.concatenate([np.full(len(e.desc), i) for i, e in enumerate(examples)])
        self.row_pen = np.concatenate([e.penalty for e in examples])
        self.nn = NearestNeighbors(algorithm="brute", metric="euclidean").fit(self.rows)
        self._pairwise = None
        self.spread, self.label_spread = self._measure_spread()
        self.row_spread = np.array([self.label_spread[self.labels[i]] for i in self.row_ex])

    def pairwise(self) -> np.ndarray:
        """Distance between every two examples (either hand)."""
        if self._pairwise is None:
            def d(A, B):
                sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
                return np.sqrt(np.maximum(sq, 0))
            D = np.minimum(d(self.X, self.X), d(self.X, self.Xm))
            np.fill_diagonal(D, np.inf)
            self._pairwise = D
        return self._pairwise

    def _measure_spread(self) -> tuple[float, dict[str, float]]:
        """Typical distance between two examples of the same sign.

        Each sign gets its own spread (a sign recorded very consistently is
        judged more strictly), pulled toward the average over all signs when it
        has few examples.
        """
        D = self.pairwise()
        per_label: dict[str, list[float]] = {}
        for i, lbl in enumerate(self.labels):
            same = np.sort(D[i, self.labels == lbl])
            same = same[np.isfinite(same)]
            if len(same):
                per_label.setdefault(lbl, []).append(same[:K].mean())
        vals = [v for vs in per_label.values() for v in vs]
        overall = float(np.median(vals)) if vals else DEFAULT_SPREAD[self.kind]
        overall = max(overall, MIN_SPREAD[self.kind])
        spreads = {}
        for lbl in set(self.labels):
            vs = per_label.get(lbl, [])
            own = float(np.median(vs)) if vs else overall
            n = len(vs)
            blended = np.sqrt((n * own ** 2 + PRIOR_WEIGHT * overall ** 2) / (n + PRIOR_WEIGHT))
            spreads[lbl] = max(float(blended), MIN_SPREAD[self.kind])
        return overall, spreads

    def scores(self, q: tuple, mirror: bool) -> dict[int, float]:
        """example index -> score (distance / spread + cut penalties) for the
        nearest examples. The best (attempt cut, example cut) pair counts.

        With `mirror`, the attempt is also compared as if signed with the other
        hand, and the closer of the two counts (left-handed signers welcome).
        """
        desc, desc_m, q_pen = q
        n = min(N_RETRIEVE, len(self.rows))
        V = np.concatenate([desc, desc_m]) if mirror else desc
        V_pen = np.concatenate([q_pen, q_pen]) if mirror else q_pen
        dist, idx = self.nn.kneighbors(V, n_neighbors=n)
        out: dict[int, float] = {}
        for v in range(len(V)):
            for dd, r in zip(dist[v], idx[v]):
                ex = int(self.row_ex[r])
                s = dd / self.row_spread[r] + self.row_pen[r] + V_pen[v]
                if s < out.get(ex, np.inf):
                    out[ex] = float(s)
        return out


def scores_to_prediction(label_scores: dict[str, float], threshold: float) -> Prediction:
    if not label_scores:
        return Prediction(label=None, confidence=0.0)
    labels = list(label_scores)
    s = np.array([label_scores[lbl] for lbl in labels])
    w = np.exp(-0.5 * s ** 2)
    total = w.sum() + np.exp(-0.5 * UNKNOWN_SCORE ** 2)
    p = w / total
    order = np.argsort(-p)
    top = order[0]
    return Prediction(
        label=labels[top],
        confidence=float(p[top]),
        alternatives=[(labels[i], float(p[i])) for i in order[1:3]],
        recognized=bool(p[top] >= threshold),
        score=float(s[top]),
    )


class Classifier:
    def __init__(self, examples: list[Example]):
        self.examples = list(examples)
        self.groups: dict[str, _Group] = {}
        for kind in ("static", "dynamic"):
            ex = [e for e in self.examples if e.kind == kind]
            if ex:
                self.groups[kind] = _Group(kind, ex)

    @property
    def labels(self) -> set[str]:
        return {e.label for e in self.examples}

    def with_extra(self, extra: list[Example]) -> "Classifier":
        return Classifier(self.examples + list(extra))

    def label_scores(self, query: Query, mirror: bool = True,
                     allowed: set[str] | None = None,
                     exclude_ids: set[str] = frozenset()) -> dict[str, float]:
        per_label: dict[str, list[float]] = {}
        for kind, g in self.groups.items():
            q = query.get(kind)
            if q is None:
                continue
            for i, score in g.scores(q, mirror).items():
                ex = g.examples[i]
                if ex.sample_id in exclude_ids:
                    continue
                if allowed is not None and ex.label not in allowed:
                    continue
                per_label.setdefault(ex.label, []).append(score)
        return {lbl: float(np.mean(sorted(v)[:K])) for lbl, v in per_label.items()}

    def predict(self, query: Query, threshold: float = 0.5, mirror: bool = True,
                allowed: set[str] | None = None) -> Prediction:
        return scores_to_prediction(self.label_scores(query, mirror, allowed), threshold)

    def flags(self) -> dict[str, str]:
        """sample_id -> label it looks more like (leave-one-out check of the library)."""
        out = {}
        for g in self.groups.values():
            spread_of = np.array([g.label_spread[lbl] for lbl in g.labels])
            D = g.pairwise() / spread_of[None, :]
            for i, ex in enumerate(g.examples):
                best_lbl, best = None, np.inf
                for lbl in set(g.labels):
                    row = np.sort(D[i, g.labels == lbl])
                    row = row[np.isfinite(row)][:K]
                    if len(row) and row.mean() < best:
                        best_lbl, best = lbl, row.mean()
                if best_lbl is not None and best_lbl != ex.label:
                    out[ex.sample_id] = best_lbl
        return out
