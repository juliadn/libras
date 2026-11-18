"""Normalization of landmarks into feature vectors.

Per frame we build one vector made of:
  * each hand's SHAPE: 20 points relative to its own wrist, divided by the hand
    size (wrist -> middle finger base). Captures the handshape and orientation.
  * each hand's POSITION: wrist relative to the middle of the shoulders, divided
    by the shoulder width. So it doesn't matter where you stand in the frame or
    how far you are from the camera.
  * hand PRESENCE flags (one-handed vs two-handed signs).
  * FACE: ~40 points (eyebrows, eyes, mouth) relative to the nose tip.
  * ELBOWS relative to the shoulders.

Every group gets a weight so the distance between two vectors reflects what
matters most (handshape > position > face).

A sign attempt (many frames) becomes either:
  * a STATIC descriptor: the average frame (for held handshapes), or
  * a DYNAMIC descriptor: the frames resampled to 32 and flattened.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landmarks import FACE_IDX, FACE_NOSE, Frame, mirror_frame

N_FRAMES = 32               # every movement is resampled to this many frames
PRESENT_LINE = 1.3          # wrist more than this many shoulder widths below the
                            # shoulders = hand resting (e.g. on the desk), ignored.
                            # Slightly below the segmenter's rest line (1.2).
MIN_HAND_PRESENCE = 0.3     # hand seen in < 30% of the frames -> treat as absent

# Feature groups: (name, size, weight)
_GROUPS = [
    ("left_shape", 60, 1.0),
    ("left_pos", 2, 0.8),
    ("left_present", 1, 0.5),
    ("right_shape", 60, 1.0),
    ("right_pos", 2, 0.8),
    ("right_present", 1, 0.5),
    ("face", 2 * len(FACE_IDX), 2.0),
    ("elbows", 4, 0.3),
]
FRAME_DIM = sum(size for _, size, _ in _GROUPS)
# Per-dimension multiplier: weight / sqrt(size) -> each group's distance is the
# RMS difference of its values times its weight.
WEIGHTS = np.concatenate([np.full(size, w / np.sqrt(size)) for _, size, w in _GROUPS])


@dataclass
class Ref:
    center: np.ndarray   # (2,) middle of the shoulders, in "height units"
    scale: float         # shoulder width, in "height units"


def to_units(points: np.ndarray, aspect: float) -> np.ndarray:
    """Normalized image coords -> isotropic units (1.0 = image height)."""
    p = np.array(points[:, :3], dtype=np.float64)
    p[:, 0] *= aspect
    p[:, 2] *= aspect   # MediaPipe's z uses roughly the same scale as x
    return p


def body_ref(frame: Frame) -> Ref | None:
    if frame.pose is None:
        return None
    p = to_units(frame.pose, frame.aspect)
    ls, rs = p[11, :2], p[12, :2]
    scale = float(np.linalg.norm(ls - rs))
    if scale < 1e-3:
        return None
    return Ref(center=(ls + rs) / 2, scale=scale)


def hand_in_body(hand: np.ndarray, aspect: float, ref: Ref) -> np.ndarray:
    """(21, 2) hand points relative to the shoulders, in shoulder widths."""
    return (to_units(hand, aspect)[:, :2] - ref.center) / ref.scale


def wrist_height(hand: np.ndarray, ref: Ref) -> float:
    """How far below the shoulder line the wrist is (in shoulder widths).

    (y is already in height units, so the image aspect doesn't matter here.)"""
    return float((hand[0, 1] - ref.center[1]) / ref.scale)


# --- Cleaning a sequence ------------------------------------------------------

def _fill_refs(frames: list[Frame]) -> list[Ref] | None:
    refs = [body_ref(f) for f in frames]
    idx = [i for i, r in enumerate(refs) if r is not None]
    if not idx:
        return None
    out = []
    for i in range(len(frames)):
        if refs[i] is not None:
            out.append(refs[i])
        else:   # nearest frame that has the body
            j = min(idx, key=lambda k: abs(k - i))
            out.append(refs[j])
    return out


def _interp_track(ts: np.ndarray, arrays: list[np.ndarray | None]) -> list[np.ndarray | None]:
    """Fill gaps in a landmark track by linear interpolation over time."""
    mask = np.array([a is not None for a in arrays])
    if not mask.any():
        return arrays
    if mask.all():
        return arrays
    shape = next(a for a in arrays if a is not None).shape
    known = np.stack([a for a in arrays if a is not None]).reshape(int(mask.sum()), -1)
    kt = ts[mask]
    filled = np.stack([np.interp(ts, kt, known[:, k]) for k in range(known.shape[1])], axis=1)
    return [filled[i].reshape(shape) for i in range(len(arrays))]


def clean_sequence(frames: list[Frame]) -> tuple[list[Frame], list[Ref]] | None:
    """Fill short tracking dropouts, drop resting hands. None if no body seen."""
    if not frames:
        return None
    refs = _fill_refs(frames)
    if refs is None:
        return None
    ts = np.array([f.t for f in frames], dtype=np.float64)
    if len(ts) > 1 and np.any(np.diff(ts) <= 0):
        ts = np.arange(len(ts), dtype=np.float64)

    hands = {}
    for side in ("left", "right"):
        track = []
        for f, r in zip(frames, refs):
            h = getattr(f, side)
            if h is not None and wrist_height(h, r) > PRESENT_LINE:
                h = None   # resting hand
            track.append(h)
        present = np.mean([h is not None for h in track])
        if present < MIN_HAND_PRESENCE:
            track = [None] * len(frames)
        else:
            track = _interp_track(ts, track)
        hands[side] = track

    faces = _interp_track(ts, [f.face for f in frames])
    poses = _interp_track(ts, [f.pose for f in frames])
    clean = [
        Frame(t=f.t, pose=poses[i], left=hands["left"][i], right=hands["right"][i],
              face=faces[i], aspect=f.aspect)
        for i, f in enumerate(frames)
    ]
    return clean, refs


# --- Per-frame vector -----------------------------------------------------------

def frame_vector(f: Frame, ref: Ref) -> np.ndarray:
    parts = []
    for hand in (f.left, f.right):
        if hand is None:
            parts += [np.zeros(60), np.zeros(2), np.zeros(1)]
            continue
        h = to_units(hand, f.aspect)
        rel = h[1:] - h[0]
        size = max(float(np.linalg.norm(h[9] - h[0])), 1e-6)
        parts += [(rel / size).ravel(), (h[0, :2] - ref.center) / ref.scale, np.ones(1)]
    if f.face is not None:
        face = to_units(f.face, f.aspect)[:, :2]
        parts.append(((face - face[FACE_NOSE]) / ref.scale).ravel())
    else:
        parts.append(np.zeros(2 * len(FACE_IDX)))
    if f.pose is not None:
        p = to_units(f.pose, f.aspect)[:, :2]
        parts.append(((p[[13, 14]] - ref.center) / ref.scale).ravel())
    else:
        parts.append(np.zeros(4))
    return np.concatenate(parts) * WEIGHTS


def _prepare(frames: list[Frame]):
    """-> (times (T,), features (T, D), hand points per frame) or None."""
    cleaned = clean_sequence(frames)
    if cleaned is None:
        return None
    clean, refs = cleaned
    X = np.stack([frame_vector(f, r) for f, r in zip(clean, refs)])
    ts = np.array([f.t for f in clean], dtype=np.float64)
    if len(ts) > 1 and np.any(np.diff(ts) <= 0):
        ts = np.arange(len(ts), dtype=np.float64)
    P = []
    for f, r in zip(clean, refs):
        hands = [hand_in_body(h, f.aspect, r) for h in (f.left, f.right) if h is not None]
        P.append(np.concatenate(hands) if hands else None)
    return ts, X, P


def sequence_matrix(frames: list[Frame]) -> tuple[np.ndarray, np.ndarray] | None:
    """-> (times (T,), features (T, FRAME_DIM)) or None if unusable."""
    m = _prepare(frames)
    return None if m is None else (m[0], m[1])


def resample(ts: np.ndarray, X: np.ndarray, n: int = N_FRAMES) -> np.ndarray:
    """Resample a (T, D) sequence to (n, D) evenly spaced in time."""
    if len(X) == 1 or ts[-1] - ts[0] <= 0:
        return np.repeat(X.mean(axis=0, keepdims=True), n, axis=0)
    grid = np.linspace(ts[0], ts[-1], n)
    return np.stack([np.interp(grid, ts, X[:, k]) for k in range(X.shape[1])], axis=1)


# --- Movement part of an attempt ------------------------------------------------

STILL_TRIM = 0.08     # hand points moving less than this (shoulder widths) = still
KEEP_STILL = 2        # frames of a still start / end that are kept


def _still_run(P: list, order) -> int:
    """How many frames, walking in `order`, stay where the first one is."""
    order = list(order)
    anchor = P[order[0]]
    n = 1
    for i in order[1:]:
        if anchor is None or P[i] is None or P[i].shape != anchor.shape:
            break
        moved = np.sort(np.linalg.norm(P[i] - anchor, axis=-1))[-3]   # ignore 1-2 jittery points
        if moved > STILL_TRIM:
            break
        n += 1
    return n


def movement_part(ts, X, P):
    """Drop the still start and end of an attempt (the held handshape before /
    after the movement), keeping a couple of frames. What remains is the movement,
    which is what tells movement signs apart (NOME = U handshape + movement)."""
    T = len(ts)
    if T <= 2 * KEEP_STILL + 1:
        return ts, X
    lead = _still_run(P, range(T))
    trail = _still_run(P, range(T - 1, -1, -1))
    a = max(0, lead - KEEP_STILL)
    b = min(T, T - trail + KEEP_STILL)
    if b - a < 2:          # the whole thing is still: keep a short piece
        a, b = max(0, T - 2 * KEEP_STILL), T
    return ts[a:b], X[a:b]


# How a live attempt may be cut before comparing it with a saved example:
#   the attempt may start with a rise / transition the example doesn't have,
#   or end with the hands going down; a saved example may start with a rise
#   that the attempt doesn't have. Each cut costs a little (in score units).
QUERY_STARTS = (0.0, 0.15, 0.3, 0.45)
QUERY_ENDS = (0.0, 0.2, 0.35)
QUERY_MIN_KEEP = 0.45          # never compare less than 45% of the attempt
QUERY_CUT_COST = 0.5           # score added per 100% of the attempt cut away
TEMPLATE_STARTS = (0.0, 0.25)
TEMPLATE_CUT_COST = 1.2        # cutting the saved example costs more


def _crops(ts, X, starts, ends, min_keep, cost):
    out, pens = [], []
    dur = ts[-1] - ts[0]
    for a in starts:
        for b in ends:
            if 1 - a - b < min_keep - 1e-9:
                continue
            if dur <= 0:
                seg_t, seg_x = ts, X
            else:
                t0, t1 = ts[0] + a * dur, ts[-1] - b * dur
                grid = np.linspace(t0, t1, max(2, int(round(len(ts) * (1 - a - b))) + 1))
                seg_t = grid
                seg_x = np.stack([np.interp(grid, ts, X[:, k]) for k in range(X.shape[1])], axis=1)
            # / sqrt(n): distance = RMS per-frame distance (same scale as static)
            out.append(resample(seg_t, seg_x).ravel() / np.sqrt(N_FRAMES))
            pens.append(cost * (a + b))
    return np.stack(out), np.array(pens)


# --- Descriptors ---------------------------------------------------------------
# describe() returns (variants (V, d), mirrored variants (V, d), penalty (V,)).
# Static: one variant. Dynamic: several cuts of the movement (see above).

def static_descriptor(frames: list[Frame]) -> np.ndarray | None:
    m = sequence_matrix(frames)
    if m is None:
        return None
    return m[1].mean(axis=0)


def dynamic_descriptor(frames: list[Frame], role: str = "template"):
    m = _prepare(frames)
    if m is None:
        return None
    ts, X = movement_part(*m)
    if role == "query":
        return _crops(ts, X, QUERY_STARTS, QUERY_ENDS, QUERY_MIN_KEEP, QUERY_CUT_COST)
    return _crops(ts, X, TEMPLATE_STARTS, (0.0,), 0.0, TEMPLATE_CUT_COST)


def describe(frames: list[Frame], kind: str, role: str = "template"):
    """Descriptor of `kind` for frames and for their mirror image (other hand).

    -> (variants (V, d), mirrored variants (V, d), penalties (V,)) or None.
    Variant 0 is always the plain, uncut one.
    """
    mirrored = [mirror_frame(f) for f in frames]
    if kind == "static":
        d, dm = static_descriptor(frames), static_descriptor(mirrored)
        if d is None:
            return None
        return d[None, :], dm[None, :], np.zeros(1)
    d = dynamic_descriptor(frames, role)
    if d is None:
        return None
    dm = dynamic_descriptor(mirrored, role)
    return d[0], dm[0], d[1]
