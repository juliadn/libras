"""Synthetic landmark frames for tests (no camera or model needed).

A person facing the camera: shoulders at y=0.45, shoulder width 0.2 (x units).
Hands are made of a wrist + 5 fingers; each "handshape" is a tuple saying which
fingers are extended (thumb, index, middle, ring, pinky).
"""
from __future__ import annotations

import numpy as np

from app.landmarks import FACE_IDX, Frame

ASPECT = 4 / 3
SHOULDER_Y = 0.45
LEFT_SHOULDER_X, RIGHT_SHOULDER_X = 0.6, 0.4   # unmirrored camera: person's left is on the image right

SHAPES = {
    "A": (1, 0, 0, 0, 0),
    "B": (0, 1, 1, 1, 1),
    "I": (0, 0, 0, 0, 1),
    "L": (1, 1, 0, 0, 0),
    "V": (0, 1, 1, 0, 0),
    "Y": (1, 0, 0, 0, 1),
    "W": (0, 1, 1, 1, 0),
}


def pose(shift=(0.0, 0.0), scale=1.0) -> np.ndarray:
    p = np.zeros((33, 4))
    p[:, 3] = 1.0
    cx, cy = 0.5, SHOULDER_Y
    pts = {0: (0.5, 0.3), 7: (0.56, 0.3), 8: (0.44, 0.3),
           11: (LEFT_SHOULDER_X, SHOULDER_Y), 12: (RIGHT_SHOULDER_X, SHOULDER_Y),
           13: (0.65, 0.6), 14: (0.35, 0.6), 15: (0.66, 0.75), 16: (0.34, 0.75),
           23: (0.57, 0.85), 24: (0.43, 0.85)}
    for i in range(33):
        x, y = pts.get(i, (0.5, 0.6))
        p[i, 0] = cx + (x - cx) * scale + shift[0]
        p[i, 1] = cy + (y - cy) * scale + shift[1]
    return p


def face(shift=(0.0, 0.0), scale=1.0) -> np.ndarray:
    rng = np.random.default_rng(42)          # fixed face geometry
    f = np.zeros((len(FACE_IDX), 3))
    f[:, 0] = 0.5 + rng.normal(0, 0.03, len(FACE_IDX))
    f[:, 1] = 0.3 + rng.normal(0, 0.03, len(FACE_IDX))
    f[:, 0] = 0.5 + (f[:, 0] - 0.5) * scale + shift[0]
    f[:, 1] = SHOULDER_Y + (f[:, 1] - SHOULDER_Y) * scale + shift[1]
    return f


def hand(shape, wrist=(0.42, 0.4), size=0.06, angle=0.0, side="right") -> np.ndarray:
    """21 hand landmarks in normalized image coords. angle rotates the hand."""
    ext = SHAPES[shape] if isinstance(shape, str) else shape
    sign = -1 if side == "right" else 1      # mirror the finger fan for the other hand
    pts = [np.zeros(2)]
    base_angles = np.radians([-55, -20, 0, 18, 35])  # thumb .. pinky, 0 = up
    for k in range(5):
        a = base_angles[k] * sign + angle
        d = np.array([np.sin(a), -np.cos(a)])
        mcp = d * (0.6 if k == 0 else 1.0)
        pts.append(mcp)
        if ext[k]:
            for step in (0.35, 0.65, 0.9):
                pts.append(mcp + d * step)
        else:   # curled back toward the palm
            pts.append(mcp + d * 0.25)
            pts.append(mcp + d * 0.1 - np.array([0, -0.15]))
            pts.append(mcp - d * 0.15 - np.array([0, -0.2]))
    xy = np.stack(pts) * size
    xy[:, 0] /= ASPECT                       # keep the hand round in pixel space
    xy += np.array(wrist)
    z = np.zeros((21, 1))
    return np.hstack([xy, z])


def frame(t, right=None, left=None, shift=(0.0, 0.0), scale=1.0, rng=None, noise=0.0) -> Frame:
    """right/left: (shape, wrist_xy) or None."""
    def jitter(a):
        if a is None or rng is None or noise == 0:
            return a
        a = a.copy()
        a[:, :2] += rng.normal(0, noise, a[:, :2].shape)
        return a

    def mk(spec, side):
        if spec is None:
            return None
        shape, wrist = spec[0], spec[1]
        angle = spec[2] if len(spec) > 2 else 0.0
        h = hand(shape, wrist=wrist, angle=angle, side=side)
        cx, cy = 0.5, SHOULDER_Y
        h[:, 0] = cx + (h[:, 0] - cx) * scale + shift[0]
        h[:, 1] = cy + (h[:, 1] - cy) * scale + shift[1]
        return h

    return Frame(t=t, pose=jitter(pose(shift, scale)), left=jitter(mk(left, "left")),
                 right=jitter(mk(right, "right")), face=jitter(face(shift, scale)), aspect=ASPECT)


UP = (0.42, 0.4)        # right hand raised in front of the right shoulder
DOWN = (0.4, 0.95)      # right hand resting low


def lerp(a, b, u):
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


def static_take(shape, n=15, fps=15, t0=0.0, rng=None, noise=0.0, side="right", wrist=UP):
    out = []
    for i in range(n):
        spec = (shape, wrist)
        kw = {"right": spec} if side == "right" else {"left": spec}
        out.append(frame(t0 + i / fps, rng=rng, noise=noise, **kw))
    return out


def sign_sequence(shape, fps=15, t0=0.0, rise=0.4, hold=0.8, rng=None, noise=0.0,
                  path=None, path_time=0.0, lower=True, angle_path=None):
    """Hand rises from below, (optionally) moves along `path`, holds, goes down."""
    frames, t = [], t0
    dt = 1 / fps
    # hands down first
    for _ in range(3):
        frames.append(frame(t, rng=rng, noise=noise)); t += dt
    n = max(1, int(rise * fps))
    for i in range(1, n + 1):
        frames.append(frame(t, right=(shape, lerp((0.45, 0.85), UP, i / n)), rng=rng, noise=noise)); t += dt
    if path is not None:
        m = max(2, int(path_time * fps))
        for i in range(m):
            u = i / (m - 1)
            ang = angle_path(u) if angle_path else 0.0
            frames.append(frame(t, right=(shape, path(u), ang), rng=rng, noise=noise)); t += dt
        end = path(1.0)
        end_ang = angle_path(1.0) if angle_path else 0.0
    else:
        end, end_ang = UP, 0.0
    for _ in range(int(hold * fps)):
        frames.append(frame(t, right=(shape, end, end_ang), rng=rng, noise=noise)); t += dt
    if lower:
        for i in range(1, 4):
            frames.append(frame(t, right=(shape, lerp(end, (0.45, 0.9), i / 3), end_ang), rng=rng, noise=noise)); t += dt
        for _ in range(4):
            frames.append(frame(t, rng=rng, noise=noise)); t += dt
    return frames


def j_path(u):
    """A small 'J' hook: down then curving to the side."""
    if u < 0.6:
        return (0.42, 0.4 + 0.12 * (u / 0.6))
    v = (u - 0.6) / 0.4
    return (0.42 + 0.06 * v, 0.52 - 0.04 * v)
