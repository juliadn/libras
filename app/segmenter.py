"""Decides when a sign starts and ends.

States:
  idle     no hand raised.
  signing  a sign is happening; frames are being collected.
  held     a sign just ended with the hands held still. Waiting for a real
           movement (a new letter in fingerspelling) before starting again, so
           holding a letter for a long time only counts once.
  wait     a sign hit the 3 s limit; waiting for a pause or for hands down.

A sign
  STARTS when a hand rises above the "rest line" (or, from `held`, moves).
  ENDS   when the hands go back down, or stay still for 0.4 s (a held
         handshape), or after 3 s.
  Very short movements (< 0.25 s) are ignored.

The same segmenter is used when teaching a movement sign and when recognizing,
so the saved examples and the live attempts are cut the same way.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .features import Ref, body_ref, hand_in_body, wrist_height
from .landmarks import Frame

REST_LINE = 1.2       # shoulder widths below the shoulders; above it = hand "up"
HOLD_TIME = 0.4       # still this long -> the sign ended (held handshape)
MAX_DURATION = 3.0    # longest sign
MIN_DURATION = 0.25   # shorter movements are ignored
STILL_DIST = 0.08     # max point wobble (shoulder widths) still counted as "still"
MOVE_DIST = 0.15      # how far a held hand must move to start a new sign
DOWN_TIME = 0.12      # hands must be down this long to end a sign
PREROLL = 0.15        # seconds of frames kept before a movement start


@dataclass
class Segment:
    frames: list[Frame]
    reason: str                 # "hold" | "down" | "timeout"
    still_from: int | None      # first frame of a still part (the held handshape)
    still_to: int | None        # one past its last frame

    @property
    def duration(self) -> float:
        return self.frames[-1].t - self.frames[0].t if self.frames else 0.0

    @property
    def still_frames(self) -> list[Frame] | None:
        if self.still_from is None:
            return None
        return self.frames[self.still_from:self.still_to]


@dataclass
class _Entry:
    frame: Frame
    hands: dict[str, np.ndarray]   # up hands only: side -> (21, 2) body coords


def _robust_max(dist: np.ndarray) -> np.ndarray:
    """3rd largest value over the 21 points (ignores 1-2 jittery landmarks,
    but still sees a finger or thumb that really moved)."""
    return np.sort(dist, axis=-1)[..., -3]


def _wobble(entries: list[_Entry]) -> float:
    """How far the hand points moved around their average position."""
    worst = 0.0
    for side in entries[0].hands:
        pts = np.stack([e.hands[side] for e in entries])
        dev = _robust_max(np.linalg.norm(pts - pts.mean(axis=0), axis=-1)).max()
        worst = max(worst, float(dev))
    return worst


def _same_hands(entries: list[_Entry]) -> bool:
    keys = set(entries[0].hands)
    return bool(keys) and all(set(e.hands) == keys for e in entries)


def find_still_window(entries: list[_Entry], length: float = 0.3) -> tuple[int, int] | None:
    """Stillest window (with hands up) of ~`length` seconds, if it is still enough."""
    best, best_w = None, None
    j = 0
    for i in range(len(entries)):
        while j < len(entries) and entries[j].frame.t - entries[i].frame.t < length:
            j += 1
        if j >= len(entries):
            break
        win = entries[i:j + 1]
        if len(win) < 3 or not _same_hands(win):
            continue
        w = _wobble(win)
        if best_w is None or w < best_w:
            best, best_w = (i, j + 1), w
    if best is None or best_w > STILL_DIST * 1.5:
        return None
    return best


class Segmenter:
    def __init__(self, rest_line: float = REST_LINE):
        self.rest_line = rest_line
        self.reset()

    def reset(self):
        self.state = "idle"
        self._recent: deque[_Entry] = deque()
        self._seg: list[_Entry] = []
        self._hold_ref: dict[str, np.ndarray] | None = None
        self._down_since: float | None = None
        self._moved_count = 0
        self._last_ref: Ref | None = None

    # -- helpers ---------------------------------------------------------------

    def _entry(self, frame: Frame) -> _Entry:
        ref = body_ref(frame)
        if ref is not None:
            self._last_ref = ref
        else:
            ref = self._last_ref
        hands = {}
        if ref is not None:
            for side in ("left", "right"):
                h = getattr(frame, side)
                if h is not None and wrist_height(h, ref) < self.rest_line:
                    hands[side] = hand_in_body(h, frame.aspect, ref)
        return _Entry(frame, hands)

    def _still_for(self, entries, hold_time: float) -> list[_Entry] | None:
        """The last `hold_time` seconds of entries, if the hands were still."""
        if not entries:
            return None
        now = entries[-1].frame.t
        win = [e for e in entries if e.frame.t >= now - hold_time]
        if len(win) < 3 or now - win[0].frame.t < hold_time * 0.8:
            return None
        if not _same_hands(win) or _wobble(win) > STILL_DIST:
            return None
        return win

    def _moved_from_hold(self, e: _Entry) -> bool:
        ref = self._hold_ref or {}
        if set(e.hands) != set(ref):
            return True
        return any(_robust_max(np.linalg.norm(e.hands[s] - ref[s], axis=-1)) > MOVE_DIST
                   for s in ref)

    def _set_held(self, window: list[_Entry]):
        self.state = "held"
        self._hold_ref = {s: np.mean([e.hands[s] for e in window], axis=0) for s in window[-1].hands}
        self._moved_count = 0

    def _emit(self, entries: list[_Entry], reason: str, still: tuple[int, int] | None) -> Segment:
        frames = [e.frame for e in entries]
        if still is None:
            still = find_still_window(entries)
        return Segment(frames, reason, *(still if still else (None, None)))

    def prime(self, frame: Frame):
        """Start fresh from the current pose (used before recording a take)."""
        self.reset()
        e = self._entry(frame)
        self._recent.append(e)
        if e.hands:
            self._set_held([e])

    @property
    def public_state(self) -> str:
        return "signing" if self.state == "signing" else "idle"

    # -- main step ------------------------------------------------------------------

    def update(self, frame: Frame) -> Segment | None:
        e = self._entry(frame)
        t = frame.t
        self._recent.append(e)
        while self._recent and self._recent[0].frame.t < t - 1.0:
            self._recent.popleft()

        if e.hands:
            self._down_since = None
        elif self._down_since is None:
            self._down_since = t
        down = self._down_since is not None and t - self._down_since >= DOWN_TIME - 1e-9

        if self.state == "idle":
            if e.hands:
                self.state = "signing"
                self._seg = [e]
            return None

        if self.state in ("held", "wait"):
            if down:
                self.state = "idle"
                return None
            if self.state == "held":
                if e.hands and self._moved_from_hold(e):
                    self._moved_count += 1
                else:
                    self._moved_count = 0
                if self._moved_count >= 2:    # two frames in a row: a real movement
                    self.state = "signing"
                    self._seg = [x for x in self._recent if x.frame.t >= t - PREROLL and x.hands]
                return None
            # wait: settle into "held" once the hands are still
            win = self._still_for(list(self._recent), HOLD_TIME)
            if win:
                self._set_held(win)
            return None

        # --- signing ---
        self._seg.append(e)
        start = self._seg[0].frame.t

        if down:
            up = [x for x in self._seg if x.hands]
            self.state = "idle"
            self._seg = []
            if not up or up[-1].frame.t - up[0].frame.t < MIN_DURATION:
                return None   # too short: ignore
            return self._emit(up, "down", None)

        if e.hands:
            win = self._still_for(self._seg, HOLD_TIME)
            if win:
                seg = self._seg
                self._seg = []
                self._set_held(win)
                return self._emit(seg, "hold", (len(seg) - len(win), len(seg)))

        if t - start >= MAX_DURATION:
            seg = [x for x in self._seg if x.hands]
            self._seg = []
            self.state = "wait"
            return self._emit(seg, "timeout", None) if seg else None
        return None
