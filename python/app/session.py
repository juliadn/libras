"""One browser connection: camera frames in, landmarks + predictions out.

Handles the two modes:
  * recognize: segmenter -> classifier -> prediction
  * teach: countdown -> record a take -> keep/discard -> save to the library
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .classifier import Example, Query, query_from_segment, query_from_static_take
from .features import PRESENT_LINE, body_ref, describe, hand_in_body, wrist_height
from .landmarks import Frame, HolisticTracker, decode_jpeg
from .library import Library, clean_label
from .segmenter import REST_LINE, Segmenter

log = logging.getLogger(__name__)

COUNTDOWN = 3.0          # seconds before each take
STATIC_TAKE = 1.0        # seconds recorded for a static sign
WAIT_FOR_MOVEMENT = 4.0  # dynamic take: give up if nothing starts by then
TARGET_TAKES = 5
REPLACE_WINDOW = 1.0     # a movement sign starting this soon after a held sign
                         # (hands still up) replaces it: NOME starts with the U
                         # handshape, so a short pause in U first shows "U".


def empty_landmarks() -> dict:
    return {"type": "landmarks", "hands": {"left": None, "right": None}, "pose": None,
            "face": None, "up": {"left": False, "right": False}, "rest_y": None,
            "fps": 0, "ms": 0}


@dataclass
class Teach:
    label: str
    kind: str
    kept: list[list[Frame]] = field(default_factory=list)
    phase: str = "ready"          # ready | countdown | recording | review
    phase_t: float | None = None
    frames: list[Frame] = field(default_factory=list)
    take: list[Frame] | None = None
    take_query: Query | None = None
    warnings: list[str] = field(default_factory=list)
    can_keep: bool = False
    segmenter: Segmenter | None = None
    auto_next: bool = True
    _extra: list[Example] = field(default_factory=list)   # kept takes as examples


class Session:
    def __init__(self, library: Library, tracker_factory=HolisticTracker):
        self.lib = library
        self.tracker = tracker_factory()
        self.segmenter = Segmenter()
        self.settings = {"threshold": 0.5, "mirror": True, "spelling": False,
                         "rest_line": REST_LINE, "paused": False}
        self.teach: Teach | None = None
        self.prompt: str | None = None
        self._prev: dict | None = None    # last recognized sign while hands stay up
        self._state: str | None = None
        self._fps = 0.0
        self._last_frame_t: float | None = None
        self._last_frame: Frame | None = None
        self._last_ref = None

    def close(self):
        try:
            self.tracker.close()
        except Exception:   # pragma: no cover - best effort
            pass

    # ------------------------------------------------------------------ dispatch

    def handle(self, msg: dict) -> list[dict]:
        kind = msg.get("type")
        try:
            if kind == "frame":
                return self.on_frame(msg)
            if kind == "settings":
                return self.on_settings(msg)
            if kind == "teach_start":
                return self.teach_start(msg.get("label", ""), msg.get("kind", "static"))
            if kind == "teach_record":
                return self.teach_record()
            if kind == "teach_keep":
                return self.teach_keep()
            if kind == "teach_discard":
                return self.teach_discard()
            if kind == "teach_save":
                return self.teach_save()
            if kind == "teach_cancel":
                return self.teach_cancel()
            if kind == "ping":
                return [{"type": "pong"}]
        except ValueError as e:
            return [{"type": "warning", "message": str(e)}]
        return [{"type": "warning", "message": f"Unknown message type: {kind}"}]

    # ------------------------------------------------------------------ frames

    def on_frame(self, msg: dict) -> list[dict]:
        t = time.monotonic()
        try:
            rgb = decode_jpeg(msg["jpeg"])
        except (KeyError, TypeError, ValueError):
            # still answer with (empty) landmarks so the browser keeps sending frames
            return [{"type": "error", "message": "Could not decode camera frame"},
                    empty_landmarks()]
        t0 = time.perf_counter()
        frame = self.tracker.process(rgb, t)
        ms = (time.perf_counter() - t0) * 1000
        return self.process_frame(frame, ms)

    def process_frame(self, frame: Frame, ms: float = 0.0) -> list[dict]:
        t = frame.t
        if self._last_frame_t is not None and t > self._last_frame_t:
            inst = 1.0 / (t - self._last_frame_t)
            self._fps = inst if self._fps == 0 else 0.85 * self._fps + 0.15 * inst
        self._last_frame_t = t
        self._last_frame = frame

        out: list[dict] = [self._landmarks_msg(frame, ms)]
        if self.teach is not None:
            out += self._teach_step(frame)
        elif not self.settings["paused"]:
            seg = self.segmenter.update(frame)
            if seg is not None:
                out.append(self._recognize(seg))
            if self.segmenter.state == "idle":
                self._prev = None            # hands went down: next sign is a new one

        state = self._public_state()
        if state != self._state:
            self._state = state
            out.append({"type": "state", "value": state})
        return out

    def _public_state(self) -> str:
        if self.teach is not None:
            if self.teach.phase == "countdown":
                return "countdown"
            if self.teach.phase == "recording":
                return "recording"
            return "idle"
        if self.settings["paused"]:
            return "paused"
        return self.segmenter.public_state

    def _landmarks_msg(self, frame: Frame, ms: float) -> dict:
        msg = frame.to_json()
        ref = body_ref(frame)
        if ref is not None:
            self._last_ref = ref
        ref = ref or self._last_ref
        up = {"left": False, "right": False}
        rest_y = None
        if ref is not None:
            line = self.settings["rest_line"]
            rest_y = float(ref.center[1] + line * ref.scale)
            for side in up:
                h = getattr(frame, side)
                up[side] = bool(h is not None and wrist_height(h, ref) < line)
        msg.update({
            "type": "landmarks",
            "up": up,
            "rest_y": None if rest_y is None else round(rest_y, 4),
            "fps": round(self._fps, 1),
            "ms": round(ms, 1),
        })
        return msg

    # ------------------------------------------------------------------ recognize

    def _allowed(self) -> set[str] | None:
        if not self.settings["spelling"]:
            return None
        letters = {lbl for lbl in self.lib.classifier.labels if len(lbl) == 1}
        return letters or None

    def _recognize(self, seg) -> dict:
        clf = self.lib.classifier
        if not clf.examples:
            msg = {"type": "prediction", "label": None, "confidence": 0.0,
                   "alternatives": [], "recognized": False, "empty": True}
        else:
            q = query_from_segment(seg)
            pred = clf.predict(q, threshold=self.settings["threshold"],
                               mirror=self.settings["mirror"], allowed=self._allowed())
            msg = pred.to_json()
        msg["reason"] = seg.reason
        msg["duration"] = round(seg.duration, 2)

        # A movement sign right after a held sign (hands never went down) is
        # usually the same sign: replace the held one instead of adding to it.
        kind = self.lib.kind_of(msg["label"]) if msg.get("label") else None
        prev = self._prev
        if (msg["recognized"] and kind == "dynamic" and prev is not None
                and seg.frames[0].t - prev["t"] <= REPLACE_WINDOW
                and (prev["kind"] == "static" or prev["label"] == msg["label"])):
            msg["replaces_previous"] = True
        self._prev = ({"label": msg["label"], "kind": kind, "t": seg.frames[-1].t}
                      if msg["recognized"] else None)
        if self.prompt:
            msg["prompt"] = self.prompt
            msg["prompt_ok"] = bool(msg["recognized"] and msg["label"] == self.prompt)
            if msg["prompt_ok"]:
                self.prompt = None
        return msg

    # ------------------------------------------------------------------ settings

    def on_settings(self, msg: dict) -> list[dict]:
        s = self.settings
        if "threshold" in msg:
            s["threshold"] = float(np.clip(float(msg["threshold"]), 0.05, 0.99))
        if "mirror" in msg:
            s["mirror"] = bool(msg["mirror"])
        if "spelling" in msg:
            s["spelling"] = bool(msg["spelling"])
        if "paused" in msg and bool(msg["paused"]) != s["paused"]:
            s["paused"] = bool(msg["paused"])
            self.segmenter.reset()
        if "rest_line" in msg:
            s["rest_line"] = float(np.clip(float(msg["rest_line"]), 0.5, PRESENT_LINE))
            self.segmenter.rest_line = s["rest_line"]
        if "prompt" in msg:
            self.prompt = msg["prompt"] or None
        return [{"type": "settings", **s}]

    # ------------------------------------------------------------------ teach

    def _teach_msg(self, **extra) -> dict:
        tch = self.teach
        if tch is None:
            return {"type": "teach", "phase": "off", **extra}
        return {
            "type": "teach", "phase": tch.phase, "label": tch.label, "kind": tch.kind,
            "count": len(tch.kept), "target": TARGET_TAKES,
            "warnings": tch.warnings if tch.phase == "review" else [],
            "can_keep": tch.can_keep, **extra,
        }

    def teach_start(self, label: str, kind: str) -> list[dict]:
        label = clean_label(label)
        if kind not in ("static", "dynamic"):
            raise ValueError("kind must be 'static' or 'dynamic'")
        out = []
        existing = self.lib.kind_of(label)
        if existing and existing != kind:
            out.append({"type": "warning", "message":
                        f"'{label}' already exists as a "
                        f"{'movement' if existing == 'dynamic' else 'static'} sign; "
                        f"new takes will be added to it."})
            kind = existing
        self.teach = Teach(label=label, kind=kind)
        self.prompt = None
        return out + [self._teach_msg(existing=len(self.lib.signs.get(label, {}).get("samples", [])))]

    def _require_teach(self) -> Teach:
        if self.teach is None:
            raise ValueError("Not in teach mode")
        return self.teach

    def teach_record(self) -> list[dict]:
        tch = self._require_teach()
        if tch.phase in ("countdown", "recording"):
            return []
        self._start_countdown(tch)
        return [self._teach_msg(remaining=COUNTDOWN)]

    def _start_countdown(self, tch: Teach):
        tch.phase = "countdown"
        tch.phase_t = None          # starts on the next camera frame
        tch.take, tch.take_query, tch.warnings, tch.can_keep = None, None, [], False

    def teach_keep(self) -> list[dict]:
        tch = self._require_teach()
        if tch.phase != "review" or not tch.can_keep or tch.take is None:
            return []
        tch.kept.append(tch.take)
        d = describe(tch.take, tch.kind)
        if d is not None:
            tch._extra.append(Example(f"_take{len(tch.kept)}", tch.label, tch.kind, *d))
        tch.take = None
        tch.phase = "ready"
        if tch.auto_next and len(tch.kept) < TARGET_TAKES:
            self._start_countdown(tch)
            return [self._teach_msg(remaining=COUNTDOWN)]
        return [self._teach_msg()]

    def teach_discard(self) -> list[dict]:
        tch = self._require_teach()
        if tch.phase in ("review", "countdown", "recording"):
            tch.take, tch.warnings, tch.can_keep = None, [], False
            tch.phase = "ready"
        return [self._teach_msg()]

    def teach_save(self) -> list[dict]:
        tch = self._require_teach()
        if not tch.kept:
            raise ValueError("Record and keep at least one take first")
        ids = self.lib.add_samples(tch.label, tch.kind, tch.kept)
        label = tch.label
        self.teach = None
        self.prompt = label
        self.segmenter.reset()
        return [
            {"type": "teach", "phase": "saved", "label": label, "count": len(ids)},
            {"type": "library_changed"},
        ]

    def teach_cancel(self) -> list[dict]:
        self.teach = None
        self.segmenter.reset()
        return [self._teach_msg()]

    def _teach_step(self, frame: Frame) -> list[dict]:
        tch = self.teach
        t = frame.t
        if tch.phase == "countdown":
            if tch.phase_t is None:
                tch.phase_t = t
            remaining = COUNTDOWN - (t - tch.phase_t)
            if remaining > 0:
                return [self._teach_msg(remaining=round(remaining, 2))]
            tch.phase, tch.phase_t, tch.frames = "recording", t, []
            if tch.kind == "dynamic":
                tch.segmenter = Segmenter(self.settings["rest_line"])
                tch.segmenter.prime(frame)
            else:
                tch.frames.append(frame)
            return [self._teach_msg(progress=0.0)]

        if tch.phase != "recording":
            return []

        if tch.kind == "static":
            tch.frames.append(frame)
            elapsed = t - tch.phase_t
            if elapsed >= STATIC_TAKE:
                return self._finish_take(tch.frames, query_from_static_take(tch.frames))
            return [self._teach_msg(progress=round(elapsed / STATIC_TAKE, 2))]

        seg = tch.segmenter.update(frame)
        if seg is not None:
            return self._finish_take(seg.frames, query_from_segment(seg))
        if tch.segmenter.state == "signing":
            tch.frames.append(frame)
            return [self._teach_msg(progress=round(min(1.0, (t - tch.frames[0].t) / 3.0), 2))]
        if t - tch.phase_t > WAIT_FOR_MOVEMENT:
            tch.phase = "ready"
            return [self._teach_msg(),
                    {"type": "warning", "message": "No movement detected. Raise your hands "
                     "above the line and sign right after 'Go!'"}]
        tch.frames = []
        return [self._teach_msg(progress=0.0)]

    def _finish_take(self, frames: list[Frame], query: Query) -> list[dict]:
        tch = self.teach
        tch.take, tch.take_query = frames, query
        tch.phase = "review"
        warnings = []

        # 1. was a hand visible (and raised) for most of the take?
        ref = None
        up_count = 0
        pts = []
        for f in frames:
            ref = body_ref(f) or ref
            hands = [h for h in (f.left, f.right)
                     if h is not None and ref is not None and wrist_height(h, ref) < PRESENT_LINE]
            if hands:
                up_count += 1
                if tch.kind == "static" and len(hands) == 1:
                    pts.append(hand_in_body(hands[0], f.aspect, ref))
        if not frames or up_count < 0.5 * len(frames):
            warnings.append("Hand not visible")

        # 2. static signs should be held still
        if tch.kind == "static" and len(pts) >= 3:
            P = np.stack(pts)
            wobble = float(np.linalg.norm(P - P.mean(0), axis=-1).max())
            if wobble > 0.25:
                warnings.append("You moved during the take. Hold the handshape still.")

        usable = describe(frames, tch.kind) is not None
        if not usable:
            warnings.append("Body not visible: step back so your shoulders are in the frame")

        # 3. does it look like another sign?
        if usable:
            clf = self.lib.classifier.with_extra(tch._extra)
            if clf.examples:
                pred = clf.predict(query, threshold=self.settings["threshold"],
                                   mirror=self.settings["mirror"])
                if pred.recognized and pred.label != tch.label:
                    warnings.append(f"This looks like {pred.label}, are you sure?")

        tch.warnings = warnings
        tch.can_keep = usable
        return [self._teach_msg(duration=round(frames[-1].t - frames[0].t, 2) if frames else 0)]
