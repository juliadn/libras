"""Build a fixture that checks the browser engine (docs/js/engine.js) gives the
same results as this Python code. Used by tools/web_parity.mjs:

  cd python
  ../.venv/bin/python -m tools.web_parity /tmp/parity.json
  node tools/web_parity.mjs /tmp/parity.json

The fixture holds, for every recorded sample:
  * its raw landmark frames,
  * the Python template descriptor (must match what the browser computes),
  * Python's label scores when the sample is used as a live attempt
    (full library and leave-one-out),
and a simulated live "stream" (samples glued together with hands-down pauses,
plus a fingerspelled J-U-L-I-A without pauses) with every prediction the
Python Session made on it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app.classifier import Query  # noqa: E402
from app.features import describe  # noqa: E402
from app.landmarks import ROOT, Frame  # noqa: E402
from app.library import Library  # noqa: E402
from app.session import Session  # noqa: E402

DT = 0.05  # 20 fps, like the Python page's frame pump


def frame_json(f: Frame) -> dict:
    def a(x):
        return None if x is None else np.asarray(x, dtype=np.float64).tolist()
    return {"t": f.t, "pose": a(f.pose), "left": a(f.left), "right": a(f.right),
            "face": a(f.face), "aspect": f.aspect}


def scores_json(s: dict) -> dict:
    return {k: float(v) for k, v in s.items()}


class _NoTracker:
    def close(self):
        pass


def build_stream(lib: Library) -> list[Frame]:
    """Samples one after another as a continuous camera stream."""
    order = []
    for label in ("ARROZ", "FEIJÃO", "NOME", "SINAL", "CASA", "A", "L"):
        order += [(lib.signs[label]["samples"][0], True)]
    # fingerspelling without lowering the hands between letters
    for letter in "JULIA":
        order += [(lib.signs[letter]["samples"][0], False)]
    order += [(lib.signs["NOME"]["samples"][1], True)]

    out: list[Frame] = []
    t = 1.0

    def rest(template: Frame, n: int):
        nonlocal t
        for _ in range(n):
            out.append(Frame(t=t, pose=template.pose, left=None, right=None,
                             face=template.face, aspect=template.aspect))
            t += DT

    for sid, pause in order:
        frames = lib.load_frames(sid)
        if pause:
            rest(frames[0], 12)
        t0 = frames[0].t
        for f in frames:
            out.append(Frame(t=t + (f.t - t0), pose=f.pose, left=f.left, right=f.right,
                             face=f.face, aspect=f.aspect))
        t = out[-1].t + DT
        if not pause:     # letters: hold the last handshape a bit (like a real signer)
            last = out[-1]
            for _ in range(6):
                out.append(Frame(t=t, pose=last.pose, left=last.left, right=last.right,
                                 face=last.face, aspect=last.aspect))
                t += DT
    rest(out[-1], 12)
    return out


def main(path: str):
    lib = Library(ROOT / "data")
    clf = lib.classifier
    samples = []
    for label, info in lib.signs.items():
        for sid in info["samples"]:
            frames = lib.load_frames(sid)
            q = Query(static=describe(frames, "static"),
                      dynamic=describe(frames, "dynamic", role="query"))
            tmpl = describe(frames, info["kind"])
            samples.append({
                "id": sid, "label": label, "kind": info["kind"],
                "frames": [frame_json(f) for f in frames],
                "template0": np.asarray(tmpl[0][0]).tolist(),
                "query_static": None if q.static is None else np.asarray(q.static[0][0]).tolist(),
                "query_dynamic_pen": None if q.dynamic is None else q.dynamic[2].tolist(),
                "scores": scores_json(clf.label_scores(q)),
                "scores_loo": scores_json(clf.label_scores(q, exclude_ids={sid})),
            })

    stream = build_stream(lib)
    runs = {}
    for spelling in (False, True):
        sess = Session(lib, tracker_factory=_NoTracker)
        sess.settings["spelling"] = spelling
        preds = []
        for i, f in enumerate(stream):
            for m in sess.process_frame(f):
                if m.get("type") == "prediction":
                    preds.append({"i": i, "label": m["label"], "confidence": m["confidence"],
                                  "recognized": m["recognized"], "reason": m["reason"],
                                  "replaces_previous": bool(m.get("replaces_previous"))})
        runs["spelling" if spelling else "words"] = preds
    Path(path).write_text(json.dumps({"samples": samples,
                                      "stream": [frame_json(f) for f in stream],
                                      "stream_predictions": runs}))
    print(f"{len(samples)} samples, stream of {len(stream)} frames -> {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "parity.json")
