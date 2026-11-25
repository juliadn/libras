"""Export the sign library for the static web app (docs/).

The browser does NOT recompute your saved examples: this script runs the exact
Python feature code on every recorded sample and writes the resulting
descriptors, so the website compares live attempts against byte-for-byte the
same templates the Python app uses.

Writes:
  docs/data/templates.json   labels, kinds, spreads, example index
  docs/data/templates.bin    float32 descriptors (little endian)

Run from the python/ folder:
  ../.venv/bin/python -m tools.export_web          (or: python -m tools.export_web)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from app import classifier as C  # noqa: E402
from app import features as F  # noqa: E402
from app import segmenter as S  # noqa: E402
from app.landmarks import ROOT  # noqa: E402
from app.library import Library  # noqa: E402

OUT = ROOT.parent / "docs" / "data"


def export(data_dir: Path = ROOT / "data", out_dir: Path = OUT) -> dict:
    lib = Library(data_dir)
    clf = lib.classifier
    out_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[np.ndarray] = []
    offset = 0
    groups = {}
    for kind, g in clf.groups.items():          # insertion order: static, dynamic
        dim = int(g.X.shape[1])
        examples = []
        for ex in g.examples:
            v = int(ex.desc.shape[0])
            block = np.concatenate([ex.desc, ex.desc_mirror]).astype("<f4")
            chunks.append(block.ravel())
            examples.append({"id": ex.sample_id, "label": ex.label, "variants": v,
                             "penalty": [float(p) for p in ex.penalty], "offset": offset})
            offset += block.size
        groups[kind] = {"dim": dim,
                        "spreads": {k: float(s) for k, s in g.label_spread.items()},
                        "examples": examples}

    meta = {
        "version": 1,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "kinds": {label: info["kind"] for label, info in lib.signs.items()},
        "groups": groups,
        "floats": offset,
        # constants the browser must agree with (checked at load time)
        "constants": {
            "N_FRAMES": F.N_FRAMES, "FRAME_DIM": F.FRAME_DIM,
            "K": C.K, "UNKNOWN_SCORE": C.UNKNOWN_SCORE, "N_RETRIEVE": C.N_RETRIEVE,
            "REST_LINE": S.REST_LINE, "HOLD_TIME": S.HOLD_TIME,
        },
    }
    (out_dir / "templates.bin").write_bytes(np.concatenate(chunks).tobytes())
    (out_dir / "templates.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    return meta


if __name__ == "__main__":
    m = export()
    n = sum(len(g["examples"]) for g in m["groups"].values())
    print(f"exported {len(m['kinds'])} signs, {n} examples, {m['floats'] * 4 / 1e6:.2f} MB -> {OUT}")
