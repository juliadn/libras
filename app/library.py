"""The sign library on disk: data/signs.json + data/samples/<id>.npz.

Each sample stores the raw landmarks of one recorded take (not the features),
so the feature code can change later without re-recording anything.
"""
from __future__ import annotations

import io
import json
import logging
import re
import secrets
import shutil
import threading
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from .classifier import Classifier, Example
from .features import describe
from .landmarks import Frame, arrays_to_frames, frames_to_arrays

log = logging.getLogger(__name__)

KINDS = ("static", "dynamic")
MAX_LABEL = 40
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,120}$")


def clean_label(label: str) -> str:
    """'  obrigado ' -> 'OBRIGADO'. Letters, digits, spaces, - and _ only."""
    label = unicodedata.normalize("NFC", str(label)).strip().upper()
    label = re.sub(r"[^\w \-]", "", label)
    label = re.sub(r"\s+", " ", label)[:MAX_LABEL].strip()
    if not label:
        raise ValueError("Sign name is empty")
    return label


def _slug(label: str) -> str:
    s = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return s or "sign"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Library:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.samples_dir = self.data_dir / "samples"
        self.index_path = self.data_dir / "signs.json"
        self._lock = threading.RLock()
        self.signs: dict[str, dict] = {}
        self._examples: dict[str, Example] = {}      # sample_id -> descriptors
        self.classifier = Classifier([])
        self.load()

    # -- persistence -------------------------------------------------------------

    def load(self):
        with self._lock:
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            if self.index_path.exists():
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.signs = data.get("signs", {})
            else:
                self.signs = {}
            self._examples = {}
            dropped = 0
            for label, info in list(self.signs.items()):
                keep = []
                for sid in info.get("samples", []):
                    if not _ID_RE.match(str(sid)) or not self._sample_path(sid).exists():
                        continue
                    keep.append(sid)
                    ex = self._load_example(sid, label, info["kind"])
                    if ex is not None:
                        self._examples[sid] = ex
                dropped += len(info.get("samples", [])) - len(keep)
                info["samples"] = keep
                if not keep:
                    del self.signs[label]
            if dropped:     # repair the index (e.g. sample files deleted by hand)
                log.warning("removed %d missing example(s) from %s", dropped, self.index_path)
                self._save_index()
            self._rebuild()

    def _save_index(self):
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": 1, "signs": self.signs}, indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.index_path)

    def _sample_path(self, sid: str) -> Path:
        if not _ID_RE.match(sid):
            raise ValueError("bad sample id")
        return self.samples_dir / f"{sid}.npz"

    def load_frames(self, sid: str) -> list[Frame]:
        with np.load(self._sample_path(sid), allow_pickle=False) as d:
            return arrays_to_frames(d)

    def _load_example(self, sid: str, label: str, kind: str) -> Example | None:
        try:
            frames = self.load_frames(sid)
        except (OSError, ValueError, KeyError) as e:
            log.warning("skipping sample %s: %s", sid, e)
            return None
        return self._make_example(sid, label, kind, frames)

    @staticmethod
    def _make_example(sid, label, kind, frames) -> Example | None:
        d = describe(frames, kind)
        if d is None:
            return None
        return Example(sid, label, kind, *d)

    def _rebuild(self):
        self.classifier = Classifier(list(self._examples.values()))
        self._flags = self.classifier.flags()

    # -- queries ---------------------------------------------------------------

    def kind_of(self, label: str) -> str | None:
        info = self.signs.get(label)
        return info["kind"] if info else None

    def summary(self) -> list[dict]:
        with self._lock:
            out = []
            for label in sorted(self.signs, key=lambda s: (len(s) > 1, s)):
                info = self.signs[label]
                out.append({
                    "label": label,
                    "kind": info["kind"],
                    "created": info.get("created"),
                    "count": len(info["samples"]),
                    "samples": [{"id": sid, "looks_like": self._flags.get(sid)}
                                for sid in info["samples"]],
                })
            return out

    def sample_json(self, sid: str) -> dict:
        frames = self.load_frames(sid)
        ex = self._examples.get(sid)
        return {
            "id": sid,
            "label": ex.label if ex else None,
            "kind": ex.kind if ex else None,
            "aspect": frames[0].aspect if frames else 4 / 3,
            "frames": [f.to_json() for f in frames],
        }

    # -- changes ---------------------------------------------------------------

    def add_samples(self, label: str, kind: str, takes: list[list[Frame]],
                    source: str = "teach") -> list[str]:
        label = clean_label(label)
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        with self._lock:
            existing = self.kind_of(label)
            if existing and existing != kind:
                raise ValueError(f"'{label}' already exists as a {existing} sign")
            info = self.signs.setdefault(label, {"kind": kind, "created": _now(), "samples": []})
            ids = []
            for frames in takes:
                ex = None
                sid = f"{_slug(label)}_{datetime.now():%Y%m%d-%H%M%S}_{secrets.token_hex(3)}"
                ex = self._make_example(sid, label, kind, frames)
                if ex is None:
                    log.warning("take for %s has no usable landmarks; skipped", label)
                    continue
                arrays = frames_to_arrays(frames)
                np.savez_compressed(self._sample_path(sid), label=np.array(label),
                                    kind=np.array(kind), source=np.array(source), **arrays)
                self._examples[sid] = ex
                info["samples"].append(sid)
                ids.append(sid)
            if not info["samples"]:
                del self.signs[label]
            self._save_index()
            self._rebuild()
            return ids

    def delete_sign(self, label: str) -> bool:
        with self._lock:
            info = self.signs.pop(label, None)
            if info is None:
                return False
            for sid in info["samples"]:
                self._examples.pop(sid, None)
                self._sample_path(sid).unlink(missing_ok=True)
            self._save_index()
            self._rebuild()
            return True

    def delete_sample(self, sid: str) -> bool:
        with self._lock:
            for label, info in list(self.signs.items()):
                if sid in info["samples"]:
                    info["samples"].remove(sid)
                    self._examples.pop(sid, None)
                    self._sample_path(sid).unlink(missing_ok=True)
                    if not info["samples"]:
                        del self.signs[label]
                    self._save_index()
                    self._rebuild()
                    return True
            return False

    # -- export / import ------------------------------------------------------------

    def export_zip(self) -> bytes:
        with self._lock:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("signs.json", json.dumps({"version": 1, "signs": self.signs},
                                                    indent=2, ensure_ascii=False))
                for info in self.signs.values():
                    for sid in info["samples"]:
                        z.write(self._sample_path(sid), f"samples/{sid}.npz")
            return buf.getvalue()

    def import_zip(self, data: bytes) -> dict:
        """Merge a library zip into this one. Existing samples are kept."""
        try:
            z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile as e:
            raise ValueError("Not a zip file") from e
        with z, self._lock:
            try:
                incoming = json.loads(z.read("signs.json").decode("utf-8")).get("signs", {})
            except KeyError as e:
                raise ValueError("signs.json missing from the zip") from e
            names = set(z.namelist())
            added, skipped, conflicts = 0, 0, []
            for raw_label, info in incoming.items():
                try:
                    label = clean_label(raw_label)
                except ValueError:
                    continue
                kind = info.get("kind")
                if kind not in KINDS:
                    continue
                existing = self.kind_of(label)
                if existing and existing != kind:
                    conflicts.append(label)
                    continue
                for sid in info.get("samples", []):
                    member = f"samples/{sid}.npz"
                    if not _ID_RE.match(str(sid)) or member not in names:
                        skipped += 1
                        continue
                    if sid in self._examples:
                        skipped += 1          # already have it
                        continue
                    dest = self._sample_path(sid)
                    with z.open(member) as src, open(dest, "wb") as out:
                        shutil.copyfileobj(src, out)
                    ex = self._load_example(sid, label, kind)
                    if ex is None:
                        dest.unlink(missing_ok=True)
                        skipped += 1
                        continue
                    entry = self.signs.setdefault(label, {"kind": kind, "created": info.get("created") or _now(), "samples": []})
                    entry["samples"].append(sid)
                    self._examples[sid] = ex
                    added += 1
            self._save_index()
            self._rebuild()
            return {"added": added, "skipped": skipped, "conflicts": conflicts}
