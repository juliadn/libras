"""MediaPipe wrapper: turns one camera frame into a `Frame` of landmarks.

All coordinates are kept exactly as MediaPipe returns them (normalized to the
image: x and y in 0..1). `features.py` takes care of making them comparable.
"""
from __future__ import annotations

import base64
import contextlib
import logging
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "holistic_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

# --- Which points we keep -------------------------------------------------

# Upper body points of MediaPipe Pose: nose, ears, shoulders, elbows, wrists, hips.
POSE_DRAW_IDX = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24]

# Mirror permutation for the full 33-point pose (left <-> right body parts).
_POSE_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
               (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
               (27, 28), (29, 30), (31, 32)]
POSE_MIRROR = list(range(33))
for _a, _b in _POSE_PAIRS:
    POSE_MIRROR[_a], POSE_MIRROR[_b] = _b, _a

# ~40 face points: eyebrows, eyes, mouth, nose tip, chin (Face Mesh indices).
# Each pair is (one side, the symmetric point on the other side).
_FACE_PAIRS = [
    (70, 300), (63, 293), (105, 334), (66, 296), (107, 336),             # eyebrows
    (33, 263), (160, 387), (158, 385), (133, 362), (153, 380), (144, 373),  # eyes
    (61, 291), (40, 270), (37, 267), (91, 321), (84, 314), (78, 308),    # mouth
]
_FACE_CENTER = [1, 0, 17, 13, 14, 152]  # nose tip, lips (outer/inner), chin
FACE_IDX = [i for pair in _FACE_PAIRS for i in pair] + _FACE_CENTER
FACE_NOSE = FACE_IDX.index(1)
_pos = {idx: k for k, idx in enumerate(FACE_IDX)}
FACE_MIRROR = list(range(len(FACE_IDX)))
for _a, _b in _FACE_PAIRS:
    FACE_MIRROR[_pos[_a]], FACE_MIRROR[_pos[_b]] = _pos[_b], _pos[_a]


@dataclass
class Frame:
    """Landmarks of one video frame (single person)."""

    t: float                        # seconds (monotonic clock)
    pose: np.ndarray | None         # (33, 4): x, y, z, visibility
    left: np.ndarray | None         # (21, 3) the person's LEFT hand
    right: np.ndarray | None        # (21, 3) the person's RIGHT hand
    face: np.ndarray | None         # (40, 3) subset given by FACE_IDX
    aspect: float                   # image width / height

    def to_json(self) -> dict:
        """Compact version sent to the browser for drawing (x, y only)."""

        def pts(a, cols=2):
            if a is None:
                return None
            return np.round(a[:, :cols], 4).tolist()

        return {
            "hands": {"left": pts(self.left), "right": pts(self.right)},
            "pose": pts(self.pose[POSE_DRAW_IDX], 4)
            if self.pose is not None else None,
            "face": pts(self.face),
        }


def mirror_frame(f: Frame) -> Frame:
    """The same frame as if it was signed with the other hand (left <-> right)."""

    def flip(a):
        if a is None:
            return None
        a = a.copy()
        a[:, 0] = 1.0 - a[:, 0]
        return a

    pose = flip(f.pose)
    if pose is not None:
        pose = pose[POSE_MIRROR]
    face = flip(f.face)
    if face is not None:
        face = face[FACE_MIRROR]
    return Frame(t=f.t, pose=pose, left=flip(f.right), right=flip(f.left),
                 face=face, aspect=f.aspect)


# --- Storage helpers (used by library.py) ----------------------------------

def frames_to_arrays(frames: list[Frame]) -> dict[str, np.ndarray]:
    """Pack frames into fixed-shape arrays (NaN where something was missing)."""
    n = len(frames)

    def stack(attr, shape):
        out = np.full((n, *shape), np.nan, dtype=np.float32)
        for i, f in enumerate(frames):
            v = getattr(f, attr)
            if v is not None:
                out[i] = v
        return out

    return {
        "t": np.array([f.t for f in frames], dtype=np.float64),
        "pose": stack("pose", (33, 4)),
        "left": stack("left", (21, 3)),
        "right": stack("right", (21, 3)),
        "face": stack("face", (len(FACE_IDX), 3)),
        "aspect": np.array([f.aspect for f in frames], dtype=np.float32),
    }


def arrays_to_frames(d) -> list[Frame]:
    def get(a, i):
        v = a[i]
        return None if np.isnan(v).any() else np.asarray(v, dtype=np.float64)

    t = d["t"]
    return [
        Frame(t=float(t[i]), pose=get(d["pose"], i), left=get(d["left"], i),
              right=get(d["right"], i), face=get(d["face"], i),
              aspect=float(d["aspect"][i]))
        for i in range(len(t))
    ]


# --- Model + tracker --------------------------------------------------------

def ensure_model(path: Path = MODEL_PATH) -> Path:
    """Download the MediaPipe holistic model the first time (13 MB)."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.warning("Downloading MediaPipe holistic model to %s ...", path)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(path)
    return path


def decode_jpeg(data: str | bytes) -> np.ndarray:
    """Base64 JPEG (optionally a data: URL) -> RGB uint8 image."""
    if isinstance(data, str):
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        data = base64.b64decode(data)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _to_array(landmarks, with_visibility=False) -> np.ndarray | None:
    if not landmarks:
        return None
    if with_visibility:
        return np.array([[p.x, p.y, p.z, p.visibility if p.visibility is not None else 1.0]
                         for p in landmarks], dtype=np.float64)
    return np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64)


@contextlib.contextmanager
def _quiet_native_logs():
    """Hide MediaPipe's C++ start-up messages ("W0000 ... Feedback manager ...").

    They are harmless but look like errors. They are written straight to the
    process's stderr, so Python's logging can't filter them: we point file
    descriptor 2 at /dev/null for a moment. Set LIBRAS_VERBOSE=1 to see them.
    """
    if os.environ.get("LIBRAS_VERBOSE"):
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


class HolisticTracker:
    """MediaPipe Holistic in VIDEO mode (uses tracking between frames).

    Not thread-safe: use one tracker per camera stream, called from one thread.
    """

    def __init__(self, model_path: Path = MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        ensure_model(model_path)
        options = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_landmarks_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
            min_face_detection_confidence=0.5,
            min_face_landmarks_confidence=0.5,
        )
        with _quiet_native_logs():
            self._landmarker = vision.HolisticLandmarker.create_from_options(options)
            # warm-up frame: the first inference prints the rest of the noise
            blank = np.zeros((64, 64, 3), np.uint8)
            self._landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=blank), 0)
        self._last_ts = 0

    def process(self, rgb: np.ndarray, t: float) -> Frame:
        ts = int(t * 1000)
        if ts <= self._last_ts:          # timestamps must strictly increase
            ts = self._last_ts + 1
        self._last_ts = ts
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(rgb))
        r = self._landmarker.detect_for_video(image, ts)
        face = _to_array(r.face_landmarks)
        if face is not None:
            face = face[FACE_IDX]
        h, w = rgb.shape[:2]
        return Frame(t=t, pose=_to_array(r.pose_landmarks, with_visibility=True),
                     left=_to_array(r.left_hand_landmarks),
                     right=_to_array(r.right_hand_landmarks),
                     face=face, aspect=w / h)

    def close(self):
        self._landmarker.close()
