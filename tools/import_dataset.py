"""Import sign videos (e.g. from MINDS-Libras or V-LIBRASIL) into the library.

Expected layout (one folder per sign, the folder name is the sign name):

    videos/
      OBRIGADO/  take1.mp4  take2.mp4 ...
      J/         a.mp4 ...

Usage (from the project folder, with the virtualenv active):

    python tools/import_dataset.py videos/ --kind dynamic
    python tools/import_dataset.py letters/ --kind static
    python tools/import_dataset.py videos/ --dry-run          # just show what would happen

Each video goes through the same pipeline as the live camera (MediaPipe ->
segmenter), so the imported examples match what the app sees when you sign.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.landmarks import Frame, HolisticTracker  # noqa: E402
from app.library import Library, clean_label  # noqa: E402
from app.segmenter import Segmenter, find_still_window  # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def read_frames(path: Path, max_width: int = 640) -> list[Frame]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / 15))           # ~15 fps, like the live app
    tracker = HolisticTracker()
    frames, i = [], 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % step == 0:
                h, w = bgr.shape[:2]
                if w > max_width:
                    bgr = cv2.resize(bgr, (max_width, round(h * max_width / w)))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(tracker.process(rgb, i / fps))
            i += 1
    finally:
        cap.release()
        tracker.close()
    return frames


def pick_take(frames: list[Frame], kind: str) -> list[Frame] | None:
    """Cut the part of the video that contains the sign."""
    seg = Segmenter()
    segments = [s for s in (seg.update(f) for f in frames) if s is not None]
    if kind == "dynamic":
        if not segments:
            return None
        return max(segments, key=lambda s: s.duration).frames
    # static: the held part of a segment, or the stillest moment of the video
    for s in segments:
        if s.still_frames:
            return s.still_frames
    entries = [seg._entry(f) for f in frames]          # noqa: SLF001
    entries = [e for e in entries if e.hands]
    win = find_still_window(entries, length=0.5) if entries else None
    return [e.frame for e in entries[win[0]:win[1]]] if win else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder with one sub-folder of videos per sign")
    ap.add_argument("--kind", choices=["static", "dynamic"], default="dynamic")
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="library folder (default: data/)")
    ap.add_argument("--max-per-sign", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lib = Library(args.data)
    folders = sorted(p for p in args.folder.iterdir() if p.is_dir())
    if not folders:
        sys.exit(f"No sub-folders found in {args.folder}")
    total = 0
    for folder in folders:
        try:
            label = clean_label(folder.name)
        except ValueError:
            continue
        existing = lib.kind_of(label)
        if existing and existing != args.kind:
            print(f"! {label}: already exists as {existing}, skipping")
            continue
        videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXT)[: args.max_per_sign]
        takes = []
        for v in videos:
            take = pick_take(read_frames(v), args.kind)
            status = f"{len(take)} frames" if take else "no sign found"
            print(f"  {label:<16} {v.name:<40} {status}")
            if take:
                takes.append(take)
        if takes and not args.dry_run:
            ids = lib.add_samples(label, args.kind, takes, source=f"import:{args.folder.name}")
            total += len(ids)
            print(f"+ {label}: added {len(ids)} example(s)")
    print(f"Done. Added {total} example(s) to {args.data}" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
