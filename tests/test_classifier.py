import numpy as np

from app.classifier import Classifier, Example, query_from_segment, query_from_static_take
from app.features import describe
from app.landmarks import mirror_frame
from app.segmenter import Segmenter
from tests.synth import UP, frame, j_path, lerp, sign_sequence, static_take


def ex(sid, label, kind, frames):
    d = describe(frames, kind)
    return Example(sid, label, kind, *d)


def segment_of(frames):
    seg = Segmenter()
    found = [s for s in (seg.update(f) for f in frames) if s is not None]
    assert found, "no segment found"
    return found[0]


def build(rng, letters=("A", "B", "L", "V"), n=5, with_j=False):
    exs = []
    for lbl in letters:
        for i in range(n):
            exs.append(ex(f"{lbl}{i}", lbl, "static", static_take(lbl, rng=rng, noise=0.003)))
    if with_j:
        for i in range(n):
            s = segment_of(sign_sequence("I", hold=0.0, path=j_path, path_time=0.7,
                                         rng=rng, noise=0.003))
            exs.append(ex(f"J{i}", "J", "dynamic", s.frames))
    return Classifier(exs)


def test_recognizes_static_letters_from_live_segments():
    rng = np.random.default_rng(0)
    clf = build(rng)
    for lbl in ("A", "B", "L", "V"):
        seg = segment_of(sign_sequence(lbl, hold=0.8, rng=rng, noise=0.003))
        pred = clf.predict(query_from_segment(seg))
        assert pred.label == lbl and pred.recognized, (lbl, pred)
        assert len(pred.alternatives) == 2


def test_unknown_sign_is_not_recognized():
    rng = np.random.default_rng(1)
    clf = build(rng, letters=("A", "B"))
    two_hands = [frame(i / 15, right=("V", UP), left=("V", (0.58, 0.4)), rng=rng, noise=0.003)
                 for i in range(15)]
    pred = clf.predict(query_from_static_take(two_hands))
    assert not pred.recognized


def test_other_hand_works_only_with_mirror():
    rng = np.random.default_rng(2)
    clf = build(rng, letters=("L", "V"))
    left_handed = [mirror_frame(f) for f in static_take("L", rng=rng, noise=0.003)]
    q = query_from_static_take(left_handed)
    assert clf.predict(q, mirror=True).label == "L"
    assert not clf.predict(q, mirror=False).recognized


def test_movement_sign_vs_static_handshape():
    rng = np.random.default_rng(3)
    clf = build(rng, letters=("I", "A", "B"), with_j=True)
    seg_i = segment_of(sign_sequence("I", hold=0.8, rng=rng, noise=0.003))
    seg_j = segment_of(sign_sequence("I", hold=0.0, path=j_path, path_time=0.7, rng=rng, noise=0.003))
    assert clf.predict(query_from_segment(seg_i)).label == "I"
    assert clf.predict(query_from_segment(seg_j)).label == "J"


def test_spelling_mode_filter():
    rng = np.random.default_rng(4)
    clf = build(rng, letters=("A", "B"))
    q = query_from_static_take(static_take("A", rng=rng, noise=0.003))
    assert clf.predict(q, allowed={"B"}).label == "B"


def test_flags_mislabeled_example():
    rng = np.random.default_rng(5)
    exs = []
    for lbl in ("A", "B"):
        for i in range(5):
            exs.append(ex(f"{lbl}{i}", lbl, "static", static_take(lbl, rng=rng, noise=0.003)))
    exs.append(ex("oops", "A", "static", static_take("B", rng=rng, noise=0.003)))
    flags = Classifier(exs).flags()
    assert flags == {"oops": "B"}


def test_empty_classifier():
    pred = Classifier([]).predict(query_from_static_take(static_take("A")))
    assert pred.label is None and not pred.recognized


# --- "U" vs "NOME": same handshape, NOME moves sideways --------------------------

def sideways(rng, start=(0.46, 0.5), end=(0.34, 0.5), shape="V", move=0.35, hold=0.5, fps=15, t0=0.0):
    """A take recorded the way Teach does it: hand already up, moves, holds."""
    frames, t = [], t0
    n = int(move * fps)
    for i in range(n):
        frames.append(frame(t, right=(shape, lerp(start, end, i / (n - 1))), rng=rng, noise=0.002)); t += 1 / fps
    for _ in range(int(hold * fps)):
        frames.append(frame(t, right=(shape, end), rng=rng, noise=0.002)); t += 1 / fps
    return frames


def live(rng, body_frames, first_xy, shape="V", fps=15):
    """Hands down -> the hand rises from below into the first position -> the sign."""
    frames, t = [], 0.0
    for _ in range(3):
        frames.append(frame(t)); t += 1 / fps
    for i in range(1, 6):
        frames.append(frame(t, right=(shape, lerp((first_xy[0], 0.9), first_xy, i / 5)), rng=rng, noise=0.002)); t += 1 / fps
    t0 = body_frames[0].t
    for f in body_frames:
        g = frame(t + f.t - t0)
        g.right = f.right
        frames.append(g)
    t = frames[-1].t
    for _ in range(4):
        t += 1 / fps
        frames.append(frame(t))
    return frames


def u_nome_library(rng):
    exs = [ex(f"U{i}", "U", "static", static_take("V", rng=rng, noise=0.002, wrist=(0.42, 0.44)))
           for i in range(5)]
    exs += [ex(f"N{i}", "NOME", "dynamic", sideways(rng)) for i in range(3)]
    return Classifier(exs)


def test_movement_sign_with_a_rise_matches_example_recorded_without_it():
    rng = np.random.default_rng(10)
    clf = u_nome_library(rng)
    attempt = live(rng, sideways(rng), (0.46, 0.5))
    pred = clf.predict(query_from_segment(segment_of(attempt)))
    assert pred.label == "NOME" and pred.recognized, pred


def test_held_handshape_is_not_taken_for_the_movement_sign():
    rng = np.random.default_rng(11)
    clf = u_nome_library(rng)
    hold = [frame(i / 15, right=("V", (0.42, 0.44)), rng=rng, noise=0.002) for i in range(12)]
    pred = clf.predict(query_from_segment(segment_of(live(rng, hold, (0.42, 0.44)))))
    assert pred.label == "U" and pred.recognized, pred
