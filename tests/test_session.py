"""Teach + recognize flows through the Session, with synthetic frames."""
import numpy as np

from app.library import Library
from app.session import Session
from tests.synth import UP, frame, j_path, lerp, sign_sequence


class FakeTracker:
    def close(self):
        pass


class Clock:
    def __init__(self):
        self.t = 100.0

    def next(self, dt=1 / 15):
        self.t += dt
        return self.t


def feed(sess, clock, specs, rng=None):
    """specs: list of dicts of frame() kwargs; returns all messages."""
    out = []
    for kw in specs:
        out += sess.process_frame(frame(clock.next(), rng=rng, noise=0.002 if rng else 0, **kw))
    return out


def of(msgs, typ):
    return [m for m in msgs if m["type"] == typ]


def teach_static(sess, clock, rng, label, shape, takes=5):
    msgs = sess.handle({"type": "teach_start", "label": label, "kind": "static"})
    assert of(msgs, "teach")[-1]["phase"] == "ready"
    msgs = sess.handle({"type": "teach_record"})
    for i in range(takes):
        # countdown 3 s + recording 1 s, holding the shape
        msgs = feed(sess, clock, [{"right": (shape, UP)}] * 70, rng)
        review = [m for m in of(msgs, "teach") if m["phase"] == "review"]
        assert review, f"take {i} never reached review"
        assert review[-1]["can_keep"]
        msgs = sess.handle({"type": "teach_keep"})
    assert of(msgs, "teach")[-1]["count"] == takes
    msgs = sess.handle({"type": "teach_save"})
    assert of(msgs, "teach")[0]["phase"] == "saved"


def test_teach_then_recognize_static(tmp_path):
    rng = np.random.default_rng(0)
    sess = Session(Library(tmp_path), tracker_factory=FakeTracker)
    clock = Clock()
    teach_static(sess, clock, rng, "a", "A")
    teach_static(sess, clock, rng, "b", "B")
    assert sess.prompt == "B"

    # hands down, then sign B: should be recognized and satisfy the prompt
    msgs = feed(sess, clock, [{}] * 5)
    rise = [{"right": ("B", lerp((0.45, 0.85), UP, i / 6))} for i in range(1, 7)]
    msgs = feed(sess, clock, rise + [{"right": ("B", UP)}] * 12, rng)
    preds = of(msgs, "prediction")
    assert len(preds) == 1
    assert preds[0]["label"] == "B" and preds[0]["recognized"] and preds[0]["prompt_ok"]
    assert "signing" in [m["value"] for m in of(msgs, "state")]


def test_teach_warns_when_take_looks_like_another_sign(tmp_path):
    rng = np.random.default_rng(1)
    sess = Session(Library(tmp_path), tracker_factory=FakeTracker)
    clock = Clock()
    teach_static(sess, clock, rng, "A", "A")
    sess.handle({"type": "teach_start", "label": "E", "kind": "static"})
    sess.handle({"type": "teach_record"})
    msgs = feed(sess, clock, [{"right": ("A", UP)}] * 70, rng)
    review = [m for m in of(msgs, "teach") if m["phase"] == "review"][-1]
    assert any("looks like A" in w for w in review["warnings"])


def test_teach_warns_hand_not_visible(tmp_path):
    sess = Session(Library(tmp_path), tracker_factory=FakeTracker)
    clock = Clock()
    sess.handle({"type": "teach_start", "label": "A", "kind": "static"})
    sess.handle({"type": "teach_record"})
    msgs = feed(sess, clock, [{}] * 70)
    review = [m for m in of(msgs, "teach") if m["phase"] == "review"][-1]
    assert "Hand not visible" in review["warnings"]


def test_teach_dynamic_sign(tmp_path):
    rng = np.random.default_rng(2)
    lib = Library(tmp_path)
    sess = Session(lib, tracker_factory=FakeTracker)
    clock = Clock()
    sess.handle({"type": "teach_start", "label": "J", "kind": "dynamic"})
    sess.handle({"type": "teach_record"})
    for _ in range(3):
        # countdown with hands down, then the J movement
        msgs = feed(sess, clock, [{}] * 46)
        msgs += [m for f in sign_sequence("I", t0=0, hold=0.0, path=j_path, path_time=0.7, rng=rng, noise=0.002)
                 for m in sess.process_frame(frame(clock.next(), right=_spec(f)))]
        assert [m for m in of(msgs, "teach") if m["phase"] == "review"]
        sess.handle({"type": "teach_keep"})
    sess.handle({"type": "teach_save"})
    assert lib.kind_of("J") == "dynamic" and len(lib.classifier.examples) == 3


def _spec(f):
    """Re-create the right-hand spec of a synth frame (shape I) at its wrist position."""
    if f.right is None:
        return None
    return ("I", (float(f.right[0, 0]), float(f.right[0, 1])))


def test_settings_and_unknown_messages(tmp_path):
    sess = Session(Library(tmp_path), tracker_factory=FakeTracker)
    out = sess.handle({"type": "settings", "threshold": 0.7, "spelling": True, "rest_line": 9})
    assert out[0]["threshold"] == 0.7 and out[0]["spelling"] and out[0]["rest_line"] <= 1.3
    assert sess.handle({"type": "nope"})[0]["type"] == "warning"
    assert sess.handle({"type": "teach_keep"})[0]["type"] == "warning"
    assert sess.handle({"type": "teach_start", "label": "  "})[0]["type"] == "warning"


def test_empty_library_prediction(tmp_path):
    sess = Session(Library(tmp_path), tracker_factory=FakeTracker)
    clock = Clock()
    feed(sess, clock, [{}] * 3)
    msgs = feed(sess, clock, [{"right": ("A", UP)}] * 15)
    preds = of(msgs, "prediction")
    assert preds and preds[0].get("empty")


def test_movement_sign_replaces_the_held_sign_it_started_with(tmp_path):
    """Pause in the U handshape, then do the NOME movement: 'U' is replaced by 'NOME'."""
    from tests.test_classifier import sideways
    rng = np.random.default_rng(3)
    lib = Library(tmp_path)
    lib.add_samples("U", "static", [[frame(i / 15, right=("V", (0.42, 0.44)), rng=rng, noise=0.002)
                                     for i in range(15)] for _ in range(5)])
    lib.add_samples("NOME", "dynamic", [sideways(rng) for _ in range(3)])
    sess = Session(lib, tracker_factory=FakeTracker)
    clock = Clock()
    feed(sess, clock, [{}] * 4)
    rise = [{"right": ("V", lerp((0.42, 0.9), (0.42, 0.44), i / 5))} for i in range(1, 6)]
    pause = [{"right": ("V", (0.42, 0.44))}] * 9                       # 0.6 s in "U"
    move = [{"right": ("V", lerp((0.42, 0.44), (0.30, 0.5), i / 5))} for i in range(1, 6)]
    hold = [{"right": ("V", (0.30, 0.5))}] * 8
    msgs = feed(sess, clock, rise + pause + move + hold, rng)
    preds = [(m["label"], m.get("replaces_previous", False)) for m in of(msgs, "prediction")]
    assert preds[-1] == ("NOME", True), preds
    assert preds[0][0] == "U", preds

    # after lowering the hands, a new NOME does not replace anything
    msgs = feed(sess, clock, [{}] * 5 + rise + move + hold, rng)
    preds = [(m["label"], m.get("replaces_previous", False)) for m in of(msgs, "prediction")]
    assert preds[-1] == ("NOME", False), preds
