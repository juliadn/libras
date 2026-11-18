import numpy as np

from app.segmenter import Segmenter
from tests.synth import UP, frame, j_path, lerp, sign_sequence


def run(frames, seg=None):
    seg = seg or Segmenter()
    out = []
    for f in frames:
        s = seg.update(f)
        if s is not None:
            out.append(s)
    return out, seg


def test_rise_and_hold_gives_one_segment_ending_in_hold():
    rng = np.random.default_rng(0)
    segs, seg = run(sign_sequence("A", hold=2.5, rng=rng, noise=0.002))
    assert [s.reason for s in segs] == ["hold"]          # long hold counts once
    s = segs[0]
    assert s.still_frames and len(s.still_frames) >= 5
    assert seg.state == "idle"                           # hands went down at the end


def test_movement_then_down():
    segs, _ = run(sign_sequence("I", hold=0.0, path=j_path, path_time=0.8))
    assert len(segs) == 1
    assert segs[0].reason == "down"
    assert segs[0].duration >= 0.8


def test_very_short_movement_is_ignored():
    frames = [frame(0.0), frame(1 / 15, right=("A", UP)), frame(2 / 15, right=("A", UP)),
              *[frame((3 + i) / 15) for i in range(5)]]
    segs, _ = run(frames)
    assert segs == []


def test_fingerspelling_two_letters_without_lowering_the_hand():
    rng = np.random.default_rng(1)
    t, dt, frames = 0.0, 1 / 15, []
    for _ in range(3):
        frames.append(frame(t)); t += dt
    for i in range(6):                                   # rise with A
        frames.append(frame(t, right=("A", lerp((0.45, 0.85), UP, (i + 1) / 6)), rng=rng, noise=0.002)); t += dt
    for _ in range(12):                                  # hold A
        frames.append(frame(t, right=("A", UP), rng=rng, noise=0.002)); t += dt
    for _ in range(3):                                   # change to B
        frames.append(frame(t, right=("B", UP), rng=rng, noise=0.002)); t += dt
    for _ in range(12):                                  # hold B
        frames.append(frame(t, right=("B", UP), rng=rng, noise=0.002)); t += dt
    segs, seg = run(frames)
    assert [s.reason for s in segs] == ["hold", "hold"]
    assert seg.state == "held"


def test_small_wobble_while_holding_does_not_retrigger():
    rng = np.random.default_rng(2)
    frames = sign_sequence("A", hold=0.8, lower=False, rng=rng, noise=0.002)
    t = frames[-1].t
    for i in range(30):
        wob = (UP[0] + 0.004 * np.sin(i), UP[1] + 0.004 * np.cos(i))
        t += 1 / 15
        frames.append(frame(t, right=("A", wob), rng=rng, noise=0.002))
    segs, seg = run(frames)
    assert len(segs) == 1 and seg.state == "held"


def test_long_movement_times_out_at_3s():
    frames, t = [], 0.0
    for i in range(90):                                   # 6 s of circling
        a = i / 4
        frames.append(frame(t, right=("B", (0.42 + 0.06 * np.cos(a), 0.4 + 0.06 * np.sin(a)))))
        t += 1 / 15
    segs, seg = run(frames)
    assert len(segs) == 1 and segs[0].reason == "timeout"
    assert 2.8 <= segs[0].duration <= 3.1
    assert seg.state == "wait"


def test_prime_with_raised_hand_waits_for_movement():
    seg = Segmenter()
    seg.prime(frame(0.0, right=("I", UP)))
    assert seg.state == "held"
    segs, _ = run([frame(i / 15, right=("I", UP)) for i in range(1, 10)], seg)
    assert segs == [] and seg.state == "held"
