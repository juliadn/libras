import numpy as np

from app import features as F
from app.landmarks import mirror_frame
from tests.synth import UP, frame, static_take


def test_mirror_twice_is_identity():
    f = frame(0, right=("L", UP), left=("B", (0.6, 0.5)))
    m2 = mirror_frame(mirror_frame(f))
    for a, b in [(f.pose, m2.pose), (f.left, m2.left), (f.right, m2.right), (f.face, m2.face)]:
        assert np.allclose(a, b)


def test_position_and_distance_to_camera_do_not_matter():
    base = F.static_descriptor(static_take("L"))
    moved = F.static_descriptor([frame(i / 15, right=("L", UP), shift=(0.1, -0.05), scale=0.7)
                                 for i in range(15)])
    assert np.linalg.norm(base - moved) < 1e-6


def test_mirrored_sign_matches_other_hand():
    right = static_take("L", side="right")
    d_right, d_right_m, _ = F.describe(right, "static")
    left = [mirror_frame(f) for f in right]
    d_left, _, _ = F.describe(left, "static")
    assert np.allclose(d_right_m, d_left)
    assert np.linalg.norm(d_right - d_left) > 0.5     # without mirroring they differ


def test_short_tracking_dropouts_are_filled():
    full = static_take("V")
    gappy = [f if i not in (3, 4, 9) else frame(f.t) for i, f in enumerate(full)]
    assert np.linalg.norm(F.static_descriptor(full) - F.static_descriptor(gappy)) < 1e-6


def test_resting_hand_is_ignored():
    one = static_take("V")
    with_rest = [frame(f.t, right=("V", UP), left=("B", (0.6, 0.97))) for f in one]
    assert np.linalg.norm(F.static_descriptor(one) - F.static_descriptor(with_rest)) < 1e-6


def test_no_body_gives_none():
    f = frame(0, right=("A", UP))
    f.pose = None
    assert F.static_descriptor([f]) is None


def moving(n_still=8, n_move=8, fps=15):
    """still -> sideways movement -> still (shape V)."""
    frames, t = [], 0.0
    xs = [0.42] * n_still + list(np.linspace(0.42, 0.30, n_move)) + [0.30] * n_still
    for x in xs:
        frames.append(frame(t, right=("V", (x, 0.45)))); t += 1 / fps
    return frames


def test_dynamic_descriptor_shapes_and_frame_rate():
    d, pen = F.dynamic_descriptor(moving())
    assert d.shape == (len(F.TEMPLATE_STARTS), F.N_FRAMES * F.FRAME_DIM)
    assert pen[0] == 0                                # variant 0 = uncut
    q, qpen = F.dynamic_descriptor(moving(), role="query")
    assert len(q) == len(qpen) > len(d)
    # same movement recorded at a different frame rate -> same descriptor
    fast, _ = F.dynamic_descriptor(moving(16, 16, fps=30))
    assert np.linalg.norm(d[0] - fast[0]) < 0.05


def test_movement_part_drops_still_start_and_end():
    m = F._prepare(moving(n_still=10, n_move=8))
    ts, X = F.movement_part(*m)
    assert 8 <= len(ts) <= 8 + 2 * F.KEEP_STILL + 2
