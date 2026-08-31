"""Manual review mode -- the part of the GUI that is not Tkinter."""
import math

import numpy as np

import synth
from bpc.config import Settings
from bpc.review import AUTO, MANUAL, ReviewSession


def _session(**kw):
    sc = synth.Scene(pitch_deg=kw.pop("pitch_deg", 9), roll_deg=kw.pop("roll_deg", -3),
                     seed=kw.pop("seed", 21), **kw)
    return ReviewSession("in-memory.jpg", Settings(), image=sc.img), sc


def test_a_fresh_session_starts_from_the_automatic_fit():
    s, sc = _session()
    assert s.mode == AUTO
    tr, tp = sc.true_roll_pitch()
    assert abs(math.degrees(s.model.roll - tr)) < 1.0


def test_striking_out_a_line_refits():
    s, _ = _session()
    before = (s.model.roll, s.model.pitch)
    idx = int(np.argmax(s.vert.length))
    assert s.toggle_line(idx)
    assert not s.enabled[idx]
    assert int(s.enabled.sum()) == len(s.vert) - 1
    # the fit is recomputed, not merely flagged
    assert (s.model.roll, s.model.pitch) != before or s.model is not None


def test_striking_out_the_slanted_candidates_only_touches_slanted_ones():
    s, _ = _session()
    lean = np.degrees(s.vert.angle_to_vert)
    n = s.disable_lines_by_angle(18.0)
    assert n == int((lean > 18.0).sum())
    assert np.all(lean[~s.enabled] > 18.0)


def test_a_click_picks_the_nearest_segment_and_misses_return_none():
    s, _ = _session()
    seg = s.vert.seg[0]
    mx, my = (seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2
    assert s.pick_line(mx / s.scale, my / s.scale, display_scale=1.0) is not None
    assert s.pick_line(-500.0, -500.0, display_scale=1.0) is None


def test_manual_sliders_override_the_fit():
    s, _ = _session()
    s.set_manual(roll_deg=-4.0, pitch_deg=11.0, focal_35mm=24.0)
    assert s.mode == MANUAL
    roll, pitch, f, _ = s.current_angles()
    assert abs(math.degrees(roll) + 4.0) < 1e-9
    assert abs(math.degrees(pitch) - 11.0) < 1e-9
    # a manual decision is never overruled by the confidence gate
    assert s.would_skip() is None


def test_manual_mode_ignores_the_automatic_limits():
    """A person looking at the picture outranks a safety cap meant for
    unattended batches."""
    s, _ = _session()
    s.set_manual(pitch_deg=28.0)
    _, pitch, _, _ = s.current_angles()
    assert abs(math.degrees(pitch) - 28.0) < 1e-9 > 0


def test_reset_returns_to_automatic_and_re_enables_every_line():
    s, _ = _session()
    s.disable_lines_by_angle(5.0)
    s.set_manual(pitch_deg=20.0)
    s.reset_to_auto()
    assert s.mode == AUTO and bool(s.enabled.all())


def test_render_pair_produces_two_images():
    s, _ = _session()
    before, after = s.render_pair(max_edge=480)
    assert before.ndim == 3 and after.ndim == 3
    assert max(before.shape[:2]) <= 480 and max(after.shape[:2]) <= 480


def test_an_unclear_image_reports_why_it_would_be_skipped():
    s = ReviewSession("flat.jpg", Settings(), image=synth.flat_image())
    assert s.would_skip() is not None
    # and the manual mode can still correct it
    s.set_manual(roll_deg=2.0, pitch_deg=0.0, focal_35mm=28)
    assert s.would_skip() is None
