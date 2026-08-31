"""Does the estimator recover a camera pose it was never told?

Accuracy targets come from measurement, not from wishes -- see
docs/accuracy.md for the numbers these thresholds are drawn from.
"""
import math

import numpy as np

import synth
from bpc.config import Settings
from bpc.pipeline import analyse


def _err(scene, settings):
    m, vert, horiz, scale, _ = analyse(scene.img, settings)
    tr, tp = scene.true_roll_pitch()
    return (math.degrees(m.roll - tr), math.degrees(m.pitch - tp), m)


def test_roll_is_recovered_precisely_and_does_not_need_the_focal_length():
    """Levelling is the half of the job that is exactly solvable: roll is
    ``atan2(u_x, -u_y)`` and both components carry a factor 1/f that cancels.
    It must therefore be right even when the focal length is unknown."""
    worst = 0.0
    for f35 in (18, 28, 50):
        for roll in (-8, -3, 0, 2.5, 6):
            sc = synth.Scene(focal_35mm=f35, pitch_deg=5, roll_deg=roll, seed=11)
            dr, _, _ = _err(sc, Settings())          # no focal length supplied
            worst = max(worst, abs(dr))
    assert worst < 0.5, f"roll error {worst:.3f} deg"


def test_pitch_is_recovered_when_the_focal_length_is_known():
    worst = 0.0
    for f35 in (18, 24, 28, 35, 50):
        for pitch in (3, 6, 10, 14):
            sc = synth.Scene(focal_35mm=f35, pitch_deg=pitch, roll_deg=1.5, seed=5)
            _, dp, _ = _err(sc, Settings().replace(focal_35mm=f35))
            worst = max(worst, abs(dp))
    assert worst < 2.0, f"pitch error {worst:.3f} deg"


def test_a_level_camera_is_recognised_as_level():
    sc = synth.Scene(pitch_deg=0, roll_deg=0, seed=2)
    dr, dp, m = _err(sc, Settings())
    assert abs(dr) < 0.2 and abs(dp) < 0.3
    total = math.degrees(math.hypot(m.roll, m.pitch))
    assert total < Settings().min_correction_deg * 4


def test_texture_without_lines_yields_no_confidence():
    """A photo with no straight edges must not be corrected on the strength of
    whatever the detector scraped together."""
    img = synth.flat_image()
    m, _, _, _, _ = analyse(img, Settings())
    assert m.confidence < Settings().min_confidence


def test_estimation_is_deterministic():
    """Two runs on one file must give one answer.  The reference RANSAC uses
    the global RNG and returned vanishing points 3 700 to 47 000 px apart across
    five identical runs of the same image."""
    sc = synth.Scene(pitch_deg=7, roll_deg=-2, seed=4)
    a = analyse(sc.img, Settings())[0]
    b = analyse(sc.img, Settings())[0]
    assert a.roll == b.roll and a.pitch == b.pitch and a.f == b.f


def test_confidence_falls_when_the_evidence_is_local():
    """A vanishing point voted for by one corner of the frame is not evidence
    about the camera, and must score lower than one supported across it."""
    full = synth.Scene(pitch_deg=9, roll_deg=1, seed=6)
    m_full = analyse(full.img, Settings())[0]
    cropped = full.img.copy()
    cropped[:, : int(full.w * 0.62)] = 235          # keep only a narrow strip
    m_part = analyse(cropped, Settings())[0]
    assert m_part.confidence < m_full.confidence


def test_gable_diagonals_do_not_become_the_vertical_direction():
    """The roof is the classic false positive: two long, strong, converging
    lines that a dominant-direction search happily calls the answer."""
    sc = synth.Scene(pitch_deg=6, roll_deg=0, seed=8)
    _, dp, m = _err(sc, Settings().replace(focal_35mm=28))
    assert abs(dp) < 2.0
    assert abs(math.degrees(m.roll)) < 2.0
