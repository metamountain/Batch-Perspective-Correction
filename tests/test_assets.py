"""Tests against real photographs in ``tests/assets``.

Synthetic scenes verify the geometry, because only there is the true camera
pose known.  What they cannot verify is the front end: whether LSD finds a
facade under real texture, JPEG blocking, foliage and lens distortion, and
whether the confidence gate fires on the right pictures.  That needs real
files, so drop architectural photos into ``tests/assets`` and these tests start
running.  Nothing here asserts a specific angle -- there is no ground truth for
a photograph -- only that the tool behaves sanely and repeatably.

A file named ``*_upright.*`` is additionally asserted to need no correction,
and ``*_skip.*`` to be refused.  Everything else is only checked for sanity.
"""
import glob
import math
import os

from bpc.config import Settings
from bpc.imageio import READABLE
from bpc.pipeline import ERROR, OK, SKIPPED, analyse, process
from bpc.review import ReviewSession

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")


def _files():
    out = []
    for f in sorted(glob.glob(os.path.join(ASSETS, "*"))):
        if os.path.splitext(f)[1].lower() in READABLE:
            out.append(f)
    return out


def _require():
    files = _files()
    if not files:
        raise SkipTest("no images in tests/assets")   # noqa: F821
    return files


def test_every_asset_is_processed_without_error():
    import tempfile
    for f in _require():
        with tempfile.TemporaryDirectory() as d:
            r = process(f, os.path.join(d, os.path.basename(f)), Settings())
            assert r.status != ERROR, f"{os.path.basename(f)}: {r.reason}"


def test_corrections_stay_within_plausible_bounds():
    """Whatever it decides, it must not propose an angle no photographer
    would have produced, and must not throw the frame away."""
    s = Settings()
    for f in _require():
        m, _, _, _, _ = analyse(_load(f), s)
        assert abs(math.degrees(m.roll)) < 25.0, os.path.basename(f)
        assert abs(math.degrees(m.pitch)) < 35.0, os.path.basename(f)


def test_results_are_repeatable_on_real_files():
    s = Settings()
    for f in _require():
        img = _load(f)
        a = analyse(img, s)[0]
        b = analyse(img, s)[0]
        assert (a.roll, a.pitch, a.f) == (b.roll, b.pitch, b.f), os.path.basename(f)


def test_files_marked_upright_are_left_alone():
    import tempfile
    hits = [f for f in _files() if "_upright" in os.path.basename(f).lower()]
    if not hits:
        raise SkipTest("no *_upright.* assets")       # noqa: F821
    for f in hits:
        with tempfile.TemporaryDirectory() as d:
            r = process(f, os.path.join(d, os.path.basename(f)), Settings())
            assert r.status == SKIPPED, f"{os.path.basename(f)} was changed: {r.line()}"


def test_files_marked_skip_are_refused():
    import tempfile
    hits = [f for f in _files() if "_skip" in os.path.basename(f).lower()]
    if not hits:
        raise SkipTest("no *_skip.* assets")          # noqa: F821
    for f in hits:
        with tempfile.TemporaryDirectory() as d:
            r = process(f, os.path.join(d, os.path.basename(f)), Settings())
            assert r.status == SKIPPED, f"{os.path.basename(f)}: {r.line()}"


def test_manual_review_opens_on_every_asset():
    for f in _require():
        s = ReviewSession(f, Settings())
        before, after = s.render_pair(400)
        assert before.size and after.size
        s.set_manual(roll_deg=1.0, pitch_deg=2.0, focal_35mm=28)
        assert s.would_skip() is None


def _load(path):
    from bpc.imageio import load
    return load(path).bgr
