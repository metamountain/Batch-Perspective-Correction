# Test assets

Drop architectural photographs here and `tests/test_assets.py` starts checking
against them. The suite skips cleanly when the folder is empty, so the
repository does not have to carry binaries.

Naming conventions the tests understand:

| name | meaning |
|---|---|
| `something_upright.jpg` | already straight; asserted to be **left unchanged** |
| `something_skip.jpg` | detection should refuse it; asserted to be **SKIPPED** |
| anything else | only checked for sanity and repeatability |

Useful cases to collect: a flat-on facade, a corner view, a strong upward tilt,
a crooked horizon with no verticals, an interior, a photo with heavy foliage in
front of the building, and a web JPEG with the EXIF stripped.
