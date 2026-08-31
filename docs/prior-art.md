# Prior art: what everyone else does, and what was taken from it

Surveyed before writing a line of this, because the problem is thirty years old
and most of the ways to get it wrong are already documented somewhere.

## The production tools

| tool | line detector | model fitted | how it decides | licence |
|---|---|---|---|---|
| **ShiftN** (Marcus Hebel) | Burns 1986 region growing | vertical vanishing point -> rotation + shift | always corrects, damping slider | LGPL, source on shiftn.de |
| **darktable `ashift`** | LSD | rotation + lensshift_h + lensshift_v + shear, Nelder-Mead | RANSAC outliers, refuses if area > 4x | GPL-3.0 |
| **Adobe Lightroom "Upright"** | undisclosed | full camera pose, patented | multiple modes, user picks | proprietary |
| **RawTherapee** | manual + auto | perspective + recovery | manual first | GPL-3.0 |
| **chsasank/Image-Rectification** | Canny + probabilistic Hough | two vanishing points -> projective + affine | never refuses | MIT |

ShiftN is the ancestor of the whole photographic line; darktable's module says
in its own commit history that it is inspired by it. Both target the same thing
this project does, so both were studied.

**Licensing:** this project is MIT and contains no code from any of the above.
darktable's `ashift.c` is GPL-3.0 and ShiftN's source is LGPL; reading them to
understand an algorithm is fine, copying them into an MIT project is not, and
nothing was copied. Constants like "assume 28 mm when unknown" are facts about
the problem, not expression.

### Taken from darktable's `ashift`

* **28 mm as the generic focal length assumption.** Independently the right
  default for architecture; useful confirmation that a production tool with
  years of user feedback landed on the same number.
* **Reject a vanishing point that falls inside the image frame.** Verticals
  that converge within the visible area describe no photograph of a building.
* **Refuse a warp that inflates the frame beyond ~4x.**
* **Discard line segments hugging the frame** (2-3 px), which are sensor edges
  and vignetting boundaries, not scene structure.
* **Nelder-Mead over the correction parameters** rather than inverting a
  vanishing point directly.
* The scale-free RANSAC residual `|v . l|` with both normalised. The angular
  residual used here is the same quantity expressed as an angle, which is what
  lets the inlier threshold be stated in degrees.

### Deliberately not taken

* **darktable's parameterisation** (rotation, lensshift_h, lensshift_v, shear).
  Fitting `roll`, `pitch` and `f` instead means every limit is stated in degrees
  of camera movement, which is what a photographer can reason about, and the
  transform is guaranteed to be a rotation, which cannot shear.
* **Self-tuning epsilon to eliminate 60 % of lines as outliers.** Sensible for
  a general editor facing arbitrary photos; too aggressive for clean
  architectural input, where 60 % of a facade's edges are genuinely inliers.
* **Correcting horizontal perspective by default.** darktable offers it;
  the brief here explicitly does not want it, and a rotation with no yaw
  component is the formal statement of "leave the horizontal perspective alone".

### Taken from ShiftN

Only the framing of the problem: correct the verticals, damp the correction,
and let the user override. Its Burns detector is the ancestor of LSD, which is
what is used here, so the lineage is honoured without copying anything.

## The research line

| method | year | approach | why not used here |
|---|---|---|---|
| Rother | 2002 | accumulator over line intersections | superseded |
| Tardif, J-Linkage | 2009 | clustering line preferences | strong, but heavier than needed once the vertical is constrained by a prior |
| Bazin & Pollefeys, 3-line RANSAC | 2012 | orthogonal triple, known focal | needs the focal length that is exactly what is missing |
| Chaudhury/DiVerdi/Ioffe | 2014 | two vanishing points, sequential RANSAC | **the reference implementation's basis**; see reference-review.md |
| Lezama et al. | 2014 | point alignments on the Gaussian sphere | ~30 s per image |
| **Lee et al.** | 2014 | MAP over vanishing points *and* camera pose jointly, with perceptual priors | **the model here follows this shape**: one joint estimate of (up, f), not two independent detections |
| Zhai et al. | 2016 | CNN global context -> horizon candidates | needs a trained model and a GPU-ish runtime |
| NeurVPS | 2019 | conic convolutions | most accurate reported, slowest |
| VaPiD | 2021 | learned optimiser | ~17x faster than NeurVPS, still a network |
| GeoCalib | 2024 | end-to-end camera calibration with a geometric optimiser | the honest answer to the unknown-focal-length problem, at the cost of a deep learning dependency |

The joint treatment demanded by the brief -- "Vertikalen + Horizont gemeinsam
als geometrisches Modell, statt zwei unabhaengige Erkennungen" -- is exactly
Lee et al.'s formulation, arrived at independently and implemented classically.

**Where a learned prior would genuinely help** is the focal length of a
stripped web JPEG, which is where this implementation is weakest (see
accuracy.md). GeoCalib or Hold-Geoffroy et al.'s perceptual measure predict it
from image content. That is the one place a network would earn a dependency,
and it is noted as future work rather than pretended away.

## Line detectors

| detector | speed | accuracy | availability |
|---|---|---|---|
| Canny + probabilistic Hough | slow, noisy directions | poor for 3 px "edgelets" | everywhere |
| **LSD** | baseline | best of the CPU methods | OpenCV core (dropped 4.1-4.7, back in 4.8+) |
| EDLines | ~10x LSD | close to LSD | not in OpenCV |
| FastLineDetector | ~10x LSD | worse in clutter | `opencv-contrib` only |
| ELSED | fastest published | best precision/recall of the CPU set | `pyelsed`, extra dependency |

LSD is the default with FLD and Hough as fallbacks, because the analysis runs on
a 1600 px copy where LSD costs a fraction of a second and accuracy is worth more
than speed. ELSED would be the upgrade if profiling ever says otherwise.

## Sources

- [chsasank/Image-Rectification](https://github.com/chsasank/Image-Rectification)
- [darktable `src/iop/ashift.c`](https://github.com/darktable-org/darktable/blob/master/src/iop/ashift.c)
- [darktable: a new module for automatic perspective correction](https://www.darktable.org/2016/03/a-new-module-for-automatic-perspective-correction/)
- [ShiftN](https://www.shiftn.de/) and the Python port [Wikinaut/shiftn](https://github.com/Wikinaut/shiftn)
- [Lee et al., Automatic Upright Adjustment of Photographs (TPAMI 2014)](https://cg.postech.ac.kr/papers/15_Automatic-Upright-Adjustment-of-Photographs.pdf)
- [Zhai et al., Detecting Vanishing Points using Global Image Context (CVPR 2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhai_Detecting_Vanishing_Points_CVPR_2016_paper.pdf)
- [NeurVPS](https://github.com/zhou13/neurvps), [VaPiD](https://vgl.ict.usc.edu/Research/VaPiD/VaPiD%20A%20Rapid%20Vanishing%20Point%20Detector%20via%20Learned%20Optimizers.pdf)
- [ELSED](https://oa.upm.es/76996/1/ELSED__Enhanced_Line_SEgment_Drawing.pdf), [A Comprehensive Review of Image Line Segment Detection](https://arxiv.org/html/2305.00264v2)
