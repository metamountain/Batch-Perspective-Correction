# How it works

## The model is three numbers

    roll    tilt of the camera about the optical axis      (radians)
    pitch   tilt out of the image plane                    (radians)
    f       focal length                                   (pixels)

Everything else is derived from them. The horizon is **not** searched for
separately: it is the polar line of the vertical vanishing point with respect to
the image of the absolute conic, `K^-T u`, so once `(roll, pitch, f)` are fixed
the horizon is fixed too. This is the joint treatment the brief asked for --
verticals and horizon as one geometric model, not two independent detections.

Horizontal lines therefore serve two purposes only: they pin down `f`, and they
cross-validate the result. They never drive the vertical estimate. That ordering
is deliberate -- windows, balconies and roof edges generate enormous numbers of
false horizontal candidates.

## Pipeline

    load  ->  orientation  ->  downscale  ->  detect  ->  classify  ->
    RANSAC  ->  focal length  ->  joint refine  ->  confidence  ->
    limit  ->  warp  ->  crop  ->  save

### 1. Load and orient
EXIF orientation is applied on load, so detection always sees an upright image,
and reset to 1 on save so viewers do not rotate the result twice.

### 2. Detect
Analysis runs on a copy no larger than 1600 px on the long edge. LSD first,
`FastLineDetector` then `Canny + HoughLinesP` as fallbacks. Segments shorter
than 3.5 % of the short edge are dropped, as are segments hugging the frame
(sensor edges and vignetting boundaries are long, straight and perfectly
vertical, so a length-weighted vote would love them).

Because the model is angles plus a focal length in pixels, the only quantity
that needs rescaling back to full resolution is `f`. Angles are scale invariant.

### 3. Classify and weight
Lines within 32 deg of the image vertical form the vertical pool, within 32 deg
of horizontal the horizontal pool. Weight is `length x angular_prior(lean)`,
a smooth Gaussian falloff rather than a hard window: a 31 deg facade edge should
not be discarded while a 29 deg rafter is fully trusted.

### 4. Vertical vanishing point, by RANSAC
Residuals are **angular**, computed from `vp[:2] - vp[2] * midpoint`, so a
vanishing point at infinity -- a level camera -- is an ordinary value. A minimal
sample is rejected unless the two lines differ enough in angle to intersect at a
well-conditioned point. Candidates must pass an orientation prior (within
~40 deg of the image vertical, seen from the principal point) and a distance
prior (outside the frame). Several distinct hypotheses survive, plus one
explicit "already parallel" hypothesis that pairwise RANSAC can never propose,
and the choice between them is deferred until the horizon can arbitrate.

Refinement is IRLS: minimise `sum_i w_i sin^2(theta_i)` via the smallest
eigenvector of `sum_i (w_i/|g_i|^2) l_i l_i^T`, reweighting each pass. The
`1/|g|^2` factor is what turns the cheap algebraic solution into the angular one.

### 5. Focal length
In order of preference:

1. `--focal-35mm`, exact;
2. EXIF `FocalLengthIn35mmFilm`;
3. geometry, from orthogonality with the horizontal vanishing points, with a
   sensitivity estimate obtained by perturbing the vertical vanishing point by
   the inlier band and watching the answer move;
4. 28 mm equivalent, the same generic assumption darktable makes.

The prior and the measurement are combined by inverse-variance weighting in log
space, never by a hard switch. See `accuracy.md` for why that matters and for
the estimator that was measured, found harmful and demoted to a flag.

### 6. Joint refinement
Nelder-Mead over `(roll, pitch)` -- and `f` too, but only when the geometry
constrains it -- minimising a Cauchy-robust cost on the unit sphere:

    sum_i w_i rho( n_i . u )  +  0.5 sum_j W_j rho( b_j . u )  +  focal prior

where `n_i = K^T l_i` are the interpretation-plane normals of the vertical
lines and `b_j = K^-1 v_j` the bearings of the horizontal vanishing points. A
world-vertical line satisfies `n . u = 0`; a world-horizontal direction
satisfies `b . u = 0`. Both constraints live on the same sphere, which is why
they compose into one cost.

### 7. Confidence
Multiplicative, so any factor can veto:

| factor | asks |
|---|---|
| inlier share | how much of the vertical evidence agrees |
| line count | are there enough independent lines |
| spread | are the inliers spread across the frame, or one window in one corner |
| horizon support | do the horizontal vanishing points land on the implied horizon |
| focal source | EXIF 1.0, geometric 0.8-1.0, guessed 0.6 |
| stability | refit on random halves; a fit that swings a degree between subsets is a coin toss, not a measurement |

Below `--min-confidence` (0.40) the image is passed through untouched. The cost
of a false negative is an unchanged photo; the cost of a false positive is a
ruined one.

### 8. Limit
Strengths, the uncertain-focal damping, then hard caps (20 deg pitch, 12 deg
roll). Corrections under 0.15 deg are reported as "already upright".

### 9. Warp
`H = K R K^-1` where `R = Rx(pitch) Rz(-roll)`. **Roll is applied first**, about
the optical axis; pitching first would tilt the axis the roll is measured
against. `R` has no yaw component, which is the formal statement of "straighten
the verticals without needlessly changing the horizontal perspective".

Being a rotation, the transform cannot shear -- unlike a general projective
transform, which has eight free parameters and no reason to correspond to
anything a camera could have done.

### 10. Crop
The largest axis-aligned rectangle of the original aspect ratio that fits inside
the warped quad, anchored at the mapped image centre, found by 40 bisection
steps on the half-width. Anchoring rather than also optimising the position
keeps the composition the photographer framed, and makes the search monotone so
bisection lands exactly on the boundary.

### 11. Save
Atomic (write to `.part`, rename). EXIF and ICC carried over, orientation reset
to 1, recorded pixel dimensions updated, stale thumbnail dropped.
