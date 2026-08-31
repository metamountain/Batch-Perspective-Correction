# What is wrong with `chsasank/Image-Rectification`

The starting point for this project. It implements Chaudhury, DiVerdi and Ioffe
(ICIP 2014) in ~530 lines of numpy and scikit-image, and the paper is sound. The
implementation is not, for architectural work. Everything below was **measured**
by running the code, not read off it.

Test harness: a synthetic building facade, 900x600, focal length 700 px, camera
pitched up 9 deg and rolled 3 deg, with window grids, a gable, a ground plane and
60 random clutter segments. The true vertical vanishing point is at
`(215, -4120)`.

## 1. A vanishing point at infinity scores zero votes

`compute_votes` starts with

```python
vp = model[:2] / model[2]
```

so a vanishing point at infinity -- `model[2] == 0` -- produces `inf`, then
`nan`, then no votes at all. Measured directly:

```
VP from two exactly parallel verticals: [0., 400., 0.]   -> w = 0.0
compute_votes returns:                  [0., 0.]         (expected: full votes)
```

This is not an edge case. **A level camera puts the vertical vanishing point at
infinity**, and a level camera is the commonest input. The consequence is worse
than a missed detection: since the correct model scores zero, RANSAC
systematically prefers some *converging* hypothesis, so the code is biased
towards inventing a correction for photographs that need none.

Fixed here by never dehomogenising: `geometry.bearing_to_vp` computes
`vp[:2] - vp[2] * midpoint`, which is the direction towards the vanishing point
for finite and infinite points alike.

## 2. It does not find the vertical direction

Five identical runs on the same image:

| run | `vp1` | distance to the true vertical VP |
|---|---|---|
| 1 | (-411, -13677) | 9 578 px |
| 2 | (-151, -11167) | 7 057 px |
| 3 | (67, -7774) | 3 657 px |
| 4 | (-2503, -51324) | **47 283 px** |
| 5 | (-638, -16240) | 12 150 px |

against an image 600 px tall. There is no orientation prior anywhere: `vp1` is
whatever direction is dominant, which on a gabled building can be the roof.

## 3. It is not deterministic

The table above is five runs of the *same* function on the *same* array. It uses
`np.random.choice` on the global RNG with no seed, so a batch re-run produces
different files. Output sizes across five runs of `rectify_image`: 965x597,
964x594, 949x641, 912x616, 1035x612.

For a batch tool this alone is disqualifying.

## 4. `algorithm='3-line'` cannot run

```python
elif algorithm == '3-line':
    focal_length = None
    vp1, vp2 = ransac_3_line(edgelets1, focal_length, ...)
```

and `ransac_3_line` immediately computes `1 / focal_length**2`. Measured:
`TypeError: unsupported operand type(s) for ** or pow(): 'NoneType' and 'int'`.
Half the documented API is dead code.

## 5. `remove_inliers` ignores its own parameter

```python
def remove_inliers(model, edgelets, threshold_inlier=10):
    inliers = compute_votes(edgelets, model, 10) > 0
```

The argument is accepted and discarded. `rectify_image` passes 10, so the bug is
invisible there -- until someone tunes it.

## 6. 3-pixel "edgelets" give noisy directions

`probabilistic_hough_line(edges, line_length=3, line_gap=2)` yields ~1 100
three-pixel fragments on the test image. A 3 px segment's direction is quantised
to a handful of possible angles, and the fit is only as good as its inputs. LSD
returns sub-pixel endpoints and an angular precision estimate.

## 7. The warp can shear

`compute_homography_and_warp` builds a general projective transform from the
vanishing line, then an affine correction to make the axes orthogonal. That has
eight degrees of freedom, and nothing constrains it to anything a camera could
have done. `clip_factor` exists to stop the result exploding, which is treating
the symptom.

Here the transform is always `K R K^-1` -- a pure camera rotation, three degrees
of freedom, of which yaw is deliberately left at zero. It cannot shear.
`test_reference.test_the_warp_is_a_camera_rotation_so_it_cannot_shear` asserts it.

## 8. It always corrects

There is no confidence, no threshold and no way to decline. For a batch of a
few hundred holiday photos of buildings, that is the difference between a tool
and a hazard.

## 9. Practicalities

* `transform.warp` returns float64 in 0..1; the `__main__` block writes it
  straight to PNG.
* No EXIF handling at all, so orientation tags are ignored on input and all
  metadata is lost on output.
* `np.linalg.lstsq(a, b)[0]` without `rcond`.
* Output is always PNG, at whatever size the homography produced, with black
  borders and no crop.

## What was kept

The overall shape of the pipeline -- detect lines, find the vertical vanishing
point by RANSAC, refine on the inliers, build a homography -- is right, and it
is the shape used here. The paper it implements is good. What it needed was an
orientation prior, an infinity-safe residual, a physical warp, a seeded RNG and
the ability to say no.
