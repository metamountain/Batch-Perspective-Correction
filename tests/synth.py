"""Synthetic architecture scenes with an exactly known camera pose.

The high-frequency-mask notes carry a hard-won lesson: a synthetic fixture
misled that project on the one question that mattered, because it was being
asked a *statistical* question (how does a histogram behave on real texture)
that synthetic data answered wrongly.

Here the question is *geometric* -- given a camera rotated by a known amount,
does the estimator recover that amount -- and for geometry a rendered scene with
an exact ground truth is the strictly better instrument, because on a real photo
nobody knows the true answer to compare against.  Real photographs are still
needed, and belong in ``tests/assets``; what they test is the front end (does
LSD find the facade at all under real texture, JPEG noise and foliage), not the
maths.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

WORLD_UP = np.array([0.0, -1.0, 0.0])


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Scene:
    """A building rendered through a camera with a known rotation."""

    def __init__(self, w=1200, h=800, focal_35mm=28.0, pitch_deg=0.0, roll_deg=0.0,
                 yaw_deg=0.0, seed=0, clutter=40, noise=3.0, corner=False,
                 texture=True):
        self.w, self.h = w, h
        self.f = focal_35mm * math.hypot(w, h) / math.hypot(36.0, 24.0)
        self.K = np.array([[self.f, 0, w / 2.0], [0, self.f, h / 2.0], [0, 0, 1.0]])
        self.pitch = math.radians(pitch_deg)
        self.roll = math.radians(roll_deg)
        self.yaw = math.radians(yaw_deg)
        # camera rotation: world -> camera
        self.R = rot_x(-self.pitch) @ rot_z(-self.roll) @ rot_y(self.yaw)
        self.rng = np.random.default_rng(seed)
        self.img = np.full((h, w, 3), 232, np.uint8)
        self._sky()
        self._facade(0.0)
        if corner:
            self._facade(math.pi / 2.4, x0=6.0)
        self._ground()
        if clutter:
            self._clutter(clutter)
        if texture:
            self._texture()
        if noise:
            n = self.rng.normal(0, noise, self.img.shape)
            self.img = np.clip(self.img.astype(float) + n, 0, 255).astype(np.uint8)

    # -- projection ------------------------------------------------------
    def project(self, P):
        p = self.K @ self.R @ np.asarray(P, dtype=float)
        if p[2] <= 1e-6:
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def seg(self, A, B, colour, thickness=2):
        a, b = self.project(A), self.project(B)
        if a is None or b is None:
            return
        cv2.line(self.img, (int(round(a[0])), int(round(a[1]))),
                 (int(round(b[0])), int(round(b[1]))), colour, thickness, cv2.LINE_AA)

    # -- content ---------------------------------------------------------
    def _sky(self):
        top = np.linspace(190, 235, self.h)[:, None]
        self.img[:] = np.dstack([np.repeat(top, self.w, 1) * 1.02,
                                 np.repeat(top, self.w, 1) * 0.99,
                                 np.repeat(top, self.w, 1) * 0.92]).clip(0, 255).astype(np.uint8)

    def _facade(self, yaw, x0=-6.0, width=12.0, top=-9.0, bottom=3.0, z=16.0):
        """A wall with a grid of windows, placed at an angle ``yaw`` about the
        world vertical so that corner views can be generated."""
        c, s = math.cos(yaw), math.sin(yaw)
        def P(u, v):
            return (x0 + u * c, v, z + u * s)
        wall = np.array([self.project(P(0, top)), self.project(P(width, top)),
                         self.project(P(width, bottom)), self.project(P(0, bottom))])
        if not np.any(wall == None):                      # noqa: E711
            cv2.fillPoly(self.img, [np.round(wall).astype(np.int32)], (176, 168, 158))
        for u in np.linspace(0, width, 9):
            self.seg(P(u, top), P(u, bottom), (86, 82, 78), 3)
        for v in np.linspace(top, bottom, 7):
            self.seg(P(0, v), P(width, v), (104, 100, 96), 2)
        # windows: extra short verticals and horizontals, like real facades
        for u in np.linspace(width / 16, width - width / 16, 8)[::1]:
            for v in np.linspace(top + 1.0, bottom - 1.2, 5):
                self.seg(P(u - 0.35, v), P(u + 0.35, v), (60, 58, 56), 2)
                self.seg(P(u - 0.35, v + 0.9), P(u + 0.35, v + 0.9), (60, 58, 56), 2)
                self.seg(P(u - 0.35, v), P(u - 0.35, v + 0.9), (52, 50, 48), 2)
                self.seg(P(u + 0.35, v), P(u + 0.35, v + 0.9), (52, 50, 48), 2)
        # gable: two strong diagonals, the classic false-vanishing-point trap
        self.seg(P(0, top), P(width / 2, top - 3.2), (48, 46, 44), 4)
        self.seg(P(width / 2, top - 3.2), P(width, top), (48, 46, 44), 4)

    def _ground(self):
        for z in np.linspace(9.0, 26.0, 7):
            self.seg((-14, 3.0, z), (14, 3.0, z), (150, 148, 145), 2)
        for x in np.linspace(-14, 14, 9):
            self.seg((x, 3.0, 9.0), (x, 3.0, 26.0), (150, 148, 145), 2)

    def _clutter(self, n):
        for _ in range(n):
            p = np.array([self.rng.uniform(-12, 12), self.rng.uniform(-8, 3),
                          self.rng.uniform(8, 24)])
            d = self.rng.normal(0, 0.8, 3)
            self.seg(p, p + d, (118, 136, 112), 2)

    def _texture(self):
        n = self.rng.normal(0, 5.0, self.img.shape[:2])
        self.img = np.clip(self.img.astype(float) + n[..., None], 0, 255).astype(np.uint8)

    # -- ground truth ----------------------------------------------------
    @property
    def up_camera(self):
        """World up expressed in camera coordinates -- what the estimator must find."""
        u = self.R @ WORLD_UP
        return u / np.linalg.norm(u)

    @property
    def vertical_vp(self):
        return self.K @ self.up_camera

    def true_roll_pitch(self):
        u = self.up_camera
        r = math.hypot(u[0], u[1])
        return math.atan2(u[0], -u[1]), math.atan2(u[2], r)

    def world_vertical_segments(self, n=7):
        """Image endpoints of lines that are vertical in the world -- used to
        measure the *effect* of a correction rather than its parameters."""
        out = []
        for x in np.linspace(-5, 5, n):
            a = self.project((x, -8.0, 16.0))
            b = self.project((x, 2.5, 16.0))
            if a and b:
                out.append([a[0], a[1], b[0], b[1]])
        return np.asarray(out)


def flat_image(w=1200, h=800, seed=0):
    """Texture with no straight lines at all: must be skipped, never corrected."""
    rng = np.random.default_rng(seed)
    img = rng.normal(140, 40, (h, w, 3))
    img = cv2.GaussianBlur(img, (0, 0), 6.0)
    img += rng.normal(0, 12, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
