"""Aurora: curtains of light drifting over a mountain silhouette.

Each curtain hangs from a base curve that folds and evolves with layered
3D noise (x, ribbon, time). Its vertical profile mimics the real thing: a
sharp, bright lower border and a long fading top, shifting from green to
violet with altitude. Fine vertical rays shimmer slowly along the curtain
and whole segments brighten and fade over minutes, so there is no loop.

All per-ribbon curves are 1D, so they are computed for every ribbon at
once with a few ``(n_ribbons, width)`` noise calls; the per-pixel work is
limited to each curtain's vertical band.
"""

from __future__ import annotations

import numpy as np
import pygame

from ..noise import fbm_grid, grid_noise, noise1
from .base import FieldScene, compose, mix_palette, ramp, smoothstep
from .stars import StarLayer

WARM = {
    "sky_top": (6, 5, 22),
    "sky_horizon": (30, 22, 58),
    "aurora_low": (110, 255, 150),
    "aurora_high": (200, 64, 168),
    "glow": (40, 110, 80),
    "far_hills": (11, 11, 28),
    "near_hills": (4, 4, 11),
    "rim": (34, 70, 56),
}
COOL = {
    "sky_top": (3, 6, 22),
    "sky_horizon": (12, 34, 64),
    "aurora_low": (60, 236, 214),
    "aurora_high": (104, 84, 236),
    "glow": (24, 90, 120),
    "far_hills": (8, 16, 32),
    "near_hills": (2, 6, 13),
    "rim": (30, 70, 96),
}

# Per ribbon: base height, fold amplitude, curtain height, brightness, tilt.
RIBBONS = np.array(
    [
        # base   amp    height  gain   tilt
        [0.50, 0.11, 0.26, 1.15, 0.10],
        [0.38, 0.09, 0.20, 0.95, -0.12],
        [0.62, 0.07, 0.14, 0.80, 0.05],
        [0.27, 0.12, 0.30, 0.55, -0.04],
    ],
    dtype=np.float32,
)
EDGE = 0.012  # thickness of the sharp lower border (screen heights)


def _ridge(n):
    """Ridged noise: sharp peaks where smooth noise crosses zero."""
    r = np.clip(1.0 - 0.55 * np.abs(n), 0.0, 1.0)
    return r * r


class AuroraScene(FieldScene):
    key = "aurora"
    title = "Aurora"
    downscale = 4.0

    def __init__(self, size, seed=0):
        self.stars = StarLayer(seed + 11, density=260, radius=(0.5, 1.3), brightness=(0.25, 0.9),
                               tints=((255, 255, 255), (200, 220, 255), (255, 236, 210)), twinkle=0.45)
        super().__init__(size, seed)

    def on_resize(self) -> None:
        rw, rh = self.res
        u, v = self.u[0], self.v[:, 0]
        # Static mountain ridges (fbm profiles along x).
        s = self.seed
        far = (
            0.83
            - 0.10 * _ridge(noise1(u * 1.7 + 3.0, s + 1))
            - 0.035 * _ridge(noise1(u * 5.3, s + 2))
            - 0.008 * noise1(u * 30.0, s + 5)
        )
        near = (
            0.93
            - 0.07 * _ridge(noise1(u * 1.1 + 9.0, s + 3))
            - 0.025 * _ridge(noise1(u * 4.1, s + 4))
            - 0.006 * noise1(u * 36.0, s + 6)
        )
        self.far_ridge = far.astype(np.float32)
        self.near_ridge = near.astype(np.float32)
        px = 1.0 / rh
        far_w = ramp(-px, px, v[:, None] - far[None, :])
        near_w = ramp(-px, px, v[:, None] - near[None, :])
        self._sky_w = ((1 - far_w) * (1 - near_w)).astype(np.float32)
        self._far_w = (far_w * (1 - near_w)).astype(np.float32)
        self._near_w = near_w.astype(np.float32)
        # Faint aurora-lit rim along the far ridge.
        self._rim = (self._far_w * np.exp(-np.maximum(v[:, None] - far[None, :], 0) / 0.008)).astype(np.float32)
        # Airglow: a broad, faint band low in the sky lit by the curtains.
        self._glow_v = np.exp(-((v - 0.62) ** 2) / 0.03)[:, None].astype(np.float32)
        self._sky_grad = (v ** 1.6)[:, None].astype(np.float32)
        self._masks = np.empty((8, rh, rw), dtype=np.float32)
        self._buf_a = np.empty((rh, rw), dtype=np.float32)
        self._buf_b = np.empty((rh, rw), dtype=np.float32)
        self.stars.layout(self.size)

    # ----------------------------------------------------------- curves
    def curtains(self, t: float):
        """Per-ribbon 1D curves: base line, height, brightness."""
        u = self.u[0]
        n = len(RIBBONS)
        rows = np.arange(n) * 9.7  # far apart in noise space = independent
        cx = self.aspect / 2
        # Large slow S-folds plus finer kinks along the lower border.
        fold = fbm_grid(u * 0.6 + 0.011 * t, rows, t / 30.0, self.seed + 21, octaves=4, lacunarity=2.3, gain=0.45)
        # Each curtain is tilted, and the tilt itself drifts over minutes,
        # so curtains cross and separate instead of staying stacked.
        tilt = RIBBONS[:, 4] + 0.10 * noise1(t / 160.0 + rows, self.seed + 26)
        base = RIBBONS[:, 0:1] + RIBBONS[:, 1:2] * fold + tilt[:, None] * (u - cx)[None, :]
        height = RIBBONS[:, 2:3] * np.clip(0.8 + 0.35 * grid_noise(u * 1.3, rows, t / 37.0, self.seed + 22), 0.35, 1.4)
        # Segments brighten and fade along the curtain, leaving dark gaps.
        seg = smoothstep(-0.35, 1.1, grid_noise(u * 0.75 - 0.004 * t, rows, t / 45.0, self.seed + 23))
        # Rays at two scales, sliding slowly along the curtain.
        rays = 0.6 * grid_noise(u * 38.0 + 0.2 * t, rows, t / 4.0, self.seed + 24)
        rays += 0.4 * grid_noise(u * 105.0 - 0.3 * t, rows, t / 2.5, self.seed + 27)
        rays = 0.55 + 0.45 * smoothstep(-1.0, 1.3, rays)
        # Slow global "substorm" breathing over minutes.
        surge = 0.75 + 0.25 * float(noise1(t / 70.0, self.seed + 25))
        brightness = (RIBBONS[:, 3:4] * surge) * seg * rays
        f32 = np.float32
        return base.astype(f32), height.astype(f32), brightness.astype(f32)

    def render(self) -> np.ndarray:
        t = self.t
        rw, rh = self.res
        v = self.v
        low = np.zeros((rh, rw), dtype=np.float32)
        high = np.zeros((rh, rw), dtype=np.float32)
        base, height, bright = self.curtains(t)

        for i in range(len(RIBBONS)):
            b, hgt, br = base[i], height[i], bright[i]
            # Only rows the curtain can reach (its tail fades to ~3% at 3.5x height).
            r0 = max(0, int((b.min() - 3.5 * hgt.max()) * rh))
            r1 = min(rh, int((b.max() + 4 * EDGE) * rh) + 2)
            if r0 >= r1:
                continue
            # In-place on preallocated buffers: this loop is the hot path.
            above, below = self._buf_a[: r1 - r0], self._buf_b[: r1 - r0]
            np.subtract(b[None, :], v[r0:r1], out=above)  # d > 0 above the lower border
            np.minimum(above, 0.0, out=below)
            below *= 1.0 / EDGE
            np.square(below, out=below)  # sharp gaussian fall-off below the border
            np.maximum(above, 0.0, out=above)
            above *= (1.0 / hgt)[None, :]  # long exponential fade above it
            below += above
            np.negative(below, out=below)
            prof = np.exp(below, out=below)
            prof *= br[None, :]
            above *= -1.6
            green = np.exp(above, out=above)  # low part of the curtain is green
            green *= prof
            low[r0:r1] += green
            prof -= green
            high[r0:r1] += prof

        # Soft-clip where curtains overlap so highlights don't flatten out.
        low = 1.0 - np.exp(-1.4 * low)
        high = 1.0 - np.exp(-0.9 * high)
        glow_x = low.mean(axis=0, keepdims=True)

        pal = mix_palette(WARM, COOL, self.mood)
        m = self._masks
        sky = self._sky_w
        m[0] = sky
        np.multiply(sky, self._sky_grad, out=m[1])
        np.multiply(low, sky, out=m[2])
        np.multiply(high, sky, out=m[3])
        m[4] = self._far_w
        m[5] = self._near_w
        # Airglow low in the sky wherever curtains are bright above it.
        np.multiply(sky, self._glow_v * glow_x, out=m[6])
        np.multiply(self._rim, glow_x * 4.0 + 0.1, out=m[7])
        return compose(
            [
                pal["sky_top"],
                pal["sky_horizon"] - pal["sky_top"],
                pal["aurora_low"],
                pal["aurora_high"],
                pal["far_hills"],
                pal["near_hills"],
                pal["glow"] * 1.5,
                pal["rim"],
            ],
            m,
        )

    def draw(self, surface: pygame.Surface) -> None:
        super().draw(surface)
        far = self.far_ridge
        rw = len(far)

        def visible(x, y):
            # Hide stars behind the hills and fade them toward the horizon.
            col = np.clip((x * len(far) / max(self.aspect, 1e-6)).astype(int), 0, rw - 1)
            return (y < far[col] - 0.01) * np.clip((far[col] - y) / 0.25, 0.0, 1.0)

        self.stars.draw(surface, self.t, offset=(self.t * 0.0004, 0.0), visible=visible)
