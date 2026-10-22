"""Ocean: layered swells rolling toward the viewer under a low sun.

The water is a stack of wave bands receding in perspective to the horizon.
Each crest line is a sum of travelling sinusoids whose wavenumbers and
speeds are mutually incommensurate, plus slow noise, so the surface never
settles into a visible cycle. Above the horizon, stratus streaks drift past
a low sun (a moon in the cool break palette), and a glitter path sparkles on
the water beneath it.

Each band is drawn from its crest line down to the next nearer crest
with anti-aliased membership masks, restricted to the rows it can touch:
a lit face that darkens toward the trough, a thin crest highlight, and a
fine ripple texture squeezed by perspective.
"""

from __future__ import annotations

import numpy as np

from ..noise import fbm_grid, grid_noise, noise1
from .base import FieldScene, compose, mix_palette, ramp, smoothstep

WARM = {
    "sky_top": (34, 24, 74),
    "sky_mid": (176, 84, 116),
    "sky_horizon": (255, 176, 122),
    "sun": (255, 226, 170),
    "cloud": (120, 60, 100),
    "cloud_lit": (255, 170, 120),
    "deep": (16, 12, 40),
    "shallow": (70, 44, 92),
    "face": (226, 118, 118),
    "reflect": (240, 150, 120),
    "crest": (255, 190, 150),
}
COOL = {
    "sky_top": (6, 16, 46),
    "sky_mid": (30, 82, 128),
    "sky_horizon": (140, 200, 214),
    "sun": (226, 240, 255),
    "cloud": (36, 60, 96),
    "cloud_lit": (170, 210, 240),
    "deep": (2, 12, 28),
    "shallow": (16, 60, 84),
    "face": (70, 160, 180),
    "reflect": (130, 196, 212),
    "crest": (180, 230, 240),
}

HORIZON = 0.52
N_BANDS = 14
SUN_RADIUS = 0.045


class OceanScene(FieldScene):
    key = "ocean"
    title = "Ocean Waves"
    downscale = 4.0

    def __init__(self, size, seed=0):
        rng = np.random.default_rng(seed)
        # Three wave trains per band: wavenumber (per screen height), speed
        # (radians/s) and phase; ratios avoid simple fractions.
        self.k = rng.uniform(0.8, 1.3, (N_BANDS, 3)) * np.array([2.3, 3.7 * np.sqrt(2), 6.1 * np.sqrt(3)])
        self.w = rng.uniform(0.7, 1.3, (N_BANDS, 3)) * np.array([0.21, -0.13, 0.33])
        self.phase = rng.uniform(0, 2 * np.pi, (N_BANDS, 3))
        self.amp = np.array([0.55, 0.3, 0.15])
        super().__init__(size, seed)

    def on_resize(self) -> None:
        rw, rh = self.res
        u, v = self.u, self.v
        self.horizon_row = int(HORIZON * rh)
        self.sun_x = self.aspect * 0.64
        sky_w = ramp(HORIZON + 0.5 / rh, HORIZON - 0.5 / rh, v) * np.ones_like(u)
        self._sky_w = sky_w.astype(np.float32)
        self._water_w = (1.0 - sky_w).astype(np.float32)
        v_mid = 0.30
        self._sky_a = (sky_w * np.clip(v / v_mid, 0, 1)).astype(np.float32)
        self._sky_b = (sky_w * np.clip((v - v_mid) / (HORIZON - v_mid), 0, 1) ** 1.4).astype(np.float32)
        # Band crest baselines: spaced in perspective (dense near horizon).
        frac = (np.arange(N_BANDS) + 1) / N_BANDS
        self.band_y = HORIZON + (1.1 - HORIZON) * frac**1.9
        depth = self.band_y - HORIZON  # ~ 1 / distance
        self.band_amp = 0.16 * depth
        self.band_scale = 1.0 / np.maximum(depth, 0.02)  # far = finer waves
        spacing = np.diff(np.concatenate([self.band_y, [self.band_y[-1] * 2 - self.band_y[-2]]]))
        self.face_len = (0.5 * spacing).astype(np.float32)
        self.crest_w = (0.0025 + 0.012 * frac).astype(np.float32)
        # Distance haze: far bands take on more of the sky's colour.
        self.band_beta = (0.9 - 0.75 * frac**0.7).astype(np.float32)
        # Water close to the horizon reflects the sky.
        self._reflect = (self._water_w * np.exp(-(v - HORIZON) / 0.045)).astype(np.float32)
        # Sun glow (sky) and its soft reflection column (water).
        d2 = (u - self.sun_x) ** 2 + (v - (HORIZON - 0.11)) ** 2
        disk = smoothstep(SUN_RADIUS + 1.0 / rh, SUN_RADIUS - 1.0 / rh, np.sqrt(d2))
        glow = 0.55 * np.exp(-d2 / 0.006) + 0.35 * np.exp(-d2 / 0.06)
        spread = 0.04 + 0.35 * (v - HORIZON)
        column = np.exp(-(((u - self.sun_x) / spread) ** 2)) * np.exp(-(v - HORIZON) / 0.35) * 0.35
        self._sun = (self._sky_w * np.clip(disk + glow, 0, 1.6) + self._water_w * column).astype(np.float32)
        self._sun_near = np.exp(-((u - self.sun_x) ** 2) / 0.25 - (v - HORIZON) ** 2 / 0.05).astype(np.float32)
        # Glitter is only computed in the columns beneath the sun.
        half = 0.45
        self._gc0 = max(0, int((self.sun_x - half) * rh))
        self._gc1 = min(rw, int((self.sun_x + half) * rh) + 1)
        gv = self.v[self.horizon_row :, 0]
        # Perspective: distance along the water ~ 1 / (rows below horizon).
        self._depth_z = (1.0 / (gv - HORIZON + 0.03)).astype(np.float64)
        gu = self.u[0, self._gc0 : self._gc1]
        spread_g = 0.03 + 0.4 * (gv[:, None] - HORIZON)
        self._glitter_env = (
            np.exp(-(((gu[None, :] - self.sun_x) / spread_g) ** 2)) * np.exp(-(gv[:, None] - HORIZON) / 0.3)
        ).astype(np.float32)
        # Fresnel-ish: water reflects more of the sky at grazing angles,
        # i.e. toward the horizon.
        self._fresnel = (0.35 + 0.65 * np.exp(-(gv - HORIZON) / 0.16))[:, None].astype(np.float32)
        self._masks = np.zeros((12, rh, rw), dtype=np.float32)

    # ---------------------------------------------------------- waves
    def crest_lines(self, t: float) -> np.ndarray:
        """Crest heights ``(N_BANDS, w)`` in screen units at time ``t``."""
        u = self.u[0].astype(np.float64)
        x = u[None, :, None] * self.band_scale[:, None, None] * 0.35
        waves = np.sin(self.k[:, None, :] * x + self.w[:, None, :] * t + self.phase[:, None, :])
        swell = (waves * self.amp).sum(axis=2)
        # Peaked crests and broad troughs, like real swell.
        swell = (np.exp(1.1 * swell) - 1.2) * 0.75
        # Slow wandering of each band's height so rows don't stay rigid.
        drift = noise1(t / 40.0 + np.arange(N_BANDS) * 3.3, self.seed + 7)[:, None] * 0.25
        lines = self.band_y[:, None] - self.band_amp[:, None] * (swell + drift)
        return lines.astype(np.float32)

    def render(self) -> np.ndarray:
        t = self.t
        rw, rh = self.res
        hr = self.horizon_row
        v_water = self.v[hr:]
        m = self._masks
        m[3:].fill(0.0)

        # --- water bands (rows below the horizon only)
        lines = self.crest_lines(t)
        px = 0.8 / rh
        n_water = rh - hr
        beta = np.zeros((n_water, rw), np.float32)
        face = np.zeros((n_water, rw), np.float32)
        crest = np.zeros((n_water, rw), np.float32)
        beta += self.band_beta[0] + 0.1  # open water before the first crest
        for k in range(N_BANDS):
            top = lines[k]
            nxt = lines[k + 1] if k + 1 < N_BANDS else None
            r0 = max(hr, int((top.min() - px) * rh))
            r1 = rh if nxt is None else min(rh, int((nxt.max() + px) * rh) + 2)
            if r0 >= r1:
                continue
            rows = self.v[r0:r1]
            below = rows - top[None, :]
            vis = ramp(-px, px, below)  # inside this band...
            if nxt is not None:
                vis *= 1.0 - ramp(-px, px, rows - nxt[None, :])  # ...until the next crest
            np.maximum(below, 0.0, out=below)
            sl = slice(r0 - hr, r1 - hr)
            # Membership masks overlap only at anti-aliased edges, so
            # "replace" each band's region with this band's values.
            keep = 1.0 - vis
            beta[sl] = beta[sl] * keep + vis * self.band_beta[k]
            face[sl] = face[sl] * keep + vis * np.exp(-below / self.face_len[k])
            crest[sl] = crest[sl] * keep + vis * np.exp(-below / self.crest_w[k])

        # Fine ripples, stretched horizontally and squeezed by perspective.
        ripple = grid_noise(self.u[0] * 7.0 + 0.02 * t, self._depth_z * 3.0, t / 5.0, self.seed + 9)
        face *= 0.75 + 0.25 * np.clip(ripple, -1.5, 1.5)
        face *= self._fresnel
        crest *= 0.35 + 0.65 * self._sun_near[hr:]  # crests glint more toward the sun

        water = self._water_w[hr:]
        m[3, hr:] = water
        np.multiply(water, beta, out=m[4, hr:])
        np.multiply(water, face, out=m[5, hr:])
        np.multiply(water, crest, out=m[6, hr:])

        # Sun glitter: sparse bright peaks of fast-twinkling noise; glints
        # are wider than tall and shrink toward the horizon.
        c0, c1 = self._gc0, self._gc1
        if c1 > c0:
            xs = self.u[0, c0:c1] * 26.0 + 0.04 * t
            sparkle = grid_noise(xs, self._depth_z * 9.0, t / 0.7, self.seed + 3)
            glint = smoothstep(1.0, 1.9, sparkle) * self._glitter_env
            m[7, hr:, c0:c1] = glint * 1.6

        # --- sky: stratus streaks drifting past the sun
        sky_rows = hr + 1
        xs = self.u[0] * 1.3 + 0.0035 * t
        ys = self.v[:sky_rows, 0] * 8.0
        streaks = fbm_grid(xs, ys, t / 90.0, self.seed + 5, octaves=3, lacunarity=2.1)
        band_v = smoothstep(0.05, 0.2, self.v[:sky_rows]) * smoothstep(HORIZON - 0.01, HORIZON - 0.09, self.v[:sky_rows])
        cloud = smoothstep(0.25, 1.5, streaks) * band_v * self._sky_w[:sky_rows]
        m[8, :sky_rows] = cloud
        np.multiply(cloud, self._sun_near[:sky_rows] * 1.5, out=m[9, :sky_rows])

        # Static layers.
        m[0] = self._sky_w
        m[1] = self._sky_a
        m[2] = self._sky_b
        m[10] = self._reflect
        np.multiply(self._sun, 1.0 - 0.7 * m[8], out=m[11])  # clouds veil the sun

        pal = mix_palette(WARM, COOL, self.mood)
        return compose(
            [
                pal["sky_top"],
                pal["sky_mid"] - pal["sky_top"],
                pal["sky_horizon"] - pal["sky_mid"],
                pal["deep"],
                pal["shallow"] - pal["deep"],
                pal["face"] * 0.6,
                pal["crest"] * 0.5,
                pal["sun"],
                pal["cloud"] - pal["sky_mid"] * 0.3,
                pal["cloud_lit"] * 0.5,
                pal["reflect"] * 0.7,
                pal["sun"],
            ],
            m,
        )
