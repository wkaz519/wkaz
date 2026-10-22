"""Nebula: slow parallax starfield over a drifting, swirling colour cloud.

The cloud is domain-warped fractal noise: a 5-octave field is resampled at
positions displaced by two further noise fields, which bends it into the
swirls of a nebula. Ridged noise (``1 - |n|``) of a second warped field
adds thin bright filaments and dark dust lanes, a low-frequency field
picks between two hues, and the densest parts glow.

Exposure is kept steady by partially standardizing the density field each
frame; otherwise the lowest octave would swing the whole screen between
empty and blown out over the minutes.

The "camera" drifts along a smooth, analytic path, so every layer's offset
is a pure function of time: the nebula moves slowest, three star layers
progressively faster (parallax). Occasional faint meteors are scheduled by
hashing time slots, so they're deterministic too.

The cloud barely moves (well under a pixel per second), so it is rendered
as keyframes 5 times a second and interpolated (see ``FieldScene``).
"""

from __future__ import annotations

import math

import numpy as np
import pygame

from ..noise import bilinear_sample, fbm_grid, lattice_values
from .base import FieldScene, mix_palette, smoothstep
from .stars import StarLayer

# Warm: magenta and violet gas with golden cores. Cool: teal and blue gas
# with icy cores. Gold/orange only ever appears where it's bright, since
# dim orange reads as muddy brown.
WARM = {
    "space": (6, 3, 15),
    "haze": (24, 9, 36),
    "cloud_a": (214, 52, 134),
    "cloud_b": (118, 52, 196),
    "core": (255, 196, 128),
    "filament": (255, 160, 190),
}
COOL = {
    "space": (2, 4, 14),
    "haze": (6, 16, 38),
    "cloud_a": (30, 150, 204),
    "cloud_b": (92, 72, 216),
    "core": (205, 242, 255),
    "filament": (150, 220, 255),
}

STAR_TINTS = ((255, 255, 255), (190, 210, 255), (255, 228, 196), (210, 200, 255))
NEBULA_SCALE = 2.0  # noise features per screen height
WARP = 0.14  # domain warp strength (screen heights)
METEOR_SLOT_S = 75.0


def camera(t: float) -> tuple[float, float]:
    """Camera displacement in screen heights: a steady drift plus gentle
    sways with incommensurate periods (never retraces its path)."""
    x = 0.0045 * t + 0.05 * math.sin(t / 97.0) + 0.03 * math.sin(t / (61.0 * math.sqrt(2)))
    y = 0.06 * math.sin(t / 113.0) + 0.035 * math.sin(t / (73.0 * math.sqrt(3)))
    return x, y


class NebulaScene(FieldScene):
    key = "nebula"
    title = "Starfield Nebula"
    downscale = 5.0
    max_internal_height = 216
    keyframe_interval = 0.2

    def __init__(self, size, seed=0):
        self.layers = [
            # (layer, parallax factor)
            (StarLayer(seed + 31, 520, (0.45, 0.9), (0.2, 0.7), STAR_TINTS, 0.35), 1.0),
            (StarLayer(seed + 32, 150, (0.7, 1.4), (0.35, 0.95), STAR_TINTS, 0.4), 2.2),
            (StarLayer(seed + 33, 28, (1.2, 2.2), (0.5, 1.0), STAR_TINTS, 0.3), 5.0),
        ]
        super().__init__(size, seed)

    def on_resize(self) -> None:
        rw, rh = self.res
        self._margin = int(math.ceil(WARP * 2.2 * rh)) + 2
        self._px = np.arange(rw, dtype=np.float32)[None, :]
        self._py = np.arange(rh, dtype=np.float32)[:, None]
        u, v = self.u, self.v
        cx = self.aspect / 2
        self._vignette = (1.0 - 0.25 * ((u - cx) / max(cx, 1e-6)) ** 2 - 0.2 * ((v - 0.5) / 0.5) ** 2).astype(np.float32)
        self._masks = np.empty((6, rh, rw), dtype=np.float32)
        for layer, _ in self.layers:
            layer.layout(self.size, wrap=1.3)

    def compute_masks(self, t: float) -> np.ndarray:
        rw, rh = self.res
        m = self._margin
        cam_x, cam_y = camera(t)
        # The nebula is far away: it moves at a fraction of the star speed.
        ox, oy = 0.35 * cam_x * NEBULA_SCALE, 0.35 * cam_y * NEBULA_SCALE
        k = NEBULA_SCALE / rh  # lattice units per internal pixel
        xs = (np.arange(rw, dtype=np.float64) + 0.5) * k + ox
        ys = (np.arange(rh, dtype=np.float64) + 0.5) * k + oy
        tt = t / 110.0

        # Domain warp: two low-frequency fields displace where we sample.
        warp_x = fbm_grid(xs * 0.6, ys * 0.6, tt * 0.8, self.seed + 41, octaves=2)
        warp_y = fbm_grid(xs * 0.6 + 19.1, ys * 0.6 - 7.3, tt * 0.8, self.seed + 42, octaves=2)
        shift = np.float32(WARP * rh)
        px = self._px + m + warp_x * shift
        py = self._py + m + warp_y * shift

        # Fields on a grid with a margin, so warped lookups stay inside.
        xs_m = (np.arange(-m, rw + m, dtype=np.float64) + 0.5) * k + ox
        ys_m = (np.arange(-m, rh + m, dtype=np.float64) + 0.5) * k + oy
        density = bilinear_sample(fbm_grid(xs_m, ys_m, tt, self.seed + 43, octaves=5, gain=0.55), px, py)
        detail = bilinear_sample(fbm_grid(xs_m * 2.1, ys_m * 2.1, tt * 1.4, self.seed + 44, octaves=3), px, py)

        # Partial per-frame standardization keeps the exposure steady.
        density = 0.4 * density + 0.6 * (density - density.mean()) / (density.std() + 1e-3)
        body = smoothstep(-0.7, 2.6, density)
        ridge = 1.0 - np.minimum(np.abs(detail), 1.0)  # 1 on the zero-crossings of detail
        filaments = ridge**8 * body
        # Dust lanes darken the cloud along a different set of ridges.
        lanes = smoothstep(0.0, 0.5, np.abs(detail - 0.6))
        body *= 0.45 + 0.55 * lanes
        hue = smoothstep(-0.55, 0.55, fbm_grid(xs * 0.4 - 3.3, ys * 0.4 + 5.1, tt * 0.6, self.seed + 45, octaves=2))

        vig = self._vignette
        out = self._masks
        out[0] = vig
        np.multiply(body, vig, out=out[1])
        out[2] = out[1]
        out[1] *= 1.0 - hue
        out[2] *= hue
        core = smoothstep(0.45, 0.95, body)
        np.multiply(core, core, out=out[3])
        out[3] *= vig
        np.multiply(filaments, vig, out=out[4])
        # Faint broad haze so dark regions aren't flat black.
        np.multiply(smoothstep(-2.0, 1.5, density), vig, out=out[5])
        return out

    def colors(self) -> list:
        pal = mix_palette(WARM, COOL, self.mood)
        return [pal["space"], pal["cloud_a"] * 0.9, pal["cloud_b"] * 0.9, pal["core"] * 0.6, pal["filament"] * 0.35, pal["haze"]]

    def draw(self, surface: pygame.Surface) -> None:
        super().draw(surface)
        cam_x, cam_y = camera(self.t)
        for layer, parallax in self.layers:
            layer.draw(surface, self.t, offset=(-cam_x * parallax, -cam_y * parallax))
        self._draw_meteors(surface)

    # ------------------------------------------------------------ meteors
    def _draw_meteors(self, surface: pygame.Surface) -> None:
        t = self.t
        w, h = surface.get_size()
        slot = math.floor(t / METEOR_SLOT_S)
        for k in (slot - 1, slot):
            r = (lattice_values(self.seed + 50, k, np.arange(7)) + 1.0) / 2.0
            if r[0] > 0.7:  # most slots get a meteor, some stay quiet
                continue
            start = k * METEOR_SLOT_S + r[1] * (METEOR_SLOT_S - 3.0)
            dur = 0.8 + 0.7 * r[2]
            p = (t - start) / dur
            if not 0.0 <= p <= 1.0:
                continue
            angle = math.radians(200 + 50 * r[3])  # heading down-left
            length = h * (0.25 + 0.2 * r[4])
            x0 = w * (0.35 + 0.6 * r[5])
            y0 = h * (0.05 + 0.4 * r[6])
            dx, dy = math.cos(angle), -math.sin(angle)
            head = (x0 + dx * length * p, y0 + dy * length * p)
            tail_len = h * 0.11 * math.sin(math.pi * min(1.0, p * 1.3)) + 2
            fade = math.sin(math.pi * p) ** 0.7 * 0.8
            segments = 10
            for i in range(segments):
                a = i / segments
                b = (i + 1) / segments
                c = int(255 * fade * (1 - a) ** 1.6)
                if c <= 0:
                    continue
                p0 = (head[0] - dx * tail_len * a, head[1] - dy * tail_len * a)
                p1 = (head[0] - dx * tail_len * b, head[1] - dy * tail_len * b)
                pygame.draw.aaline(surface, (c, c, min(255, c + 20)), p0, p1)
