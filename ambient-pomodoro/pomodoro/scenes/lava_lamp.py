"""Lava lamp: metaball wax that rises, merges, splits and sinks.

Each blob follows its own slow rise/sink cycle. The periods are all
different and the phase is perturbed by smooth noise, so blobs hesitate,
overtake each other and never fall back into a repeating pattern. Wax pools
rest just below the bottom edge (and thinly above the top), so rising blobs
visibly bud off the bottom pool and merge into the top one.

The field uses a compact Wyvill kernel ``(1 - d²/R²)³`` rather than the
classic ``1/d²``: nearby blobs fuse with a smooth neck that thins and snaps
as they separate, and distant blobs don't influence each other at all,
which also lets every blob be splatted into just its bounding box.
"""

from __future__ import annotations

import numpy as np

from ..noise import noise1
from .base import FieldScene, compose, mix_palette, ramp, smoothstep

WARM = {
    "bg_top": (12, 3, 24),
    "bg_bottom": (46, 8, 46),
    "lamp": (255, 84, 36),
    "wax_hot": (255, 124, 34),
    "wax_cool": (222, 34, 128),
    "core": (255, 214, 140),
    "halo": (255, 60, 100),
}
COOL = {
    "bg_top": (3, 7, 24),
    "bg_bottom": (8, 24, 52),
    "lamp": (40, 150, 255),
    "wax_hot": (38, 210, 196),
    "wax_cool": (104, 80, 232),
    "core": (196, 255, 244),
    "halo": (70, 130, 255),
}

THRESHOLD = 0.22  # iso-level of the summed field that forms the wax surface
# For a lone blob the surface sits at this fraction of the kernel radius.
_SURFACE_FRACTION = float(np.sqrt(1.0 - THRESHOLD ** (1.0 / 3.0)))

N_BLOBS = 40  # spread over a wide strip; only the visible ones cost anything
N_POOL = 18
STRIP = 3.9  # half-width of the strip (in screen heights) blobs live in


class LavaLampScene(FieldScene):
    key = "lava_lamp"
    title = "Lava Lamp"
    downscale = 3.0
    max_internal_height = 360

    def __init__(self, size, seed=0):
        rng = np.random.default_rng(seed)
        # Blob anchors are fixed in "screen height" units around the centre,
        # so resizing reveals or hides blobs at the edges instead of
        # reshuffling the whole lamp.
        spacing = 2 * STRIP / N_BLOBS
        self.offset = -STRIP + (np.arange(N_BLOBS) + 0.5) * spacing + rng.uniform(-0.4, 0.4, N_BLOBS) * spacing
        self.radius = rng.uniform(0.055, 0.13, N_BLOBS)  # visible radius
        self.period = rng.uniform(85.0, 170.0, N_BLOBS)  # seconds per rise + sink
        self.phase = rng.uniform(0.0, 2 * np.pi, N_BLOBS)
        self.travel = rng.uniform(0.55, 1.0, N_BLOBS)  # fraction of the height
        self.wander = rng.uniform(0.03, 0.10, N_BLOBS)
        self.blob_seed = rng.integers(0, 2**31 - 1, N_BLOBS)
        # Lumpy shapes: each blob carries a smaller satellite orbiting slowly.
        self.sat_period = rng.uniform(25.0, 60.0, N_BLOBS) * rng.choice([-1, 1], N_BLOBS)
        self.sat_size = rng.uniform(0.45, 0.7, N_BLOBS)
        pool_spacing = 2 * STRIP / N_POOL
        self.pool_offset = -STRIP + (np.arange(N_POOL) + 0.5) * pool_spacing
        self.pool_radius = rng.uniform(0.15, 0.22, N_POOL)
        self.pool_seed = rng.integers(0, 2**31 - 1, N_POOL)
        super().__init__(size, seed)

    def on_resize(self) -> None:
        u, v = self.u, self.v
        cx = self.aspect / 2
        # Light from the lamp's heater below the bottom edge.
        self._lamp = np.exp(-((u - cx) ** 2) / (0.9 * self.aspect) ** 2 - (v - 1.15) ** 2 / 0.22).astype(np.float32)
        nx = (u - cx) / max(cx, 1e-6)
        ny = (v - 0.5) / 0.5
        self._vignette = (1.0 - 0.22 * nx**2 - 0.18 * ny**2).astype(np.float32)
        self._height_mix = smoothstep(0.1, 0.95, v).astype(np.float32)
        self._field = np.zeros((self.res[1], self.res[0]), dtype=np.float32)
        # Highlight probe: how far (in pixels) toward the light to look.
        self._probe = max(1, round(self.res[1] / 120))
        self._masks = np.empty((8, self.res[1], self.res[0]), dtype=np.float32)

    # ------------------------------------------------------------ motion
    def _blob_y(self, t):
        wobble = 0.8 * noise1(t / 60.0 + self.phase, self.blob_seed)
        phi = 2 * np.pi * t / self.period + self.phase + wobble
        lift = 0.5 - 0.5 * np.cos(phi)
        lift = 0.5 * lift + 0.5 * smoothstep(0.0, 1.0, lift)  # linger a little at the ends
        return 1.04 - self.travel * 1.1 * lift

    def blobs(self, t: float):
        """Return arrays (x, y, kernel_radius, stretch) for all blobs."""
        # Evaluate t and t ± 0.5 s in one vectorized call for the velocity.
        y_prev, y, y_next = self._blob_y(np.array([[t - 0.5], [t], [t + 0.5]]))
        speed = np.abs(y_next - y_prev)
        stretch = 1.0 + np.clip(speed / 0.035, 0.0, 1.0) * 0.35  # elongate while moving
        x = self.aspect / 2 + self.offset + self.wander * noise1(t / 80.0 + 3.1, self.blob_seed + 1)
        breathe = 1.0 + 0.12 * noise1(t / 30.0 + 7.7, self.blob_seed + 2)
        kernel = self.radius * breathe / _SURFACE_FRACTION
        return x, y, kernel, stretch

    # ------------------------------------------------------------ render
    def _splat(self, cx, cy, radius, stretch=1.0) -> None:
        """Add one elliptical kernel to the field, touching only its bbox."""
        cx, cy, radius, stretch = float(cx), float(cy), float(radius), float(stretch)
        rw, rh = self.res
        sq = stretch**0.5
        hx, hy = radius / sq, radius * sq
        c0 = max(0, int((cx - hx) * rh))
        c1 = min(rw, int((cx + hx) * rh) + 2)
        r0 = max(0, int((cy - hy) * rh))
        r1 = min(rh, int((cy + hy) * rh) + 2)
        if c0 >= c1 or r0 >= r1:
            return
        inv = 1.0 / (radius * radius)
        dx = self.u[0, c0:c1] - cx
        dy = self.v[r0:r1, 0] - cy
        q = (dx * dx * (stretch * inv))[None, :] + (dy * dy * (inv / stretch))[:, None]
        k = np.maximum(1.0 - q, 0.0, dtype=np.float32)
        self._field[r0:r1, c0:c1] += k * k * k

    def build_field(self, t: float) -> np.ndarray:
        field = self._field
        field.fill(0.0)
        cx = self.aspect / 2
        # Bottom pool: big, slowly heaving lumps mostly below the edge.
        heave = noise1(t / 50.0 + self.pool_offset, self.pool_seed)
        for i in range(N_POOL):
            r = self.pool_radius[i] / _SURFACE_FRACTION
            self._splat(cx + self.pool_offset[i], 1.0 + 0.6 * r + 0.035 * heave[i], r, 1.6)
        # A thin film collects along the top.
        for i in range(0, N_POOL, 2):
            r = 0.55 * self.pool_radius[i] / _SURFACE_FRACTION
            self._splat(cx + self.pool_offset[i] + 0.1, -0.78 * r + 0.02 * heave[i], r, 0.7)

        x, y, kernel, stretch = self.blobs(t)
        ang = 2 * np.pi * t / self.sat_period + self.phase
        off = 0.62 * kernel * _SURFACE_FRACTION
        for i in range(N_BLOBS):
            self._splat(x[i], y[i], kernel[i], stretch[i])
            self._splat(
                x[i] + off[i] * np.cos(ang[i]),
                y[i] + off[i] * np.sin(ang[i]) * stretch[i],
                self.sat_size[i] * kernel[i],
                stretch[i],
            )
        return field

    def render(self) -> np.ndarray:
        f = self.build_field(self.t)
        pal = mix_palette(WARM, COOL, self.mood)
        vig = self._vignette
        hm = self._height_mix

        wax = ramp(THRESHOLD - 0.02, THRESHOLD + 0.02, f)
        depth = smoothstep(THRESHOLD, THRESHOLD + 0.55, f)
        aura = smoothstep(0.0, THRESHOLD, f)
        aura *= aura
        # Glossy highlight: where stepping toward the light (up-left) leaves
        # the wax quickly, the surface faces the light.
        k = self._probe
        facing = np.zeros_like(f)
        facing[k:, k:] = f[k:, k:] - f[:-k, :-k]
        gloss = ramp(0.015 * k, 0.07 * k, facing)
        gloss *= 1.0 - depth

        # Weight masks; colours are applied in one matrix product below.
        m = self._masks
        bg_w = np.multiply(vig, 1.0 - wax, out=m[0])  # background
        np.multiply(bg_w, self.v, out=m[1])  # ... its vertical gradient
        np.multiply(bg_w, self._lamp, out=m[2])  # ... heater glow
        wax_w = wax * vig
        # Wax is hottest (orange) near the heater and cools (magenta) as it rises.
        shade = np.multiply(depth, 0.45, out=m[3])
        shade += 0.55 + 0.30 * hm
        shade *= wax_w
        np.multiply(shade, hm, out=m[4])
        np.multiply(depth, depth, out=m[5])
        m[5] *= wax_w
        np.multiply(aura, bg_w, out=m[6])  # aura only outside the wax
        np.multiply(gloss, wax_w, out=m[7])

        return compose(
            [
                pal["bg_top"],
                pal["bg_bottom"] - pal["bg_top"],
                pal["lamp"] * 0.28,
                pal["wax_cool"],
                pal["wax_hot"] - pal["wax_cool"],
                pal["core"] * 0.42,
                pal["halo"] * 0.22,
                (0.35, 0.35, 0.35),
            ],
            m,
        )
