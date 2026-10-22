"""Dusk: a minimal, very cheap scene and the simplest example to copy.

A three-stop vertical gradient whose colors drift slowly, plus two huge,
soft light orbs wandering on Lissajous paths. All periods are mutually
incommensurate (ratios of irrational numbers), so the combination never
lines up into a visible loop.
"""

from __future__ import annotations

import numpy as np

from .base import FieldScene, mix_palette

WARM = {
    "top": (28, 14, 48),
    "mid": (92, 34, 78),
    "bottom": (196, 92, 74),
    "orb_a": (255, 150, 90),
    "orb_b": (230, 80, 140),
}
COOL = {
    "top": (8, 18, 44),
    "mid": (24, 62, 98),
    "bottom": (70, 150, 160),
    "orb_a": (120, 200, 255),
    "orb_b": (120, 110, 230),
}


class GradientScene(FieldScene):
    key = "dusk"
    title = "Dusk Gradient"
    # Everything here is very soft, so a tiny internal buffer is plenty.
    downscale = 12.0
    max_internal_height = 120

    def render(self) -> np.ndarray:
        t = self.t
        pal = mix_palette(WARM, COOL, self.mood)
        u, v = self.u, self.v

        # Gradient stop position breathes slowly.
        mid_pos = 0.55 + 0.08 * np.sin(t / 97.0) + 0.04 * np.sin(t / (61.0 * np.sqrt(2)))
        upper = np.clip(v / mid_pos, 0.0, 1.0) ** 1.3
        lower = np.clip((v - mid_pos) / (1.0 - mid_pos), 0.0, 1.0) ** 1.1
        column = pal["top"] + (pal["mid"] - pal["top"]) * upper[None] + (pal["bottom"] - pal["mid"]) * lower[None]
        img = np.repeat(column, self.res[0], axis=2)

        for i, (name, radius) in enumerate((("orb_a", 0.55), ("orb_b", 0.45))):
            phase = self.seed * 0.37 + i * 2.1
            cx = self.aspect * (0.5 + 0.38 * np.sin(t / (83.0 + 29 * i) + phase))
            cy = 0.5 + 0.30 * np.sin(t / (71.0 * np.sqrt(3) + 17 * i) + 1.3 * phase)
            d2 = (u - cx) ** 2 + (v - cy) ** 2
            glow = np.exp(-d2 / (radius * radius)) * np.float32(0.30 + 0.08 * np.sin(t / 41.0 + i))
            img += glow[None] * pal[name]
        return img
