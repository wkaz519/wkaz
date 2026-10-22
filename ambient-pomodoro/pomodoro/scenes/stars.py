"""Crisp star sprites drawn at full window resolution.

Stars are a few pixels wide, so rendering them in a scene's low-resolution
buffer would smear them into blobs. Instead they're blitted additively on
top of the upscaled image from a small cache of pre-rendered sprites (a few
sizes × brightness levels × tints), which costs ~1µs per star.
"""

from __future__ import annotations

import math

import numpy as np
import pygame

LEVELS = 10  # brightness quantization steps for twinkling
_PHI = 0.6180339887


def _star_sprite(radius: float, brightness: float, tint) -> pygame.Surface:
    """Soft round star: gaussian core plus a faint halo, black background
    (blitted with BLEND_ADD, so black is transparent)."""
    size = max(3, math.ceil(radius * 6)) | 1
    c = (size - 1) / 2
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d2 = (xx - c) ** 2 + (yy - c) ** 2
    core = np.exp(-d2 / (2 * max(radius * 0.6, 0.35) ** 2))
    halo = 0.12 * np.exp(-d2 / (2 * (radius * 1.8) ** 2))
    a = np.clip((core + halo) * brightness, 0.0, 1.0)
    rgb = (a[..., None] * np.asarray(tint, np.float32)[None, None, :]).astype(np.uint8)
    surf = pygame.Surface((size, size))
    pygame.surfarray.blit_array(surf, rgb.transpose(1, 0, 2))
    return surf


class StarLayer:
    """A population of stars with sizes, tints and independent twinkle.

    Positions are generated in normalized units so the layer can be laid
    out again for any window size without changing the star pattern.
    """

    def __init__(
        self,
        seed: int,
        density: float,
        radius: tuple[float, float],
        brightness: tuple[float, float],
        tints=((255, 255, 255),),
        twinkle: float = 0.35,
    ) -> None:
        self.seed = seed
        self.density = density  # stars per (screen height)²
        self.radius_range = radius
        self.brightness_range = brightness
        self.tints = tints
        self.twinkle = twinkle
        self.size = (0, 0)
        self.positions = np.zeros((0, 2), np.float32)

    def layout(self, size: tuple[int, int], wrap: float = 1.0) -> None:
        """Place stars for a window of ``size`` pixels.

        ``wrap`` > 1 makes the star field larger than the window so it can
        scroll (parallax) and wrap around without visible seams.
        """
        w, h = size
        self.size = size
        aspect = w / h
        rng = np.random.default_rng(self.seed)
        n = int(self.density * aspect * wrap * wrap)
        self.field = (aspect * wrap, wrap)
        self.pos_norm = rng.random((n, 2)) * np.array(self.field)
        scale = h / 1080.0
        self.radius = rng.uniform(*self.radius_range, n) * max(0.5, scale)
        base = rng.uniform(0, 1, n) ** 2.2  # most stars faint, few bright
        lo, hi = self.brightness_range
        self.base = lo + (hi - lo) * base
        self.tint_index = rng.integers(0, len(self.tints), n)
        self.freq = rng.uniform(0.4, 2.2, n)
        self.phase = rng.uniform(0, 2 * np.pi, n)
        self.tw_amount = rng.uniform(0.2, 1.0, n) * self.twinkle
        # Bucket radii into a few sprite sizes.
        sizes = np.unique(np.round(self.radius * 2) / 2)
        self.size_values = np.maximum(sizes, 0.5)
        self.size_index = np.abs(self.radius[:, None] - self.size_values[None, :]).argmin(axis=1)
        self._sprites: dict = {}

    def sprite(self, size_i: int, level: int, tint_i: int) -> pygame.Surface:
        key = (size_i, level, tint_i)
        s = self._sprites.get(key)
        if s is None:
            # One gaussian per size and tint at full brightness; dimmer
            # levels are cheap multiplied copies of it.
            base = self._sprites.get((size_i, LEVELS - 1, tint_i))
            if base is None:
                base = _star_sprite(float(self.size_values[size_i]), 1.0, self.tints[tint_i])
                self._sprites[(size_i, LEVELS - 1, tint_i)] = base
            if level == LEVELS - 1:
                return base
            s = base.copy()
            k = round(255 * (level + 1) / LEVELS)
            s.fill((k, k, k), special_flags=pygame.BLEND_MULT)
            self._sprites[key] = s
        return s

    def draw(
        self,
        surface: pygame.Surface,
        t: float,
        offset: tuple[float, float] = (0.0, 0.0),
        visible=None,
        gain: float = 1.0,
    ) -> None:
        """Blit the stars, scrolled by ``offset`` (in screen heights).

        ``visible`` optionally maps normalized (x, y) arrays to a [0, 1]
        brightness factor (e.g. to hide stars behind mountains).
        """
        if len(self.pos_norm) == 0:
            return
        w, h = self.size
        fw, fh = self.field
        x = self.pos_norm[:, 0] + offset[0]
        laps = np.floor(x / fw)
        x -= laps * fw
        # Each time a star wraps around horizontally it re-enters at a new
        # height (golden-ratio steps), so scrolling layers never visibly
        # bring the same star pattern back.
        y = np.mod(self.pos_norm[:, 1] + offset[1] + _PHI * fh * laps, fh)
        on_screen = (x < w / h + 0.02) & (y < 1.02)
        b = self.base * (1.0 - self.tw_amount * (0.5 + 0.5 * np.sin(self.freq * t + self.phase))) * gain
        if visible is not None:
            b = b * visible(x, y)
        keep = on_screen & (b > 0.5 / LEVELS)
        if not keep.any():
            return
        level = np.clip((b[keep] * LEVELS).astype(int) - 1, 0, LEVELS - 1)
        si = self.size_index[keep]
        ti = self.tint_index[keep]
        px = x[keep] * h
        py = y[keep] * h
        seq = []
        for k in range(len(level)):
            spr = self.sprite(si[k], level[k], ti[k])
            half = spr.get_width() // 2
            seq.append((spr, (int(px[k]) - half, int(py[k]) - half), None, pygame.BLEND_ADD))
        surface.blits(seq, doreturn=False)
