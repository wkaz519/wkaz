"""Scene base classes.

A scene is an animated background with three responsibilities:

* ``update(dt)`` advances its clock,
* ``draw(surface)`` paints the full surface (opaque),
* ``resize(size)`` adapts to a new window size.

Scenes are written so that the picture is a pure function of the scene
clock ``t`` (plus ``mood`` and the seed). That keeps long sessions free of
numerical drift and lets ``seek(t)`` jump straight to any moment, which the
snapshot tool uses to render t = 25 minutes without simulating it.

``mood`` is a float in [0, 1] set by the app every frame: 0 is the warm
focus palette, 1 the cool break palette. The app eases it smoothly, so
scenes simply interpolate their palettes with it.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pygame


class Scene:
    key: ClassVar[str] = ""
    title: ClassVar[str] = ""

    def __init__(self, size: tuple[int, int], seed: int = 0) -> None:
        self.seed = seed
        self.t = 0.0
        self.mood = 0.0
        self.size = (0, 0)
        self.resize(size)

    def resize(self, size: tuple[int, int]) -> None:
        self.size = (max(1, int(size[0])), max(1, int(size[1])))

    def update(self, dt: float) -> None:
        self.t += dt

    def seek(self, t: float) -> None:
        self.t = float(t)

    def draw(self, surface: pygame.Surface) -> None:
        raise NotImplementedError


class FieldScene(Scene):
    """A scene rendered with NumPy at reduced resolution and smooth-upscaled.

    Subclasses implement :meth:`render`, returning a float32 *planar*
    ``(3, h, w)`` RGB image in [0, 1] at ``self.res``. Planar layout matters
    for speed: combining an ``(h, w)`` mask with a colour then runs NumPy's
    inner loop along whole rows instead of over 3-element pixels (measured
    ~5x faster than ``(h, w, 3)`` broadcasting).

    Convenience coordinate arrays are provided, measured in units of the
    internal *height* so shapes stay round at any aspect ratio:

    * ``self.u`` – x coordinates, shape ``(1, w)``, from 0 to ``aspect``
    * ``self.v`` – y coordinates, shape ``(h, 1)``, from 0 (top) to 1 (bottom)
    """

    # Internal resolution is the window size divided by ``downscale``, but
    # never taller than ``max_internal_height`` (keeps 4K windows cheap).
    downscale: ClassVar[float] = 4.0
    max_internal_height: ClassVar[int] = 300
    min_internal_height: ClassVar[int] = 36

    def resize(self, size: tuple[int, int]) -> None:
        super().resize(size)
        w, h = self.size
        rh = int(np.clip(round(h / self.downscale), self.min_internal_height, self.max_internal_height))
        rh = min(rh, h) if h >= 8 else rh
        rw = max(1, round(rh * w / h))
        self.res = (rw, rh)
        self.aspect = rw / rh
        self.u = ((np.arange(rw, dtype=np.float32) + 0.5) / rh)[None, :]
        self.v = ((np.arange(rh, dtype=np.float32) + 0.5) / rh)[:, None]
        self._bgra = np.zeros((rh, rw, 4), dtype=np.uint8)
        self._bgra[..., 3] = 255
        # Static sub-LSB dither: hides 8-bit banding in dark gradients.
        rng = np.random.default_rng(1234)
        self._dither = (rng.random((3, rh, rw), dtype=np.float32) - rng.random((3, rh, rw), dtype=np.float32)) * 0.75
        self._scratch = np.empty((3, rh, rw), dtype=np.float32)
        self.on_resize()

    def on_resize(self) -> None:
        """Hook for subclasses to rebuild size-dependent caches."""

    def render(self) -> np.ndarray:
        raise NotImplementedError

    def draw(self, surface: pygame.Surface) -> None:
        self.blit_scaled(surface, self.render())

    def to_surface(self, rgb: np.ndarray) -> pygame.Surface:
        s = self._scratch
        np.multiply(rgb, 255.0, out=s)
        s += self._dither
        np.clip(s, 0.0, 255.0, out=s)
        self._bgra[..., 0] = s[2]
        self._bgra[..., 1] = s[1]
        self._bgra[..., 2] = s[0]
        low = pygame.image.frombuffer(self._bgra, self.res, "BGRA")
        # Converting to the display's pixel format lets smoothscale take its
        # fast path (and drops the alpha channel we don't need).
        return low.convert() if pygame.display.get_surface() is not None else low.copy()

    def blit_scaled(self, surface: pygame.Surface, rgb: np.ndarray) -> None:
        low = self.to_surface(rgb)
        size = surface.get_size()
        try:
            pygame.transform.smoothscale(low, size, surface)
        except ValueError:  # destination format mismatch: scale then blit
            surface.blit(pygame.transform.smoothscale(low, size), (0, 0))


# ---------------------------------------------------------------- helpers


def lerp(a, b, t):
    return a + (b - a) * t


def mix_palette(warm: dict, cool: dict, mood: float) -> dict:
    """Interpolate two palettes ``{name: (r, g, b)}`` given as 0-255 ints.

    Colours come back as float32 ``(3, 1, 1)`` arrays in [0, 1], ready to
    broadcast against planar ``(h, w)`` masks: ``pal["sky"] * mask[None]``.
    """
    m = np.float32(mood)
    out = {}
    for k in warm:
        a = np.asarray(warm[k], np.float32) / 255.0
        b = np.asarray(cool[k], np.float32) / 255.0
        out[k] = (a + (b - a) * m)[:, None, None]
    return out


def smoothstep(edge0, edge1, x):
    t = np.asarray((x - edge0) * (1.0 / (edge1 - edge0)))
    np.clip(t, 0.0, 1.0, out=t)
    out = t * t
    t *= -2.0
    t += 3.0
    out *= t
    return out


def ramp(edge0, edge1, x):
    """Linear 0→1 ramp; a cheaper smoothstep for edges only ~1px wide."""
    t = np.asarray((x - edge0) * (1.0 / (edge1 - edge0)))
    return np.clip(t, 0.0, 1.0, out=t)


def compose(colors, masks: np.ndarray) -> np.ndarray:
    """Mix colours by weight masks in one BLAS call.

    ``img = sum_k colors[k] * masks[k]`` for ``K`` colours ``(r, g, b)`` and
    masks of shape ``(K, h, w)``, returned planar ``(3, h, w)``. Scene
    shading that is linear in its colours (gradients, tints, glows) can be
    factored this way: a colour that varies with height becomes two
    masks, ``m`` and ``m * v``. This replaces many full-colour array passes
    (memory-bound) with single-channel ones plus a tiny matrix product.
    """
    k, h, w = masks.shape
    c = np.empty((3, k), dtype=np.float32)
    for i, col in enumerate(colors):
        c[:, i] = np.asarray(col, dtype=np.float32).ravel()
    return (c @ masks.reshape(k, h * w)).reshape(3, h, w)


def vertical_gradient(v: np.ndarray, top, bottom) -> np.ndarray:
    """Planar ``(3, h, 1)`` gradient from ``top`` to ``bottom`` colours."""
    top = np.asarray(top, np.float32)[:, None, None]
    bottom = np.asarray(bottom, np.float32)[:, None, None]
    return top + (bottom - top) * v[None]
