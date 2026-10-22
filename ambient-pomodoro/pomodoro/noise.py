"""Vectorized smooth noise for the ambient scenes (NumPy only).

Value noise with cubic B-spline interpolation over a hashed lattice:

* The lattice value at integer coordinates comes from an integer hash, so
  the noise is infinite and never repeats along any axis, including time.
  (A classic 256-entry permutation table would loop.)
* B-spline weights are non-negative and sum to 1, so the result is C2
  smooth (no creases) and bounded by the lattice value range.
* Sampling an axis-aligned pixel grid is separable:
  ``noise = Wy @ L(t) @ Wx.T``, where ``Wx``/``Wy`` hold the
  interpolation weights per column/row and ``L(t)`` is the lattice slice
  interpolated in time. That turns per-pixel work into two small matrix
  multiplies, roughly 10-50x faster than per-pixel Perlin noise in NumPy.

All public functions return values normalized to roughly unit standard
deviation (the hard bound is about ±2.5 for 1D, ±5 for 3D).
"""

from __future__ import annotations

import numpy as np

_M1 = np.uint32(0x7FEB352D)
_M2 = np.uint32(0x846CA68B)
_PRIMES = (np.uint32(0x9E3779B1), np.uint32(0x85EBCA77), np.uint32(0xC2B2AE3D), np.uint32(0x27D4EB2F))


def _mix32(x: np.ndarray) -> np.ndarray:
    """lowbias32 integer finalizer (Chris Wellons); good avalanche, cheap."""
    x = x ^ (x >> np.uint32(16))
    x = x * _M1
    x = x ^ (x >> np.uint32(15))
    x = x * _M2
    return x ^ (x >> np.uint32(16))


def hash_coords(seed: int, *coords) -> np.ndarray:
    """Hash integer coordinate arrays (broadcast together) to uint32."""
    with np.errstate(over="ignore"):
        h = _mix32(np.asarray(seed & 0xFFFFFFFF, dtype=np.uint32))
        for axis, c in enumerate(coords):
            c = np.asarray(c, dtype=np.int64).astype(np.uint32)
            h = _mix32(h ^ (c * _PRIMES[axis % 4]))
    return h


def lattice_values(seed: int, *coords) -> np.ndarray:
    """Pseudo-random float32 values in [-1, 1) at integer coordinates."""
    h = hash_coords(seed, *coords)
    return (h >> np.uint32(8)).astype(np.float32) * np.float32(2.0 / (1 << 24)) - np.float32(1.0)


def bspline_weights(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Uniform cubic B-spline weights for lattice offsets -1, 0, +1, +2."""
    f2 = f * f
    f3 = f2 * f
    w0 = (1.0 - f) ** 3 / 6.0
    w1 = (3.0 * f3 - 6.0 * f2 + 4.0) / 6.0
    w2 = (-3.0 * f3 + 3.0 * f2 + 3.0 * f + 1.0) / 6.0
    w3 = f3 / 6.0
    return w0, w1, w2, w3


def _spline_variance_factor() -> float:
    f = np.linspace(0.0, 1.0, 4001)
    return float(np.mean(sum(w * w for w in bspline_weights(f))))


# Variance of uniform[-1, 1] lattice values is 1/3; interpolation shrinks it
# by this factor per dimension. Used to normalize outputs to unit std.
_C1 = _spline_variance_factor()
_STD_1D = float(np.sqrt(_C1 / 3.0))
_STD_3D = float(np.sqrt(_C1**3 / 3.0))


def noise1(x, seed: int = 0) -> np.ndarray:
    """Smooth 1D noise; ``x`` may be a scalar or array (per-element seeds ok)."""
    x = np.asarray(x, dtype=np.float64)
    i = np.floor(x)
    f = x - i
    i = i.astype(np.int64)
    out = np.zeros(np.broadcast(x, np.asarray(seed)).shape, dtype=np.float64)
    for k, w in enumerate(bspline_weights(f)):
        out += w * _lattice_1d(seed, i + (k - 1))
    return out / _STD_1D


def _lattice_1d(seed, i):
    seed = np.asarray(seed)
    if seed.ndim == 0:
        return lattice_values(int(seed), i)
    # Vector of seeds: hash each seed separately, then mix in the coordinate.
    with np.errstate(over="ignore"):
        h = _mix32(seed.astype(np.int64).astype(np.uint32))
        h = _mix32(h ^ (np.asarray(i, dtype=np.int64).astype(np.uint32) * _PRIMES[0]))
    return (h >> np.uint32(8)).astype(np.float32) * np.float32(2.0 / (1 << 24)) - np.float32(1.0)


def _weight_matrix(coords: np.ndarray) -> tuple[np.ndarray, int]:
    """Sparse-as-dense B-spline weight matrix for 1D sample positions.

    Returns ``(M, origin)`` where ``M[j, k]`` weights lattice index
    ``origin + k`` for sample ``j``.
    """
    coords = np.asarray(coords, dtype=np.float64).ravel()
    i = np.floor(coords)
    f = (coords - i).astype(np.float32)
    i = i.astype(np.int64)
    origin = int(i.min()) - 1
    n = int(i.max()) + 3 - origin
    m = np.zeros((coords.size, n), dtype=np.float32)
    rows = np.arange(coords.size)
    base = i - 1 - origin
    for k, w in enumerate(bspline_weights(f)):
        m[rows, base + k] = w
    return m, origin


_slice_cache: dict[tuple[int, int, int, int, int, int], np.ndarray] = {}


def _lattice_slice(seed: int, ox: int, oy: int, nx: int, ny: int, kt: int) -> np.ndarray:
    """Lattice values for one integer time step, cached because consecutive
    frames almost always reuse the same few slices."""
    key = (seed, ox, oy, nx, ny, kt)
    lat = _slice_cache.get(key)
    if lat is None:
        if len(_slice_cache) > 512:
            _slice_cache.clear()
        ix = np.arange(ox, ox + nx, dtype=np.int64)[None, :]
        iy = np.arange(oy, oy + ny, dtype=np.int64)[:, None]
        lat = _slice_cache[key] = lattice_values(seed, ix, iy, kt)
    return lat


def grid_noise(xs, ys, t: float, seed: int = 0) -> np.ndarray:
    """One octave of 3D noise sampled on the grid ``xs × ys`` at time ``t``.

    ``xs``/``ys`` are 1D arrays of lattice-space coordinates (a lattice cell
    is one feature size). Returns float32 of shape ``(len(ys), len(xs))``.
    """
    wx, ox = _weight_matrix(xs)
    wy, oy = _weight_matrix(ys)
    kt = int(np.floor(t))
    ft = np.float32(t - kt)
    lat = np.zeros((wy.shape[1], wx.shape[1]), dtype=np.float32)
    for k, w in enumerate(bspline_weights(ft)):
        lat += np.float32(w) * _lattice_slice(seed, ox, oy, wx.shape[1], wy.shape[1], kt + k - 1)
    out = (wy @ lat) @ wx.T
    out *= np.float32(1.0 / _STD_3D)
    return out


def fbm_grid(
    xs,
    ys,
    t: float,
    seed: int = 0,
    octaves: int = 4,
    lacunarity: float = 2.03,
    gain: float = 0.5,
    time_lacunarity: float = 1.6,
) -> np.ndarray:
    """Fractal sum of :func:`grid_noise` octaves, normalized to ~unit std.

    Each octave is finer by ``lacunarity`` and evolves faster by
    ``time_lacunarity``. Non-integer lacunarity plus a per-octave offset
    keeps the lattices from lining up into a visible grid.
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    total = None
    amp, freq, tf, norm = 1.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        off = 17.31 * o
        n = grid_noise(xs * freq + off, ys * freq - off * 0.7, t * tf + off * 0.37, seed + 1013 * o)
        total = n * np.float32(amp) if total is None else total + n * np.float32(amp)
        norm += amp * amp
        amp *= gain
        freq *= lacunarity
        tf *= time_lacunarity
    return total * np.float32(1.0 / np.sqrt(norm))


def bilinear_sample(field: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Sample a 2D field at fractional pixel positions (edges clamped).

    Used for domain warping, where sample positions are no longer an
    axis-aligned grid.
    """
    h, w = field.shape
    px = np.clip(px, 0.0, w - 1.001)
    py = np.clip(py, 0.0, h - 1.001)
    x0 = px.astype(np.int32)
    y0 = py.astype(np.int32)
    fx = (px - x0).astype(np.float32)
    fy = (py - y0).astype(np.float32)
    flat = field.ravel()
    idx = y0 * w + x0
    a = flat.take(idx)
    b = flat.take(idx + 1)
    c = flat.take(idx + w)
    d = flat.take(idx + w + 1)
    top = a + (b - a) * fx
    bottom = c + (d - c) * fx
    return top + (bottom - top) * fy
