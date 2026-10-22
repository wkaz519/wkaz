import numpy as np
import pytest

from pomodoro.noise import bilinear_sample, bspline_weights, fbm_grid, grid_noise, noise1


def test_bspline_weights_partition_of_unity():
    f = np.linspace(0, 1, 101)
    w = bspline_weights(f)
    assert np.allclose(sum(w), 1.0)
    assert all((wi >= -1e-12).all() for wi in w)


def test_noise1_deterministic_smooth_and_normalized():
    x = np.linspace(0, 2000, 200_001)
    a = noise1(x, seed=5)
    assert np.array_equal(a, noise1(x, seed=5))
    assert not np.allclose(a, noise1(x, seed=6))
    assert 0.8 < a.std() < 1.2
    assert abs(a.mean()) < 0.1
    assert np.abs(np.diff(a)).max() < 0.05  # Lipschitz: slope stays below ~5 per unit
    assert np.abs(a).max() < 2.6


def test_noise1_vector_seeds_match_scalar_seeds():
    x = np.array([0.3, 12.7, 99.1])
    seeds = np.array([1, 2, 3])
    expected = [noise1(xi, int(s)) for xi, s in zip(x, seeds)]
    assert np.allclose(noise1(x, seeds), expected)


def test_noise1_does_not_repeat():
    """No short period: far-apart stretches are uncorrelated."""
    x = np.linspace(0, 300, 3001)
    a, b = noise1(x, 1), noise1(x + 256.0, 1)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.3


def test_grid_noise_shape_and_normalization():
    xs = np.linspace(0, 40, 400)
    ys = np.linspace(0, 20, 200)
    n = grid_noise(xs, ys, 3.3, seed=1)
    assert n.shape == (200, 400) and n.dtype == np.float32
    assert 0.6 < n.std() < 1.4


def test_grid_noise_depends_only_on_absolute_coordinates():
    """The lattice window chosen for a grid must not change the values."""
    xs = np.linspace(0, 10, 101)
    ys = np.linspace(-3, 4, 71)
    full = grid_noise(xs, ys, 7.25, seed=9)
    part = grid_noise(xs[40:60], ys[10:30], 7.25, seed=9)
    assert np.allclose(full[10:30, 40:60], part, atol=1e-5)


def test_grid_noise_continuous_in_time():
    xs = np.linspace(0, 8, 64)
    ys = np.linspace(0, 8, 64)
    a = grid_noise(xs, ys, 4.999, seed=2)
    b = grid_noise(xs, ys, 5.001, seed=2)
    assert np.abs(a - b).max() < 0.02


def test_fbm_grid_normalized():
    xs = np.linspace(0, 6, 300)
    ys = np.linspace(0, 3, 150)
    f = fbm_grid(xs, ys, 1.0, seed=3, octaves=5)
    assert 0.4 < f.std() < 1.4
    assert np.isfinite(f).all()


def test_bilinear_sample_exact_at_pixels_and_interpolates():
    field = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert bilinear_sample(field, np.array([2.0]), np.array([1.0]))[0] == pytest.approx(6.0)
    assert bilinear_sample(field, np.array([0.5]), np.array([0.5]))[0] == pytest.approx(2.5)
    # Clamped outside the field.
    assert bilinear_sample(field, np.array([-5.0]), np.array([-5.0]))[0] == pytest.approx(0.0)
