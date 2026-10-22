"""Smoke tests for every registered scene (rendered headless at small sizes)."""

import numpy as np
import pygame
import pytest

from pomodoro.scenes import SCENES, get_scene_class, next_scene_key, scene_keys

SIZE = (320, 180)


@pytest.fixture(scope="module")
def screen():
    pygame.display.init()
    surf = pygame.display.set_mode(SIZE)
    yield surf
    pygame.display.quit()


def render(scene, surface, t, mood=0.0):
    scene.seek(t)
    scene.mood = mood
    scene.draw(surface)
    return pygame.surfarray.array3d(surface).astype(np.float32)


def test_registry_keys_unique_and_titled():
    keys = scene_keys()
    assert len(keys) == len(set(keys)) == len(SCENES)
    assert all(cls.title for cls in SCENES)


def test_registry_lookup_and_cycling():
    first, *_ = scene_keys()
    assert get_scene_class("does-not-exist").key == first
    assert next_scene_key(first, len(SCENES)) == first
    assert next_scene_key("does-not-exist") == first


@pytest.mark.parametrize("cls", SCENES, ids=lambda c: c.key)
def test_scene_renders_non_trivial_image(cls, screen):
    img = render(cls(SIZE, seed=1), screen, 30.0)
    assert img.std() > 2.0, "image is (nearly) a flat color"
    assert img.max() > 30, "image is (nearly) black"


@pytest.mark.parametrize("cls", SCENES, ids=lambda c: c.key)
def test_scene_changes_over_time(cls, screen):
    scene = cls(SIZE, seed=1)
    a = render(scene, screen, 60.0)
    b = render(scene, screen, 600.0)
    assert np.abs(a - b).mean() > 3.0


@pytest.mark.parametrize("cls", SCENES, ids=lambda c: c.key)
def test_scene_is_deterministic_function_of_time(cls, screen):
    a = render(cls(SIZE, seed=3), screen, 123.4)
    scene = cls(SIZE, seed=3)
    for _ in range(10):
        scene.update(1 / 30)
    scene.seek(123.4)
    b = render(scene, screen, 123.4)
    assert np.abs(a - b).max() <= 1.0


@pytest.mark.parametrize("cls", SCENES, ids=lambda c: c.key)
def test_mood_shifts_palette(cls, screen):
    scene = cls(SIZE, seed=1)
    warm = render(scene, screen, 100.0, mood=0.0)
    cool = render(scene, screen, 100.0, mood=1.0)
    # Cool palette has relatively more blue than red.
    warm_bias = warm[..., 2].mean() - warm[..., 0].mean()
    cool_bias = cool[..., 2].mean() - cool[..., 0].mean()
    assert cool_bias > warm_bias + 2.0


@pytest.mark.parametrize("cls", SCENES, ids=lambda c: c.key)
def test_scene_survives_resize_to_odd_sizes(cls):
    scene = cls(SIZE, seed=1)
    for size in [(101, 333), (640, 120), (64, 48), SIZE]:
        scene.resize(size)
        target = pygame.Surface(size)
        scene.draw(target)
        assert target.get_size() == size


@pytest.mark.parametrize("cls", SCENES, ids=lambda c: c.key)
def test_scene_handles_long_sessions(cls, screen):
    """Hours in, values stay finite and the picture is still alive."""
    img = render(cls(SIZE, seed=2), screen, 4 * 3600.0)
    assert np.isfinite(img).all() and img.std() > 2.0
