"""End-to-end smoke test of the App driven by synthetic input events."""

import json

import pygame
import pytest

from pomodoro.app import App
from pomodoro.scenes import scene_keys
from pomodoro.settings import Settings
from pomodoro.timer import SessionType


@pytest.fixture
def app(tmp_path):
    path = tmp_path / "settings.json"
    a = App(Settings(window_width=640, window_height=400), path, seed=0)
    yield a
    if pygame.get_init():
        pygame.quit()


def press(app, key, mod=0):
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=key, mod=mod, unicode="", scancode=0))
    frame(app)


def frame(app, dt=1 / 60):
    app.handle_events()
    app.tick_timer(dt)
    app.update(dt)
    app.draw(app.screen)


def test_keyboard_controls(app):
    assert not app.timer.running
    press(app, pygame.K_SPACE)
    assert app.timer.running
    app.tick_timer(65)
    assert app.timer.remaining == pytest.approx(25 * 60 - 65, abs=0.1)

    press(app, pygame.K_r)
    assert not app.timer.running and app.timer.remaining == 25 * 60

    press(app, pygame.K_s)
    assert app.timer.session is SessionType.SHORT_BREAK

    first = app.scene.key
    press(app, pygame.K_v)
    assert app.scene.key == scene_keys()[1 % len(scene_keys())]
    for _ in range(120):  # finish the crossfade
        frame(app)
    assert app.fading_out is None
    press(app, pygame.K_v, pygame.KMOD_SHIFT)
    assert app.scene.key == first

    muted = app.chime.muted
    press(app, pygame.K_m)
    assert app.chime.muted is not muted


def test_mood_eases_toward_break_palette(app):
    assert app.mood == 0.0
    app.timer.skip()
    for _ in range(60):
        frame(app, 0.1)
    assert 0.8 < app.mood <= 1.0


def test_resize_and_fullscreen(app):
    app.screen = pygame.display.set_mode((800, 500), pygame.RESIZABLE)
    app.on_resize()
    assert app.scene.size == (800, 500)
    press(app, pygame.K_f)
    assert app.fullscreen
    assert app.scene.size == app.screen.get_size()
    press(app, pygame.K_f)
    assert not app.fullscreen
    assert app.screen.get_size() == (800, 500)


def test_quit_persists_settings(app):
    app.timer.start()
    press(app, pygame.K_m)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    app.run()
    saved = json.loads(app.settings_path.read_text())
    assert saved["muted"] is True
    assert saved["window_width"] == 640 and saved["window_height"] == 400
    assert saved["scene"] in scene_keys()


def test_escape_quits(app):
    press(app, pygame.K_ESCAPE)
    assert not app.running
