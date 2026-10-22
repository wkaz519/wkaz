"""End-to-end smoke test of the App driven by synthetic input events."""

import json

import pygame
import pytest

from pomodoro.app import App, QualityGovernor
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


def run_governor(gov, work_ms, seconds, fps=60):
    changes = 0
    for _ in range(int(seconds * fps)):
        changes += gov.observe(work_ms / 1000, 1 / fps, 1 / fps)
    return changes


def test_governor_degrades_on_sustained_overload_then_recovers():
    gov = QualityGovernor()
    run_governor(gov, 25, 1.5)  # warm-up grace period: no reaction yet
    assert gov.quality == 1.0
    run_governor(gov, 25, 6)
    assert gov.quality < 1.0
    lowered = gov.quality
    run_governor(gov, 12, 8)  # within budget but not idle: hold steady
    assert gov.quality == lowered
    run_governor(gov, 4, 60)  # lots of headroom: climb back to full
    assert gov.quality == 1.0


def test_governor_ignores_brief_spikes_and_respects_floor():
    gov = QualityGovernor()
    run_governor(gov, 5, 3)
    for _ in range(10):  # a hitch now and then (e.g. nebula keyframes)
        run_governor(gov, 40, 0.1)
        run_governor(gov, 5, 1)
    assert gov.quality == 1.0
    run_governor(gov, 200, 120)
    assert gov.quality == pytest.approx(QualityGovernor.MIN_QUALITY, abs=0.08)
    assert gov.quality >= QualityGovernor.MIN_QUALITY


def test_lower_quality_shrinks_internal_resolution(app):
    scene = app.scene
    full = scene.res
    app.governor.quality = 0.6
    scene.set_quality(0.6)
    assert scene.res[1] < full[1]
    frame(app)
