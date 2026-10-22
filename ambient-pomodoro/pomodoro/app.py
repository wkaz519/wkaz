"""The interactive application: window, input, main loop."""

from __future__ import annotations

import logging
import math
import random
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pygame

from .audio import ChimePlayer
from .scenes import Scene, get_scene_class, next_scene_key
from .settings import Settings, save_settings
from .timer import PomodoroTimer, SessionEnded, SessionType, format_remaining
from .ui import Overlay, OverlayState

log = logging.getLogger(__name__)

SCENE_FADE_S = 1.6  # crossfade when switching visuals
MOOD_TIME_CONSTANT_S = 2.5  # palette easing between focus and break
CURSOR_HIDE_S = 2.5
MAX_SCENE_DT = 0.1  # clamp animation steps after stalls (timer uses real dt)
MIN_WINDOW = (480, 320)
# Frame rate the quality governor defends. For slow ambient visuals,
# sharpness matters more than going from 50 to 60 fps, so resolution only
# drops when a machine would fall below roughly this rate.
GOVERNOR_FPS = 45


class QualityGovernor:
    """Lowers scene render resolution when frames run over budget.

    Watches the time spent updating and drawing each frame (not the frame
    cap's sleep). Sustained use above 90% of the frame budget steps quality
    down; sustained use below 50% steps it back up. The wide gap between the
    two thresholds means one step can't flip-flop.
    """

    STEP = 0.85
    MIN_QUALITY = 0.5
    DEGRADE_AFTER_S = 3.0
    RECOVER_AFTER_S = 10.0

    def __init__(self) -> None:
        self.quality = 1.0
        self.ema: float | None = None
        self._over = 0.0
        self._under = 0.0
        self._grace = 2.0

    def hold(self, seconds: float = 2.0) -> None:
        """Ignore measurements for a while (warm-up after a change)."""
        self._grace = max(self._grace, seconds)
        self.ema = None
        self._over = self._under = 0.0

    def observe(self, work_s: float, dt: float, budget_s: float) -> bool:
        """Record one frame; returns True when ``quality`` changed."""
        if self._grace > 0:
            self._grace -= dt
            return False
        self.ema = work_s if self.ema is None else self.ema + (work_s - self.ema) * 0.1
        self._over = self._over + dt if self.ema > 0.9 * budget_s else 0.0
        self._under = self._under + dt if self.ema < 0.5 * budget_s else 0.0
        new = self.quality
        if self._over >= self.DEGRADE_AFTER_S:
            new = max(self.MIN_QUALITY, self.quality * self.STEP)
        elif self._under >= self.RECOVER_AFTER_S:
            new = min(1.0, self.quality / self.STEP)
        if abs(new - self.quality) < 1e-6:
            if self._over >= self.DEGRADE_AFTER_S or self._under >= self.RECOVER_AFTER_S:
                self._over = self._under = 0.0
            return False
        self.quality = new
        self.hold()
        return True


def set_min_window_size(size: tuple[int, int]) -> None:
    """Best effort: pygame-ce exposes this via its Window API."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pygame.Window.from_display_module().minimum_size = size
    except (AttributeError, pygame.error):
        pass


def mood_for(session: SessionType) -> float:
    """0 = warm focus palette, 1 = cool break palette."""
    return 0.0 if session is SessionType.FOCUS else 1.0


def window_icon() -> pygame.Surface:
    """A tiny generated icon: a warm glowing disc."""
    size = 64
    icon = pygame.Surface((size, size), pygame.SRCALPHA)
    for r in range(size // 2, 0, -1):
        k = r / (size / 2)
        color = (int(255 - 40 * k), int(120 + 60 * (1 - k)), int(90 + 40 * (1 - k)), 255)
        pygame.draw.circle(icon, color, (size // 2, size // 2), r)
    return icon


class App:
    def __init__(
        self,
        settings: Settings,
        settings_path: Path | None = None,
        *,
        fullscreen: bool = False,
        show_fps: bool = False,
        seed: int | None = None,
    ) -> None:
        self.settings = settings
        self.settings_path = settings_path
        self.show_fps = show_fps
        self.seed = random.randrange(1 << 30) if seed is None else seed

        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        pygame.display.set_caption("Ambient Pomodoro")
        pygame.display.set_icon(window_icon())
        self.windowed_size = (
            max(MIN_WINDOW[0], settings.window_width),
            max(MIN_WINDOW[1], settings.window_height),
        )
        self.fullscreen = False
        self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
        set_min_window_size(MIN_WINDOW)
        self.governor = QualityGovernor()
        if fullscreen:
            self.toggle_fullscreen()

        self.timer = PomodoroTimer(settings.timer_config())
        self.chime = ChimePlayer(volume=settings.volume, muted=settings.muted)
        self.overlay = Overlay(self.screen.get_size())
        self.mood = mood_for(self.timer.session)
        self.clock = pygame.time.Clock()
        self.running = True
        self.elapsed = 0.0
        self.last_mouse_move = 0.0
        self._caption = ""

        cls = get_scene_class(settings.scene)
        self.settings.scene = cls.key
        self.scene = self._make_scene(cls)
        # Scenes are built (and warmed up) on a worker thread when switching
        # so the old one keeps animating instead of stalling for ~50 ms.
        self._builder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scene-builder")
        self._pending: tuple[type[Scene], Future] | None = None
        # Crossfades fade out a still snapshot of the previous scene: every
        # scene moves slowly enough that a frozen 1.6 s fade-out is
        # indistinguishable from a live one, at half the rendering cost.
        self.fading_out: pygame.Surface | None = None
        self.fade = 1.0
        self.overlay.show_toast(cls.title, 2.5)

    # ------------------------------------------------------------- lifecycle
    def run(self) -> None:
        last = time.perf_counter()
        try:
            while self.running:
                now = time.perf_counter()
                dt, last = now - last, now
                self.handle_events()
                self.tick_timer(dt)
                work_start = time.perf_counter()
                self.update(min(dt, MAX_SCENE_DT))
                self.draw(self.screen)
                self.govern_quality(time.perf_counter() - work_start, dt)
                pygame.display.flip()
                # Barely render while minimized; the timer uses real time anyway.
                self.clock.tick(self.settings.fps if pygame.display.get_active() else 5)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.persist()
        self._builder.shutdown(wait=True, cancel_futures=True)
        pygame.quit()

    def persist(self) -> None:
        if not self.fullscreen:
            self.settings.window_width, self.settings.window_height = self.screen.get_size()
        else:
            self.settings.window_width, self.settings.window_height = self.windowed_size
        self.settings.muted = self.chime.muted
        save_settings(self.settings, self.settings_path)

    # ---------------------------------------------------------------- scenes
    def govern_quality(self, work_s: float, dt: float) -> None:
        if self.fading_out is not None:  # don't count the crossfade's extra work
            return
        budget = 1.0 / min(self.settings.fps, GOVERNOR_FPS)
        if self.governor.observe(work_s, dt, budget):
            log.info("Render quality -> %.2f", self.governor.quality)
            self.scene.set_quality(self.governor.quality)

    def _make_scene(self, cls: type[Scene]) -> Scene:
        scene = cls(self.screen.get_size(), seed=self.seed)
        if hasattr(self, "governor"):
            scene.set_quality(self.governor.quality)
            self.governor.hold()
        scene.seek(self.elapsed)
        scene.mood = self.mood
        return scene

    def switch_scene(self, step: int = 1) -> None:
        current = self._pending[0] if self._pending else type(self.scene)
        cls = get_scene_class(next_scene_key(current.key, step))
        self.settings.scene = cls.key
        self.overlay.show_toast(cls.title)
        save_settings(self.settings, self.settings_path)
        if cls is type(self.scene) and self._pending is None:
            return
        size, seed, quality, t = self.screen.get_size(), self.seed, self.governor.quality, self.elapsed

        def build() -> Scene:
            scene = cls(size, seed=seed)
            scene.set_quality(quality)
            scene.seek(t)
            scene.prepare()
            return scene

        # A newer request supersedes one still building; its result is dropped.
        self._pending = (cls, self._builder.submit(build))

    def adopt_pending_scene(self) -> None:
        """Start the crossfade once a scene finished building."""
        if self._pending is None or not self._pending[1].done():
            return
        cls, future = self._pending
        self._pending = None
        if cls is type(self.scene):  # switched away and back before it was ready
            return
        scene = future.result()
        if scene.size != self.screen.get_size():
            scene.resize(self.screen.get_size())
        scene.seek(self.elapsed)
        scene.mood = self.mood
        # Capture what's on screen now (mid-fade composites included).
        snapshot = pygame.Surface(self.screen.get_size()).convert(self.screen)
        self.draw_scenes(snapshot)
        self.fading_out = snapshot
        self.scene = scene
        self.fade = 0.0
        self.governor.hold()

    # ----------------------------------------------------------------- input
    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.on_key(event)
            elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
                self.on_resize()
            elif event.type == pygame.MOUSEMOTION:
                self.last_mouse_move = self.elapsed
                pygame.mouse.set_visible(True)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.overlay.panel_rect.collidepoint(event.pos):
                    self.toggle_timer()

    def on_key(self, event: pygame.event.Event) -> None:
        key = event.key
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            self.toggle_timer()
        elif key == pygame.K_r:
            self.timer.reset()
            self.overlay.show_toast(f"{self.timer.session.label} reset")
        elif key == pygame.K_s:
            self.handle_session_end(self.timer.skip())
        elif key == pygame.K_v:
            self.switch_scene(-1 if event.mod & pygame.KMOD_SHIFT else 1)
        elif key == pygame.K_f:
            self.toggle_fullscreen()
        elif key == pygame.K_m:
            muted = self.chime.toggle_mute()
            self.settings.muted = muted
            self.overlay.show_toast("Sound off" if muted else "Sound on")
            save_settings(self.settings, self.settings_path)

    def toggle_timer(self) -> None:
        running = self.timer.toggle()
        self.overlay.show_toast("Started" if running else "Paused", 1.2)

    def toggle_fullscreen(self) -> None:
        if self.fullscreen:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE)
            set_min_window_size(MIN_WINDOW)
        else:
            self.windowed_size = self.screen.get_size()
            # (0, 0) = desktop resolution, which pygame maps to borderless
            # "desktop" fullscreen instead of a video mode switch.
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        self.fullscreen = not self.fullscreen
        if hasattr(self, "scene"):
            self.on_resize()

    def on_resize(self) -> None:
        self.screen = pygame.display.get_surface()
        size = self.screen.get_size()
        if not self.fullscreen:
            self.windowed_size = size
        if size == self.overlay.size and self.scene.size == size:
            return  # pygame 2 reports one resize as several events
        if self.scene.size != size:
            self.scene.resize(size)
        self.fading_out = None  # snapshot no longer fits; just cut
        self.overlay.resize(size)
        self.governor.hold()

    # ---------------------------------------------------------------- update
    def tick_timer(self, dt: float) -> None:
        events = self.timer.tick(dt)
        if events:
            self.handle_session_end(events[-1])

    def handle_session_end(self, event: SessionEnded) -> None:
        if not event.skipped:
            self.chime.play("break_end" if event.ended.is_break else "focus_end")
        if event.skipped:
            self.overlay.show_toast(f"Skipped to {event.next.label}")
        elif event.next is SessionType.FOCUS:
            self.overlay.show_toast("Back to focus")
        else:
            self.overlay.show_toast(f"Time for a {event.next.label.lower()}")

    def update(self, dt: float) -> None:
        self.elapsed += dt
        self.adopt_pending_scene()
        target = mood_for(self.timer.session)
        self.mood += (target - self.mood) * (1.0 - math.exp(-dt / MOOD_TIME_CONSTANT_S))
        self.scene.mood = self.mood
        self.scene.update(dt)
        if self.fading_out is not None:
            self.fade = min(1.0, self.fade + dt / SCENE_FADE_S)
            if self.fade >= 1.0:
                self.fading_out = None
        idle = self.elapsed - self.last_mouse_move
        if idle > CURSOR_HIDE_S and self.timer.running:
            pygame.mouse.set_visible(False)
        self.overlay.update(dt, self.overlay_state())
        self.update_caption()

    def overlay_state(self) -> OverlayState:
        t = self.timer
        idle = self.elapsed - self.last_mouse_move
        return OverlayState(
            session_label=t.session.label,
            time_text=format_remaining(t.remaining),
            running=t.running,
            progress=t.progress,
            dots_filled=t.dots_filled,
            cycle_length=t.cycle_length,
            in_focus=t.session is SessionType.FOCUS,
            mood=self.mood,
            muted=self.chime.muted,
            show_hints=(not t.running) or idle < CURSOR_HIDE_S,
            fps=self.clock.get_fps() if self.show_fps else None,
            quality=self.governor.quality,
        )

    def update_caption(self) -> None:
        t = self.timer
        state = "" if t.running else " (paused)"
        caption = f"{format_remaining(t.remaining)} · {t.session.label}{state} — Ambient Pomodoro"
        if caption != self._caption:
            pygame.display.set_caption(caption)
            self._caption = caption

    # ------------------------------------------------------------------ draw
    def draw_scenes(self, surface: pygame.Surface) -> None:
        self.scene.draw(surface)
        if self.fading_out is not None:
            eased = self.fade * self.fade * (3 - 2 * self.fade)
            self.fading_out.set_alpha(int(255 * (1.0 - eased)))
            surface.blit(self.fading_out, (0, 0))

    def draw(self, surface: pygame.Surface) -> None:
        self.draw_scenes(surface)
        self.overlay.draw(surface, self.overlay_state())
