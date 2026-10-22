"""The interactive application: window, input, main loop."""

from __future__ import annotations

import logging
import math
import random
import time
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
        self.fading_out: Scene | None = None
        self.fade = 1.0
        self._fade_buffer: pygame.Surface | None = None
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
                self.update(min(dt, MAX_SCENE_DT))
                self.draw(self.screen)
                pygame.display.flip()
                # Barely render while minimized; the timer uses real time anyway.
                self.clock.tick(self.settings.fps if pygame.display.get_active() else 5)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self.persist()
        pygame.quit()

    def persist(self) -> None:
        if not self.fullscreen:
            self.settings.window_width, self.settings.window_height = self.screen.get_size()
        else:
            self.settings.window_width, self.settings.window_height = self.windowed_size
        self.settings.muted = self.chime.muted
        save_settings(self.settings, self.settings_path)

    # ---------------------------------------------------------------- scenes
    def _make_scene(self, cls: type[Scene]) -> Scene:
        scene = cls(self.screen.get_size(), seed=self.seed)
        scene.seek(self.elapsed)
        scene.mood = self.mood
        return scene

    def switch_scene(self, step: int = 1) -> None:
        cls = get_scene_class(next_scene_key(self.scene.key, step))
        if cls.key == self.scene.key:
            return
        # If a fade is already in progress, drop the oldest scene.
        self.fading_out = self.scene
        self.scene = self._make_scene(cls)
        self.fade = 0.0
        self.settings.scene = cls.key
        self.overlay.show_toast(cls.title)
        save_settings(self.settings, self.settings_path)

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
        w, h = self.screen.get_size()
        if not self.fullscreen and (w < MIN_WINDOW[0] or h < MIN_WINDOW[1]):
            self.screen = pygame.display.set_mode((max(w, MIN_WINDOW[0]), max(h, MIN_WINDOW[1])), pygame.RESIZABLE)
        size = self.screen.get_size()
        if not self.fullscreen:
            self.windowed_size = size
        for scene in (self.scene, self.fading_out):
            if scene is not None and scene.size != size:
                scene.resize(size)
        self.overlay.resize(size)
        self._fade_buffer = None

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
        target = mood_for(self.timer.session)
        self.mood += (target - self.mood) * (1.0 - math.exp(-dt / MOOD_TIME_CONSTANT_S))
        for scene in (self.scene, self.fading_out):
            if scene is not None:
                scene.mood = self.mood
                scene.update(dt)
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
        )

    def update_caption(self) -> None:
        t = self.timer
        state = "" if t.running else " (paused)"
        caption = f"{format_remaining(t.remaining)} · {t.session.label}{state} — Ambient Pomodoro"
        if caption != self._caption:
            pygame.display.set_caption(caption)
            self._caption = caption

    # ------------------------------------------------------------------ draw
    def draw(self, surface: pygame.Surface) -> None:
        if self.fading_out is None:
            self.scene.draw(surface)
        else:
            self.fading_out.draw(surface)
            if self._fade_buffer is None or self._fade_buffer.get_size() != surface.get_size():
                self._fade_buffer = pygame.Surface(surface.get_size()).convert(surface)
            self.scene.draw(self._fade_buffer)
            eased = self.fade * self.fade * (3 - 2 * self.fade)
            self._fade_buffer.set_alpha(int(255 * eased))
            surface.blit(self._fade_buffer, (0, 0))
        self.overlay.draw(surface, self.overlay_state())
