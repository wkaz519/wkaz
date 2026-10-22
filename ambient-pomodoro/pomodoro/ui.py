"""Timer overlay: frosted panel, countdown, session label, cycle dots, toasts.

Legibility over arbitrary backgrounds comes from three layers: the area
behind the panel is blurred and darkened ("frosted glass"), the digits get a
soft drop shadow, and all text is near-white.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pygame

# Font families tried in order; the first one installed wins, else pygame's
# bundled font. Names are pygame's normalized family names.
FONT_CANDIDATES = (
    "interdisplay",
    "inter",
    "helveticaneue",
    "segoeui",
    "avenirnext",
    "sfns",
    "dejavusans",
)
# Windows registers each weight as its own family name.
WEIGHT_FAMILIES = {"light": ("segoeuilight",), "regular": ()}
_AVOID_STYLES = ("thin", "hairline", "italic", "oblique", "black", "heavy", "condensed")

TEXT = (250, 246, 240)
WARM_ACCENT = (255, 178, 124)
COOL_ACCENT = (140, 208, 255)
PANEL_TINT = (10, 8, 22, 92)
PANEL_BORDER = (255, 255, 255, 30)
HINT_TEXT = "Space start/pause  ·  R reset  ·  S skip  ·  V visual  ·  F fullscreen  ·  M mute  ·  Esc quit"


@dataclass
class OverlayState:
    session_label: str
    time_text: str
    running: bool
    progress: float
    dots_filled: int
    cycle_length: int
    in_focus: bool
    mood: float = 0.0
    muted: bool = False
    show_hints: bool = False
    fps: float | None = None
    quality: float = 1.0


def find_font_path(weight: str = "regular") -> str | None:
    """Locate a nice installed font file of roughly the requested weight.

    ``match_font`` may return any face of a family (e.g. "Thin" on Linux),
    so look for a sibling file with the wanted weight in its name.
    """
    available = set(pygame.font.get_fonts())
    wanted = ("Light", "Regular") if weight == "light" else ("Regular", "Book", "")
    for name in WEIGHT_FAMILIES.get(weight, ()) + FONT_CANDIDATES:
        if name not in available:
            continue
        path = pygame.font.match_font(name)
        if not path:
            continue
        p = Path(path)
        family = p.stem.split("-")[0]
        siblings = {f.stem.lower(): f for f in p.parent.glob(f"{family}*") if f.suffix.lower() in (".ttf", ".otf")}
        for w in wanted:
            for stem in (f"{family}-{w}", f"{family}{w}"):
                if stem.lower() in siblings:
                    return str(siblings[stem.lower()])
        if not any(bad in p.stem.lower() for bad in _AVOID_STYLES):
            return path
    return None


def soft_blur(surface: pygame.Surface, radius: int) -> pygame.Surface:
    """Cheap blur: area-average down by ``radius`` then bilinear back up.

    Upsampling in two steps hides the diamond artifacts that a single large
    bilinear stretch leaves behind.
    """
    w, h = surface.get_size()
    radius = max(1, int(radius))
    small = pygame.transform.smoothscale(surface, (max(1, w // radius), max(1, h // radius)))
    if radius >= 4:
        small = pygame.transform.smoothscale(small, (max(1, 2 * w // radius), max(1, 2 * h // radius)))
    return pygame.transform.smoothscale(small, (w, h))


def rounded_rect_surface(size, radius, color, border_color=None, border=0, supersample=3) -> pygame.Surface:
    """Anti-aliased rounded rectangle (drawn large, then scaled down)."""
    w, h = size
    k = supersample
    big = pygame.Surface((w * k, h * k), pygame.SRCALPHA)
    pygame.draw.rect(big, color, big.get_rect(), border_radius=radius * k)
    if border_color and border:
        pygame.draw.rect(big, border_color, big.get_rect(), width=border * k, border_radius=radius * k)
    return pygame.transform.smoothscale(big, (w, h))


def circle_sprite(radius: float, ring: float = 0.0, supersample: int = 4) -> pygame.Surface:
    """White anti-aliased disc (``ring`` = 0) or ring of the given thickness."""
    size = max(2, math.ceil(radius * 2 + 2))
    k = supersample
    big = pygame.Surface((size * k, size * k), pygame.SRCALPHA)
    center = (size * k / 2, size * k / 2)
    width = 0 if ring <= 0 else max(1, round(ring * k))
    pygame.draw.circle(big, (255, 255, 255, 255), center, radius * k, width)
    return pygame.transform.smoothscale(big, (size, size))


def tinted(sprite: pygame.Surface, color, alpha: float = 1.0) -> pygame.Surface:
    s = sprite.copy()
    s.fill((*color, int(255 * max(0.0, min(1.0, alpha)))), special_flags=pygame.BLEND_RGBA_MULT)
    return s


def mix_rgb(a, b, t: float):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


class Overlay:
    def __init__(self, size: tuple[int, int]) -> None:
        pygame.font.init()
        self._font_path = find_font_path("regular")
        self._digit_font_path = find_font_path("light") or self._font_path
        self.hint_alpha = 1.0
        self._toast_text = ""
        self._toast_age = 0.0
        self._toast_duration = 0.0
        self._pulse = 0.0
        self.panel_rect = pygame.Rect(0, 0, 0, 0)
        self.resize(size)

    # ----------------------------------------------------------------- setup
    def _font(self, px: float, path: str | None = None) -> pygame.font.Font:
        return pygame.font.Font(path or self._font_path, max(8, int(px)))

    def resize(self, size: tuple[int, int]) -> None:
        self.size = (max(1, size[0]), max(1, size[1]))
        w, h = self.size
        self.unit = s = max(120.0, min(h, w * 9 / 16))
        self.font_digits = self._font(s * 0.23, self._digit_font_path)
        self.font_label = self._font(s * 0.034)
        self.font_small = self._font(s * 0.021)
        self._glyphs: dict = {}
        self._timer_cache: tuple[str, pygame.Surface, pygame.Surface] | None = None
        self._text_cache: dict = {}
        self._sprites: dict = {}
        self._panel_key = None
        self._digit_cell = max(self.font_digits.size(d)[0] for d in "0123456789")
        metrics = [m for m in self.font_digits.metrics("0123456789:") if m]
        ascent = self.font_digits.get_ascent()
        self._ink_top = max(0, ascent - max(m[3] for m in metrics))
        self._ink_bottom = ascent - min(m[2] for m in metrics)
        r = s * 0.0085
        self._dot = circle_sprite(r)
        self._ring = circle_sprite(r, ring=max(1.2, r * 0.3))
        self._dot_spacing = s * 0.034

    def _cached(self, key, factory):
        """Memoize rendered sprites until the next resize."""
        value = self._sprites.get(key)
        if value is None:
            if len(self._sprites) > 128:
                self._sprites.clear()
            value = self._sprites[key] = factory()
        return value

    def _text(self, font: pygame.font.Font, text: str, color=TEXT) -> pygame.Surface:
        key = (id(font), text, color)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 64:
                self._text_cache.clear()
            surf = font.render(text, True, color)
            self._text_cache[key] = surf
        return surf

    def _tracked(self, font: pygame.font.Font, text: str, tracking: float, color=TEXT) -> pygame.Surface:
        """Render with extra letter spacing (for the small-caps session label)."""
        key = ("tracked", id(font), text, tracking, color)
        surf = self._text_cache.get(key)
        if surf is None:
            glyphs = [font.render(ch, True, color) for ch in text]
            width = sum(g.get_width() for g in glyphs) + int(tracking * max(0, len(glyphs) - 1))
            surf = pygame.Surface((max(1, width), font.get_height()), pygame.SRCALPHA)
            x = 0
            for g in glyphs:
                surf.blit(g, (x, 0))
                x += g.get_width() + int(tracking)
            self._text_cache[key] = surf
        return surf

    def _timer_surfaces(self, text: str) -> tuple[pygame.Surface, pygame.Surface]:
        """Countdown with fixed-width digit cells (no jitter as digits change)
        plus its pre-blurred drop shadow. Cached until the text changes."""
        if self._timer_cache and self._timer_cache[0] == text:
            return self._timer_cache[1], self._timer_cache[2]
        font = self.font_digits
        cells = []
        for ch in text:
            g = self._glyphs.get(ch)
            if g is None:
                g = self._glyphs[ch] = (font.render(ch, True, TEXT), font.render(ch, True, (0, 0, 0)))
            cell = self._digit_cell if ch.isdigit() else int(g[0].get_width() * 0.92)
            cells.append((g, cell))
        width = sum(c for _, c in cells)
        ink_h = self._ink_bottom - self._ink_top
        pad = int(self.unit * 0.03)
        text_surf = pygame.Surface((width, ink_h), pygame.SRCALPHA)
        shadow = pygame.Surface((width + 2 * pad, ink_h + 2 * pad), pygame.SRCALPHA)
        x = 0
        for (light, dark), cell in cells:
            gx = x + (cell - light.get_width()) // 2
            text_surf.blit(light, (gx, -self._ink_top))
            shadow.blit(dark, (gx + pad, pad - self._ink_top))
            x += cell
        shadow = soft_blur(soft_blur(shadow, max(2, pad // 3)), max(2, pad // 4))
        self._timer_cache = (text, text_surf, shadow)
        return text_surf, shadow

    def _panel_surfaces(self, size: tuple[int, int]):
        if self._panel_key != size:
            radius = int(self.unit * 0.045)
            self._panel_mask = rounded_rect_surface(size, radius, (255, 255, 255, 255))
            self._panel_tint = rounded_rect_surface(size, radius, PANEL_TINT, PANEL_BORDER, border=1)
            self._panel_key = size
        return self._panel_mask, self._panel_tint

    # ---------------------------------------------------------------- update
    def show_toast(self, text: str, duration: float = 2.2) -> None:
        self._toast_text = text
        self._toast_age = 0.0
        self._toast_duration = duration

    def update(self, dt: float, state: OverlayState) -> None:
        target = 1.0 if state.show_hints else 0.0
        rate = 2.5 if target > self.hint_alpha else 0.8
        self.hint_alpha += max(-rate * dt, min(rate * dt, target - self.hint_alpha))
        self._toast_age += dt
        self._pulse += dt

    # ------------------------------------------------------------------ draw
    def draw(self, surface: pygame.Surface, state: OverlayState) -> None:
        s = self.unit
        sw, sh = surface.get_size()
        accent = mix_rgb(WARM_ACCENT, COOL_ACCENT, state.mood)

        digits, shadow = self._timer_surfaces(state.time_text)
        label_text = state.session_label.upper() + ("" if state.running else "  ·  PAUSED")
        label = self._tracked(self.font_label, label_text, s * 0.006, mix_rgb(TEXT, accent, 0.35))

        pad_x, pad_top, pad_bottom = s * 0.07, s * 0.045, s * 0.05
        gap_label, gap_dots, gap_bar = s * 0.012, s * 0.03, s * 0.032
        dot_h = self._dot.get_height()
        bar_h = max(2, int(s * 0.0045))
        content_w = max(digits.get_width(), label.get_width())
        panel_w = int(content_w + 2 * pad_x)
        panel_h = int(pad_top + label.get_height() + gap_label + digits.get_height() + gap_dots + dot_h + gap_bar + bar_h + pad_bottom)
        panel = pygame.Rect(0, 0, panel_w, panel_h)
        panel.center = (sw // 2, sh // 2)
        self.panel_rect = panel

        self._draw_panel(surface, panel)

        y = panel.top + pad_top
        surface.blit(label, (panel.centerx - label.get_width() // 2, y))
        y += label.get_height() + gap_label

        dx = panel.centerx - digits.get_width() // 2
        pad = (shadow.get_width() - digits.get_width()) // 2
        shadow.set_alpha(150)
        surface.blit(shadow, (dx - pad, y - pad + int(s * 0.008)))
        digits.set_alpha(255 if state.running else 165)
        surface.blit(digits, (dx, int(y)))
        y += digits.get_height() + gap_dots

        self._draw_dots(surface, panel.centerx, int(y), state, accent)
        y += dot_h + gap_bar

        bar = pygame.Rect(0, 0, int(panel_w - 2 * pad_x), bar_h)
        bar.midtop = (panel.centerx, int(y))
        track = self._cached(("track", bar.size), lambda: rounded_rect_surface(bar.size, bar_h // 2, (255, 255, 255, 36), supersample=2))
        surface.blit(track, bar)
        filled = int(bar.width * max(0.0, min(1.0, state.progress)))
        if filled >= 1:
            fill = self._cached(("fill", bar.size, accent), lambda: rounded_rect_surface(bar.size, bar_h // 2, (*accent, 230), supersample=2))
            surface.blit(fill, bar.topleft, area=pygame.Rect(0, 0, filled, bar_h))

        self._draw_toast(surface, panel)
        self._draw_chrome(surface, state)

    def _draw_panel(self, surface: pygame.Surface, panel: pygame.Rect) -> None:
        clipped = panel.clip(surface.get_rect())
        if clipped.width < 4 or clipped.height < 4:
            return
        mask, tint = self._panel_surfaces(panel.size)
        behind = surface.subsurface(clipped)
        blurred = soft_blur(behind, 10)
        frosted = pygame.Surface(panel.size, pygame.SRCALPHA)
        frosted.blit(blurred, (clipped.x - panel.x, clipped.y - panel.y))
        frosted.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        surface.blit(frosted, panel)
        surface.blit(tint, panel)

    def _draw_dots(self, surface, cx: int, y: int, state: OverlayState, accent) -> None:
        n = max(1, state.cycle_length)
        if n > 12:  # very long cycles: just show "k / n"
            txt = self._text(self.font_small, f"{state.dots_filled} / {n}")
            surface.blit(txt, (cx - txt.get_width() // 2, y))
            return
        size = self._dot.get_width()
        x0 = cx - (self._dot_spacing * (n - 1)) / 2 - size / 2
        filled = self._cached(("dot", accent), lambda: tinted(self._dot, accent))
        ring = self._cached(("ring",), lambda: tinted(self._ring, TEXT, 0.55))
        for i in range(n):
            pos = (int(x0 + i * self._dot_spacing), y)
            if i < state.dots_filled:
                surface.blit(filled, pos)
                continue
            surface.blit(ring, pos)
            if state.in_focus and i == state.dots_filled:
                # The session in progress: a gently breathing fill.
                breath = 0.5 + 0.5 * math.sin(self._pulse * 2.2) if state.running else 0.4
                fill = self._cached(("dot", accent), lambda: tinted(self._dot, accent))
                fill.set_alpha(int(255 * (0.25 + 0.45 * breath)))
                surface.blit(fill, pos)
                fill.set_alpha(255)

    def _draw_toast(self, surface: pygame.Surface, panel: pygame.Rect) -> None:
        if not self._toast_text or self._toast_age >= self._toast_duration:
            return
        age, dur = self._toast_age, self._toast_duration
        alpha = min(1.0, age / 0.15, (dur - age) / 0.6)
        text = self._text(self.font_label, self._toast_text)
        pad_x, pad_y = int(self.unit * 0.025), int(self.unit * 0.012)
        size = (text.get_width() + 2 * pad_x, text.get_height() + 2 * pad_y)
        def make_pill():
            pill = rounded_rect_surface(size, size[1] // 2, (10, 8, 22, 140), (255, 255, 255, 28), border=1, supersample=2)
            pill.blit(text, (pad_x, pad_y))
            return pill

        pill = self._cached(("toast", self._toast_text), make_pill)
        pill.set_alpha(int(255 * max(0.0, alpha)))
        surface.blit(pill, (panel.centerx - size[0] // 2, panel.bottom + int(self.unit * 0.04)))

    def _draw_chrome(self, surface: pygame.Surface, state: OverlayState) -> None:
        sw, sh = surface.get_size()
        margin = int(self.unit * 0.03)
        if self.hint_alpha > 0.01:
            def make_hint():
                hint = self._text(self.font_small, HINT_TEXT)
                if hint.get_width() > sw - 2 * margin:
                    hint = pygame.transform.smoothscale_by(hint, (sw - 2 * margin) / hint.get_width())
                return hint, soft_blur(self._shadow_for(hint), 3)

            hint, backing = self._cached(("hint", sw), make_hint)
            for surf, offset in ((backing, 1), (hint, 0)):
                surf.set_alpha(int(255 * self.hint_alpha * (0.8 if offset else 0.72)))
                surface.blit(surf, (sw // 2 - hint.get_width() // 2 + offset, sh - margin - hint.get_height() + offset))
        if state.muted:
            muted = self._text(self.font_small, "Sound off")
            muted.set_alpha(150)
            surface.blit(muted, (sw - margin - muted.get_width(), margin))
        if state.fps is not None:
            label = f"{state.fps:5.1f} fps"
            if state.quality < 0.999:
                label += f"  ·  render {state.quality:.0%}"
            fps = self._text(self.font_small, label)
            surface.blit(fps, (margin, margin))

    def _shadow_for(self, surf: pygame.Surface) -> pygame.Surface:
        shadow = surf.copy()
        shadow.fill((0, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
        return shadow
