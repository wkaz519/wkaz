"""Headless developer tools: scene snapshots and a frame-time benchmark.

Both run with SDL's dummy video driver, so they work on CI machines and
over SSH without a display.
"""

from __future__ import annotations

import os
import statistics
import time
from pathlib import Path


def init_headless(size: tuple[int, int]):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    pygame.display.init()
    pygame.font.init()
    return pygame.display.set_mode(size)


def _selected_scenes(keys: list[str] | None):
    from .scenes import SCENES

    if not keys:
        return list(SCENES)
    by_key = {cls.key: cls for cls in SCENES}
    unknown = [k for k in keys if k not in by_key]
    if unknown:
        raise SystemExit(f"Unknown scene(s): {', '.join(unknown)}. Available: {', '.join(by_key)}")
    return [by_key[k] for k in keys]


def overlay_state_at(t: float, mood: float):
    """Overlay state for a timer that has been running for ``t`` seconds."""
    from .timer import PomodoroTimer, SessionType, format_remaining
    from .ui import OverlayState

    timer = PomodoroTimer()
    timer.start()
    timer.tick(t)
    return OverlayState(
        session_label=timer.session.label,
        time_text=format_remaining(timer.remaining),
        running=True,
        progress=timer.progress,
        dots_filled=timer.dots_filled,
        cycle_length=timer.cycle_length,
        in_focus=timer.session is SessionType.FOCUS,
        mood=mood,
    )


def render_frames(
    out_dir: Path,
    size: tuple[int, int] = (1280, 720),
    times: list[float] | None = None,
    scene_keys: list[str] | None = None,
    seed: int = 0,
) -> list[Path]:
    """Save PNG snapshots of each scene at several timestamps.

    For every scene this writes ``<key>_t<seconds>s.png`` (focus palette),
    ``<key>_break.png`` (break palette), ``<key>_ui.png`` (with the timer
    overlay) and finally a ``contact_sheet.png`` of everything.
    """
    import pygame

    from .ui import Overlay

    times = times if times is not None else [0, 60, 600, 1500]
    screen = init_headless(size)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = Overlay(size)
    written: list[Path] = []
    rows: list[tuple[str, list[tuple[str, pygame.Surface]]]] = []
    thumb = (384, round(384 * size[1] / size[0]))
    mid_t = times[len(times) // 2] if times else 0

    for cls in _selected_scenes(scene_keys):
        scene = cls(size, seed=seed)
        cells = []

        def snap(name: str, label: str, t: float, mood: float, ui: bool = False) -> None:
            scene.seek(t)
            scene.mood = mood
            scene.draw(screen)
            if ui:
                overlay.draw(screen, overlay_state_at(t, mood))
            path = out_dir / f"{cls.key}_{name}.png"
            pygame.image.save(screen, str(path))
            written.append(path)
            cells.append((label, pygame.transform.smoothscale(screen, thumb)))

        for t in times:
            snap(f"t{int(t):04d}s", f"t = {_fmt_time(t)}", t, 0.0)
        snap("break", f"break palette, t = {_fmt_time(mid_t)}", mid_t, 1.0)
        snap("ui", f"with timer, t = {_fmt_time(mid_t)}", mid_t, 0.0, ui=True)
        rows.append((cls.title, cells))
        print(f"  {cls.title:<22} {len(cells)} images")

    sheet = _contact_sheet(rows, thumb)
    sheet_path = out_dir / "contact_sheet.png"
    pygame.image.save(sheet, str(sheet_path))
    written.append(sheet_path)
    print(f"Wrote {len(written)} files to {out_dir}")
    return written


def _fmt_time(t: float) -> str:
    t = int(t)
    return f"{t // 60}m{t % 60:02d}s" if t >= 60 else f"{t}s"


def _contact_sheet(rows, thumb):
    import pygame

    font = pygame.font.Font(None, 22)
    gap, label_h, title_w = 8, 22, 150
    cols = max(len(cells) for _, cells in rows)
    width = title_w + cols * (thumb[0] + gap) + gap
    height = gap + len(rows) * (thumb[1] + label_h + gap)
    sheet = pygame.Surface((width, height))
    sheet.fill((18, 18, 22))
    for r, (title, cells) in enumerate(rows):
        y = gap + r * (thumb[1] + label_h + gap)
        sheet.blit(font.render(title, True, (235, 235, 235)), (gap, y + thumb[1] // 2))
        for c, (label, image) in enumerate(cells):
            x = title_w + c * (thumb[0] + gap)
            sheet.blit(image, (x, y))
            sheet.blit(font.render(label, True, (170, 170, 180)), (x, y + thumb[1] + 3))
    return sheet


def benchmark(
    size: tuple[int, int] = (1920, 1080),
    frames: int = 120,
    scene_keys: list[str] | None = None,
    target_fps: float = 30.0,
) -> dict[str, dict[str, float]]:
    """Report per-scene frame times (scene alone and with the overlay)."""
    from .ui import Overlay

    screen = init_headless(size)
    overlay = Overlay(size)
    state = overlay_state_at(600, 0.0)
    budget_ms = 1000.0 / target_fps
    results: dict[str, dict[str, float]] = {}
    print(f"Benchmark at {size[0]}x{size[1]}, {frames} frames per scene (budget {budget_ms:.1f} ms)")
    print(f"  {'scene':<22} {'internal':>9} {'avg ms':>8} {'p95 ms':>8} {'+UI ms':>8} {'~fps':>7}")
    for cls in _selected_scenes(scene_keys):
        scene = cls(size, seed=0)
        scene.seek(300.0)
        for _ in range(5):  # warm-up (caches, first-touch allocations)
            scene.update(1 / 60)
            scene.draw(screen)
        samples, with_ui = [], []
        for _ in range(frames):
            t0 = time.perf_counter()
            scene.update(1 / 60)
            scene.draw(screen)
            t1 = time.perf_counter()
            overlay.draw(screen, state)
            t2 = time.perf_counter()
            samples.append((t1 - t0) * 1000)
            with_ui.append((t2 - t0) * 1000)
        avg = statistics.fmean(samples)
        p95 = sorted(samples)[int(0.95 * (len(samples) - 1))]
        avg_ui = statistics.fmean(with_ui)
        res = getattr(scene, "res", size)
        results[cls.key] = {"avg_ms": avg, "p95_ms": p95, "avg_with_ui_ms": avg_ui}
        flag = "" if avg_ui <= budget_ms else "  <-- over budget"
        print(
            f"  {cls.title:<22} {res[0]:>4}x{res[1]:<4} {avg:8.2f} {p95:8.2f} {avg_ui:8.2f} "
            f"{1000 / avg_ui:7.1f}{flag}"
        )
    return results
