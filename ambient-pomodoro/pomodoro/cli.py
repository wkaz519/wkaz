"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(p) for p in text.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from None
    if w < 16 or h < 16:
        raise argparse.ArgumentTypeError("size must be at least 16x16")
    return w, h


def _float_list(text: str) -> list[float]:
    try:
        return [float(p) for p in text.split(",") if p.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated seconds, got {text!r}") from None


def _str_list(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def build_parser() -> argparse.ArgumentParser:
    from .scenes import scene_keys

    p = argparse.ArgumentParser(
        prog="ambient-pomodoro",
        description="A Pomodoro timer over slowly evolving ambient visuals.",
        epilog="Keys: Space start/pause, R reset, S skip, V/Shift+V visual, F fullscreen, M mute, Esc quit.",
    )
    timer = p.add_argument_group("timer (saved as your new defaults)")
    timer.add_argument("--focus", type=float, metavar="MIN", help="focus length in minutes")
    timer.add_argument("--short", type=float, metavar="MIN", help="short break length in minutes")
    timer.add_argument("--long", type=float, metavar="MIN", help="long break length in minutes")
    timer.add_argument("--cycle", type=int, metavar="N", help="focus sessions before a long break")
    timer.add_argument(
        "--auto-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="start the next session automatically when one ends",
    )

    app = p.add_argument_group("app")
    app.add_argument("--scene", choices=scene_keys(), help="visual to start with")
    app.add_argument("--fullscreen", action="store_true", help="start in fullscreen")
    app.add_argument("--fps", type=int, help="frame-rate cap (default 60; 30 saves battery)")
    app.add_argument("--show-fps", action="store_true", help="show an FPS counter")
    app.add_argument("--seed", type=int, help="fix the visuals' random seed")
    app.add_argument("--config", type=Path, metavar="PATH", help="settings file to use")
    app.add_argument("--list-scenes", action="store_true", help="list available visuals and exit")

    dev = p.add_argument_group("developer tools (headless)")
    dev.add_argument("--render-frames", action="store_true", help="save PNG snapshots of every scene")
    dev.add_argument("--benchmark", action="store_true", help="report average frame time per scene")
    dev.add_argument("--out", type=Path, default=Path("snapshots"), help="snapshot directory")
    dev.add_argument("--size", type=_size, help="render size, e.g. 1920x1080")
    dev.add_argument("--times", type=_float_list, default=[0, 60, 600, 1500], help="snapshot times in seconds")
    dev.add_argument("--scenes", type=_str_list, help="comma-separated scene keys (default: all)")
    dev.add_argument("--frames", type=int, default=120, help="benchmark frames per scene")
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    from .scenes import SCENES

    if args.list_scenes:
        for cls in SCENES:
            print(f"{cls.key:<12} {cls.title}")
        return 0

    if args.render_frames:
        from .devtools import render_frames

        render_frames(args.out, args.size or (1280, 720), args.times, args.scenes, seed=args.seed or 0)
        return 0

    if args.benchmark:
        from .devtools import benchmark

        benchmark(args.size or (1920, 1080), args.frames, args.scenes)
        return 0

    from .settings import default_settings_path, load_settings, save_settings

    path = args.config or default_settings_path()
    settings = load_settings(path)
    overrides = {
        "focus_minutes": args.focus,
        "short_break_minutes": args.short,
        "long_break_minutes": args.long,
        "sessions_before_long_break": args.cycle,
        "auto_start": args.auto_start,
        "scene": args.scene,
        "fps": args.fps,
    }
    from .settings import Settings

    for name, value in overrides.items():
        if value is None:
            continue
        candidate = Settings.from_dict({**settings.to_dict(), name: value})
        if getattr(candidate, name) != value:
            print(f"error: invalid value for {name}: {value!r}", file=sys.stderr)
            return 2
        setattr(settings, name, value)
    save_settings(settings, path)

    from .app import App

    App(settings, path, fullscreen=args.fullscreen, show_fps=args.show_fps, seed=args.seed).run()
    return 0
