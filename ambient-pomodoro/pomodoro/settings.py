"""User settings persisted as JSON in the platform's config directory."""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .timer import TimerConfig

log = logging.getLogger(__name__)

APP_NAME = "AmbientPomodoro"
CONFIG_ENV_VAR = "AMBIENT_POMODORO_CONFIG"
SETTINGS_FILENAME = "settings.json"


def default_config_dir() -> Path:
    """Per-user config directory following each platform's conventions."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA")
        return Path(base) / APP_NAME if base else Path.home() / "AppData" / "Roaming" / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_CONFIG_HOME")
    return (Path(base) if base else Path.home() / ".config") / "ambient-pomodoro"


def default_settings_path() -> Path:
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return Path(override).expanduser()
    return default_config_dir() / SETTINGS_FILENAME


@dataclass
class Settings:
    focus_minutes: float = 25.0
    short_break_minutes: float = 5.0
    long_break_minutes: float = 15.0
    sessions_before_long_break: int = 4
    auto_start: bool = True
    scene: str = "lava_lamp"
    muted: bool = False
    volume: float = 0.6
    window_width: int = 1280
    window_height: int = 720
    fps: int = 60

    def timer_config(self) -> TimerConfig:
        return TimerConfig.from_minutes(
            focus=self.focus_minutes,
            short_break=self.short_break_minutes,
            long_break=self.long_break_minutes,
            sessions_before_long_break=self.sessions_before_long_break,
            auto_start=self.auto_start,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> Settings:
        """Build settings from parsed JSON, falling back to defaults per field.

        Unknown keys are ignored and invalid values (wrong type or out of
        range) are replaced by their default, so a hand-edited file with one
        typo doesn't throw away everything else.
        """
        settings = cls()
        if not isinstance(data, dict):
            return settings
        for f in fields(cls):
            if f.name not in data:
                continue
            value = _validate(f.name, data[f.name])
            if value is None:
                log.warning("Ignoring invalid setting %s=%r", f.name, data[f.name])
            else:
                setattr(settings, f.name, value)
        return settings


# name -> (kind, min, max)
_RULES: dict[str, tuple[str, float, float]] = {
    "focus_minutes": ("float", 1 / 60, 24 * 60),
    "short_break_minutes": ("float", 1 / 60, 24 * 60),
    "long_break_minutes": ("float", 1 / 60, 24 * 60),
    "sessions_before_long_break": ("int", 1, 100),
    "auto_start": ("bool", 0, 0),
    "scene": ("str", 0, 0),
    "muted": ("bool", 0, 0),
    "volume": ("float", 0.0, 1.0),
    "window_width": ("int", 320, 16384),
    "window_height": ("int", 240, 16384),
    "fps": ("int", 10, 240),
}


def _validate(name: str, value: Any) -> Any:
    kind, lo, hi = _RULES[name]
    if kind == "bool":
        return value if isinstance(value, bool) else None
    if kind == "str":
        return value if isinstance(value, str) and value else None
    # bool is a subclass of int; don't accept true/false as numbers.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or not (lo <= value <= hi):
        return None
    if kind == "int":
        return int(value) if float(value).is_integer() else None
    return float(value)


def load_settings(path: Path | None = None) -> Settings:
    path = path or default_settings_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return Settings()
    except OSError as exc:
        log.warning("Could not read settings from %s: %s", path, exc)
        return Settings()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("Settings file %s is not valid JSON (%s); using defaults", path, exc)
        return Settings()
    return Settings.from_dict(data)


def save_settings(settings: Settings, path: Path | None = None) -> bool:
    """Write settings atomically. Returns False (and logs) on failure."""
    path = path or default_settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".settings-", suffix=".json", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(settings.to_dict(), fh, indent=2, sort_keys=True)
                fh.write("\n")
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    except OSError as exc:
        log.warning("Could not save settings to %s: %s", path, exc)
        return False
    return True
