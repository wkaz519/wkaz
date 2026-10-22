import json

import pytest

from pomodoro import settings as settings_mod
from pomodoro.settings import Settings, load_settings, save_settings


def test_missing_file_gives_defaults(tmp_path):
    s = load_settings(tmp_path / "nope.json")
    assert s == Settings()


def test_round_trip(tmp_path):
    path = tmp_path / "sub" / "settings.json"
    s = Settings(focus_minutes=50, short_break_minutes=10, scene="aurora", muted=True, window_width=1600, window_height=900)
    assert save_settings(s, path)
    assert load_settings(path) == s
    assert json.loads(path.read_text())["focus_minutes"] == 50


def test_corrupt_json_gives_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not json")
    assert load_settings(path) == Settings()


def test_non_object_json_gives_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("[1, 2, 3]")
    assert load_settings(path) == Settings()


def test_invalid_fields_fall_back_individually(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "focus_minutes": -3,  # out of range
                "short_break_minutes": 7,  # fine
                "sessions_before_long_break": 2.5,  # not an int
                "muted": "yes",  # not a bool
                "volume": True,  # bool is not a number here
                "scene": "",  # empty
                "window_width": 2000,
                "unknown_key": 123,
            }
        )
    )
    s = load_settings(path)
    assert s.focus_minutes == 25
    assert s.short_break_minutes == 7
    assert s.sessions_before_long_break == 4
    assert s.muted is False
    assert s.volume == 0.6
    assert s.scene == "lava_lamp"
    assert s.window_width == 2000


def test_integral_float_accepted_for_int_field():
    assert Settings.from_dict({"sessions_before_long_break": 3.0}).sessions_before_long_break == 3


def test_timer_config_from_settings():
    cfg = Settings(focus_minutes=30, short_break_minutes=6, long_break_minutes=20, sessions_before_long_break=3).timer_config()
    assert (cfg.focus, cfg.short_break, cfg.long_break, cfg.sessions_before_long_break) == (1800, 360, 1200, 3)


def test_save_failure_returns_false(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert save_settings(Settings(), blocker / "settings.json") is False


def test_env_var_overrides_path(tmp_path, monkeypatch):
    monkeypatch.setenv(settings_mod.CONFIG_ENV_VAR, str(tmp_path / "custom.json"))
    assert settings_mod.default_settings_path() == tmp_path / "custom.json"


@pytest.mark.parametrize(
    "platform, expected_tail",
    [("win32", "AmbientPomodoro"), ("darwin", "Library/Application Support/AmbientPomodoro"), ("linux", "ambient-pomodoro")],
)
def test_platform_config_dirs(monkeypatch, tmp_path, platform, expected_tail):
    monkeypatch.setattr(settings_mod.sys, "platform", platform)
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert settings_mod.default_config_dir().as_posix().endswith(expected_tail)
