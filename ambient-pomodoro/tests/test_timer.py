import pytest

from pomodoro.timer import (
    PomodoroTimer,
    SessionEnded,
    SessionType,
    TimerConfig,
    format_remaining,
)

F, S, L = SessionType.FOCUS, SessionType.SHORT_BREAK, SessionType.LONG_BREAK


def make_timer(**kwargs) -> PomodoroTimer:
    return PomodoroTimer(TimerConfig.from_minutes(**kwargs))


def finish_session(timer: PomodoroTimer) -> list[SessionEnded]:
    """Run the clock exactly to the end of the current session."""
    timer.start()
    return timer.tick(timer.remaining)


# --------------------------------------------------------------------- config


def test_default_config_is_standard_pomodoro():
    cfg = TimerConfig()
    assert cfg.focus == 25 * 60
    assert cfg.short_break == 5 * 60
    assert cfg.long_break == 15 * 60
    assert cfg.sessions_before_long_break == 4


def test_from_minutes_converts_to_seconds():
    cfg = TimerConfig.from_minutes(focus=50, short_break=10, long_break=30, sessions_before_long_break=3)
    assert (cfg.focus, cfg.short_break, cfg.long_break) == (3000, 600, 1800)
    assert cfg.duration(F) == 3000
    assert cfg.duration(S) == 600
    assert cfg.duration(L) == 1800


@pytest.mark.parametrize(
    "kwargs",
    [
        {"focus": 0},
        {"short_break": -5},
        {"long_break": float("nan")},
        {"focus": float("inf")},
        {"sessions_before_long_break": 0},
        {"sessions_before_long_break": 2.5},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        TimerConfig(**kwargs)


def test_session_labels():
    assert F.label == "Focus"
    assert S.label == "Short Break"
    assert L.label == "Long Break"
    assert not F.is_break and S.is_break and L.is_break


# ---------------------------------------------------------------- start/pause


def test_initial_state():
    t = PomodoroTimer()
    assert t.session is F
    assert t.remaining == 25 * 60
    assert not t.running
    assert t.completed_in_cycle == 0
    assert t.progress == 0.0


def test_tick_does_nothing_while_paused():
    t = PomodoroTimer()
    assert t.tick(10) == []
    assert t.remaining == 25 * 60


def test_negative_or_zero_dt_is_ignored():
    t = PomodoroTimer()
    t.start()
    t.tick(-5)
    t.tick(0)
    assert t.remaining == 25 * 60


def test_running_timer_counts_down():
    t = PomodoroTimer()
    t.start()
    assert t.tick(1.5) == []
    assert t.remaining == pytest.approx(25 * 60 - 1.5)
    assert t.elapsed == pytest.approx(1.5)
    assert t.progress == pytest.approx(1.5 / 1500)


def test_pause_and_resume_preserves_remaining_time():
    t = PomodoroTimer()
    t.start()
    t.tick(100)
    t.pause()
    t.tick(500)  # wall time passes while paused
    assert t.remaining == pytest.approx(1400)
    t.start()
    t.tick(50)
    assert t.remaining == pytest.approx(1350)


def test_toggle_flips_running_state():
    t = PomodoroTimer()
    assert t.toggle() is True and t.running
    assert t.toggle() is False and not t.running


# ---------------------------------------------------------------- transitions


def test_focus_end_moves_to_short_break():
    t = PomodoroTimer()
    events = finish_session(t)
    assert events == [SessionEnded(ended=F, next=S, skipped=False)]
    assert t.session is S
    assert t.remaining == 5 * 60
    assert t.completed_in_cycle == 1


def test_auto_start_keeps_running_and_carries_leftover_time():
    t = PomodoroTimer(TimerConfig(auto_start=True))
    t.start()
    events = t.tick(25 * 60 + 7)
    assert [e.ended for e in events] == [F]
    assert t.running
    assert t.session is S
    assert t.remaining == pytest.approx(5 * 60 - 7)


def test_without_auto_start_next_session_waits_paused():
    t = PomodoroTimer(TimerConfig(auto_start=False))
    t.start()
    events = t.tick(25 * 60 + 7)
    assert [e.ended for e in events] == [F]
    assert not t.running
    assert t.session is S
    assert t.remaining == 5 * 60  # leftover discarded


def test_long_break_after_every_fourth_focus_session():
    t = PomodoroTimer()
    sequence = [t.session]
    for _ in range(9):
        finish_session(t)
        sequence.append(t.session)
    assert sequence == [F, S, F, S, F, S, F, L, F, S]


def test_cycle_counter_fills_then_resets_after_long_break():
    t = PomodoroTimer()
    seen = []
    for _ in range(9):
        seen.append((t.session, t.dots_filled))
        finish_session(t)
    assert seen == [
        (F, 0), (S, 1), (F, 1), (S, 2), (F, 2), (S, 3), (F, 3), (L, 4), (F, 0),
    ]


def test_custom_long_break_cadence():
    t = make_timer(focus=1, short_break=1, long_break=2, sessions_before_long_break=2)
    sequence = []
    for _ in range(8):
        finish_session(t)
        sequence.append(t.session)
    assert sequence == [S, F, L, F, S, F, L, F]
    assert t.remaining == 60


def test_huge_dt_spans_multiple_sessions_in_order():
    t = make_timer(focus=1, short_break=1, long_break=1, sessions_before_long_break=2)
    t.start()
    events = t.tick(4 * 60 + 30)
    assert [(e.ended, e.next) for e in events] == [(F, S), (S, F), (F, L), (L, F)]
    assert t.session is F
    assert t.remaining == pytest.approx(30)
    assert t.running


def test_absurd_dt_is_bounded():
    t = make_timer(focus=1 / 60, short_break=1 / 60, long_break=1 / 60)
    t.start()
    events = t.tick(1e9)
    assert len(events) == 100  # MAX_TRANSITIONS_PER_TICK


# ----------------------------------------------------------------------- skip


def test_skip_focus_goes_to_break_and_counts_toward_cycle():
    t = PomodoroTimer()
    event = t.skip()
    assert event == SessionEnded(ended=F, next=S, skipped=True)
    assert t.session is S
    assert t.completed_in_cycle == 1
    assert t.remaining == 5 * 60


def test_skip_break_returns_to_focus():
    t = PomodoroTimer()
    t.skip()
    event = t.skip()
    assert event.ended is S and event.next is F
    assert t.session is F
    assert t.completed_in_cycle == 1


def test_skipping_through_a_cycle_reaches_long_break_and_resets():
    t = PomodoroTimer()
    sessions = []
    for _ in range(8):
        t.skip()
        sessions.append(t.session)
    assert sessions == [S, F, S, F, S, F, L, F]
    assert t.completed_in_cycle == 0


def test_skip_preserves_running_state():
    t = PomodoroTimer()
    t.skip()
    assert not t.running
    t.start()
    t.tick(30)
    t.skip()
    assert t.running
    assert t.remaining == 25 * 60  # fresh session, no time consumed


def test_skip_mid_session_discards_remaining_time():
    t = PomodoroTimer()
    t.start()
    t.tick(600)
    t.skip()
    assert t.session is S
    assert t.remaining == 5 * 60


# ---------------------------------------------------------------------- reset


def test_reset_restores_full_duration_and_pauses():
    t = PomodoroTimer()
    t.start()
    t.tick(321)
    t.reset()
    assert t.remaining == 25 * 60
    assert not t.running
    assert t.session is F


def test_reset_keeps_session_and_cycle_position():
    t = PomodoroTimer()
    finish_session(t)
    finish_session(t)
    finish_session(t)  # now in second short break
    t.tick(60)
    t.reset()
    assert t.session is S
    assert t.completed_in_cycle == 2
    assert t.remaining == 5 * 60


# ------------------------------------------------------------------ set_config


def test_set_config_before_start_uses_new_duration():
    t = PomodoroTimer()
    t.set_config(TimerConfig.from_minutes(focus=50))
    assert t.remaining == 50 * 60


def test_set_config_mid_session_keeps_elapsed_time():
    t = PomodoroTimer()
    t.start()
    t.tick(10 * 60)
    t.set_config(TimerConfig.from_minutes(focus=30))
    assert t.remaining == pytest.approx(20 * 60)


def test_set_config_shorter_than_elapsed_ends_on_next_tick():
    t = PomodoroTimer()
    t.start()
    t.tick(20 * 60)
    t.set_config(TimerConfig.from_minutes(focus=15))
    assert t.remaining == 0
    events = t.tick(0.01)
    assert [e.ended for e in events] == [F]


# ----------------------------------------------------------------- formatting


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (1500, "25:00"),
        (1499.2, "25:00"),
        (1499.0, "24:59"),
        (60.0, "01:00"),
        (59.01, "01:00"),
        (0.4, "00:01"),
        (0.0, "00:00"),
        (-3, "00:00"),
        (3600, "60:00"),
        (100 * 60, "100:00"),
    ],
)
def test_format_remaining(seconds, expected):
    assert format_remaining(seconds) == expected
