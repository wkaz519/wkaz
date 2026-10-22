"""Pure Pomodoro timer logic.

This module deliberately has no pygame dependency so it can be unit-tested
and reasoned about in isolation. Time only advances when the caller feeds
elapsed seconds into :meth:`PomodoroTimer.tick`.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

# Safety valve for absurd ``dt`` values (e.g. resuming from a very long
# system sleep with tiny session lengths): never process more than this many
# session transitions in a single tick.
MAX_TRANSITIONS_PER_TICK = 100


class SessionType(enum.Enum):
    FOCUS = "focus"
    SHORT_BREAK = "short_break"
    LONG_BREAK = "long_break"

    @property
    def label(self) -> str:
        return _LABELS[self]

    @property
    def is_break(self) -> bool:
        return self is not SessionType.FOCUS


_LABELS = {
    SessionType.FOCUS: "Focus",
    SessionType.SHORT_BREAK: "Short Break",
    SessionType.LONG_BREAK: "Long Break",
}


@dataclass(frozen=True)
class TimerConfig:
    """Session lengths (in seconds) and cycle behaviour."""

    focus: float = 25 * 60
    short_break: float = 5 * 60
    long_break: float = 15 * 60
    sessions_before_long_break: int = 4
    # Start the next session automatically when one ends. When False the
    # next session is loaded but stays paused until the user starts it.
    auto_start: bool = True

    def __post_init__(self) -> None:
        for name in ("focus", "short_break", "long_break"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
                raise ValueError(f"{name} duration must be a positive number of seconds, got {value!r}")
        if not isinstance(self.sessions_before_long_break, int) or self.sessions_before_long_break < 1:
            raise ValueError(
                "sessions_before_long_break must be an integer >= 1, "
                f"got {self.sessions_before_long_break!r}"
            )

    @classmethod
    def from_minutes(
        cls,
        focus: float = 25,
        short_break: float = 5,
        long_break: float = 15,
        sessions_before_long_break: int = 4,
        auto_start: bool = True,
    ) -> TimerConfig:
        return cls(
            focus=focus * 60,
            short_break=short_break * 60,
            long_break=long_break * 60,
            sessions_before_long_break=sessions_before_long_break,
            auto_start=auto_start,
        )

    def duration(self, session: SessionType) -> float:
        if session is SessionType.FOCUS:
            return self.focus
        if session is SessionType.SHORT_BREAK:
            return self.short_break
        return self.long_break


@dataclass(frozen=True)
class SessionEnded:
    """Emitted whenever a session finishes (naturally or by skipping)."""

    ended: SessionType
    next: SessionType
    skipped: bool


class PomodoroTimer:
    """Pomodoro state machine: focus → short break → … → long break.

    A long break follows every ``sessions_before_long_break`` focus sessions.
    Skipping a session counts as finishing it, so the cycle cadence (and the
    session dots in the UI) behave the same whether a session ran out or was
    skipped.
    """

    def __init__(self, config: TimerConfig | None = None) -> None:
        self.config = config or TimerConfig()
        self.session = SessionType.FOCUS
        self.remaining = self.config.duration(self.session)
        self.running = False
        # Focus sessions finished in the current cycle (resets after a long break).
        self.completed_in_cycle = 0

    # ------------------------------------------------------------------ state
    @property
    def duration(self) -> float:
        return self.config.duration(self.session)

    @property
    def elapsed(self) -> float:
        return self.duration - self.remaining

    @property
    def progress(self) -> float:
        """Fraction of the current session that has elapsed, in [0, 1]."""
        return min(1.0, max(0.0, self.elapsed / self.duration))

    @property
    def cycle_length(self) -> int:
        return self.config.sessions_before_long_break

    @property
    def dots_filled(self) -> int:
        """Number of completed focus sessions to show in the cycle counter."""
        return min(self.completed_in_cycle, self.cycle_length)

    # --------------------------------------------------------------- controls
    def start(self) -> None:
        self.running = True

    def pause(self) -> None:
        self.running = False

    def toggle(self) -> bool:
        """Start if paused, pause if running. Returns the new running state."""
        self.running = not self.running
        return self.running

    def reset(self) -> None:
        """Restart the current session from its full duration, paused."""
        self.remaining = self.duration
        self.running = False

    def skip(self) -> SessionEnded:
        """Finish the current session immediately.

        The running state carries over: skipping while running starts the
        next session right away, skipping while paused leaves it paused.
        """
        return self._advance(skipped=True)

    def set_config(self, config: TimerConfig) -> None:
        """Apply new durations, keeping the time already spent in this session."""
        elapsed = self.elapsed
        self.config = config
        self.remaining = max(0.0, self.duration - elapsed)

    # ------------------------------------------------------------------- time
    def tick(self, dt: float) -> list[SessionEnded]:
        """Advance the clock by ``dt`` seconds of wall time.

        Returns the sessions that ended during this tick (usually none, one
        at a session boundary, possibly several after a very long ``dt``).
        Time left over after a boundary carries into the next session when
        ``auto_start`` is on and is discarded otherwise.
        """
        events: list[SessionEnded] = []
        if not self.running or dt <= 0:
            return events
        for _ in range(MAX_TRANSITIONS_PER_TICK):
            if dt < self.remaining:
                self.remaining -= dt
                break
            dt -= self.remaining
            events.append(self._advance(skipped=False))
            self.running = self.config.auto_start
            if not self.running:
                break
        return events

    def _advance(self, skipped: bool) -> SessionEnded:
        ended = self.session
        if ended is SessionType.FOCUS:
            self.completed_in_cycle += 1
            if self.completed_in_cycle >= self.cycle_length:
                upcoming = SessionType.LONG_BREAK
            else:
                upcoming = SessionType.SHORT_BREAK
        else:
            if ended is SessionType.LONG_BREAK:
                self.completed_in_cycle = 0
            upcoming = SessionType.FOCUS
        self.session = upcoming
        self.remaining = self.duration
        return SessionEnded(ended=ended, next=upcoming, skipped=skipped)


def format_remaining(seconds: float) -> str:
    """Format a countdown as ``MM:SS``.

    Rounds *up* to whole seconds so a fresh 25-minute session reads "25:00"
    and "00:00" only appears once the session has actually ended.
    """
    total = max(0, math.ceil(seconds - 1e-6))
    minutes, secs = divmod(total, 60)
    return f"{minutes:02d}:{secs:02d}"
