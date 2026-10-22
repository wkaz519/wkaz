"""Programmatically generated chimes (no audio files needed).

Synthesis is pure NumPy so it can be tested without an audio device;
:class:`ChimePlayer` wraps the result in pygame mixer sounds and degrades to
silence if no audio output is available.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# Played when a focus session ends (time to rest) and when a break ends
# (time to focus): a soft descending and ascending major triad respectively.
CHIME_NOTES_HZ = {
    "focus_end": (783.99, 659.25, 523.25),  # G5 E5 C5
    "break_end": (523.25, 659.25, 783.99),  # C5 E5 G5
}

# Bell-like partials: (frequency ratio, relative amplitude, decay time in s).
# Slightly inharmonic upper partials give a glassy, singing-bowl quality.
_PARTIALS = (
    (1.0, 1.00, 1.60),
    (2.0, 0.16, 0.90),
    (2.76, 0.20, 0.60),
    (5.40, 0.05, 0.30),
)
_NOTE_SPACING_S = 0.24
_TAIL_S = 2.4
_ATTACK_S = 0.012


def synthesize_note(freq: float, sample_rate: int, length_s: float) -> np.ndarray:
    t = np.arange(int(sample_rate * length_s), dtype=np.float64) / sample_rate
    out = np.zeros_like(t)
    for ratio, amp, decay in _PARTIALS:
        f = freq * ratio
        # Two slightly detuned copies beat gently, like a real struck bell.
        tone = np.sin(2 * np.pi * f * t) + 0.6 * np.sin(2 * np.pi * (f + 1.3) * t + 0.7)
        out += amp * tone * np.exp(-t / decay)
    attack = np.clip(t / _ATTACK_S, 0.0, 1.0)
    out *= 0.5 - 0.5 * np.cos(np.pi * attack)  # click-free raised-cosine attack
    return out


def synthesize_chime(sample_rate: int = 44100, kind: str = "focus_end") -> np.ndarray:
    """Return a mono float32 chime in [-1, 1]."""
    notes = CHIME_NOTES_HZ[kind]
    total_s = _NOTE_SPACING_S * (len(notes) - 1) + _TAIL_S
    out = np.zeros(int(sample_rate * total_s), dtype=np.float64)
    for i, freq in enumerate(notes):
        start = int(i * _NOTE_SPACING_S * sample_rate)
        note = synthesize_note(freq, sample_rate, total_s - i * _NOTE_SPACING_S)
        # Later notes slightly softer so the phrase settles.
        out[start : start + len(note)] += note[: len(out) - start] * (1.0 - 0.12 * i)
    # Fade the last 300 ms to exactly zero to avoid a click at the end.
    fade = min(len(out), int(0.3 * sample_rate))
    out[-fade:] *= np.linspace(1.0, 0.0, fade) ** 2
    out *= 0.8 / max(1e-9, float(np.max(np.abs(out))))
    return out.astype(np.float32)


def to_pcm(samples: np.ndarray, size: int, channels: int) -> np.ndarray:
    """Convert mono float samples to the mixer's sample format/channel layout.

    ``size`` follows pygame's convention: negative means signed, the absolute
    value is bits per sample, and 32 means float.
    """
    if size == 32:
        pcm = samples.astype(np.float32)
    elif size == -16:
        pcm = (samples * 32767).astype(np.int16)
    elif size == 16:
        pcm = (samples * 32767 + 32768).astype(np.uint16)
    elif size == -8:
        pcm = (samples * 127).astype(np.int8)
    elif size == 8:
        pcm = (samples * 127 + 128).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported mixer sample size: {size}")
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    return np.ascontiguousarray(pcm)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = 44100) -> None:
    """Save mono float samples as a 16-bit WAV (handy for previewing)."""
    pcm = to_pcm(samples, -16, 1)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


class ChimePlayer:
    """Plays chimes through pygame's mixer; silently no-ops without audio."""

    def __init__(self, volume: float = 0.6, muted: bool = False) -> None:
        self.muted = muted
        self.volume = volume
        self._sounds: dict = {}
        try:
            import pygame

            if not pygame.mixer.get_init():
                pygame.mixer.init()
            freq, size, channels = pygame.mixer.get_init()
            for kind in CHIME_NOTES_HZ:
                pcm = to_pcm(synthesize_chime(freq, kind), size, channels)
                sound = pygame.sndarray.make_sound(pcm)
                sound.set_volume(volume)
                self._sounds[kind] = sound
        except Exception as exc:  # no audio device, unsupported format, ...
            log.warning("Audio unavailable, chimes disabled: %s", exc)
            self._sounds = {}

    @property
    def available(self) -> bool:
        return bool(self._sounds)

    def toggle_mute(self) -> bool:
        self.muted = not self.muted
        return self.muted

    def play(self, kind: str = "focus_end") -> None:
        if self.muted or kind not in self._sounds:
            return
        self._sounds[kind].play()
