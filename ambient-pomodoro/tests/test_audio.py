import wave

import numpy as np
import pytest

from pomodoro.audio import CHIME_NOTES_HZ, ChimePlayer, synthesize_chime, to_pcm, write_wav


@pytest.mark.parametrize("kind", list(CHIME_NOTES_HZ))
def test_chime_is_bounded_audible_and_click_free(kind):
    sr = 22050
    x = synthesize_chime(sr, kind)
    assert x.dtype == np.float32
    assert 1.5 * sr < len(x) < 5 * sr
    assert np.max(np.abs(x)) == pytest.approx(0.8, abs=1e-3)
    assert abs(x[0]) < 1e-3 and abs(x[-1]) < 1e-3  # no click at either end
    # Gentle: the tail is much quieter than the attack.
    quarter = len(x) // 4
    assert np.sqrt(np.mean(x[-quarter:] ** 2)) < 0.5 * np.sqrt(np.mean(x[:quarter] ** 2))


def test_chime_pitch_matches_first_note():
    sr = 44100
    x = synthesize_chime(sr, "break_end")[: int(0.2 * sr)]
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    peak_hz = np.argmax(spectrum) * sr / len(x)
    assert peak_hz == pytest.approx(CHIME_NOTES_HZ["break_end"][0], rel=0.03)


@pytest.mark.parametrize("size, dtype", [(-16, np.int16), (16, np.uint16), (-8, np.int8), (8, np.uint8), (32, np.float32)])
def test_pcm_formats(size, dtype):
    pcm = to_pcm(np.array([-0.8, 0.0, 0.8], np.float32), size, 2)
    assert pcm.dtype == dtype and pcm.shape == (3, 2)


def test_write_wav(tmp_path):
    path = tmp_path / "chime.wav"
    write_wav(path, synthesize_chime(8000), 8000)
    with wave.open(str(path)) as w:
        assert w.getframerate() == 8000 and w.getnchannels() == 1 and w.getnframes() > 8000


def test_player_respects_mute():
    player = ChimePlayer(muted=True)
    player.play("focus_end")  # must not raise, with or without a device
    assert player.toggle_mute() is False
    player.play("focus_end")
