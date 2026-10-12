import numpy as np
import pytest
import soundfile as sf

from beat_agent.analysis.numbering import number
from beat_agent.analysis.schema import BeatGrid, Confidence, Tempo, TimeSignature
from beat_agent.io.audio import AudioInputError, open_source, read_range, write_like
from beat_agent.io.audition import click_wave, write_click_audition


def _grid(times, downs, dur):
    t = np.array(times)
    pos = np.array([0 if x in downs else 1 for x in times])
    beats, bars, db = number(t, pos, np.zeros(len(t), int), np.ones(len(t)), 2, dur)
    return BeatGrid(source={}, duration=dur, tempo=Tempo(bpm=120, type="constant", segments=[]),
                    time_signature=TimeSignature(numerator=2, denominator=4, beats_per_bar=2),
                    beats=beats, downbeats=db, bars=bars, no_beat_regions=[],
                    confidence=Confidence(overall=1, beat=1, downbeat=1, coverage=1, how=""))


@pytest.mark.parametrize("sr,ch,subtype", [(8000, 1, "PCM_16"), (44100, 2, "PCM_24"), (96000, 2, "FLOAT"),
                                           (48000, 6, "PCM_32")])
def test_click_audition_keeps_format_and_places_clicks(tmp_path, sr, ch, subtype):
    dur = 3.0
    sf.write(tmp_path / "in.wav", np.zeros((int(sr * dur), ch), np.float32), sr, subtype=subtype)
    src = open_source(tmp_path / "in.wav")
    times = [0.5, 1.0, 1.5, 2.0, 2.5]
    info = write_click_audition(src, _grid(times, {0.5, 1.5, 2.5}, dur), tmp_path / "click.wav")
    assert info["n_clicks"] == 5 and info["n_accents"] == 3
    y, sr2 = sf.read(tmp_path / "click.wav", always_2d=True)
    i = sf.info(tmp_path / "click.wav")
    assert (sr2, i.channels, i.subtype) == (sr, ch, subtype)
    # silent input: the first non-zero sample of each click is the beat sample
    for t in times:
        k = int(round(t * sr))
        assert np.abs(y[k - 5:k, 0]).max() == 0
        assert np.abs(y[k:k + 50, 0]).max() > 0.1, t


def test_click_spanning_block_boundary(tmp_path, monkeypatch):
    """A click starting just before the end of one read block must ring on into the next."""
    from beat_agent.io import audio as aio
    real = aio.iter_blocks
    monkeypatch.setattr("beat_agent.io.audition.iter_blocks", lambda src, *a, **k: real(src, block=1000))
    sr = 8000
    sf.write(tmp_path / "in.wav", np.zeros((8000, 1), np.float32), sr, subtype="FLOAT")
    src = open_source(tmp_path / "in.wav")
    t = 990 / sr  # 10 samples before the first block edge
    write_click_audition(src, _grid([t, 0.5, 0.75, 0.9], {t}, 1.0), tmp_path / "c.wav")
    y, _ = sf.read(tmp_path / "c.wav")
    c = click_wave(sr, True)
    np.testing.assert_allclose(y[990:990 + len(c)], c, atol=1e-6)


def test_read_range_pads_outside_file(tmp_path):
    x = (np.arange(100, dtype=np.float32) / 100)[:, None]
    sf.write(tmp_path / "a.wav", x, 1000, subtype="FLOAT")
    src = open_source(tmp_path / "a.wav")
    r = read_range(src, -10, 110)
    assert r.shape == (120, 1)
    assert np.all(r[:10] == 0) and np.all(r[110:] == 0)
    np.testing.assert_allclose(r[10:110, 0], x[:, 0])


def test_pcm_writer_clips_instead_of_wrapping(tmp_path):
    sf.write(tmp_path / "a.wav", np.zeros((10, 1), np.float32), 8000, subtype="PCM_16")
    src = open_source(tmp_path / "a.wav")
    write_like(tmp_path / "b.wav", np.array([[1.5], [-1.5], [0.5]], np.float32), src)
    y, _ = sf.read(tmp_path / "b.wav")
    assert y[0] > 0.99 and y[1] < -0.99   # wrapped values would flip sign


def test_untouched_samples_are_bit_identical(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.integers(-30000, 30000, (5000, 2)).astype(np.int16)
    sf.write(tmp_path / "a.wav", x, 44100, subtype="PCM_16")
    src = open_source(tmp_path / "a.wav")
    write_like(tmp_path / "b.wav", read_range(src, 0, 5000), src)
    y, _ = sf.read(tmp_path / "b.wav", dtype="int16")
    assert np.array_equal(x, y)


@pytest.mark.parametrize("content", [b"", b"not audio at all" * 10])
def test_bad_files_raise_clear_error(tmp_path, content):
    p = tmp_path / "bad.wav"
    p.write_bytes(content)
    with pytest.raises(AudioInputError):
        open_source(p)


def test_missing_file(tmp_path):
    with pytest.raises(AudioInputError, match="not found"):
        open_source(tmp_path / "nope.mp3")
