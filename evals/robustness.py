"""Generalisation + robustness suite on audio generated in code.

Every case renders audio with exact ground truth, runs the *full* public
pipeline (`analyze` + click audition), and checks the grid against the truth.
Statuses:
  pass     correct grid (or the expected clear message for beatless input)
  partial  usable but with a known, reported limitation (octave error with the
           right alternative exposed, 6/8 <-> 12/8 hypermeter, drift > 10 ms)
  fail     wrong grid or crash
Run:  uv run python -m evals.robustness [--quick] [--out evals/results]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

from evals import synth

TARGET_BEAT_F = 0.90
TARGET_DOWNBEAT_F = 0.80
TARGET_DRIFT_MS = 10.0


@dataclass
class Case:
    name: str
    group: str
    make: callable                      # () -> (audio, truth, sr, subtype)
    fmt: str = "wav"                    # wav | flac | mp3 | m4a | ogg
    slow: bool = False
    expect: str = "grid"                # grid | message


@dataclass
class Result:
    name: str
    group: str
    status: str
    checks: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    runtime_s: float = 0.0
    audio_s: float = 0.0


def _music(seed=0, sr=44100, channels=2, subtype="PCM_16", **kw):
    def f():
        a, t = synth.make_track(sr=sr, channels=channels, seed=seed, **kw)
        return a, t, sr, subtype
    return f


def cases(quick: bool = False) -> list[Case]:
    out: list[Case] = []
    # --- generalisation: random tempo / meter / offset, never seen before
    # Fixed default seed so reports are reproducible; set BEAT_AGENT_SYNTH_SEED
    # to draw a fresh, never-seen set of tempos / meters / offsets.
    import os
    rng = np.random.default_rng(int(os.environ.get("BEAT_AGENT_SYNTH_SEED", "20261012")))
    for i in range(6 if quick else 12):
        seed = int(rng.integers(1_000_000))
        kw = synth.random_case(seed, min_bpm=70, max_bpm=175)
        out.append(Case(f"random_{i:02d}_{kw['meter'].replace('/', '-')}_{kw['segments'][0][1]:.1f}bpm", "generalisation",
                        _music(**kw)))
    # --- robustness: tempo extremes, meters, changes
    for bpm in (60, 90, 128, 160, 200):
        out.append(Case(f"tempo_{bpm}", "robustness", _music(seed=bpm, segments=((96, bpm, bpm),), start_s=0.7)))
    for meter, bpm in (("3/4", 132), ("4/4", 104), ("6/8", 84)):
        bpb = synth.METERS[meter][0]
        out.append(Case(f"meter_{meter.replace('/', '-')}_{bpm}", "robustness",
                        _music(seed=bpm + bpb, segments=((bpb * 24, bpm, bpm),), meter=meter, start_s=1.1)))
    out.append(Case("tempo_step_118_to_126", "robustness",
                    _music(seed=5, segments=((64, 118, 118), (64, 126, 126)), start_s=0.5)))
    out.append(Case("tempo_step_140_to_120", "robustness",
                    _music(seed=6, segments=((64, 140, 140), (64, 120, 120)), start_s=0.5)))
    out.append(Case("tempo_ramp_96_to_120", "robustness", _music(seed=7, segments=((96, 96, 120),), start_s=0.5)))
    out.append(Case("swing_4-4_100", "robustness", _music(seed=8, segments=((96, 100, 100),), swing=1.0)))
    out.append(Case("pad_intro_45s", "robustness", _music(seed=9, segments=((96, 122, 122),), intro_pad_s=45.0)))
    out.append(Case("silence_20s", "robustness", lambda: (*synth.make_silence(), 44100, "PCM_16"), expect="message"))
    out.append(Case("white_noise_20s", "robustness", lambda: (*synth.make_noise(), 44100, "PCM_16"), expect="message"))
    out.append(Case("short_3s", "robustness", _music(seed=10, segments=((6, 120, 120),), start_s=0.2, tail_s=0.3)))
    out.append(Case("sr_8k_mono", "formats", _music(seed=11, sr=8000, channels=1, segments=((64, 126, 126),))))
    out.append(Case("sr_96k_stereo_24bit", "formats", _music(seed=12, sr=96000, subtype="PCM_24", segments=((64, 140, 140),))))
    out.append(Case("float32_6ch_48k", "formats", _music(seed=13, sr=48000, channels=6, subtype="FLOAT", segments=((64, 112, 112),))))
    out.append(Case("flac_16bit", "formats", _music(seed=14, segments=((64, 124, 124),)), fmt="flac"))
    out.append(Case("mp3_roundtrip", "formats", _music(seed=15, segments=((64, 124, 124),)), fmt="mp3"))
    out.append(Case("m4a_aac_roundtrip", "formats", _music(seed=16, segments=((64, 124, 124),)), fmt="m4a"))
    out.append(Case("ogg_vorbis", "formats", _music(seed=17, segments=((64, 124, 124),)), fmt="ogg"))
    out.append(Case("long_12min", "robustness",
                    _music(seed=18, channels=1, segments=((4 * 372, 124, 124),), four_on_floor=True), slow=True))
    return out


# --------------------------------------------------------------------- render
def _write(case: Case, audio, sr, subtype, d: Path) -> Path:
    wav = d / f"{case.name}.wav"
    sf.write(str(wav), audio, sr, subtype=subtype)
    if case.fmt == "wav":
        return wav
    if case.fmt == "flac":
        p = d / f"{case.name}.flac"
        sf.write(str(p), audio, sr, subtype="PCM_16", format="FLAC")
        return p
    if case.fmt == "ogg":
        # libsndfile's Vorbis encoder overflows the stack on Windows when handed
        # one large buffer; write in blocks.
        p = d / f"{case.name}.ogg"
        with sf.SoundFile(str(p), "w", sr, audio.shape[1], format="OGG", subtype="VORBIS") as f:
            for i in range(0, len(audio), 8192):
                f.write(audio[i:i + 8192])
        return p
    from beat_agent.io.audio import ffmpeg_exe
    exe = ffmpeg_exe()
    p = d / f"{case.name}.{case.fmt}"
    codec = ["-c:a", "libmp3lame", "-b:a", "192k"] if case.fmt == "mp3" else ["-c:a", "aac", "-b:a", "192k"]
    subprocess.run([exe, "-v", "error", "-y", "-i", str(wav), *codec, str(p)], check=True)
    return p


# --------------------------------------------------------------------- checks
def _level_truth(truth: synth.Truth, grid_bpm: float, truth_bpm: float):
    """Truth beats at the grid's metrical level, if that level is legitimate."""
    r = grid_bpm / truth_bpm
    if truth.subdivision == 3 and abs(r - 3) < 0.15:
        b = truth.beats
        sub = np.concatenate([b[:-1, None] + np.diff(b)[:, None] * np.arange(3)[None, :] / 3]).ravel()
        return np.append(sub, b[-1]), "eighth-note level (compound meter)"
    return truth.beats, None


def check(case: Case, grid, truth: synth.Truth, audition_info: dict, src) -> Result:
    res = Result(case.name, case.group, "pass")
    c, notes = res.checks, res.notes
    est = grid.beat_times()
    if case.expect == "message":
        c["n_beats"] = len(est)
        c["message"] = grid.messages[0] if grid.messages else None
        if len(est) or not grid.messages:
            res.status = "fail"
            notes.append("beats emitted for beatless input" if len(est) else "no message")
        return res
    if not len(est):
        res.status = "fail"
        notes.append("no grid: " + "; ".join(grid.messages))
        return res
    truth_bpm = float(60.0 / np.median(np.diff(truth.beats)))
    ref, level_note = _level_truth(truth, grid.tempo.bpm or truth_bpm, truth_bpm)
    if level_note:
        notes.append(level_note)
    f = mir_eval.beat.f_measure(ref, est, 0.07)
    c["beat_f"] = round(float(f), 4)
    if f < TARGET_BEAT_F:
        r = (grid.tempo.bpm or 0) / truth_bpm
        alt = {a.factor for a in grid.tempo.alternatives}
        if abs(r - 0.5) < 0.05 or abs(r - 2) < 0.1:
            fixed = mir_eval.beat.f_measure(truth.beats[::2] if r < 1 else ref, est if r < 1 else est[::2], 0.07)
            res.status = "partial"
            notes.append(f"tempo octave error: tracked {grid.tempo.bpm} BPM vs {truth_bpm:.1f}; "
                         f"x{1 / r:.0f} alternative {'exposed' if (2.0 if r < 1 else 0.5) in alt else 'MISSING'}; "
                         f"F at the tracked level {fixed:.3f}")
        else:
            res.status = "fail"
    # timing error of matched beats
    if len(ref):
        nearest = np.array([ref[np.argmin(np.abs(ref - t))] for t in est])
        err = est - nearest
        m = np.abs(err) <= 0.07
        if m.any():
            c["max_abs_err_ms"] = round(float(1e3 * np.abs(err[m]).max()), 2)
            c["mean_err_ms"] = round(float(1e3 * err[m].mean()), 2)
            if truth.is_constant and grid.tempo.type == "constant" and c["max_abs_err_ms"] > TARGET_DRIFT_MS:
                notes.append(f"constant grid off by up to {c['max_abs_err_ms']} ms (> {TARGET_DRIFT_MS} ms)")
                if res.status == "pass":
                    res.status = "partial"
    # no beats in the intro / before the music
    early = est[est < truth.beat_start_s - 0.1]
    c["beats_before_music"] = int(len(early))
    if len(early):
        res.status = "fail"
        notes.append(f"{len(early)} beats emitted before the music starts at {truth.beat_start_s:.2f}s")
    # tempo type
    want = "constant" if truth.is_constant else "variable"
    c["tempo_type"] = grid.tempo.type
    if grid.tempo.type != want:
        res.status = "fail" if res.status != "fail" else res.status
        notes.append(f"tempo type {grid.tempo.type}, expected {want}")
    # step changes located
    steps = [s for s in truth.tempo_segments[1:]] if len(truth.tempo_segments) > 1 and all(
        s[2] == s[3] for s in truth.tempo_segments) else []
    if steps:
        t_change = truth.beats[steps[0][0]]
        found = [ch for ch in grid.tempo.changes if ch["kind"] == "step"]
        c["tempo_change_found"] = bool(found)
        if found:
            d = min(abs(ch["time"] - t_change) for ch in found)
            c["tempo_change_err_s"] = round(float(d), 4)
            if d > 2 * 60.0 / truth_bpm:
                res.status = "fail"
                notes.append(f"tempo change located {d:.2f}s from truth")
        else:
            res.status = "fail"
            notes.append("tempo step not detected")
    # meter + downbeats
    ts = grid.time_signature
    want_ts = truth.meter
    got_ts = f"{ts.numerator}/{ts.denominator}" if ts.numerator else None
    c["meter"] = got_ts
    if truth.duration_s < 8.0:
        notes.append("short clip: meter not required")
    else:
        if got_ts != want_ts:
            compound = {"6/8", "12/8"}
            if got_ts in compound and want_ts in compound:
                notes.append(f"hypermeter: {got_ts} for {want_ts}")
                if res.status == "pass":
                    res.status = "partial"
            else:
                res.status = "fail"
                notes.append(f"meter {got_ts}, expected {want_ts}")
        fd = mir_eval.beat.f_measure(truth.downbeats, np.array(grid.downbeats), 0.07) if grid.downbeats else 0.0
        c["downbeat_f"] = round(float(fd), 4)
        if fd < TARGET_DOWNBEAT_F and res.status == "pass":
            res.status = "fail"
            notes.append("downbeats wrong")
    # export format preserved
    info = sf.info(audition_info["path"])
    c["audition_format"] = f"{info.samplerate}Hz/{info.channels}ch/{info.subtype}"
    if info.samplerate != src.samplerate or info.channels != src.channels or info.subtype != src.export_subtype:
        res.status = "fail"
        notes.append(f"audition format {c['audition_format']} != source")
    return res


def run(quick=False, include_slow=False, out_dir="evals/results", work_dir=None, only=None) -> list[Result]:
    from beat_agent.analysis.pipeline import analyze
    from beat_agent.io.audio import open_source
    from beat_agent.io.audition import write_click_audition
    work = Path(work_dir or "evals/out/robustness")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    results = []
    for case in cases(quick):
        if case.slow and not include_slow:
            continue
        if only and case.name not in only:
            continue
        try:
            audio, truth, sr, subtype = case.make()
            path = _write(case, audio, sr, subtype, work)
            del audio
            t0 = time.perf_counter()
            src = open_source(path)
            grid = analyze(path, src)
            rt = time.perf_counter() - t0
            aud = write_click_audition(src, grid, work / f"{case.name}.click.wav")
            (work / f"{case.name}.grid.json").write_text(grid.model_dump_json())
            (work / f"{case.name}.truth.json").write_text(json.dumps(truth.to_dict()))
            r = check(case, grid, truth, aud, src)
            r.runtime_s, r.audio_s = round(rt, 2), round(src.duration, 2)
        except Exception as e:  # a crash is a failure, recorded, never hidden
            r = Result(case.name, case.group, "fail", notes=[f"crash: {e!r}", traceback.format_exc(limit=3)])
        print(f"[{r.status:7s}] {r.name:34s} {r.checks} {'; '.join(r.notes)}", flush=True)
        results.append(r)
    _write_report(results, Path(out_dir))
    return results


def _write_report(results: list[Result], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "robustness.json").write_text(json.dumps([r.__dict__ for r in results], indent=1))
    lines = ["# Generalisation and robustness (generated audio)", "",
             "Written by `python -m evals.robustness`. Ground truth is exact (generated in code).", "",
             "| case | group | status | beat F | downbeat F | max err ms | meter | tempo | runtime s / audio s | notes |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r.checks
        lines.append(f"| {r.name} | {r.group} | **{r.status}** | {c.get('beat_f', '-')} | {c.get('downbeat_f', '-')} | "
                     f"{c.get('max_abs_err_ms', '-')} | {c.get('meter', '-')} | {c.get('tempo_type', '-')} | "
                     f"{r.runtime_s} / {r.audio_s} | {'; '.join(n.splitlines()[0] for n in r.notes) or ''} |")
    counts = {s: sum(1 for r in results if r.status == s) for s in ("pass", "partial", "fail")}
    lines += ["", f"Totals: {counts['pass']} pass, {counts['partial']} partial, {counts['fail']} fail of {len(results)}."]
    (out / "robustness.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--slow", action="store_true", help="include the 12-minute clip")
    ap.add_argument("--out", default="evals/results")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    run(quick=a.quick, include_slow=a.slow, out_dir=a.out, only=a.only)
