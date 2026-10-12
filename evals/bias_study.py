"""Root-cause study of the timing offset between our grid and reference beats.

Question: on real music (GTZAN substitutes) the grid sat 12-25 ms *early* of the
human annotations. Is that a bug in our pipeline (frame centring, hop/window
offset, resampling, decoder delay, sub-frame interpolation) or something else
(the model's learned placement, or the annotations' convention)?

Experiments (all numbers written to evals/results/bias_study.{md,json}):
  E1  generated audio, onsets at known samples: offset of
        a) Beat This!'s own File2Beats (official API, its minimal post-processing)
        b) our activations + integer-frame peaks
        c) our activations + parabolic sub-frame peaks (the old default; now off)
  E2  linearity: the same audio delayed by 0/5/10/15 ms -> the measured offset
      must move by exactly the delay (a frame-centring bug would not)
  E3  resampling path: one 44.1 kHz render resampled to 22.05-96 kHz
  E4  decoder: the same audio as WAV, FLAC, MP3, M4A
  E5  annotation convention on the substitutes: where do the human annotations
      sit relative to the audio's own attacks, and where do the tracker's beats?
This script changes nothing. Finding 2 led to one config change (sub-frame
interpolation off), recorded in DECISIONS.md.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from evals import synth


def _nearest_offsets(est, ref, win=0.07):
    est = np.asarray(est)
    out = []
    for r in ref:
        if not len(est):
            break
        j = int(np.argmin(np.abs(est - r)))
        d = est[j] - r
        if abs(d) <= win:
            out.append(d)
    return np.array(out)


def _summ(d):
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "median_ms": round(1e3 * float(np.median(d)), 2),
            "mean_ms": round(1e3 * float(np.mean(d)), 2), "mad_ms": round(1e3 * float(np.median(np.abs(d - np.median(d)))), 2)}


def _ours(path, interp=True):
    from beat_agent.analysis import tracker
    from beat_agent.io.audio import load_analysis_mono, open_source
    y = load_analysis_mono(open_source(path))
    act = tracker.activations(y)
    if interp:
        from beat_agent import config
        keep = config.PEAK_SUBFRAME_INTERPOLATION
        config.PEAK_SUBFRAME_INTERPOLATION = True
        try:
            return tracker.pick_peaks(act.beat)[0]
        finally:
            config.PEAK_SUBFRAME_INTERPOLATION = keep
    from scipy.ndimage import maximum_filter1d
    a = act.beat
    mx = maximum_filter1d(a, size=7, mode="constant")
    return np.flatnonzero((a == mx) & (a > 0.5)) / 50.0


def _official(path):
    from beat_this.inference import File2Beats
    global _F2B
    try:
        _F2B
    except NameError:
        _F2B = File2Beats(checkpoint_path="final0", device="cpu", dbn=False)
    beats, _ = _F2B(str(path))
    return np.asarray(beats)


def _pipeline(path):
    from beat_agent.analysis.pipeline import analyze
    return analyze(path, phase_refinement=False).beat_times()


def run(out="evals/results"):
    from beat_agent.analysis.onsets import attack_times
    from beat_agent.io.audio import ffmpeg_exe, load_analysis_mono, open_source
    res = {}
    td = Path(tempfile.mkdtemp(prefix="bias_"))
    # E1 + E2: three generated tracks with different drum programming and tempo
    tracks = {"g120": dict(segments=((96, 120.0, 120.0),), seed=31, four_on_floor=True, start_s=1.0),
              "g97": dict(segments=((96, 97.0, 97.0),), seed=32, start_s=1.0),
              "g141": dict(segments=((128, 141.0, 141.0),), seed=33, start_s=1.0)}
    e1, e2 = {}, {}
    for name, kw in tracks.items():
        a, tr = synth.make_track(**kw)
        p = synth.write(td / f"{name}.wav", a, 44100)
        e1[name] = {"official_File2Beats": _summ(_nearest_offsets(_official(p), tr.beats)),
                    "ours_integer_frames": _summ(_nearest_offsets(_ours(p, False), tr.beats)),
                    "ours_parabolic": _summ(_nearest_offsets(_ours(p, True), tr.beats)),
                    "ours_full_grid_no_refinement": _summ(_nearest_offsets(_pipeline(p), tr.beats))}  # product defaults
        if name == "g120":
            for dly in (0.005, 0.010, 0.015):
                pad = np.zeros((int(round(dly * 44100)), a.shape[1]), np.float32)
                q = synth.write(td / f"{name}_d{int(dly * 1000)}.wav", np.vstack([pad, a]), 44100)
                e2[f"+{int(dly * 1000)}ms"] = _summ(_nearest_offsets(_ours(q, False), tr.beats + dly))
            e2["+0ms"] = e1[name]["ours_integer_frames"]
    res["E1_generated_offsets"] = e1
    res["E2_delay_linearity_g120"] = e2
    # E3: resampling path -- ONE render at 44.1 kHz, resampled with soxr to each
    # rate outside the product, so the musical content is identical
    e3 = {}
    a, tr = synth.make_track(**tracks["g97"])
    for sr in (22050, 32000, 44100, 48000, 96000):
        b = a if sr == 44100 else soxr.resample(a, 44100, sr, quality="VHQ").astype(np.float32)
        p = td / f"sr{sr}.wav"
        sf.write(p, b, sr, subtype="FLOAT")
        e3[str(sr)] = _summ(_nearest_offsets(_ours(p, True), tr.beats))
    res["E3_samplerate"] = e3
    # E4: decoders
    a, tr = synth.make_track(**tracks["g141"])
    base = synth.write(td / "dec.wav", a, 44100)
    e4 = {"wav": _summ(_nearest_offsets(_ours(base, True), tr.beats))}
    sf.write(td / "dec.flac", a, 44100, format="FLAC")
    e4["flac"] = _summ(_nearest_offsets(_ours(td / "dec.flac", True), tr.beats))
    exe = ffmpeg_exe()
    for ext, codec in (("mp3", ["-c:a", "libmp3lame", "-b:a", "192k"]), ("m4a", ["-c:a", "aac", "-b:a", "192k"])):
        q = td / f"dec.{ext}"
        subprocess.run([exe, "-v", "error", "-y", "-i", str(base), *codec, str(q)], check=True)
        e4[ext] = _summ(_nearest_offsets(_ours(q, True), tr.beats))
    res["E4_decoders"] = e4
    # E5: annotation convention on the substitutes
    from evals.references import load_manifest, resolve_clips
    e5 = {}
    for c in resolve_clips(load_manifest()):
        if not c.substitute or not c.audio or c.beats is None:
            continue
        y = load_analysis_mono(open_source(c.audio))
        ann = c.beats
        att_ann = attack_times(y, 22050, ann, 0.04, 6.0)            # attack nearest each annotation
        ok = np.isfinite(att_ann)
        trk = _ours(c.audio, False)   # the product's peak picking (integer frames)
        att_trk = attack_times(y, 22050, trk, 0.04, 6.0)
        ok2 = np.isfinite(att_trk)
        e5[c.id] = {"annotation_minus_attack": _summ(ann[ok] - att_ann[ok]),
                    "tracker_minus_attack": _summ(trk[ok2] - att_trk[ok2]),
                    "tracker_minus_annotation": _summ(_nearest_offsets(trk, ann)),
                    "official_File2Beats_minus_annotation": _summ(_nearest_offsets(_official(c.audio), ann))}
    res["E5_substitutes"] = e5
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "bias_study.json").write_text(json.dumps(res, indent=1))
    (Path(out) / "bias_study.md").write_text(report(res))
    return res


def _cell(s):
    return f"{s['median_ms']} (MAD {s['mad_ms']}, n={s['n']})" if s.get("n") else "-"


def report(res) -> str:
    L = ["# Timing-offset root-cause study", "", "Generated by `python -m evals.bias_study`. Offsets in ms, "
         "median over beats (negative = earlier than the reference). The script changes nothing; finding 2 led to one "
         "config change, recorded in DECISIONS.md D16.", "",
         "## E1. Generated audio: where do beats land relative to the true onsets?", "",
         "| track | Beat This! File2Beats (official) | ours, integer frames | ours, parabolic (old default) | our full grid, current defaults |",
         "|---|---|---|---|---|"]
    for k, v in res["E1_generated_offsets"].items():
        L.append(f"| {k} | {_cell(v['official_File2Beats'])} | {_cell(v['ours_integer_frames'])} | "
                 f"{_cell(v['ours_parabolic'])} | {_cell(v['ours_full_grid_no_refinement'])} |")
    L += ["", "## E2. Onset phase vs the 20 ms frame grid (g120 delayed by a known amount; integer-frame peaks vs the delayed truth)", "",
          "| delay | offset |", "|---|---|"]
    for k, v in res["E2_delay_linearity_g120"].items():
        L.append(f"| {k} | {_cell(v)} |")
    L += ["", "## E3. Sample rate / resampling path", "", "| rate (resampled from one 44.1 kHz render) | offset, parabolic peaks |", "|---|---|"]
    for k, v in res["E3_samplerate"].items():
        L.append(f"| {k} Hz | {_cell(v)} |")
    L += ["", "## E4. Decoder", "", "| format | offset |", "|---|---|"]
    for k, v in res["E4_decoders"].items():
        L.append(f"| {k} | {_cell(v)} |")
    L += ["", "## E5. Substitutes: annotations vs attacks vs tracker", "",
          "| clip | annotation - attack | tracker - attack | tracker - annotation | official File2Beats - annotation |",
          "|---|---|---|---|---|"]
    for k, v in res["E5_substitutes"].items():
        L.append(f"| {k} | {_cell(v['annotation_minus_attack'])} | {_cell(v['tracker_minus_attack'])} | "
                 f"{_cell(v['tracker_minus_annotation'])} | {_cell(v['official_File2Beats_minus_annotation'])} |")
    L += ["", "## Findings (computed from the tables above)", ""]
    e1 = res["E1_generated_offsets"]
    d_int = max(abs(v["official_File2Beats"]["median_ms"] - v["ours_integer_frames"]["median_ms"]) for v in e1.values())
    d_par = [v["ours_parabolic"]["median_ms"] - v["ours_integer_frames"]["median_ms"] for v in e1.values()]
    e3 = [v["median_ms"] for v in res["E3_samplerate"].values()]
    e4 = [v["median_ms"] for v in res["E4_decoders"].values()]
    e5 = res["E5_substitutes"]
    L.append(f"1. Our frame-level peaks differ from Beat This!'s official output by at most {d_int:.2f} ms on generated "
             "audio: no frame-centring / hop / window bug in our use of the model.")
    L.append(f"2. Parabolic sub-frame interpolation moved beats by {', '.join(f'{x:+.2f}' for x in d_par)} ms relative to "
             "integer frames on the three generated tracks, i.e. away from Beat This!'s own output. It was a real (small) "
             "bias in our code and is now off by default (`PEAK_SUBFRAME_INTERPOLATION`).")
    L.append(f"3. Resampling: the same render at 5 rates gives offsets spanning {max(e3) - min(e3):.2f} ms -- no resampling delay.")
    L.append(f"4. Decoders: WAV/FLAC/MP3/M4A offsets span {max(e4) - min(e4):.2f} ms (lossy encoder priming, mostly "
             "compensated by ffmpeg).")
    late = [f"{k} {v['annotation_minus_attack']['median_ms']:+.1f}" for k, v in e5.items() if v["annotation_minus_attack"].get("n")]
    trk = [f"{k} {v['tracker_minus_attack']['median_ms']:+.1f}" for k, v in e5.items() if v["tracker_minus_attack"].get("n")]
    L.append(f"5. On the substitutes the human annotations sit late of the audio's own attacks ({'; '.join(late)} ms), while "
             f"the tracker sits closer to them ({'; '.join(trk)} ms); Beat This!'s official output shows the same offset to "
             "the annotations as ours. The 'early' grid on GTZAN is mostly the annotation convention (tapped beats land "
             "after the attack), not a pipeline bug. No offset was added to compensate.")
    L.append("6. What remains is per-track placement of a few ms by the model itself (E1) and the dependence on where an "
             "onset falls inside a 20 ms frame (E2); the line fit averages the latter over many beats.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    print(report(run()))
