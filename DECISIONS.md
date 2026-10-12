# Decisions

Status: initial version written at Phase 0. Each decision is revisited at the end
of the phase that implements it; "measured" notes link to eval output.

## Key decisions

### D1. Beat tracker: Beat This! activations + our own grid post-processing
- **Options:** Beat This! with its "minimal" peak picking; Beat This! + madmom DBN;
  madmom RNN+DBN; BeatNet; librosa.
- **Choice:** Beat This! (`final0`, CPU) for frame-wise beat/downbeat activations, then our
  own post-processing: peak picking with sub-frame (parabolic) interpolation → beat
  regions → tempo segmentation → per-segment linear fit for constant-tempo segments →
  downbeat/meter decoding with a small HMM over bar positions.
- **Trade-off:** more code of our own than calling `File2Beats`, but the brief's targets
  (constant vs variable flag, tempo segments, ≤10 ms drift, no hallucinated beats in a
  drumless intro, 6/8) are all post-processing questions the stock decoders don't answer.
  madmom does not build on Python 3.11/Windows without a compiler, so the DBN is out.
- **Would change my mind:** if the regularised grid is *worse* than raw peaks on real
  steady-tempo music (e.g. subtle tempo drift in "constant" tracks), the constant-segment
  test must get stricter.
- librosa's tracker is kept only as the baseline row in the eval table.

### D2. Constant-tempo grids are fitted, not peak-picked
Beat This! frames are 20 ms; peak times jitter by up to ±10 ms. A 6-minute track has
~750 beats, so a least-squares line through them pins phase and period far more
precisely than any single peak. A segment is only treated as constant if the residual,
smoothed over 16 beats, stays within 10 ms (the brief's own drift tolerance) — so a live
band that drifts is *not* forced onto a straight line.

### D3. Edits are a virtual clip timeline over the immutable source
Each edit state is a list of clips `(source_start, source_end, gain envelope)`. Rendering
streams from the original file. Consequences: undo/redo is a list of snapshots (cheap);
untouched samples are bit-identical to the input; the edited grid is the source grid
mapped through the same clip list (never re-detected), so "the grid stays true" by
construction. Crossfades are centred on the splice point and use source audio just
beyond each clip's edge, so output length — and every grid time — is exact.

### D4. The LLM never writes JavaScript (Part 3)
Score JSON → deterministic HTML template. Generated JS running in a user's browser is a
code-execution risk (network calls, exfiltration, infinite loops), is not reproducible
run to run, and cannot be unit-tested. Typed ops (`SetTrackMute`, `SetPatchParam`,
`SetTempo`) cover the brief's chat changes.

### D5. Sync uses a Rubber Band time map, not a single ratio
B is stretched with `rubberband --timemap` mapping each of B's grid beats onto the
corresponding beat of A's grid. For two constant-tempo tracks this is exactly the global
ratio + offset (which we also report); for variable-tempo tracks it stays locked instead
of drifting.

### D6. LLM: Gemini 3 with JSON-schema structured output, behind a provider interface
Gemini via `google-genai`, model from `GEMINI_MODEL` (default `gemini-3.5-flash`), checked
against `models.list` at startup; 2.0/2.5 models are refused (2.5 shuts down
16 Oct 2026). Output is validated with pydantic; one retry with the validation error;
never execute a partially parsed plan. Responses cached on (model, prompt, grid state) so
eval re-runs cost nothing.

### D7. Basic Pitch via ONNX, not TensorFlow
Basic Pitch hard-requires TensorFlow on Python 3.11. Its wheel ships the same model as
ONNX; we override TensorFlow out (`[tool.uv] override-dependencies`) and run
onnxruntime. Smaller install, same model.

## Ambiguities and the reading we use

The agent states the reading in its echo whenever a command touches one of these.

1. **Indexing.** *Beats are 0-based positions; bars are 1-based units.*
   Beat 0 is the first downbeat of the track (the start of bar 1). Beats in a pickup
   before the first downbeat get negative indices and bar 0. `beat_in_bar` is 1-based
   (count "1, 2, 3, 4"). Why mixed: the brief says "beat 0" (so beats are 0-based), and
   "A1's bar 33" is the DJ phrase boundary after 32 bars, which only works 1-based.
2. **Beat ranges are half-open:** "beat 16–32" = from beat 16 up to (not including) beat
   32 = 16 beats. With beat 0 on a downbeat, in 4/4 that is exactly bars 5–8. Inclusive
   would be 17 beats, which nobody means. **Bar ranges are inclusive:** "bars 200–210"
   = 11 bars, "bar 5" = one bar.
3. **"Cut beat 16–32 and repeat it twice"** = the range ends up playing **twice in total**,
   in place (the original occurrence + one copy immediately after). "Cut" here selects the
   range rather than deleting it — deleting and then repeating would leave the range
   playing exactly as often as before. The echo states this reading and offers the
   alternative ("original + 2 copies") in one line.
4. **"Fade out the last 8 beats"** = the last 8 beats of the *current edited* track,
   ending at the end of the audio. Shape: equal-power (cos) fade to silence.
5. **"Loop the first 4 bars four times, then let the track play on"** = bars 1–4 play
   **four times in total** (original + 3 repeats), then the track continues from bar 5.
6. **"The last bar before the tempo change"** requires a detected tempo segment boundary.
   If none is detected the agent says so and does not guess. If there are several it
   asks which.
7. **"Chop out the boring part"** is not measurable as stated. The agent pushes back,
   proposes a candidate chosen by a measured criterion (lowest-energy/novelty bar range),
   and does **not** cut until the user confirms.
8. **Out-of-range ranges** ("delete bars 200–210" on a shorter track) are rejected with
   the actual bar count. Never clamped.
9. **Bars on beats-only material and in 6/8.** Bars always come from our own downbeat
   decoding, whatever the reference contains. In compound meters a "beat" is the felt
   pulse (dotted quarter): 6/8 = 2 beats per bar with ternary subdivision. The grid JSON
   reports meter confidence; low confidence is echoed whenever a bar address is used.
10. **Downbeats during a drumless intro.** No beats or downbeats are emitted in a region
    without a detectable pulse; the region is listed in `no_beat_regions`. Bar 1 is the
    first downbeat after it. We do not extrapolate the grid backwards into the intro.

## What we deliberately do not build
(filled in at Phase 4)
