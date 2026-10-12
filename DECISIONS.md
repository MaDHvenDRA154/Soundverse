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

### D8. Phase refinement onto measured attacks
- **Problem (measured):** on generated audio the tracker's grids were consistently offset by -25 to +7 ms
  depending on the track (20 ms frames + learned placement), which alone breaks a 10 ms drift target.
- **Choice:** per tempo segment, detect the steepest energy rise within +-40 ms of each beat (1 ms
  resolution) and shift the segment by the median offset, *only* when that median has a standard error
  <= 2 ms, is at least 2 SE from zero and is <= 30 ms. Tempo and spacing are never changed.
- **Trade-off / risk:** generated truth sits exactly on the attack, so this step flatters the generated
  results (max error 1-6 ms after vs up to 28 ms before). Human annotations may sit a few ms off the
  attack. The Part 1 eval therefore always reports an ablation column ("no phase refinement").
- **Would change my mind:** if on the real test pack the ablation scores better, turn it off
  (`PHASE_REFINEMENT = False`) -- a decision for the reference data, not something to tune per clip.

### D9. Downbeats: network activation x harmonic change, decoded jointly with the meter
- The Beat This! downbeat head alone read fast generated 3/4 as 4/4 (4 of 7 cases). Adding chord-change
  evidence (chroma novelty entering each beat) as an equal-weight geometric mix fixed 2 of those 4 and
  changed nothing else (notes/experiments.md, E1). The cue switches itself off on static harmony.
- Meter = beats-per-bar in {2, 3, 4, 6} chosen by Viterbi likelihood, then simple vs compound from
  high-band subdivision energy (E2). In compound meters the beat level is whatever the tracker locked to:
  dotted quarter (2/bar for 6/8) or eighth (6/bar); `time_signature.beat_unit` says which.
- **Known limitation:** 6/8 vs 12/8 hypermeter -- on generated 6/8 the network marks every other bar, so
  we report 12/8 and only half the downbeats (downbeat F 0.67). Reported as "partial".

### D10. Step tempo changes are rebuilt from the two fitted lines
The tracker keeps the old tempo for a few beats after a change and sometimes adds a beat. The change beat
is placed where a beat of the old line and a beat of the new line coincide, and the grid around it is
generated from the lines (old before, new from the crossing). Measured on generated 140 -> 120 BPM:
change located within 0.3 ms (was 2 s off), max beat error 2.6 ms (was 69 ms).

### D7. Basic Pitch via ONNX, not TensorFlow
Basic Pitch hard-requires TensorFlow on Python 3.11. Its wheel ships the same model as
ONNX; we override TensorFlow out (`[tool.uv] override-dependencies`) and run
onnxruntime. Smaller install, same model.

### D11. One turn = one atomic EDL, addresses resolved op by op against the current grid
All ops of a turn are applied to a copy of the state; any failure (out of range, empty range, a missing tempo
change) discards the whole turn and the user is told nothing changed. Each op resolves against the grid as it
stands after the previous op; to keep "remove bar 5 and bar 9" meaning the *current* bars 5 and 9, the LLM is
told to emit several ranges from the latest to the earliest. Undo/redo restore whole turns and name the turn.

### D12. Vague targets are measured and proposed, never cut
"Chop out the boring part" maps to `propose_cut`: code measures the quietest contiguous stretch of complete
bars (8-bar phrase by default, a quarter of short tracks), reports its level against the track average, and
cuts only after an explicit confirmation turn (`confirm_proposal`). If the quietest stretch is less than
3 dB below the average, the agent says it has no measured basis and asks (on the substitute disco clip the
best candidate was only 0.9 dB quieter).

### D13. Tempo-octave correction keeps the bar lines
"The tempo is half/double that" changes the beat level only; downbeats stay where the downbeat decoder put
them, beats-per-bar scales (6 -> 3 when halving), and halving is refused when bars have an odd number of
beats. (First version kept beats-per-bar fixed like DJ software, which silently renumbered every bar.)

### D14. Splice quality is judged against the source, with a negative control
Clicks are measured as the sample step at the splice relative to the natural steps around both source
points (plus audible HF excess); every Part 2 report also renders the same edit with a hard cut to show the
detector can see a click on that material. 5 ms equal-power crossfades centred on the splice keep lengths
and grid times exact.

### D15. Substitute real-music clips while the test pack is missing
Five GTZAN excerpts (rule fixed before scoring: track 00000 of disco, hiphop, rock, jazz, country) with
public beat/downbeat annotations, fetched by a script and not committed (no licence stated). GTZAN is held
out from the Beat This! checkpoint we use. They are reported in their own rows, labelled SUBSTITUTE, and no
parameter was changed in response to them -- including phase refinement, which made the hiphop substitute
worse (max error 27.8 -> 36.4 ms); that decision is left for the real test pack.

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
   decoding, whatever the reference contains. In compound meters the beat is the level the
   tracker locked to -- dotted quarter (6/8 = 2 beats per bar) or eighth (6 per bar); the
   grid says which in `time_signature.beat_unit`, and "beat N" in chat means that level.
   (Changed from the Phase 0 plan of always using the dotted quarter: the tracker locks to
   eighths at slow dotted-quarter tempos, and silently converting would make chat beat
   numbers disagree with the click audition.) Meter confidence is echoed whenever it is low.
10. **Downbeats during a drumless intro.** No beats or downbeats are emitted in a region
    without a detectable pulse; the region is listed in `no_beat_regions`. Bar 1 is the
    first downbeat after it. We do not extrapolate the grid backwards into the intro.

11. **"Make it louder"** without an amount -> the agent asks for the dB (observed with Gemini: clarify).
12. **"Cut X and repeat it"** selects X; **"Cut beat 0 to beat 8"** removes. "Cut" alone means remove;
    "cut ... and repeat/loop" means select. Pickup beats before beat 0 are kept and the echo says so.

## What we deliberately do not build
(filled in at Phase 4)
