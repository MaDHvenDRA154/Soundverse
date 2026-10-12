# Decisions

Status: final pass at Phase 4 (2026-10-13). Each decision was revisited at the end of the phase that
implemented it; "measured" notes link to eval output. The official test pack was not available: every
measured statement below is about SUBSTITUTE or generated audio.

## Key decisions

### D1. Beat tracker: Beat This! activations + our own grid post-processing
- **Options:** Beat This! with its "minimal" peak picking; Beat This! + madmom DBN;
  madmom RNN+DBN; BeatNet; librosa.
- **Choice:** Beat This! (`final0`, CPU) for frame-wise beat/downbeat activations, then our
  own post-processing: integer-frame peak picking (as Beat This! itself does; sub-frame
  interpolation was tried and measured to bias beats late, see D16) → beat regions → tempo segmentation → per-segment linear fit for constant-tempo segments →
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
Gemini via `google-genai`, model from `GEMINI_MODEL`, checked against `models.list` at startup with no
fallback to an alias (`gemini-flash-latest` could point at a non-Gemini-3 model); 2.0/2.5 models are refused
(2.5 shuts down 16 Oct 2026). Output is validated with pydantic; one retry with the validation error; never
execute a partially parsed plan. Responses are cached on (model, system prompt, contents incl. grid state,
schema), so eval re-runs cost nothing. **The cache is committed** (`evals/llm_cache/`: answer text, token
counts, model, fetch time; no prompt, key or request data), so a fresh clone replays the exact answers behind
the committed results. The eval harness defaults to `LLM_CACHE=replay` (no key, no API call, a miss is an
error); `LLM_CACHE=off` (`--live`, `make eval-live`) makes fresh calls with your key, and results may then
differ slightly. Every report prints where its answers came from and when they were fetched. The preview app
and `beat-agent chat` are not evals: they always call the model live with the user's key, never touch the
committed cache, and explain how to add a key when none is set.
- **Model used for every committed result: `gemini-3.1-flash-lite` on the free tier** (its own daily quota;
  the earlier `gemini-3.5-flash` runs hit that model's 20-requests/day free limit). It is also the default.
  **Reviewers can set `GEMINI_MODEL` to any Gemini 3 model** listed for their key; answers are then fetched
  fresh (the cache key includes the model) and the evals report which model ran.
- A full re-run needs 20 calls (17 Part 2 turns + 3 Part 3 chat turns; up to 40 if every answer needs its
  validation retry). Cost is reported as an upper bound at the highest listed Gemini 3 Flash price; on the
  free tier the actual charge is $0.
- **No partial LLM results are published.** If the provider fails mid-run (daily quota, network), the eval
  stops calling, writes what it has to `evals/out/part2/part2.partial.json` (or `part3.partial.json`) and
  leaves `evals/results/` untouched; the answers that did arrive are cached, so the next run after the quota
  reset replays them and makes only the missing calls.

### D7. Basic Pitch via ONNX, not TensorFlow
Basic Pitch hard-requires TensorFlow on Python 3.11. Its wheel ships the same model as
ONNX; we override TensorFlow out (`[tool.uv] override-dependencies`) and run
onnxruntime. Smaller install, same model.

### D8. Phase refinement onto measured attacks
- **Problem (measured):** on generated audio the tracker's grids were consistently offset by -25 to +7 ms
  depending on the track (20 ms frames + learned placement), which alone breaks a 10 ms drift target.
- **Choice:** per tempo segment, detect the steepest energy rise within +-40 ms of each beat (1 ms
  resolution) and shift the segment by the median offset, *only* when that median has a standard error
  <= 2 ms, is at least 2 SE from zero and is <= 30 ms. Tempo and spacing are never changed.
- **Trade-off / risk:** generated truth sits exactly on the attack, so this step flatters the generated
  results (max error 1-6 ms after vs up to 28 ms before). Human annotations may sit a few ms off the
  attack.
- **Status: OFF by default (`PHASE_REFINEMENT = False`), kept as an option.** The only real-music
  evidence -- the GTZAN hiphop substitute, the one clip where it fired -- got worse (max error 27.8 ->
  36.4 ms against the human annotations). One clip is weak evidence, but it is the only real evidence, and
  the bias study (D16) shows human annotations sit *after* the attack, which is exactly what aligning to
  attacks moves away from. The Part 1 eval always reports both settings (default and "refinement ON").
- **Would change my mind:** the real test pack scoring better with it on.

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

### D11. One turn = one atomic EDL, addresses resolved op by op against the current grid
All ops of a turn are applied to a copy of the state; any failure (out of range, empty range, a missing tempo
change) discards the whole turn and the user is told nothing changed. Each op resolves against the grid as it
stands after the previous op; to keep "remove bar 5 and bar 9" meaning the *current* bars 5 and 9, the LLM is
told to emit several ranges from the latest to the earliest. Undo/redo restore whole turns and name the turn.

### D12. Vague targets are measured and proposed, never cut
"Chop out the boring part" maps to `propose_cut`: code measures the quietest (lowest RMS) contiguous stretch of complete
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
- **Measured limit:** the crossfaded splices score 0 of 13 flagged, but the hard-cut control is flagged on
  only 5 of 13 (all five in S1 on disco); on the other clips the detector does not flag the hard cut either,
  so there it cannot tell a click from the music (why was not investigated). So "0 clicks" is strong evidence on S1 and
  weak evidence elsewhere; the hard-cut renders are kept next to each edit for listening.

### D15. Substitute real-music clips while the test pack is missing
Five GTZAN excerpts (rule fixed before scoring: track 00000 of disco, hiphop, rock, jazz, country) with
public beat/downbeat annotations, fetched by a script and not committed (no licence stated). GTZAN is held
out from the Beat This! checkpoint we use. They are reported in their own rows, labelled SUBSTITUTE, and no
parameter was changed in response to them -- including phase refinement, which made the hiphop substitute
worse (max error 27.8 -> 36.4 ms); that decision is left for the real test pack.

### D16. The "early" grid on real music: root cause investigated, no offset added
Investigated in `evals/bias_study.py` -> [evals/results/bias_study.md](evals/results/bias_study.md):
our integer-frame peaks match Beat This!'s official `File2Beats` exactly; resampling (one render at 5
rates) adds no delay; decoders shift 1-3 ms; the remaining per-track offset on generated audio comes from
the model's own placement and from where an onset falls inside a 20 ms frame. On the GTZAN substitutes the
human annotations sit 11-29 ms *after* the audio's own drum attacks on disco/hiphop/country, while the
tracker sits within a few ms of them, and the official Beat This! output shows the same offset to the
annotations. So the 12-25 ms "early" offset is mostly the annotation convention, not a pipeline bug, and
no empirical offset was added. One real (small) bias in our code was found and removed: parabolic sub-frame
interpolation moved beats away from the model's own integer-frame output by up to ~5 ms (+4.8 / -2.1 /
+1.4 ms on the three generated tracks), so it is off (`PEAK_SUBFRAME_INTERPOLATION = False`).

### D17. Part 3: score JSON in the middle, a fixed template renders it
Audio -> grid (Part 1) -> Demucs (htdemucs, CPU) stems -> drums by band-split onset detection
(kick < 150 Hz, snare 150-5000 Hz, hats > 6 kHz, each hit refined to ~1 ms) and Basic Pitch (ONNX) on the
bass / other / vocal stems -> soft quantisation (50% towards the nearest 16th or triplet of the grid) ->
versioned score JSON (notes in beats for playback, seconds for evaluation). A fixed HTML/JS template turns
the score into the page: oscillators, seeded filtered noise, envelopes and FM only; a look-ahead scheduler
on AudioContext time for seamless loops; per-track mixer; the same voice code renders offline through
OfflineAudioContext for the evals. Chat changes are typed score ops (`set_mute`, `set_gain`,
`brightness up/down`, `set_tempo`) applied by code; the LLM never writes JavaScript (D4). A test fails the
build if a page contains a URL, `fetch`, `import`, a media element, a `src=` or base64 audio, and the
offline render fails if the page makes any network request.

### D18. A short clip is its own loop
With no bar range asked for, a clip of up to 16 bars is looped whole: a whole number of beats at the
measured beat period, tempo set so the loop is exactly the clip length, notes quantised to the grid's
beat phase. Bars are labels only in this mode, so a wrong meter (the piano substitute reads as 6/8) cannot
shorten the loop. Longer tracks use complete bars of the grid (default: the first 8; `--bars`, `--start-bar`).
If there is no beat grid at all, the tempo comes from onset autocorrelation and the score says so.

### D19. Part 3 substitutes from public MIDI
P1-P4 are missing, so `evals/substitutes/fetch_midi.py` builds stand-ins with selection rules fixed in code
before any transcription was scored: a Groove MIDI 2-bar drum loop, a Lakh MIDI 4-bar drums + bass excerpt,
POP909 song 001 (8 bars, melody + piano), and the next qualifying dense Lakh file (8 bars, 7 instruments,
used without a reference). They are rendered with FluidSynth + the GM soundfont TimGM6mb (reverb and chorus
off) and trimmed to exactly their bars. The GM kit and piano are far cleaner than real recordings, so these
numbers are likely optimistic for real P1-P4 material.
- **The render is pinned** because it is an eval input: FluidSynth 2.3.1 (Debian 2.3.1-2 in the image) and
  `TimGM6mb.sf2` sha256 `c5378b62...c854` (timgm6mb-soundfont 1.3-5), checked at image build and by the fetch
  script. Measured: FluidSynth 2.4.4 (the Windows build first used) rendered the same MIDI 50-63 dB off and
  moved one transcribed note on two clips; even the Windows 2.3.1 build is not bit-identical to Debian's. The
  committed Part 3 numbers are the Docker image's; `part3.md` states the toolchain that rendered its inputs.

### D20. Licences: what is committed
- **LLM answers** (`evals/llm_cache/`): our own Gemini outputs, committed so the results replay; they hold
  no source audio. `python -m evals.cache_tool check` guards against anything key-like.
- **GTZAN** (Parts 1-2 substitutes): no licence is stated for the audio or the TempoBeatDownbeat
  annotations, so neither they nor anything rendered from them (edited audio, click auditions, hard-cut
  controls, stems) is committed. Only the eval reports (numbers and chat transcripts) are.
- **Groove MIDI Dataset** (subP1): CC BY 4.0 (Google LLC), original performances recorded for the dataset.
  Committed with attribution in `evals/samples/subP1/` (input render, reference MIDI, score JSON, page,
  offline render).
- **Lakh MIDI** (subP2, subP4): the dataset is CC BY 4.0, but its author states he did not make the files
  and they cannot be attributed per file; the two excerpts transcribe commercial songs (Linda Ronstadt
  recordings). The dataset licence does not cover those compositions, so nothing derived from them is
  committed.
- **POP909** (subP3): MIT on the dataset, but song 001 is a commercial pop composition the MIT licence does
  not cover. Not committed.
- **TimGM6mb** soundfont (GPL-2) renders all four; it is installed (Debian package, or downloaded by the
  fetch script), never committed. Rendered audio is credited in each `ATTRIBUTION.md`.
- The decision lives in data (`commit_outputs` in `evals/manifest.yaml`); `python -m evals.samples`
  copies only what it allows, and `make reproduce` rebuilds everything else. Details: `evals/README.md`.

### D21. Part 3 render offset: measured, explained, not compensated
The recreation of subP1 plays a median 14.5 ms (17.4 ms absolute) after the original. Decomposed in
`evals/render_offset_study.py` -> [evals/results/render_offset_study.md](evals/results/render_offset_study.md):
raw drum detections are +1 ms from the reference note-ons; the 0.5 soft quantise moves them +11 ms because
the grid it pulls towards sits +13.5 ms after the note-ons (the tracker's own placement on a 2-bar clip,
where the line fit has only ~8 beats to average 20 ms frames over; cf. D16); the page plays every sound
+6.04 ms after its scheduled time, which is Chrome's DynamicsCompressor look-ahead (0.05 ms with it bypassed;
the scheduler is exact); synth attack shape is detected ~4 ms *sooner* than the soundfont's, so it is not
the cause. Predicted +14.1 ms vs measured +14.5 ms. Nothing was changed to shrink the number. The
compressor delay is a real, constant playback latency (inaudible on its own); replacing the compressor with
a look-ahead-free limiter would remove it, and is left as a decision for review rather than made to improve
an eval number.

### D22. Demucs is seeded
Demucs' API default (`shifts=1`) offsets the input by a random amount from Python's global `random` on every
call, so stems and every Part 3 number changed between identical runs (drum onset F 0.72 vs 0.73 on subP1).
The shift is kept and seeded per call (`DEMUCS_SHIFT_SEED`), restoring the caller's random state; two runs
now give identical scores.

### D23. One command per audience
`make reproduce` (fetch every substitute, then `make eval`) for a clean clone; `make eval` regenerates
`evals/out`, `evals/results`, `evals/samples` and the README status block from existing inputs. The README
status table is generated from the result files (`python -m evals.status`), so no reported number in it is
typed by hand.

## Indexing at a glance

| Thing | Convention | Example (4/4 track, first downbeat at 1.741 s, 3 pickup beats before it) |
|---|---|---|
| beat index | 0-based position; **beat 0 = first downbeat** | beats -3, -2, -1, **0** (1.741 s), 1, 2, ... |
| pickup beats | beats before the first downbeat get **negative** indices | beats -3..-1 (0.000-1.741 s) |
| bar number | 1-based; **bar 1 starts at beat 0** | bar 1 = beats 0-3, bar 2 = beats 4-7 |
| pickup bar | the incomplete bar before bar 1 is **bar 0** | bar 0 = beats -3..-1 |
| beat_in_bar | 1-based count, 1 = downbeat | beat 5 is bar 2, beat_in_bar 2 |
| beat range | **half-open** [start, end): end beat not included | "beats 0-8" = beats 0..7 = bars 1-2 |
| bar range | **inclusive** | "bars 5-8" = 4 bars; "bar 5" = 1 bar |
| compound meter | "beat" = the level the tracker locked to (`beat_unit`, stated in the echo) | 6/8 at the eighth: 6 beats per bar |

How it appears in an echo (real output, session S5 on the rock substitute, "Cut beat 0 to beat 8"):

```
Removed beats 0-8 (beat 0 up to, not including, beat 8: 8 beats): beats 0-7, bars 1-2,
1.741-5.646 s (3.905 s). Beats -3 to -1 are a pickup before bar 1 (0.000-1.741 s); they come
before beat 0 and are kept.
Length 0:30.013 -> 0:26.108.
Grid: 4/4, 1 beat = quarter note, 120.90 BPM (variable); beats -3-49, bars 1-13. Beats are 0-based,
bars 1-based; beat ranges exclude their end beat, bar ranges include it.
```

Worked example on that track: beat 0 is at 1.741 s and is beat_in_bar 1 of bar 1; "remove bar 2" removes
beats 4-7 (the half-open beat range [4, 8)); afterwards the old bar 3 is bar 2, because every address is
resolved against the edited grid.

## Ambiguities and the reading we use

The agent states the reading in its echo whenever a command touches one of these.

1. **Indexing.** *Beats are 0-based positions; bars are 1-based units.* Full table above
   ("Indexing at a glance"). Why mixed: the brief says "beat 0" (so beats are 0-based), and
   "A1's bar 33" is the DJ phrase boundary after 32 bars, which only works 1-based.
2. **Beat ranges are half-open:** "beat 16–32" = from beat 16 up to (not including) beat
   32 = 16 beats. With beat 0 on a downbeat, in 4/4 that is exactly bars 5–8. Inclusive
   would be 17 beats, which nobody means. **Bar ranges are inclusive:** "bars 200–210"
   = 11 bars, "bar 5" = one bar.
3. **"Cut beat 16–32 and repeat it twice" = 2 occurrences in total**, back to back in place (the original
   plus one copy right after it), encoded explicitly as `repeat` with `times_total: 2`. Why: "twice" counts
   how many times you hear it ("play it twice" = you hear it two times); reading it as "original + 2 more"
   would make "repeat it once" two plays and "repeat it twice" three. "Cut" here selects the range rather
   than deleting it -- deleting then repeating would leave the range playing exactly as often as before, so
   that reading cannot be what was meant. Because the other reading exists, every repeat echo states the
   count ("plays 2 times in total ... original + 1 copy") and adds one line: "If you meant the original plus
   2 more, that is 3 plays: say 'play it 3 times in total'."
4. **"Fade out the last 8 beats"** = the last 8 beats of the *current edited* track,
   ending at the end of the audio. Shape: equal-power (cos) by default; the model may pick `linear` when
   the user names no shape (gemini-3.1-flash-lite did, in S1), and the echo always states the shape used.
5. **"Loop the first 4 bars four times, then let the track play on"** = bars 1–4 play
   **four times in total** (original + 3 repeats), then the track continues from bar 5.
6. **"The last bar before the tempo change"** requires a detected tempo segment boundary.
   If none is detected the agent says so and does not guess. If there are several it
   asks which.
7. **"Chop out the boring part"** is not measurable as stated. The agent pushes back,
   proposes a candidate chosen by a measured criterion (the lowest-RMS stretch of whole bars),
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

- **No LLM-produced numbers about the audio and no LLM-written code.** The model never sees beat times and
  cannot express seconds; Part 3 pages come from a fixed template (D4).
- **No empirical timing offsets** to make an eval number look better (D16, D21).
- **No madmom DBN and no GPU path.** madmom does not build on Python 3.11/Windows without a compiler;
  everything runs on a laptop CPU.
- **No "boring" detector beyond loudness.** Vague targets are proposed from a stated measurement and cut
  only after confirmation (D12).
- **No automatic hypermeter (6/8 vs 12/8) or tempo-octave choice.** Both are exposed as alternatives the
  user can pick in chat (D9, D13).
- **No time-stretching inside Part 2 edits.** Edits splice the source; stretching is Part 1 sync only.
- **No sample playback in Part 3.** The page uses oscillators and seeded noise only, so it carries no audio
  from the source.
- **No accounts, persistence or multi-user state in the preview app.** Sessions live in memory; it is a
  local review tool.
- **No committed third-party audio** whose licence does not clearly allow it (D20).
