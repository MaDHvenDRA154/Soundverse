# Morning review (2026-10-13 night run)

**State: nothing merged, nothing pushed.** Everything is on branch `final-verification`; `main` is untouched
at `b124194`. Review with `git diff main final-verification` (54 files: code, docs, results, 20 cache entries).

| | |
|---|---|
| branch | `final-verification`, head `b882522` (+ the commit adding this file) |
| results produced by | `8e5e62b` (no code changed after it; `b882522` only adds results, docs, notes and prunes the cache) |
| main can fast-forward | yes (`main` is an ancestor of the branch) |

## 1. The final clean-clone run

- Fresh `git clone --branch final-verification` into a new directory; **no `.env`, no `.cache`, nothing copied
  in**, `GEMINI_API_KEY` and `LLM_CACHE` stripped from the environment, new compose project (new
  model-weight volume); all bind mounts were the clone's own directories.
- `docker compose build` then `docker compose run --rm -T app make reproduce` (default eval mode = replay).
- **0 live LLM calls**: Part 2 replayed 17/17, Part 3 chat 3/3, from the committed `evals/llm_cache/`, with no
  key present; the cache was not written to (43 files before and after).
- Substitute renders: FluidSynth 2.3.1 + TimGM6mb.sf2 sha256 `c5378b62...c854`, `matches_pinned: true`.
- Exit 0; the only warnings are Basic Pitch's notes about optional CoreML/TFLite/TensorFlow backends.

Timings (local time, this laptop: Ryzen 9 8940HX, Docker 32 CPUs / 7.4 GB):

| stage | time |
|---|---|
| image build (cached layers from the identical earlier build) | 15 s |
| GTZAN fetch | 4 min 22 s |
| MIDI substitutes + FluidSynth renders | 12 s |
| robustness / Part 1 / Part 2 / Part 3 / studies | 2:52 / 1:43 / 0:50 / 1:14 / 0:54 |
| **`make reproduce` total** | **12 min 14 s** |

## 2. Result diffs

### 2a. Final run vs the first Docker run (both Linux; that one ran `aff9c66` with a copied-in cache)

**Every measured value is identical** (all six results files). The only differences are new fields:
`fetched_at` and `cache_key` on each of the 17 Part 2 attempts, `llm_calls` on the 3 Part 3 chat turns,
`llm_provenance` (Part 2 and Part 3 chat), `substitute_toolchain`, and the ledger renamed
`cumulative` -> `cache_inventory`. Wall-clock fields differ as expected.

Byte-level: GTZAN audio, MIDI excerpts, FluidSynth renders, `score.json` (4 clips) and pages identical. Stem
WAVs differ only in 2 bytes, the timestamp libsndfile writes into the float-WAV PEAK chunk; their audio is
bit-identical. The browser renders (`render_1loop.wav`) differ by one 16-bit LSB on 0.01-0.1% of samples
(109-118 dB below the signal): Chromium's OfflineAudioContext is not bit-deterministic. Three renders of
the same page in the same container differ the same way; the cause inside Chromium was not investigated.
Every onset metric was nevertheless identical across the two runs.

### 2b. Final run vs what is committed on `main` (the local Windows run)

| field | main (Windows) -> final (Docker) | cause |
|---|---|---|
| subP3 pitched / all-notes F; notes | 0.6222 -> 0.625; 142 -> 141 (P 0.493 -> 0.4965) | substitute WAVs rendered by FluidSynth 2.4.4 (Windows) vs 2.3.1 (Docker): same MIDI, -50 to -63 dB different audio, one note crosses a threshold. Confirmed: the Linux pipeline on the Windows-rendered WAV gives the Windows notes exactly |
| subP3 render signed offset | 2.9 -> 5.8 ms | same cause (one 5.8 ms detector hop) |
| subP4 bass notes; render onset F | 51 -> 50; 0.9286 -> 0.922 (72 -> 73 render onsets) | same cause |
| subP3 / subP4 page bytes | 29243 -> 29131; 80353 -> 80237 | one note fewer in each score |
| render offset study, subP2 hop-32 means | 10.49 -> 10.44; 9.26 -> 9.19 ms | same cause |
| bias study E4 M4A; robustness `m4a_aac_roundtrip` | mean 2.75 -> 2.51 ms; max err 5.36 -> 4.16 ms | ffmpeg AAC encoder: 7.1 (Windows, imageio-ffmpeg) vs 5.1.9 (Debian) |
| sync `a_start_s` | 62.736382622962445 -> ...246 | float rounding in the 14th decimal |
| paths in JSON | `\` -> `/` | Windows vs Linux |
| LLM attempts `cached`, per-session usage | live -> replayed | main's run made the 20 calls; this one replayed them |
| new fields | absent -> present | provenance, cache keys, toolchain, ledger rename (see 2a) |

No chat echo, op, splice score, grid check or Part 1 / robustness verdict differs.

## 3. LLM cache

- Committed: `evals/llm_cache/`, **20 entries** (17 Part 2 + 3 Part 3 chat), all `gemini-3.1-flash-lite`,
  fetched 2026-10-12 19:00-19:19 UTC. Each holds `text`, `usage`, `model`, `fetched_at` only.
- Pruned (`python -m evals.cache_tool prune`): 23 entries no result uses: the 20 stale `gemini-3.5-flash`
  answers and 3 Part 3 chat answers from before Demucs was seeded. **They are still in history** (added in
  `64be5d6`, deleted in `b882522`); they passed the same secret scan. Squash the branch on merge if you want
  them out of history.
- The dropped 44th entry (`e2bebca9...0733`, "remove bars 200 through 210") was my app smoke test in the
  first clone; it was never migrated and no result references it.
- The pruned cache was re-checked locally with no key: Part 2 17/17 and Part 3 chat 3/3 replayed, 0 live, the
  Part 3 echoes identical to the committed ones.

## 4. Secret scan (full history, all branches)

13 commits (all branches, including the commit that adds this file; ~1.3 MB of patches) + all 127 tracked files: the key value appears nowhere; no key-like pattern
(`AIza...`, `ghp_`, `sk-`, `xox?-`, private-key headers); `.env` was never committed; `.env.example` only
ever had `GEMINI_API_KEY=` empty; `python -m evals.cache_tool check` passes. Tests: 82 passed.

## 5. Doc diff summary (main -> branch)

- **README.md**: quickstart table "use it (app, live, needs a key)" vs "reproduce (evals, replay, no key)" vs
  "re-run live"; cached-replay paragraph with model and fetch date; test-pack note (`LLM_CACHE=on` + key);
  **first-run timing note**; headline table rows "LLM calls, this run" and "LLM cache inventory" (not a call
  count); provenance lines in the status block; regenerated numbers.
- **evals/README.md**: committed-vs-regenerated table (adds `evals/llm_cache/`); eval cache-mode table
  (replay default / `on` / `off`); app and CLI are always live; pinned FluidSynth + soundfont table with the
  measured effect of a different version; Chromium LSB note; subP3/subP4 failure-analysis numbers updated.
- **DECISIONS.md**: D6 (committed cache, eval default replay, app/CLI live); D19 (render pinned); D20 (LLM
  answers committed); D22 (deterministic stems/scores/pages; Chromium LSB).
- **Dockerfile**: `fluidsynth`/`libfluidsynth3=2.3.1-2`, `timgm6mb-soundfont=1.3-5`, `sha256sum -c` of the
  soundfont. **docker-compose.yml**: `LLM_CACHE` defaults to `replay` (eval harness only).
- **Makefile**: `eval-replay`, `eval-live`.
- **notes/**: mistakes log #32-36; demo script rewritten with exact commands (section 8).

## 6. Mistakes log, one line each

"How caught" is from the log. Rows marked **(your review)** came from your questions; for the others,
check whether you caught them yourself before relying on them in AI_USAGE.md.

| # | what was wrong | how caught |
|---|---|---|
| 1 | No-hardcoding guard regexes had `\b` turned into a backspace by a heredoc; guard tests passed while broken | guard self-test failed |
| 2 | Grid trusted every tracker peak; Beat This! ticks 1-3 beats into silence | experiment vs generated truth |
| 3 | Beat repair's running median followed metrical-level switches (6/8) instead of repairing them | synthetic 6/8 gave 4 "free" segments |
| 4 | Step-tempo model rejected real steps because boundary beats are smoothed by the tracker | 120->132 step classified free, 60 ms errors |
| 5 | Kick detector counted 372 kicks in 60 s at 124 BPM; first sync "measurement" was detector noise | kick count vs tempo x duration |
| 6 | `attack_times` hop rounding (22 samples != 1 ms) drifted 70 ms by 30 s; phase refinement never fired | refinement shift 0.0 everywhere |
| 7 | Config comment cited a scratch experiment not in the repo | re-reading config before commit |
| 8 | Step grid emitted the crossing beat twice; beats not increasing after refinement | mir_eval "increasing order" error |
| 9 | Multi-op turn resolved later ops against the pre-turn grid (stale cache key) | re-reading code before first test |
| 10 | First click metric flagged correct beat-aligned splices (kick attacks) | crossfade tests failed |
| 11 | Step-change location pulled up to 9 beats off by tracker transition errors | bar-before-change test, robustness 140->120 |
| 12 | Tempo-octave correction renumbered every bar (kept beats-per-bar fixed) | reading the real Gemini S6 echo |
| 13 | Scripted manifest edit matched "sessions:" in a comment and mangled the header | read-back of the file |
| 14 | Click detector could not see hard cuts (negative control 0/5): "0 clicks" was not evidence | adding the hard-cut control |
| 15 | LLM client retried a per-day 429 with sleeps | app hung ~30 s on chat |
| 16 | Generated drum samples truncated mid-decay: every generated kick/snare clicked | transcriber saw a "snare" 350 ms after each kick |
| 17 | Edit scripts read files as cp1252, so non-ASCII patterns never matched | repeated assertion errors |
| 18 | Parabolic sub-frame peak interpolation (mine) biased beats 1-4.5 ms late | bias study E1 |
| 19 | First drum transcriber fired on the noise floor; onsets ~8 ms early | test on generated hits |
| 20 | Wrong "first beat missing" diagnosis; added a lead-in that created a phantom beat | checking the clip's first non-zero sample |
| 21 | Render timing reported exactly 0.0 ms (512-sample hop quantisation) | a number that was too good |
| 22 | Docstring word "reference" tripped the no-hardcoding guard (reworded, guard not loosened) | the guard test |
| 23 | Step-crossing search took a coincidence at the LCM of the periods: change 7 beats late | robustness regression |
| 24 | Adaptive guard also ran at region edges: beats dropped, tempo extrapolated (~400 ms errors) | bar-before-change test |
| 25 | Step split across a tracker dropout became "free"/"across_gap" | same test after #24 |
| 26 | Demucs `shifts=1` random offset: Part 3 numbers changed between identical runs | two runs gave 0.7209 vs 0.7294 |
| 27 | Render-offset finding said attack shape added +1.8 ms; net of the compressor it is -4.2 ms | re-reading the arithmetic |
| 28 | Model check fell back to the `gemini-flash-latest` alias (could be non-Gemini-3) | Phase 4 review of the model check |
| 29 | evals/README draft stated causes nobody had measured | checking each sentence against results files |
| 30 | Part 1 drift table had 8 header cells, 7 separators | reading the generated part1.md |
| 31 | Grid header printed "bars 1-0" when the meter was unknown | Docker app smoke test |
| 32 | "Cumulative spend" counted cache files and called them real calls | **(your review)** of the clean-clone diff |
| 33 | Part 3 substitute renders depended on the installed FluidSynth (2.4.4 vs 2.3.1) | clean-clone vs local diff (the diff was your request; I traced the cause) |
| 34 | The first "clean-clone" check was seeded with the copied local cache and substitutes | **(your review)**: how could a gitignored cache replay? |
| 35 | Cache-mode default `on` for every caller: app/CLI would replay eval answers and write to the tracked cache | **(your review)**: replay should be eval-only |
| 36 | Claimed Part 3 "deterministic"; Chromium's offline render varies by 1 LSB run to run | byte-comparing two Docker runs before merge |

## 7. Push to a new public GitHub repo (after you approve)

Decide first:
- The repo contains **`Copy of Soundverse AI - Applied AI Engineer Assignment Beat Agent.pdf`** and
  **`BRIEF.md`** (Soundverse's assignment text). Publishing the company's brief may not be wanted; remove
  them before pushing if not (`git rm` + commit; they stay in history unless you rewrite it -- simplest is to
  remove them and squash, or ask Soundverse).
- All 12 commits carry **`madhvendra.mitblr2023@learner.manipal.edu`**; it becomes public. To use GitHub's
  no-reply address instead you would have to rewrite history (`git filter-repo --mailmap`); otherwise leave it.
- Optional: squash-merge to drop the 23 pruned cache entries from history (section 3).

Then (PowerShell or Git Bash, in the repo):

```bash
# 1. merge (fast-forward) and re-run the checks on main
git switch main
git merge --ff-only final-verification
uv run python -m pytest -q -m "not slow and not llm"
uv run python -m evals.cache_tool check
git status                      # must be clean; .env must NOT appear (it is gitignored)

# 2. create the empty repo: https://github.com/new
#    name e.g. beat-agent, Public, do NOT add a README / .gitignore / licence (the repo has them)

# 3. push
git remote add origin https://github.com/<your-username>/beat-agent.git
git push -u origin main
#    (Git Credential Manager opens a browser login the first time)

# 4. optional cleanup
git branch -d final-verification
```

Afterwards, open the repo page and check: README renders (mermaid diagram, status table links), `.env` is
absent, `evals/llm_cache/` has 20 files.

## 8. Demo script (also in `notes/demo_script.md`)

Before recording: `cp .env.example .env` and add `GEMINI_API_KEY` (the app calls Gemini live);
`docker compose run --rm app make reproduce`; `docker compose up -d`. Tabs: http://localhost:8000, README
status table, `evals/samples/subP1/page.html`, editor, terminal. The app's chat is live, so its wording can
differ slightly from the committed transcripts in `evals/results/part2.md` (fallback if the API fails).

| time | segment | show | say |
|---|---|---|---|
| 0:00-0:45 | What it is | README top + headline table | Grid you can trust, chat edits in beats/bars, audio -> Web Audio page. LLM interprets, code measures. Test pack missing: SUBSTITUTE / generated audio, labelled. |
| 0:45-2:15 | Click audition | `docker compose run --rm app uv run beat-agent analyze evals/substitutes/audio/gtzan_disco_00000.wav --out audio/demo_disco`; play `audio/demo_disco/gtzan_disco_00000.click.wav`; `evals/out/part1/sub-disco/grid_diagnostics.png`; `audio/demo_disco/gtzan_disco_00000.grid.json` -> `conventions` | Clicks on the kick, accented downbeats; beat 0 = first downbeat, bars 1-based. 6-min generated track: max drift 0.91 ms, 8.5 s. GTZAN taps sit late of the attacks (bias study), no offset added. |
| 2:15-3:30 | Sync render | `evals/out/part1/sync_G-house-partner_to_G-house-6min/overlap_mix.wav`, `b_stretched.wav`, `sync.json` | Rubber Band time map beat onto beat; kick offsets median 1.7 ms, max 3.5 ms (121 pairs). |
| 3:30-5:30 | Two chat edits + pushback | App: upload `evals/substitutes/audio/gtzan_disco_00000.wav`; `Cut beat 16-32 and repeat it twice.` -> echo, play, Undo; `Chop out the boring part.`; optional `Delete bars 200-210.` | Echo in beats, bars, seconds; "twice" = 2 plays with the alternative stated; grid mapped not re-detected (re-tracked beat F >= 0.98); 0/13 splices flagged. "Boring" not measurable: quietest bars only 0.9 dB below average, nothing cut. Out-of-range rejected with the real bar count. |
| 5:30-7:00 | Recreation vs original | `evals/samples/subP1/original.wav`, `page.html` (play, loop, mute drums), `score.json`, `evals/results/part3.md` chat | Score JSON -> fixed template; LLM never writes JS; safety test. Drum onsets F 0.72, labels 0.33, render onsets 0.98; 14.5 ms late (render offset study). Chat ops 3/3 applied. |
| 7:00-8:15 | Code structure | `beat_agent/{analysis,edit,agent,transcribe,render,sync,app}`, `config.py`, `tests/test_no_hardcoding.py`, `evals/status.py`, `evals/llm_cache/` | Clip timeline over the untouched source; provider interface + pydantic + one retry; thresholds in config with reasons; README table generated; committed LLM answers make a fresh clone reproduce the results with no key. |
| 8:15-9:30 | Top 3 problems | README "Known problems" | Meter on hard material (fast 3/4, 12/8 for 6/8, jazz double time, downbeat F 0.07); Part 3 drum labels (F 0.33-0.48) and 14.5 ms late render; substitutes only, click check weak outside S1 (hard cuts 5/13). |
| 9:30-10:00 | Close | `evals/README.md` committed vs regenerated | `make reproduce` rebuilds everything else; `LLM_CACHE=off` re-runs the chat evals live. |

## 9. README first-run timing note (as added)

> **First run takes a while** (measured on this laptop: AMD Ryzen 9 8940HX, Docker with 32 CPUs / 7.4 GB):
> the cold `docker compose build` took more than 10 minutes (PyTorch, Demucs, Basic Pitch, Playwright
> Chromium; a rebuild with cached layers takes ~15 s); `make reproduce` then took 12 min 14 s: fetching the
> GTZAN excerpts (streamed from a 1.2 GB tarball) 4 min 22 s, the MIDI substitutes 12 s, and the evals 7 min
> 35 s (robustness 2:52, Part 1 1:43, Part 2 0:50, Part 3 1:14 including the first download of the Demucs
> weights, studies 0:54). In the app the first analysis also downloads the Beat This! weights (131 s the first
> time on a fresh volume).

The cold-build figure is a lower bound: that build outran a 10-minute tool timeout and was not timed to the
end.

## 10. Still open (your call)

- **Compressor latency** (DECISIONS D21): the page plays every note 6.04 ms late (Chrome's
  DynamicsCompressor look-ahead). A look-ahead-free limiter would remove it; left unchanged so as not to tune
  an eval number.
- The brief PDF / BRIEF.md, the commit email, squashing (section 7).
- Local leftovers you may delete: scratch clones `clean`, `final`, `verify`, `verify2` under the session
  scratchpad; Docker volumes `clean_model-cache`, `final_model-cache`, `verify_model-cache`,
  `verify2_model-cache`, `soundverseassignment_model-cache` (`docker volume rm ...`); images `clean-app`,
  `final-app`, `verify-app`, `verify2-app`, `soundverseassignment-app` (other projects' images untouched). Your local `.cache/llm` still holds the old 43 entries; nothing
  reads it any more.
- The 131 s first analysis in the app came from one run on a fresh volume; I did not separate the weight
  download from the analysis itself.
