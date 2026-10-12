# Beat Agent — one-page plan

## Inputs found at Phase 0 (2026-10-12)
- The assignment brief (text + original PDF): present locally; not part of the repo.
- `testpack/`: **absent.** Nothing can be scored against reference annotations or MIDI
  until it is added. All development and tests run on audio generated in code; the eval
  harness marks accuracy rows "not measured" for clips without a reference.
- `audio/`: empty.
- Machine: Windows 11, 32 logical cores, Python 3.11.9, uv 0.11, Node 24, Docker installed
  (daemon not running at start). No system ffmpeg / rubberband / make: ffmpeg comes from
  `imageio-ffmpeg`, Rubber Band 4.0.0 CLI is downloaded into `.tools/bin` (gitignored);
  the Docker image installs both from Debian. `make` targets run inside the container.

## Phase order and gates
| Phase | Deliverable | Gate (shown to user before moving on) |
|---|---|---|
| 0 | scaffold, Docker, Makefile, config, DECISIONS, no-hardcoding guard | guard passes |
| 1 | beat grid JSON, click audition, sync render, part1 eval | table of beat/downbeat F, drift, sync offsets, runtime (synthetic now; test pack when supplied) |
| 2 | typed EDL, executor, grid transform, undo/redo, Gemini agent, UI | transcripts + echoes of the six sessions |
| 3 | stems → score JSON → deterministic HTML → offline render | note F (P1–P3), onset timing |
| 4 | README status table, DECISIONS, evals README, clean-clone check | — |

## Core architecture in one paragraph
Analysis turns audio into a **grid** (beats with musical addresses, downbeats, meter,
tempo segments, confidence) purely by signal processing + Beat This!. Edits are an
**EDL of typed ops in beat/bar addresses**; the executor resolves addresses to samples
using the current grid and maintains a **virtual timeline of clips over the immutable
source**, so the edited grid is the source grid mapped through the same clip list (never
re-detected) and undo is a snapshot. The LLM only maps text → ops; it never sees or emits
seconds. Part 3 writes a versioned **score JSON** that a deterministic template renders
to HTML; the LLM edits the score through typed ops, never JavaScript.

## Biggest risks
1. No test pack → real accuracy unknown; synthetic tracks are easier than real music.
2. Beat This! timing resolution is 20 ms frames; the 10 ms drift target relies on the
   regularised (linear-fit) grid for constant-tempo segments.
3. Gemini free-tier quotas: eval responses are cached on (model, prompt, grid state).
4. Basic Pitch / Demucs on CPU are slow; Part 3 clips are short (≤16 bars) by design.
