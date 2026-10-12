# Demo script (target 9:30, hard stop 10:00)

## Before recording (not on camera)

```bash
cp .env.example .env                      # GEMINI_API_KEY=... (the app calls Gemini live)
docker compose run --rm app make reproduce  # substitutes + every eval output below (replays the committed LLM cache)
docker compose up -d                      # app on http://localhost:8000
```

Open in tabs: http://localhost:8000 · `README.md` (status table) · `evals/samples/subP1/page.html` · an editor on
the repo · a terminal in the repo. Every number said aloud is in the README status table. The app's chat is
live, so its wording can differ slightly from the committed transcripts in `evals/results/part2.md` (the
fallback if the API is unavailable).

## Script

| time | segment | show (exact files / commands) | say |
|---|---|---|---|
| 0:00-0:45 | What it is | `README.md` top, then the headline table | One agent: a beat grid you can trust, chat edits in beats and bars, audio -> a Web Audio page. The LLM interprets; code measures and executes. The test pack was missing, so everything shown is SUBSTITUTE (public datasets) or generated audio, labelled. |
| 0:45-2:15 | **Click audition** | `docker compose run --rm app uv run beat-agent analyze evals/substitutes/audio/gtzan_disco_00000.wav --out audio/demo_disco`; play `audio/demo_disco/gtzan_disco_00000.click.wav` (accented click = downbeat); open `evals/out/part1/sub-disco/grid_diagnostics.png`; scroll `audio/demo_disco/gtzan_disco_00000.grid.json` to `conventions` | Clicks sit on the kick. The grid states its conventions: beat 0 = first downbeat, bars 1-based. Generated 6-minute track: max drift 0.91 ms, analysed in 8.5 s on a laptop CPU. On GTZAN the grid sits 12-17 ms before the human taps; the bias study (`evals/results/bias_study.md`) shows the taps are late of the attacks, so no offset was added. |
| 2:15-3:30 | **Sync render** | play `evals/out/part1/sync_G-house-partner_to_G-house-6min/overlap_mix.wav`, then `b_stretched.wav`; show `sync.json` `kick_alignment` | B is stretched with a Rubber Band time map, beat onto beat, not one ratio. Kick offsets: median 1.7 ms, max 3.5 ms over 121 pairs. |
| 3:30-5:30 | **Two chat edits, one pushback** | App: upload `evals/substitutes/audio/gtzan_disco_00000.wav`. Type `Cut beat 16-32 and repeat it twice.` -> read the echo, play, click **Undo**. Type `Chop out the boring part.` -> read the pushback. Optional: `Delete bars 200-210.` | (1) The echo states beats, bars and seconds, and that "twice" means 2 plays in total, with the alternative reading. The edited grid is mapped, not re-detected; re-tracking the edited audio agrees (beat F >= 0.98). Splices are crossfaded: 0 of 13 flagged. (2) "Boring" is not measurable: it measures the quietest bars, finds them only 0.9 dB below average, says that is no basis, and cuts nothing until told. Out-of-range bars are rejected with the real bar count. |
| 5:30-7:00 | **Recreation next to its original** | play `evals/samples/subP1/original.wav`; open `evals/samples/subP1/page.html` (Play, loop, mute Drums); show `evals/samples/subP1/score.json`; `evals/results/part3.md` chat section | Demucs + band-split drums + Basic Pitch -> score JSON -> a fixed template; the LLM never writes JavaScript, and a test fails if a page contains a URL, fetch, import or audio data. Numbers: drum onsets F 0.72, labels F 0.33, render onsets F 0.98; it plays 14.5 ms late, explained in `render_offset_study.md`. Chat changes ("mute the drums", "brighter bass", "play it at 140") are typed score ops: 3 of 3 applied. |
| 7:00-8:15 | **Code structure** | editor tree: `beat_agent/{analysis,edit,agent,transcribe,render,sync,app}`; `beat_agent/config.py`; `tests/test_no_hardcoding.py`; `evals/status.py`; `evals/llm_cache/` | Analysis -> grid JSON; edits are a clip timeline over the untouched source; the agent is a provider interface + pydantic plan + one retry; every threshold lives in config.py with its reason; the README table is generated from eval files; the evals replay committed LLM answers, so a fresh clone reproduces the results with no key. |
| 8:15-9:30 | **Top 3 problems** | `README.md` "Known problems" | (1) Meter on hard material: fast 3/4 fails, 6/8 reads as 12/8, jazz locks to double time (downbeat F 0.07). (2) Part 3 drum labels are weak (F 0.33-0.48) and the render is 14.5 ms late. (3) Substitutes only, and the click check is weak outside S1: hard cuts flagged on 5 of 13 splices. |
| 9:30-10:00 | Close | `evals/README.md` "What is committed" | `make reproduce` rebuilds everything that is not committed; `LLM_CACHE=off` re-runs the chat evals live. |
