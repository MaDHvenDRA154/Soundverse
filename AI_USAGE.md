# AI usage

## Tools used

| Tool | Role |
|---|---|
| **Claude Code** | Wrote nearly all the code, tests, eval harness and first drafts of the docs, working in phases with a gate report at the end of each. |
| **Claude (chat)** | Occasional sounding board for wording prompts and sanity-checking reports. |
| **Gemini (`gemini-3.1-flash-lite`)** | Not a coding tool: the runtime LLM inside Beat Agent that turns chat messages into typed edit operations. |

## What I delegated vs. what I decided

**Delegated to Claude Code:** implementation of all three parts, the test suites, the Docker setup, the eval harness, and first drafts of README / DECISIONS.

**What I decided or set as constraints up front:**
- **The LLM/code boundary:** the LLM only maps text to typed ops; it never produces tempo, beat times or notes, and never writes JavaScript.
- **No hardcoding:** no test-pack IDs or tuned constants in the package, enforced by a grep test and a synthetic-audio robustness suite, because the reviewers re-run on held-out audio.
- **Honest reporting:** every number must come from an eval file. Since the official test pack was never available to me, everything is reported on generated audio and clearly labelled public-dataset substitutes, never presented as the test pack.
- **Gates:** build in phases and stop at each gate for my review before continuing.

**What I reviewed:**
- Every phase gate report. I checked each claim against the files and sent back corrections before allowing the next phase.
- The live app: I ran it on my own tracks and tested edits and pushback by hand.
- **TODO:** list the files or functions you actually read line by line (e.g. the edit-op schema, the EDL executor's splice/crossfade code, the grid transform after edits, the score → HTML renderer's safety scan). Only list what you really read.
- **TODO:** listening checks. For each file you listened to, one line on what you heard, e.g. "S1 splice at 16.89 s: no click; the hard-cut control clicks clearly."

## Cases where the AI was wrong or subtly off

**1. The "clean clone" wasn't clean.**
Claude Code reported a clean-clone run that replayed every LLM answer from the cache, but the cache was gitignored and had never been committed. I noticed those two claims can't both be true. When I questioned it, it turned out it had copied my local cache and substitute files into the clone. The fix: commit the secret-free cache to `evals/llm_cache/`, then re-run from a fresh `git clone` with no key, nothing copied in, and replay-only mode, where a cache miss is an error. That run made 0 live calls and matched the committed results.

**2. The interactive app would have replayed canned eval answers.**
After the cache was committed, I asked whether replay mode applied only to the evals. It didn't: every caller defaulted to the cache, so the app and `beat-agent chat` would have answered new user messages from stored eval answers and written into the tracked cache. Now the eval harness defaults to replay, while the app and CLI always call the model live and show a clear message if no key is set.

**3. A systematic 12–25 ms "early" bias was nearly left as a tuning question.**
On real-music substitutes the grid sat consistently early against the human annotations. Claude Code flagged it but held off on changes because I had said "don't tune to substitutes". I treated a one-directional offset as a likely bug, and asked for a root-cause study and no empirical offset. The study ruled out frame centring, resampling and decoder delay, and found the annotations themselves sit after the drum attacks. It also found a real ~5 ms bias in our own sub-frame peak interpolation, which is now off by default.

**4. Rubber Band only existed on my machine.**
Phase 1 sync worked because Claude Code had downloaded the Rubber Band binary into a gitignored `.tools/` folder. I spotted that a reviewer's checkout would have had no time-stretcher and had it moved into the Docker image. It's now installed in the Dockerfile, and sync is verified inside the container.

**5. Stale results would have been committed next to new code.**
The Gemini free-tier quota ran out, so Claude Code restored the old Part 2 results, which came from earlier analysis defaults. I required a full re-run on the final code before submission, so that no reported number disagrees with the files.

**6. A 6 ms latency in the Web Audio output path.**
The recreation page's master compressor (Chrome's look-ahead) added ~6 ms to every note, directly hurting rhythm, which the brief ranks first. I chose to remove it rather than compensate, since the latency is browser-specific. The render delay on the drum loop dropped from +14.5 ms to +8.0 ms.

**7. Secret safety I didn't take on trust.**
At one point I put my real API key in `.env.example`, which is a committed file. I didn't rely on the AI's statement that the file was safe. I set hard rules: never stage `.env.example` while it holds a key, stage files only by explicit path (no `git add .`), and grep every staged diff for the key before each commit. I also required a scan of the full history before any push, and repeated it after the history rewrite. The final scan found no key or key-like string in any commit, and `.env` was never committed.

**8. A repeat edit that shortened the track (found by using the app).**
On a 6/8 pop track, "Cut beat 16–32 and repeat it twice" left the track 7 ms *shorter*, when a repeat should add several seconds. I caught it from the length in the Undo echo. The cause: the live model answered with two ops, [remove beats 16–32, repeat beats 16–32 ×2], and the executor ran them literally. The remove deleted the range, then the repeat (resolved on the edited grid) doubled the *next* 16 beats. The eval case passed only because its cached answer was a single repeat op. Fix: a remove immediately followed by a repeat of the same range in one turn is now read as "select X" (the remove is dropped and the echo says so), with regression tests on a 6/8 ramp grid checking edited length = original + (times − 1) × range length.

## Claude Code's own mistake log

Claude Code also logged its own bugs as it found them: 37 entries in [notes/ai_mistakes_log.md](notes/ai_mistakes_log.md). Examples:
- a no-hardcoding test that passed while its patterns were broken;
- a kick detector counting ~3× too many kicks, so the first sync numbers measured noise;
- tempo-change detection 7 beats late because two tempos realign every few beats;
- a "0.0 ms" render-timing result caused by coarse detector frames.

Apart from #37 (case 8 above), those were caught by the AI's own tests, not by me, and I list them separately for that reason.

## What I'd do differently

- **Define "clean clone" precisely from day one** (fresh `git clone`, no key, nothing copied in, new volumes) instead of discovering at the end that earlier checks had borrowed local state.
- **Get the test pack, or real annotated music, before tuning anything.** Generated audio flattered several choices (phase refinement, sub-frame interpolation) that real music later showed were neutral or harmful.
- **Use the live app on real tracks earlier.** The repeat-edit bug on a 6/8 track (case 8) only showed up when I tested by hand; the eval sessions on 4/4 audio passed.
- **Pay for the API tier during development.** The free-tier daily limit left committed results stale at one point and forced extra re-run planning.
