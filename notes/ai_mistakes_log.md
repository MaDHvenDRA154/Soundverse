# AI mistakes log

Running log, kept by the AI assistant while building. Each entry: what was wrong,
how it was caught, the fix. Raw material for AI_USAGE.md -- the human author
reviews and writes that file.

| # | Phase | What was wrong | How it was caught | Fix |
|---|---|---|---|---|
| 1 | 0 | The no-hardcoding guard's regexes were written through a Python string in a shell heredoc; `\b` became a literal backspace (0x08), so the word-boundary patterns could never match. The two real guard tests **passed** with the broken patterns. | The guard's own self-test (`test_guard_actually_detects_leaks`, which asserts the patterns fire on known leaks) failed. `cat -A` showed `^H` bytes. | Rewrote the block as a raw string, added an assertion that the file contains no 0x08. Lesson: every guard test needs a "does it catch a known-bad input" case. |
