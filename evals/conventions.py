"""Scoring conventions for the eval harness (never imported by beat_agent/)."""
BEAT_F_WINDOW_S = 0.070      # mir_eval.beat.f_measure default (+-70 ms)
NOTE_ONSET_WINDOW_S = 0.050  # mir_eval.transcription onset tolerance (+-50 ms)
# mir_eval trims beats in the first 5 s by default. We report both the trimmed
# (comparable to published numbers) and untrimmed (a DJ grid must hold from 0 s).
MIR_EVAL_TRIM_S = 5.0
