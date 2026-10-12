"""Every tunable threshold in Beat Agent lives here.

Rule: each value is justified by a general property of audio, the models we use,
or a published evaluation convention -- never by how it scored on a particular
clip. If you change a value, change its justification too.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Input / output
# --------------------------------------------------------------------------
# Beat This! was trained on 22.05 kHz mono log-mel spectrograms; analysis is
# done at that rate regardless of the input rate. Exports keep the input rate.
ANALYSIS_SR = 22050
# Streamed reads keep memory flat on long/high-rate files: 2**18 frames is
# ~6 s at 44.1 kHz, ~3 MB per channel as float32.
IO_BLOCK_FRAMES = 2 ** 18
# Lossy or exotic inputs are decoded once to WAV. 16-bit is what MP3/AAC/OGG
# decoders are normally assumed to deliver; documented in DECISIONS.md.
LOSSY_EXPORT_SUBTYPE = "PCM_16"
# Formats libsndfile reads natively with sample-accurate seeking. Anything
# else is decoded through ffmpeg first.
NATIVE_FORMATS = {"WAV", "FLAC", "AIFF", "W64", "RF64", "WAVEX"}
# Inputs shorter than this cannot contain two bars of even fast music, so we
# still analyse them but never claim a meter (see analysis.meter).
MIN_DURATION_FOR_METER_S = 4.0

# --------------------------------------------------------------------------
# Beat tracking (Beat This!)
# --------------------------------------------------------------------------
BEAT_THIS_CHECKPOINT = "final0"   # the authors' recommended general model
# Peak threshold on the sigmoid activation: 0.5 == logit 0, the threshold the
# Beat This! "minimal" post-processing uses. Peaks below it are not beats.
PEAK_PROB_THRESHOLD = 0.5
# Peaks closer than this are merged (Beat This! uses a +-70 ms max filter,
# equivalent to a 214 BPM ceiling at the beat level).
PEAK_MIN_SEPARATION_S = 0.07

# --------------------------------------------------------------------------
# Beat regions / "is there a beat here at all?"
# --------------------------------------------------------------------------
# A run of tracked beats only becomes a grid if it has at least this many
# beats: 8 beats = two bars of 4/4, the shortest stretch in which "a tempo"
# means anything.
MIN_BEATS_PER_REGION = 8
# ...except in clips shorter than 8 s, which may not contain 8 beats at all
# (a 3 s clip at 120 BPM has 6). There, 4 beats -- three intervals, the least
# that can show regularity -- may form a grid.
SHORT_CLIP_S = 8.0
SHORT_CLIP_MIN_BEATS = 4
# Two consecutive tracked beats further apart than this many times the local
# median beat period break the region (an intro, breakdown or silence).
# 2.5 periods tolerates one missed beat (2.0) plus jitter.
REGION_GAP_FACTOR = 2.5
# Within a region, inter-beat intervals must be regular: coefficient of
# variation above this means the "beats" are not a pulse (noise, speech).
# Human performance with rubato stays well below 0.15; random peaks are ~0.5.
REGION_MAX_IBI_CV = 0.20
# Mean beat activation required for a region to count as beated. Real beats
# activate the network far above the 0.5 detection threshold.
REGION_MIN_MEAN_ACTIVATION = 0.6
# Edge trimming: the tracker keeps a pulse going for a few beats into silence
# or a sustained pad (seen on generated audio: activation ~0.9 in digital
# silence). A beat at a region edge is kept only if the spectral flux under
# it is at least this fraction of the region's median on-beat flux, i.e. some
# event actually happens there. Interior beats are never trimmed.
EDGE_ONSET_MIN_REL = 0.15
# Window for "onset under the beat": +-40 ms, about two tracker frames.
BEAT_SUPPORT_WINDOW_S = 0.04

# --------------------------------------------------------------------------
# Repair of single tracker errors inside a region
# --------------------------------------------------------------------------
# An interval >= 1.75 local periods is a missed beat (2 periods, allowing 12%
# jitter). Beyond REGION_GAP_FACTOR it is a gap, not a miss.
REPAIR_MISSED_FACTOR = 1.75
# An interval < 0.6 local periods holds an extra peak: no music changes tempo
# by 40% within a single beat.
REPAIR_EXTRA_FACTOR = 0.6
# During line fits, beats further than 25% of a period from the line are
# outliers (half the distance to the neighbouring subdivision).
FIT_OUTLIER_BEAT_FRACTION = 0.25
# ...and a fit is only accepted if at least 90% of beats are inliers.
CONSTANT_MIN_INLIER_FRACTION = 0.9

# --------------------------------------------------------------------------
# Tempo segmentation (constant vs variable)
# --------------------------------------------------------------------------
# Shortest tempo segment we will split off: 16 beats (four 4/4 bars). Shorter
# deviations are fills or rubato, not a tempo change.
MIN_SEGMENT_BEATS = 16
# A split is only accepted if adjacent segments differ by at least this
# fraction in tempo. 2% is below what a DJ hears as "a different tempo"
# (record decks are calibrated in 1-2% steps) but well above tracker noise
# on the median of 16+ intervals.
TEMPO_CHANGE_MIN_RATIO = 0.02
# A segment is "constant" when a straight-line fit of beat time vs beat index
# leaves no structured residual: the residual smoothed over one 4/4 phrase
# (16 beats) must stay within this many ms. 10 ms is the brief's own drift
# tolerance for a DJ-usable grid; anything above it audibly flams.
CONSTANT_MAX_SMOOTHED_RESIDUAL_S = 0.010
CONSTANT_RESIDUAL_SMOOTH_BEATS = 16
# ...and its raw (unsmoothed) residual RMS must be consistent with the
# tracker's quantisation noise: 20 ms frames -> 5.8 ms RMS, times 2.5 for
# model jitter.
CONSTANT_MAX_RESIDUAL_RMS_S = 0.015
# Beats this close to a detected step change, or to the start/end of a beat
# region, are excluded from the constant test and placed by the line fitted
# to the stable part: the network integrates context across a change and
# lacks context at a region edge, and in both cases the nearest ~4 beats come
# out 5-50 ms off (measured on generated audio). 4 beats = one 4/4 bar.
STEP_TRANSITION_GUARD_BEATS = 4
# A tracked beat counts as explained by a tempo line when it is within 30 ms of
# the line's nearest beat: 1.5 tracker frames, under half the 70 ms beat window.
STEP_INLIER_S = 0.030
# Phase refinement (grid.refine_phase): shift each tempo segment onto the
# audio's attacks when the evidence is statistically clear.
PHASE_REFINEMENT = True
# Search +-40 ms around each beat (two tracker frames) for an attack.
ATTACK_SEARCH_S = 0.04
# An "attack" is a >= 6 dB rise of 4 ms energy within 3 ms: percussive onsets
# do this, sustained notes and noise do not.
ATTACK_MIN_RISE_DB = 6.0
# At least half the beats of a segment need an attack to estimate a shift.
PHASE_MIN_SUPPORT_FRACTION = 0.5
# Apply the median offset only if its standard error is <= 2 ms (it must be
# measured more precisely than the 10 ms drift tolerance it serves)...
PHASE_MAX_SE_S = 0.002
# ...and only if it is <= 30 ms (1.5 tracker frames). A larger disagreement
# means the attacks are not the beat (e.g. pushed hats), so do nothing.
PHASE_MAX_SHIFT_S = 0.030
# Variable (non-constant) segments use the tracked beats, smoothed by a local
# quadratic (Savitzky-Golay) over this many beats each side: 9 beats averages
# frame quantisation down ~3x, and a local quadratic still follows ramps.
VARIABLE_SMOOTH_HALF_WINDOW = 4

# --------------------------------------------------------------------------
# Meter / downbeats
# --------------------------------------------------------------------------
# Candidate beats-per-bar at the tracked beat level. 2 covers 2/4 and 6/8
# (two dotted-quarter beats), 3 covers 3/4 (and 9/8), 4 covers 4/4 and 12/8,
# 6 covers 6/8 when the tracker locks to the eighth note (it does at slow
# dotted-quarter tempos).
METER_CANDIDATES = (2, 3, 4, 6)
# Per-beat probability that the bar restarts early (a dropped beat, an
# edit in the source, a live band's extra beat). Small, so the decoder only
# breaks bar structure when the downbeat evidence clearly demands it.
DOWNBEAT_RESTART_PROB = 0.005
# Mild prior favouring 4 beats per bar, the most common case, used only to
# break near ties. Expressed as log-likelihood bonus per bar.
METER_PRIOR_LOGBONUS = {2: 0.0, 3: 0.0, 4: 0.05, 6: 0.0}
# Weight of the network's downbeat activation vs the harmonic-change cue
# (chord change entering the beat) in the downbeat decoder, as a geometric
# mix. Equal weight: on generated audio the network alone misread 3/4 as 4/4
# in 4 of 6 fast cases while chords changed exactly on bar lines; 0.5 fixed 2
# of those 4 and changed no other case (notes/experiments.md, E1).
DOWNBEAT_NETWORK_WEIGHT = 0.5
# Harmony is only used when it actually changes: if the 95th-percentile
# beat-to-beat chroma change (1 - cosine) is below this, the track has
# static harmony and the cue is neutral (0.5).
CHROMA_MIN_CHANGE = 0.05
# Ternary vs binary subdivision decides 6/8 vs 2/4 (and 12/8 vs 4/4).
# triplet = min(E(1/3), E(2/3)) / E(0), duplet = E(1/2) / E(0) on high-band
# onset flux. Compound needs triplet >= 1.25 x duplet (a clear 25% margin)
# AND triplet >= 10% of the on-beat energy (something is really there).
COMPOUND_RATIO_THRESHOLD = 1.25
SUBDIVISION_MIN_REL = 0.10
# Subdivisions live in hats/shakers/strums: flux above 4 kHz, 512-point
# frames (23 ms) with 2.5 ms hop so eighth-note triplets at 180 BPM (111 ms
# apart) stay separate.
SUBDIVISION_FMIN_HZ = 4000.0
SUBDIVISION_N_FFT = 512
SUBDIVISION_HOP_S = 0.0025
# Onset envelope used for beat support / edge trimming.
ONSET_HOP_S = 0.005

# --------------------------------------------------------------------------
# Tempo octave (half/double time)
# --------------------------------------------------------------------------
# We report the tracker's metrical level and expose x0.5 / x2 alternatives
# with their comb-filter support. The tracker's choice is kept unless the user
# corrects it by chat; we do not auto-switch (see DECISIONS.md).
TEMPO_OCTAVE_FACTORS = (0.5, 2.0)

# --------------------------------------------------------------------------
# Click audition
# --------------------------------------------------------------------------
CLICK_FREQ_HZ = 1000.0          # ordinary beat
CLICK_ACCENT_FREQ_HZ = 2000.0   # downbeat: an octave up is unmistakable
CLICK_DURATION_S = 0.030
CLICK_DECAY_S = 0.006           # exponential decay -> sharp, well-localised attack
CLICK_GAIN = 0.35               # peak, linear
CLICK_ACCENT_GAIN = 0.55
# Original is attenuated by this so music + click rarely clips (-3 dB).
AUDITION_MUSIC_GAIN = 0.7

# --------------------------------------------------------------------------
# Sync
# --------------------------------------------------------------------------
SYNC_OVERLAP_S = 60.0
# Kick detection for the sync metric: energy below this frequency.
KICK_LOWPASS_HZ = 150.0
# Kick onset = first sample where the low-band envelope reaches this fraction
# of the local peak; a level-crossing is far more precise than a frame-based
# onset function (sub-ms instead of a hop).
KICK_ONSET_FRACTION = 0.3
# Envelope smoothing time constant for kick detection.
KICK_ENVELOPE_S = 0.002
# Ignore kicks whose envelope rise is under this fraction of the strongest
# rises in the excerpt (99.9th percentile).
KICK_MIN_REL_LEVEL = 0.25
# Rise measured over 15 ms on an envelope averaged over 15 ms: a kick reaches
# its peak within ~5-15 ms; averaging over 15 ms removes the beating between
# kick and bass (both live below 150 Hz, beat frequencies >= ~60 Hz).
KICK_RISE_S = 0.015
KICK_DETECT_SMOOTH_S = 0.015
# A detection must be a new event: the envelope after the rise is at least
# 2x the minimum of the preceding 30 ms (6 dB), i.e. not a wobble on a decay.
KICK_NEW_ONSET_RATIO = 2.0
KICK_NEW_ONSET_LOOKBACK_S = 0.030
# Two kicks closer than 70 ms are one hit (that is a 32nd note at 107 BPM).
KICK_MIN_SPACING_S = 0.070
# Kicks of A and B are paired only if within this fraction of a beat; anything
# further apart is a different drum hit, not a misalignment.
KICK_PAIR_MAX_BEAT_FRACTION = 0.25

# --------------------------------------------------------------------------
# Edit executor (Part 2)
# --------------------------------------------------------------------------
# Equal-power crossfade at every splice, centred on the splice point so the
# output length (and the grid) is exactly the sum of the kept ranges.
# 5 ms is below the ~10 ms where a crossfade smears a transient audibly, and
# long enough to remove the step discontinuity.
SPLICE_CROSSFADE_S = 0.005
DEFAULT_FADE_SHAPE = "equal_power"

# --------------------------------------------------------------------------
# Chat editing limits and echo
# --------------------------------------------------------------------------
# Repeats beyond 16 plays (a 64-bar loop of 4 bars) are almost certainly a
# misparse; ask for it in steps instead.
MAX_REPEAT_TOTAL = 16
# Refuse edits that would make the output longer than 30 minutes (keeps
# renders within laptop disk/RAM; far beyond any DJ edit).
MAX_OUTPUT_S = 30 * 60
# "Boring part" proposals measure a stretch of this many bars by default: 8
# bars is the standard phrase length in pop/dance music (capped at a quarter
# of the track for short clips).
PROPOSE_CUT_BARS = 8
# A "quietest stretch" is only offered as a real candidate if it is at least
# 3 dB below the track average -- about the smallest level change listeners
# reliably call "noticeably quieter". Below that the agent says it has no
# measured basis and asks.
PROPOSE_MIN_CONTRAST_DB = 3.0
# The echo warns about a tempo-octave alternative when the half-beat points
# carry >= 75% of the on-beat onset strength (double time nearly as well
# supported) or when alternate beats differ by more than 2:1 (strong-weak
# pattern: half time plausible).
OCTAVE_DOUBLE_PLAUSIBLE = 0.75
OCTAVE_HALF_PLAUSIBLE = 0.5
