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
# For variable (non-constant) segments the tracked beats are used directly,
# lightly smoothed with a local linear fit over this many beats each side.
# 2 beats either side removes frame quantisation without erasing rubato.
VARIABLE_SMOOTH_HALF_WINDOW = 2

# --------------------------------------------------------------------------
# Meter / downbeats
# --------------------------------------------------------------------------
# Candidate beats-per-bar at the tracked beat level. 2 covers 2/4 and 6/8
# (two dotted-quarter beats), 3 covers 3/4 (and 9/8), 4 covers 4/4 and 12/8.
METER_CANDIDATES = (2, 3, 4)
# Per-beat probability that the bar restarts early (a dropped beat, an
# edit in the source, a live band's extra beat). Small, so the decoder only
# breaks bar structure when the downbeat evidence clearly demands it.
DOWNBEAT_RESTART_PROB = 0.005
# Mild prior favouring 4 beats per bar, the most common case, used only to
# break near ties. Expressed as log-likelihood bonus per bar.
METER_PRIOR_LOGBONUS = {2: 0.0, 3: 0.0, 4: 0.05}
# Ternary vs binary subdivision decides 6/8 vs 2/4 (and 12/8 vs 4/4): ratio
# of onset energy at beat thirds vs beat halves. 1.0 == neither; we
# require a clear 25% margin before calling a meter compound.
COMPOUND_RATIO_THRESHOLD = 1.25
# Onset envelope used for subdivision analysis.
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
# Ignore kicks quieter than this relative to the loudest kick in the render.
KICK_MIN_REL_LEVEL = 0.25
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
