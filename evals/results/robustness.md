# Generalisation and robustness (generated audio)

Written by `python -m evals.robustness`. Ground truth is exact (generated in code).

| case | group | status | beat F | downbeat F | max err ms | meter | tempo | runtime s / audio s | notes |
|---|---|---|---|---|---|---|---|---|---|
| random_00_3-4_96.2bpm | generalisation | **pass** | 1.0 | 1.0 | 2.4 | 3/4 | constant | 15.96 / 47.91 |  |
| random_01_4-4_87.2bpm | generalisation | **pass** | 1.0 | 1.0 | 4.69 | 4/4 | constant | 4.23 / 70.4 |  |
| random_02_6-8_108.7bpm | generalisation | **partial** | 1.0 | 0.6667 | 3.46 | 12/8 | constant | 0.89 / 28.17 | hypermeter: 12/8 for 6/8 |
| random_03_6-8_102.8bpm | generalisation | **partial** | 1.0 | 0.6667 | 1.64 | 12/8 | constant | 1.58 / 32.46 | hypermeter: 12/8 for 6/8 |
| random_04_3-4_113.9bpm | generalisation | **pass** | 1.0 | 1.0 | 1.29 | 3/4 | constant | 2.14 / 42.3 |  |
| random_05_3-4_138.8bpm | generalisation | **fail** | 1.0 | 0.2857 | 1.31 | 4/4 | constant | 1.77 / 35.46 | meter 4/4, expected 3/4 |
| random_06_3-4_72.6bpm | generalisation | **pass** | 1.0 | 1.0 | 3.74 | 3/4 | constant | 2.97 / 62.86 |  |
| random_07_6-8_71.8bpm | generalisation | **partial** | 0.993 | 1.0 | 15.13 | 6/8 | constant | 1.66 / 43.83 | eighth-note level (compound meter); constant grid off by up to 15.13 ms (> 10.0 ms) |
| random_08_3-4_149.8bpm | generalisation | **fail** | 1.0 | 0.6667 | 3.18 | 6/8 | constant | 1.56 / 31.23 | meter 6/8, expected 3/4 |
| random_09_4-4_105.2bpm | generalisation | **pass** | 1.0 | 1.0 | 5.03 | 4/4 | constant | 1.9 / 57.22 |  |
| random_10_4-4_102.8bpm | generalisation | **pass** | 1.0 | 1.0 | 1.97 | 4/4 | constant | 2.66 / 59.64 |  |
| random_11_6-8_88.1bpm | generalisation | **partial** | 1.0 | 0.6667 | 6.5 | 12/8 | constant | 1.45 / 36.82 | hypermeter: 12/8 for 6/8 |
| tempo_60 | robustness | **partial** | 0.669 | 0.6667 | 3.95 | 4/4 | constant | 4.99 / 98.2 | tempo octave error: tracked 119.991 BPM vs 60.0; x1 alternative exposed; F at the tracked level 1.000 |
| tempo_90 | robustness | **pass** | 1.0 | 1.0 | 5.72 | 4/4 | constant | 2.43 / 66.2 |  |
| tempo_128 | robustness | **pass** | 1.0 | 1.0 | 0.92 | 4/4 | constant | 1.59 / 47.2 |  |
| tempo_160 | robustness | **pass** | 1.0 | 1.0 | 1.25 | 4/4 | constant | 1.51 / 38.2 |  |
| tempo_200 | robustness | **partial** | 0.6667 | 0.6667 | 2.3 | 4/4 | constant | 1.42 / 31.0 | tempo octave error: tracked 99.99 BPM vs 200.0; x2 alternative exposed; F at the tracked level 1.000 |
| meter_3-4_132 | robustness | **pass** | 1.0 | 1.0 | 1.38 | 3/4 | constant | 1.38 / 35.33 |  |
| meter_4-4_104 | robustness | **pass** | 1.0 | 1.0 | 1.75 | 4/4 | constant | 1.63 / 57.98 |  |
| meter_6-8_84 | robustness | **partial** | 1.0 | 0.6667 | 4.09 | 12/8 | constant | 1.33 / 36.89 | hypermeter: 12/8 for 6/8 |
| tempo_step_118_to_126 | robustness | **pass** | 1.0 | 1.0 | 3.98 | 4/4 | variable | 2.36 / 65.02 |  |
| tempo_step_140_to_120 | robustness | **pass** | 1.0 | 1.0 | 2.37 | 4/4 | variable | 2.31 / 61.43 |  |
| tempo_ramp_96_to_120 | robustness | **pass** | 0.9688 | 0.9583 | 65.17 | 4/4 | variable | 1.73 / 55.56 |  |
| swing_4-4_100 | robustness | **pass** | 1.0 | 1.0 | 1.11 | 4/4 | constant | 2.37 / 59.6 |  |
| pad_intro_45s | robustness | **pass** | 1.0 | 1.0 | 1.14 | 4/4 | constant | 3.38 / 94.21 |  |
| silence_20s | robustness | **pass** | - | - | - | - | - | 0.43 / 20.0 |  |
| white_noise_20s | robustness | **pass** | - | - | - | - | - | 0.55 / 20.0 |  |
| short_3s | robustness | **pass** | 1.0 | - | 0.96 | None | constant | 0.07 / 2.5 | short clip: meter not required |
| sr_8k_mono | formats | **pass** | 1.0 | 1.0 | 1.41 | 4/4 | constant | 1.18 / 32.48 |  |
| sr_96k_stereo_24bit | formats | **pass** | 1.0 | 1.0 | 1.8 | 4/4 | constant | 1.15 / 29.43 |  |
| float32_6ch_48k | formats | **pass** | 1.0 | 1.0 | 1.93 | 4/4 | constant | 1.61 / 36.29 |  |
| flac_16bit | formats | **pass** | 1.0 | 1.0 | 2.75 | 4/4 | constant | 1.38 / 32.97 |  |
| mp3_roundtrip | formats | **pass** | 1.0 | 1.0 | 2.61 | 4/4 | constant | 1.87 / 32.97 |  |
| m4a_aac_roundtrip | formats | **pass** | 1.0 | 1.0 | 4.25 | 4/4 | constant | 1.49 / 32.97 |  |
| ogg_vorbis | formats | **pass** | 1.0 | 1.0 | 3.26 | 4/4 | constant | 1.98 / 32.97 |  |
| long_12min | robustness | **pass** | 1.0 | 1.0 | 1.1 | 4/4 | constant | 47.0 / 722.0 |  |

Totals: 27 pass, 7 partial, 2 fail of 36.
