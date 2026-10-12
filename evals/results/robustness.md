# Generalisation and robustness (generated audio)

Written by `python -m evals.robustness`. Ground truth is exact (generated in code).

| case | group | status | beat F | downbeat F | max err ms | meter | tempo | runtime s / audio s | notes |
|---|---|---|---|---|---|---|---|---|---|
| random_00_3-4_96.2bpm | generalisation | **pass** | 1.0 | 1.0 | 4.89 | 3/4 | constant | 52.6 / 47.91 |  |
| random_01_4-4_87.2bpm | generalisation | **pass** | 1.0 | 1.0 | 6.49 | 4/4 | constant | 2.33 / 70.4 |  |
| random_02_6-8_108.7bpm | generalisation | **partial** | 1.0 | 0.6667 | 13.88 | 12/8 | constant | 0.92 / 28.17 | constant grid off by up to 13.88 ms (> 10.0 ms); hypermeter: 12/8 for 6/8 |
| random_03_6-8_102.8bpm | generalisation | **partial** | 1.0 | 0.6667 | 15.34 | 12/8 | constant | 1.47 / 32.46 | constant grid off by up to 15.34 ms (> 10.0 ms); hypermeter: 12/8 for 6/8 |
| random_04_3-4_113.9bpm | generalisation | **pass** | 1.0 | 1.0 | 4.42 | 3/4 | constant | 1.49 / 42.3 |  |
| random_05_3-4_138.8bpm | generalisation | **fail** | 1.0 | 0.2857 | 6.4 | 4/4 | constant | 1.62 / 35.46 | meter 4/4, expected 3/4 |
| random_06_3-4_72.6bpm | generalisation | **pass** | 1.0 | 1.0 | 3.01 | 3/4 | constant | 2.4 / 62.86 |  |
| random_07_6-8_71.8bpm | generalisation | **partial** | 0.993 | 1.0 | 12.21 | 6/8 | constant | 1.92 / 43.83 | eighth-note level (compound meter); constant grid off by up to 12.21 ms (> 10.0 ms) |
| random_08_3-4_149.8bpm | generalisation | **fail** | 1.0 | 0.6667 | 8.41 | 6/8 | constant | 1.3 / 31.23 | meter 6/8, expected 3/4 |
| random_09_4-4_105.2bpm | generalisation | **partial** | 1.0 | 1.0 | 20.48 | 4/4 | constant | 1.84 / 57.22 | constant grid off by up to 20.48 ms (> 10.0 ms) |
| random_10_4-4_102.8bpm | generalisation | **pass** | 1.0 | 1.0 | 9.8 | 4/4 | constant | 2.42 / 59.64 |  |
| random_11_6-8_88.1bpm | generalisation | **partial** | 1.0 | 0.6667 | 6.19 | 12/8 | constant | 1.42 / 36.82 | hypermeter: 12/8 for 6/8 |
| tempo_60 | robustness | **pass** | 1.0 | 1.0 | 0.0 | 4/4 | constant | 3.28 / 98.2 |  |
| tempo_90 | robustness | **pass** | 1.0 | 1.0 | 7.78 | 4/4 | constant | 2.52 / 66.2 |  |
| tempo_128 | robustness | **pass** | 1.0 | 1.0 | 2.09 | 4/4 | constant | 1.55 / 47.2 |  |
| tempo_160 | robustness | **pass** | 1.0 | 1.0 | 2.78 | 4/4 | constant | 1.42 / 38.2 |  |
| tempo_200 | robustness | **partial** | 0.6667 | 0.6667 | 0.0 | 4/4 | constant | 1.43 / 31.0 | tempo octave error: tracked 100.0 BPM vs 200.0; x2 alternative exposed; F at the tracked level 1.000 |
| meter_3-4_132 | robustness | **pass** | 1.0 | 1.0 | 5.18 | 3/4 | constant | 1.4 / 35.33 |  |
| meter_4-4_104 | robustness | **partial** | 1.0 | 1.0 | 11.14 | 4/4 | constant | 1.94 / 57.98 | constant grid off by up to 11.14 ms (> 10.0 ms) |
| meter_6-8_84 | robustness | **partial** | 1.0 | 0.6667 | 17.43 | 12/8 | constant | 1.48 / 36.89 | constant grid off by up to 17.43 ms (> 10.0 ms); hypermeter: 12/8 for 6/8 |
| tempo_step_118_to_126 | robustness | **pass** | 1.0 | 1.0 | 9.57 | 4/4 | variable | 2.29 / 65.02 |  |
| tempo_step_140_to_120 | robustness | **pass** | 1.0 | 1.0 | 10.23 | 4/4 | variable | 2.29 / 61.43 |  |
| tempo_ramp_96_to_120 | robustness | **pass** | 1.0 | 1.0 | 16.66 | 4/4 | variable | 1.86 / 55.56 |  |
| swing_4-4_100 | robustness | **pass** | 1.0 | 1.0 | 8.28 | 4/4 | constant | 2.24 / 59.6 |  |
| pad_intro_45s | robustness | **pass** | 1.0 | 1.0 | 5.67 | 4/4 | constant | 3.39 / 94.21 |  |
| silence_20s | robustness | **pass** | - | - | - | - | - | 0.46 / 20.0 |  |
| white_noise_20s | robustness | **pass** | - | - | - | - | - | 0.58 / 20.0 |  |
| short_3s | robustness | **pass** | 1.0 | - | 0.0 | None | constant | 0.1 / 2.5 | short clip: meter not required |
| sr_8k_mono | formats | **pass** | 1.0 | 1.0 | 4.27 | 4/4 | constant | 1.31 / 32.48 |  |
| sr_96k_stereo_24bit | formats | **pass** | 1.0 | 1.0 | 3.43 | 4/4 | constant | 1.06 / 29.43 |  |
| float32_6ch_48k | formats | **pass** | 1.0 | 1.0 | 6.56 | 4/4 | constant | 1.49 / 36.29 |  |
| flac_16bit | formats | **pass** | 1.0 | 1.0 | 1.48 | 4/4 | constant | 1.22 / 32.97 |  |
| mp3_roundtrip | formats | **pass** | 1.0 | 1.0 | 1.74 | 4/4 | constant | 1.54 / 32.97 |  |
| m4a_aac_roundtrip | formats | **pass** | 1.0 | 1.0 | 4.16 | 4/4 | constant | 1.54 / 32.97 |  |
| ogg_vorbis | formats | **pass** | 1.0 | 1.0 | 2.59 | 4/4 | constant | 1.53 / 32.97 |  |
| long_12min | robustness | **pass** | 1.0 | 1.0 | 2.15 | 4/4 | constant | 24.51 / 722.0 |  |

Totals: 26 pass, 8 partial, 2 fail of 36.
