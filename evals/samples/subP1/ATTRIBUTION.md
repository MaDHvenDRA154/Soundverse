# subP1 -- SUBSTITUTE (stands in for P1; not the test pack)

Groove MIDI 2-bar drum loop (GM render)

- Source licence: CC BY 4.0 (Groove MIDI Dataset, Google LLC); original performances recorded for the dataset
- Cite: Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, David Bamman. Learning to Groove with Inverse Sequence Transformations. ICML 2019. https://magenta.withgoogle.com/datasets/groove
- `original.wav` is the excerpt MIDI rendered with FluidSynth and the TimGM6mb General MIDI soundfont
  (GPL-2, Tim Brechbill; Debian package `timgm6mb-soundfont`).
- `score.json`, `page.html`, `render_*.wav` are this repo's recreation of `original.wav`
  (`python -m evals.run part3`); `reference.mid` is the excerpt used as the reference.
