# MIMO Shock Suite, Draft 0.1: Problem Folder and Run File

Draft, not released. The design and its reasons are in the research record
(`07_Research/MIMO_Shock_Benchmark/Design.md`).

## The Folder

One folder a structure draw, named as the MIMO Random v0.2 folder it is built on
(`frame-12x6-realistic_s00`), made by `make_shock_problem.py`.

| File | What |
|---|---|
| `structure.json` | The Random folder's file, byte for byte (SHA-256 checked against the Random problem's record) |
| `reference_model.py`, `shock_model.py` | Copies of the code: the level rule in time is `reference_model.simulate_level` |
| `target_<kind>.json` | One per target kind (`srs`, `half_sine`, `sawtooth`): scale, SRS points (controls x 17, third octaves 49.6 to 2000 Hz, Q 10), fine points (65, twelfth octaves), band, peak in g, declared deviation from the specification, the reference file's SHA-256 |
| `reference_<kind>.f8` | The full-level waveform at each control, (controls, 16384), little-endian float64, m/s^2, 8,192 Hz |
| `problem.json` | Identity (as Random's), the ladder, the tolerances, the target table with SHA-256, the checks made at generation (highest reachable frequency against 4,096 Hz, the sizing, the state reached at full level and 6 dB over) |

## The Run File

`batch.py` writes one JSON a run: `job` (problem, kind, law, noise, seed), `code_sha256`, the
problem's and the target's SHA-256, `run` (every pulse: kind, level, over, worst, absolute, the SRS
errors per control and point to 0.01 dB, the twelfth-octave over, drive RMS and peak, law
seconds, the state reached: median and largest peak ratio, largest drop and rise; classical
pulses: band and velocity per control), the law's report, and `card` (`shock_scorer.score_run`:
strict, ends, level, exposure, card, classical). No time histories.
