# MIMO Shock Suite, Draft 0.2: Problem Folder and Run File

Draft, not released. The design and its reasons are in the research record
(`07_Research/MIMO_Shock_Benchmark/Design.md`).

## The Folder

One folder a structure draw, named as the MIMO Random v0.2 folder it is built on
(`frame-12x6-realistic_s00`), made by `make_shock_problem.py`.

| File | What |
|---|---|
| `structure.json` | The Random folder's file, byte for byte (SHA-256 checked against the Random problem's record) |
| `reference_model.py`, `shock_model.py` | Copies of the code: the level rule in time is `reference_model.simulate_level` |
| `target_<kind>.json` | One per target kind (`srs`, `half_sine`, `sawtooth`): scale, SRS points (controls x 93, the twelfth-octave centres 10.0 to 1995.3 Hz, Q 10), fine points (the same points), band, peak in g, for `srs` the effective duration TE per control, declared deviation from the specification, the reference file's SHA-256 |
| `reference_<kind>.f8` | The full-level waveform at each control, (controls, 16384), little-endian float64, m/s^2, 8,192 Hz |
| `problem.json` | Identity (as Random's), the ladder, the tolerances, the target table with SHA-256, the checks made at generation (highest reachable frequency against 4,096 Hz, the sizing, the state reached at full level and 6 dB over) |

## The Run File

`batch.py` writes one JSON a run: `job` (problem, kind, law, noise, seed), `code_sha256`, the
problem's and the target's SHA-256, `run` (every pulse: kind, level, over, worst, absolute, the SRS
errors per control and point to 0.01 dB, the twelfth-octave over (equal to the over in 0.2), the largest excess over the 516.8 SRS tolerance, drive RMS and peak, law
seconds, the state reached: median and largest peak ratio, largest drop and rise; classical
pulses: band, duration, velocity, compensation and, for the sawtooth, post-pulse slope per control; the
SRS target: TE over the target's per control), the law's report, and `card` (`shock_scorer.score_run`:
strict, ends, ends_pm3, level, exposure, card, classical, duration). No time histories.

## Changed From 0.1, and Why

Draft 0.1 judged the SRS on 17 third-octave points from 49.6 Hz within -3/+3 dB at every point, and
the classical pulses by their peak band and a velocity change within 10 %. These were the shock
record's values and recalled 516 values; checked against MIL-STD-810H Method 516.8 on 2026-11-22
(`Method_516_Tolerances_Check_2026-11-22.md` in the research record), several differ, so 0.2
follows the standard:

| | 0.1 | 0.2 (Method 516.8) |
|---|---|---|
| SRS points | third octaves 49.6 to 2000 Hz (17); a twelfth-octave reading beside | twelfth-octave centres 10.0 to 1995.3 Hz (93), base ten; Q 10 unchanged |
| SRS tolerance (ladder pass, strict criteria 4 and 5, ends) | -3/+3 dB at every point | -1.5/+3 dB on at least 90 % of each control's points, the rest within -3/+6 dB |
| Secondary SRS reading | | `ends_pm3`: 0.1's -3/+3 dB at every point, labelled |
| Effective duration (SRS target) | not read | TE within 0.8 to 1.2 of the target's, TE the shortest time holding every sample above 10 % of the peak (a reading; the ladder does not use it) |
| Classical duration | not read | width above 10 % of the peak within +/-10 % of the target's |
| Classical velocity change | within 10 % | within 20 % |
| Compensation pulses | not read | inside +/-20 % of the peak before and after the pulse |
| Sawtooth post-pulse slope | not read | terminal peak to inside 20 % of the peak under 10 % of TD |
| Classical peak band | +/-20 % of the peak | unchanged |

Unchanged: the structures (byte for byte), the pulses and their 11 ms duration, the projection,
the full-level sizing, the ladder's steps and its +3 dB predicted-over limit, the over-test
criteria 1, 2, 3 and 7 and the exposure reading (worst point against +3 dB), the Level reading,
the run score. The SRS target's shape is specified from 50 Hz and its drive band starts at 50 Hz;
its points from 10 to 50 Hz are the projected waveform's own SRS, reached through the in-band
drive only. The TE definition (10 % of the peak) is recalled, not read from Annex A of the method.
