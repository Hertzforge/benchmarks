# MIMO Shock Suite, Draft 0.2: Problem Folder and Run File

Draft, not released.

## The Folder

One folder a structure draw, named as the MIMO Random v0.2 folder it is built on
(`frame-12x6-realistic_s00`), made by `make_shock_problem.py`.

| File | What |
|---|---|
| `structure.json` | The Random folder's file, byte for byte (SHA-256 checked against the Random problem's record) |
| `reference_model.py`, `shock_model.py` | Copies of the code: the level rule in time is `reference_model.simulate_level` |
| `target_<kind>.json` | One per target kind (`srs`, `half_sine`, `sawtooth`): scale, SRS points (controls x 93, the twelfth-octave centres 10.0 to 1995.3 Hz, Q 10), fine points (the same points), band, peak in g, for `srs` the effective duration TE per control (at 10 % of the peak, as generated; the run recomputes the target's TE by 516.8), declared deviation from the specification, the reference file's SHA-256 |
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
the classical pulses by their peak band and a velocity change within 10 %. These were the values of an
earlier, unpublished study and recalled 516 values; checked against MIL-STD-810H Method 516.8 on
2026-11-22, several differ, so 0.2 follows the standard:

| | 0.1 | 0.2 (Method 516.8) |
|---|---|---|
| SRS points | third octaves 49.6 to 2000 Hz (17); a twelfth-octave reading beside | twelfth-octave centres 10.0 to 1995.3 Hz (93), base ten; Q 10 unchanged |
| SRS tolerance (ladder pass, strict criteria 4 and 5, ends) | -3/+3 dB at every point | -1.5/+3 dB on at least 90 % of each control's points, the rest within -3/+6 dB |
| Secondary SRS reading | | `ends_pm3`: 0.1's -3/+3 dB at every point, labelled |
| Effective duration (SRS target) | not read | TE within 0.8 to 1.2 of the target's on every control, TE by Method 516.8 Annex A 1.3.2: the minimum time holding every magnitude above the peak over the crest factor (3 when the target gives none); the target's TE computed the same way from its waveform at run time, each pulse's TE, the ratio and the target's TE saved in the run file (a reading; the ladder does not use it). Draft 0.2's runs before 2026-11-22 used 10 % of the peak and saved only the ratio |
| Classical duration | not read | width above 10 % of the peak within +/-10 % of the target's |
| Classical velocity change | within 10 % | within 20 % |
| Compensation pulses | not read | inside +/-20 % of the peak before and after the pulse |
| Sawtooth post-pulse slope | not read | terminal peak to inside 20 % of the peak under 10 % of TD |
| Classical peak band | +/-20 % of the peak | unchanged |
| Full-level sizing (2026-11-25, still 0.2) | the scale whose median change over the article's modes equals the Random full level's, by bisection between 1e-6 and 1e6 with an exact comparison and no check at the bounds | the same, except: a target above the level rule's reachable change (the median change at the upper bound 1e6) by at most 1e-9 (relative) is sized to the reachable change itself, the smallest scale at which the median reaches it, as a target equal to it always was; a target further above, or a search that still ends at a bound, is refused. Why: the rule's change levels off (each drop at min(1, r), a rise at its cap); on structures 8 and 74 (6x6 strong, softening only) the target exceeded that plateau by 5e-17 (rounding), so the exact comparison never held and the search ran to 1e6 (about 100,000 g). Two other problems (audit 6x6 realistic softening 74, strong softening 72) have a target equal to or just under the plateau and were sized correctly; a rule that sizes to a fraction of the plateau would move them, this one does not. Checked by regenerating all 610 problems: every other target's scale is unchanged byte for byte. The sizing record of a problem so sized adds `reachable_change` and `sized_to`; no field the law or the scorer reads changes, so the version stays 0.2 |

Unchanged: the structures (byte for byte), the pulses and their 11 ms duration, the projection,
the full-level sizing (but for the bound case in the table), the ladder's steps and its +3 dB predicted-over limit, the over-test
criteria 1, 2, 3 and 7 and the exposure reading (worst point against +3 dB), the Level reading,
the run score. The SRS target's shape is specified from 50 Hz and its drive band starts at 50 Hz;
its points from 10 to 50 Hz are the projected waveform's own SRS, reached through the in-band
drive only. The TE definition (10 % of the peak) is recalled, not read from Annex A of the method.
