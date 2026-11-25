# MIMO Sine Problems (Draft 0.1)

Status: draft, not released. This file describes the MIMO Sine suite of the Hertzforge Open
Benchmark Set: what a problem folder holds, how each number in it was chosen, the case files the
runner reads, the run fields the scorer reads, and what the scorer reports. The structures are the
MIMO Random suite's (version 0.2), byte for byte; `FORMAT.md` at the root of this repository
describes them and the simulator's level rule.

Code, all in `sine/`:

| File | What |
|---|---|
| `sine_model.py` | The suite's constants and model: the band, the reference, the full level, the tone rules, the tolerances, the drives the checks play |
| `make_sine_problem.py` | Makes the problem folders, with the generation checks |
| `plan.py` | The matrix: structures, tests, laws, jobs |
| `sine_scorer.py` | The scorer: several readings, never one verdict |
| `batch.py` | Runs the matrix through the runner and the scorer |
| `check_tone_state.py` | Holds the steady-tone level rule to `reference_model.simulate_level` on one problem |
| `publication_check.py` | Every generated or shipped file: no avoided word, no local path, no listed name |
| `case_schema.py`, `msnb-case-1.schema.json` | The runner's case schema (copied from the runner's draft) and a checker for it |
| `tests/` | `python3 -m unittest discover -s sine/tests`; set `MSNB_RANDOM_PROBLEMS` to the Random problems folder |

## 1. The Matrix

| Block | Rigs (shakers x controls; folder) | Tiers | Structures | Tests |
|---|---|---|---|---|
| A | 3x12 (`frame-12x3`), 6x12 (`frame-12x6`), 12x24 (`frame-24x12`), 6x6 (`frame-6x6`), 12x12 (`frame-12x12`) | linear, realistic, strong | the Random study's twenty structures a rig and tier (seeds 0 to 9, 32, 33, 12 to 15, 34, 17, 35, 19): 300 | T1 to T5 |
| D | 6x12, 6x6 | softening only (`_soft`) and hardening only (`_hard`), realistic and strong | the first ten of those seeds: 80 | T1 to T4 |
| L | 6x12, 6x6 | realistic, strong | the first ten seeds: 40, all in block A | L1 |

380 problem folders; the limit block adds a test to 40 of them. No held-back structure is made.

Laws and the tests each is run on (the laws' manifests): `matrix` all tests; `paired` and `ratio`
T3 and T4; `steepest_descent` and `secant` T4. That is 4,140 jobs (`plan.py`).

The shipped MIMO Sine engine takes at most 16 controls and 8 drives. The 12x24 rig (24 controls,
12 drives) and the 12x12 rig (12 drives) are over those limits. Their problems are made and
`checks.engine.within_channel_limits` says so; whether they are run is the lead's decision.

## 2. A Problem Folder

Named as the Random folder it is built on (`frame-12x6-realistic_s00`).

| File | What |
|---|---|
| `structure.json` | The Random problem's, byte for byte; its SHA-256 is checked against the Random problem's record when made |
| `reference_model.py` | The repository's reference model (the simulator, restated), copied |
| `sine_model.py` | The suite's model, copied, so the folder alone recomputes everything in it |
| `problem.json` | The record: identity, stage, effects, band, full level, the reference's summary, the tests, the files, the checks, the article in service (copied from the Random problem) |
| `reference.json` | The reference: the specification, the projection, the declared deviation, the grid check, the condition of the response |
| `reference_hz.f8` | The reference grid, Hz: 20 x 2^(k/384) up to 2000, then 2000 exactly (2,553 points) |
| `reference_amplitude.f8` | Each control's reference amplitude at full level, m/s^2 peak, (points, controls), little-endian float64 |
| `reference_phase_deg.f8` | Each control's reference phase against the tone, degrees in (-180, 180], (points, controls) |
| `case_reference.json` | The same reference in the runner's form, `msnb-reference/1`, 9 significant digits |
| `ideal_drive.c16` | The least-squares drive at full level, V peak, (points, drives), complex128; not given to the law |
| `scoring/ideal_held_out.c16` | The held-out gauges' small-signal response to the ideal drive, m/s^2 peak, (points, gauges) |
| `scoring/stress_modal.f8` | The Random problem's modal stress matrix, byte for byte (SHA-256 checked) |
| `test_T1.json` ... `test_T5.json`, `test_L1.json` | Each test; a test not built says so and why (`built: false`) |
| `cases/<test>__<law>.json` | The runner's case files (section 7) |

Given to the law: the controls' responses as the engine measures them, the drives, the
reference, the test and its ladder, and for L1 the limit channel and its bound. Kept for scoring:
the held-out gauges, the ideal drive, the modal stress matrix.

## 3. The Band and the Reference

The band is 20 to 2000 Hz, every Random v0.2 problem's band (`grid.band_hz`, read at
generation; a problem whose band differs is refused).

The specification is flat in acceleration over the band, the same amplitude at every control,
in phase at every control. Flat in acceleration is kept because the band starts at 20 Hz, where a
constant acceleration is still a modest displacement (6.7 g at 20 Hz is 4.2 mm peak), and because
the Random suite's targets are judged in acceleration too.

The reference is the specification projected onto what the system can reach in its small-signal
model (MIMO Shock Default 3): at each grid frequency, the least-squares drive
`d = H^+ s` (Tikhonov regularization at 1e-3 of the largest singular value) and its response at
the controls `R = H d`, where `H` is the small-signal response at the controls
(`reference_model.frf`) and `s` the specification. On a square rig `R = s`; on a rig with more
controls than drives `R` is the nearest reachable pattern, and the gap `R / s` is stored as the
declared deviation (`reference.json`, `declared_deviation`: dB min, max, rms, and the share of
points and controls more than 6 dB and 20 dB under). A common delay cancels in `H H^+`. No floor
is put on the projected reference: where it is deep under the specification it stays there.

Interpolation between grid points: amplitude log-log, phase linear against log frequency the
shorter way round, held beyond the ends (the engine's). `reference.json` `grid_check` gives the
interpolated reference at every grid midpoint against the projection computed there.

## 4. The Full Level

As MIMO Shock Design section 1, for a tone. The full level is the specification amplitude `s`
(m/s^2 peak, the same at every control) at which the median, over the article's modes in band,
of the size of the frequency change at the resonance the tone passes equals the median
`|frequency_change|` of the Random problem's full level (`level_targets.frequency_change`, the
same modes). The modes are the article's modes in band (modes with a level rule; on a linear
structure, the modes in band other than the rigid-body modes).

How a steady tone drives the level rule (written for `sine_model.tone_ratio_curve`): under a tone
at `f` with complex drive `d`, mode `m` is driven by `F_m = drive_shape_m . d`, and its amplitude
is `|F_m| / |w_m(r)^2 - w^2 + 2j zeta_m(r) w_m(r) w|`, where `w_m(r)` and `zeta_m(r)` are the
level rule's frequency and damping at amplitude ratio `r` (`reference_model.level_state`). The
rule's envelope is a peak follower of `sqrt(q^2 + (v / w_m)^2)`; for a steady tone this is the
modal amplitude times `max(1, f / f_m)`. Each mode is independent in modal coordinates, so the
state is a fixed point per mode: `r = amplitude(r) max(1, f / f_m) / reference_amplitude_m`. Where
several fixed points exist, the suite takes the smallest, the one a slow rise of the level from
zero reaches. For a mode, the curve `M(r)` (the smallest specification scale that holds the mode
at ratio `r` or more over the frequencies within 15 % of its small-signal frequency) is tabulated
on 1,601 ratios from 1e-5 to 1e3; its inverse gives the mode's ratio at any scale, the rule gives
its frequency change, and a bisection finds `s`. A quasi-steady figure: a sweep that gives a mode
too little time leaves it under this state.

Checked against `simulate_level` (experiment MSNB-B-E1, `check_tone_state.py`): on
frame-12x6-realistic s00, five tones, the largest envelope over the last tone period against the
predicted fixed point agreed to within 3 % for every mode within a quarter octave of the tone
(the modes the rule is read at), and the frequency change of the largest mode at each tone agreed
to 1e-5 or better. Modes far above a tone swing within each tone period (down to 0.15 of their
peak at seven times the tone): their state is not steady under a tone.

A linear structure takes its realistic member's full level. A single-effect structure (`_soft`,
`_hard`) takes its two-effect member's at the same tier: the hardening-only members' Random
median change is 0 or near it (frame-12x6-realistic_hard s03: every mode 0), so the rule cannot
size them. `problem.json` `full_level.sizing` says which (`sized` or `from_member`).

`checks.state` gives the quasi-steady state at full level and at 6 dB over: each mode's ratio,
change, damping factor, the largest drop and rise, and the highest frequency reached.

## 5. The Tests

Every test runs a ladder before its judged level: levels -12, -9, -6, -3 and 0 dB against full
level (the engine's default level table), each rung but the last held 2 s (the runner's default
rung), the last held to the end of the test (the runner's `ladder`). A one-level run on a realistic
structure aborts at the end of the settle window (the runner's experiment MSNB-A-E1), so every test
climbs. The engine runs the ladder on the test's own clock (`hertzforge-mimo-sine` `schedule.rs`:
the ladder ticks beside the test plan; a dwell's clock, a step's clock and the ladder's stop in a
settle window, a sweep's does not). So the lower rungs take the first part of the test: a sweep
runs its first 8 s and more below full level (1 octave a minute: from 20 Hz to about 22 Hz, more
with the settle windows), a stepped test its first steps. A dwell's case holds the rungs' 8 s on
top of the dwell (`dwell_s` = the T4 seconds + 8), so the dwell at full level is the T4 seconds
after the last settle window. The scorer counts a sweep's and a stepped test's coverage from where
full level began and reports the share run below full level first (`ladder_share`). The
engine identifies at its own level (-20 dB, its default) first. Judged: updates at full level, the
law's answer used, outside the engine's settle window (the runner's `judged`), and for a step or a
dwell outside the suite's own settle window, counted from the first full-level update of the step
or dwell. Measurement noise: 1e-3 of the specification amplitude, m/s^2 rms on every output read,
one seed per problem and test (CRC-32 of `<problem>/<test>`), the same for every law.

| Test | What | Rule |
|---|---|---|
| T1 | Log sweep up, 20 to 2000 Hz, one octave a minute | 398.6 s |
| T2 | Log sweep up, four octaves a minute | 99.7 s |
| T3 | Stepped sine, 12 steps an octave, 20 x 2^(k/12) and 2000 Hz (81 steps) | Each step held `step_s` = 10 time constants `1 / (zeta w)` of the slowest mode in band, at least 20 cycles of the lowest step (1 s); the first 6 time constants of each step are not judged |
| T4 | Dwell at the first resonance in band | Held 30 time constants of the dwell mode, at least 10 s; the first 20 not judged |
| T5 | Log sweep down, 2000 to 20 Hz, one octave a minute, block A only | Built because the runner's `FORMAT_RUN.md` says the engine runs a down-only sweep ("A down-only sweep is the engine's own") |
| L1 | T1 with one limit channel | Section 6 |

The step rule (Derivations LD-5): a mode's time constant is `tau_m = 1 / (zeta_m w_m)`; the
engine's loop-speed rule keeps the loop at `2 tau_m` or slower, so three loop time constants (6
`tau_m`) settle the loop to 5 %, and four more are judged. The runner holds every step for the same
`step_s`, so the time is set by the slowest mode in band (the largest `tau_m` in seconds). 12 steps
an octave puts a step every 5.9 % in frequency.

The dwell: the first resonance in band is the lowest peak of the largest singular value of the
small-signal response at the controls (1/1536-octave grid) standing 3 dB over its flanks. The
dwell mode is the mode nearest it; `test_T4.json` records the mode, its quasi-steady state at full
level, and how far its resonance at full level sits from the tone in half-power half-bandwidths.
The dwell time: 20 time constants of the dwell mode to settle at full level (ten loop time
constants at the loop-speed rule's `2 tau_m`), 10 more to judge, at least 10 s (1,000 updates at a
10 ms update).

## 6. The Limit Block (L1)

One limit channel on a held-out gauge. For each held-out gauge, the peaks of its full-level
small-signal response to the ideal drive (1/1536-octave grid); the gauge whose highest peak stands
furthest over its second (at least 1 dB) is the limit channel; its bound is flat over the band at
the geometric mean of those two peaks. So at the small signal the response passes the bound
around one resonance only, by half the gap. `test_L1.json` `limit` records the gauge, the gap, the
bite (dB over the bound at the top peak), the regions over the bound and their span.

## 7. Case Files (the Runner's `msnb-case/1`)

The runner and its formats are Package A's (`crates/sds-control/examples/msnb_run/FORMAT_RUN.md`
of the Hertzforge repository, draft commit 1c0a4a24); the schema is copied here as
`msnb-case-1.schema.json` and every case is checked against it (`case_schema.py`). Paths are
relative to the problems folder, the runner's `--root`.

| Key | The suite writes |
|---|---|
| `case_id` | `<problem>__<test>__<law>` |
| `structure` | `<problem>/structure.json` and its SHA-256 |
| `controls` | The structure's control outputs, in file order |
| `fs_hz` | 8192 |
| `law` | `id`, `params` `{}` (the law's defaults) |
| `test` | sweep: `band_hz`, `rate_oct_per_min`, `direction`; stepped: `band_hz`, `steps_per_octave`, `step_s`; dwell: `freq_hz`, `dwell_s` (the T4 seconds plus the 8 s of the lower rungs, section 5) |
| `reference` | `<problem>/case_reference.json`, its SHA-256, `phase` `auto` |
| `full_level_scale` | 1 (the reference file is at full level) |
| `ladder` | `levels_db` [-12, -9, -6, -3, 0], `rung` `{"seconds": 2}` |
| `noise` | `rms_m_s2` (1e-3 of the specification), `seed` |
| `limits` | L1 only: the gauge's output name and the flat bound on [20, 2000] Hz |
| `protective` | `default` (what a user's run gets) |

Left to the runner's defaults: `tick_s` (0.01 s), `drives` (ceiling 9 V, abort 10 V),
`identification_seed`, `max_test_s`. The engine's own identification is used (no `--ideal-frf`).

## 8. What the Scorer Reads From a Run

The run folder's `run.json` (`msnb-run/1`): `case_id`; `outcome` (`class`, `engine_outcome`,
`guard`, `channel`, `frequency_hz`, `time_s`, `level_db`, `message`); `history` (`time_s`,
`frequency_hz`, `level_db`, `judged`, `control_re/_im`, `reference_re/_im`, `drive_re/_im`,
`limit_re/_im`, `limit_bound`, `bin_error_db_min/_max`); `true_modes` (`tracked`, `modes`,
`frequency_hz`); `summary` (`largest_drive_vpk`, `over_ceiling_samples`). `read_run` in
`sine_scorer.py` is the one place that reads them.

The runner records the controls and limit channels only. The held-out gauges as tested are
estimated: the run's drive at each point through the modal model, the modes the runner followed
at the frequencies it reports, their damping from the level rule at that frequency (the smallest
ratio giving that change), every other mode small signal.

## 9. The Scorer's Readings

`python3 sine/sine_scorer.py <problem folder> <run folder> [--test T1] [--out card.json]`. The card
holds readings, never one verdict. Each judged point weighs its share of the judged run: for a
sweep each point stands for its 1/48-octave bin; for a step or a dwell, time.

| Reading | What |
|---|---|
| `amplitude` | Each control's amplitude over the reference along the judged run, against each band of section 10: share inside, worst excursion (dB beyond the band), where (Hz, control), the error there. Between the sweep's points the run's bin extremes are read. The composites are the average of the control amplitudes over the average of the reference amplitudes |
| `phase` | Each control's phase against the reference: p95 and worst, and the share within 10 degrees as an analogy only (MIL-STD-810H Method 527.2 Annex E 5.2, random cross-spectra). Not judged |
| `resonances` | For each article mode in band, its half-power band (at least 1/96 octave each side; as tested where the runner follows the mode): the worst error at the controls and held-out gauges on the way through it; counts of resonances over 1.5, 3 and 6 dB |
| `practice` | The proposed practice's report card (Hertzforge Journal 2026-001), read for a tone: line 1 level within 1.5 dB; line 2 pattern graded A (3 dB), B (4.5), C (6); line 3 over-test alarm at +6 dB; line 4 share of held-out bands more than 3 dB under; line 5 declared deviation and law error; line 6 shakers, controls, condition, exact-fit flag. `departures` lists every departure from the article's statement |
| `stress` | Per element, the largest peak fibre stress over the judged run as tested against the same under the ideal drive at the same frequencies |
| `frequency` | Sweep: realized rate against the stated, share of the sweep within +/-10 %, duration within 5 %. Dwell and steps: largest frequency error against +/-2 % (IEC) and +/-0.1 % (MIL), dwell duration within 5 % (the full-level time from where the engine's settle window closed) |
| `completion` | The outcome and why the run ended; the share of the test covered at full level (a sweep or a stepped test counted from where full level began; `ladder_share` the share run below full level before it) |
| `drive` | Largest drive norm and drive voltage; effort over the ideal drive (dB) |
| `state` | Stage 2: each followed article mode's largest frequency change as tested, beside the article in service (the Random problem's `service_state`) and the quasi-steady state at full level |
| `limit` | L1: the limit channel against its bound, worst and share over |

## 10. Tolerances

Read from the standards' text (MIL-STD-810H Method 514.8; IEC 60068-2-6:2007, the publicly
available preview, read through clause 4.3 and the contents). Every value the scorer uses is read.

| Item | Tolerance | Clause | Status |
|---|---|---|---|
| Check point (each control) | +/-25 % up to 500 Hz, +/-50 % above | IEC 60068-2-6:2007 4.1.4.2 | read |
| Reference point (multipoint: fictitious, the average of the check points) | +/-15 % | IEC 60068-2-6:2007 4.1.4.1, 4.2.1, 4.2.1.1 | read |
| Composite control (average, maximum or minimum of the control channels) | +/-10 % | MIL-STD-810H 514.8 Table 514.8-III, note 1 | read |
| Multi-point average control, each channel | +/-25 % below 500 Hz, +/-50 % above | MIL-STD-810H 514.8 Table 514.8-III | read |
| Multi-point maxi control, each channel | +10 %/-25 % below 500 Hz, +10 %/-50 % above | MIL-STD-810H 514.8 Table 514.8-III | read |
| Frequency, sweeping | +/-1 Hz 5 to 50 Hz, +/-2 % above 50 Hz | IEC 60068-2-6:2007 4.1.5.1 | read |
| Frequency, fixed | +/-2 % | IEC 60068-2-6:2007 4.1.5.2 a) | read |
| Frequency | +/-0.1 % | MIL-STD-810H 514.8 Table 514.8-III | read |
| Sweep rate | one octave a minute +/-10 % (IEC); +/-10 % of the stated rate (MIL) | IEC 60068-2-6:2007 4.1.6; MIL-STD-810H 514.8 Table 514.8-III | read |
| Test time duration | +/-5 % | MIL-STD-810H 514.8 Table 514.8-III | read |
| Phase between fixing points | "substantially in phase" (no number) | IEC 60068-2-6:2007 4.1.1; MIL-STD-810H 514.8 4.2.2.2 | read; reported, not judged |
| Control system uncertainty | not over one third of the Table 514.8-III tolerances | MIL-STD-810H 514.8 4.2.2.2 | read; not used (a simulator has none) |
| Cross-axis motion | 50 % to 500 Hz, 100 % above | IEC 60068-2-6:2007 4.1.2.1; MIL-STD-810H 514.8 Table 514.8-III | read; not judged |
| Signal tolerance (distortion) | 5 % | IEC 60068-2-6:2007 4.1.3 | read; not judged (the run holds the tracked fundamental only) |
| Critical frequency | +/-0.5 % above 100 Hz | IEC 60068-2-6:2007 4.1.5.3 | read; not used |
| Severities, endurance durations, the vibration response investigation and resonance dwell procedure | | IEC 60068-2-6:2007 5.1 to 5.3, 8.2, 8.3, Annex A | recalled, not read (outside the preview); not used |

IEC 60068-2-6 4.1.4 allows a wider tolerance, set by the relevant specification, "at low
frequencies or with large size or high mass specimens"; the suite applies the clause values as
written.

## 11. Generation Checks

A problem is refused, and the log says so, when: its `structure.json` does not match the Random
problem's record; a mode can reach `FS / 2` = 4096 Hz or more (`reference_model.max_frequency`,
the simulator's bound); its band is not the suite's; the reference or the ideal drive is not finite
and positive; the full-level ideal drive or the same drive 6 dB higher does not play to a finite
response through `simulate_level` (a sweep at 8 octaves a minute over the band and 3 s at the dwell
frequency, each ramped in over 0.5 s with a raised cosine, because a tone switched on from rest
rings every mode); the limit block finds no gauge with a peak 1 dB over its second. The play's
peaks are recorded (`checks.play`). The engine's limits (16 controls, 8 drives, 10 V a drive) are
recorded, not refused (`checks.engine`).

The generation log's last line is `GENERATION END: <made> made, <refused> refused, of <n>, <time>`.

## 12. The Batch

`python3 sine/batch.py --problems <problems> --out <runs> --name <batch> --runner <msnb_run>
[--workers 6] [--only ...] [--tests ...] [--laws ...] [--min-free-gb 50]`. One run folder a job:
the runner's `run.json`, the batch's `batch.json` (the suite's code hash, the runner's and the
case's SHA-256, the runner's exit status) and the scorer's `card.json` (with the code hash). The
batch stops starting jobs when the free disk is under 50 GB. The log names each job and gives no
outcome; its last line is `BATCH END <batch>: N done, F failed, S skipped`.
`--runner` may be the Python harness (`sine/pyloop/harness.py`, section 13); a `.py` runner is run
with the batch's own Python.

## 13. The Python Harness

`python3 sine/pyloop/harness.py <case> --root <problems> --out <run folder>` runs one case with no
Hertzforge build and writes `run.json` in the runner's `msnb-run/1` layout (and `timing.json`),
so the same scorer reads either. Exit status as the runner's: 0 completed, 1 another end, 2
refused (nothing written). Agreement between the harness and the engine has not been checked; it
is a separate step.

The laws (`sine/pyloop/`) are the research testbed's (`msine_gym` and its `build_m` reference
compositions). Changed from the testbed: the imports; the controller keeps only its quasi-steady
loop (what the parity vectors were made with); `identification` and `structures` keep only the
classes the laws read; the loop-speed rule and the secant's settle count read the tracker's delay
factor and a transport delay from the setup (the testbed's `1 / pi` and 0 when the setup gives
none, as the shipped law math's defaults). `paired` is `PairedRef`, `ratio` `RatioPerControl`
(pairing by reference), `steepest_descent` `SteepestDescent`, `secant` `SecantComplexGain` (carry
factor), each at the shipped manifest's defaults (`params` `{}`; other parameters are refused).
`matrix` is `ShippedMatrix` (`pyloop/matrix_shipped.py`): `RefMatrix` (free phase in phase) with
the shipped law's update restated from `matrix.rs`: the reference's change carried backward from
the reference told at the last update, the correction aimed at the reference as the trackers read
it (`seen_reference`), the cut part of the last carry carried again (`recarry`), the notch scale
on the reference aimed at.

They are held to the shipped laws' parity vectors (`tests/data`, copied from the Hertzforge
repository at 9a4c7b80): 30 runs (each law and schedule, the matrix runs for `RefMatrix` and
`ShippedMatrix`), largest relative difference of drive and controls 5.6e-15
(`tests/test_pyloop.py`; a different law is shown to fail by more than 0.1). The vectors have a
flat reference, the phase specified or in phase, no guards and the quasi-steady loop: they cannot
see the shipped matrix's backward carry, seen reference or recarry (a test shows `RefMatrix` and
`ShippedMatrix` part on a sloped reference), nor the loop delays or the notch scale. What is not
restated of the shipped laws: the modal fit's group limits, its splitting of large groups and the
windows it leaves out (where the shipped matrix carries nothing); the Python fit fits every group
jointly and the matrix carries everywhere. The Python fit over a sweep's band took 210 s on
frame-12x12 s00 (23 modes fitted); a dwell's fit finds no mode (the band is the dwell frequency at
both ends, as the engine's) and uses the smoothed identification.

What the harness restates from the shipped engine (`hertzforge-mimo-sine` `core.rs`,
`schedule.rs`; `hertzforge-dsp` `tone.rs`):

| Part | As restated |
|---|---|
| Update | The case's `tick_s` in whole samples; the drive ramps linearly over a tick; the tone's frequency is held through a tick and is phase-continuous |
| Structure | `reference_model.simulate_level` on the controls and limit channels, the structure's own modes and level rule, continued tick to tick; white noise of `noise.rms_m_s2` added, seeded by `noise.seed` |
| Tracker | Four cascaded one-pole sections on the heterodyned signal, `alpha = B_n / (0.15625 fs)`, `B_n` 10 % of the frequency, at most 100 Hz, set each update |
| Schedule | The ladder on the test's clock; the dwell's, a step's and the ladder's clocks stop in a settle window, a sweep's does not; the stepped grid from the low edge, `steps_per_octave` an octave |
| Settle window | `2 x 1.3 / B_n + 3 x` the identification's group delay (largest element form, five-line smoothing, the largest within +/-2 `B_n`), opened at engage, at every level move and at every step |
| Level carry | The drive in force scaled with each move of the level, by at most 1 dB down and min(1 dB, 240 dB/s x tick) up an update; the law told `engage` (hold) while the window is open |
| First drive | At engage and at every step the law engages again; its drive bounded to the regularized inverse drive (1e-3) plus 6 dB, or 5 % of the ceilings' norm on an untrusted line |
| Laws' loop delays | Tracker delay factor 0.625 (the engine's four sections); transport 0 (the identification holds the structure's delays; no pipeline) |
| Aimed reference | The reference times the notch scale for every law (`law.rs` `aimed_reference`) |
| Judging | Control error against the reference (notched, with limit channels); abort after `abort_ticks` updates in a row outside the abort band, not counted in a settle window; a limit channel over its abort band ends the run at once; `judged` = at the last level, not settling, the law's answer used |
| Caps and guards | Change caps per drive against the drive as played; divergence stop after `divergence_s` of binding caps; a drive over its ceiling scales the whole vector; drive abort; 3 law misses a law fault; the runner's time limit `1.5 x (test + rungs) + 60 s` |
| Protective | `default` and `widened` as the runner's table |
| Run file | History decimated as the runner's (1/48-octave bins from the band's low edge at each level, with the bins' counts and error extremes; every update for a step or a dwell); `true_modes` from the simulator's own envelopes at the end of each tick, for the modes in the identification's band; outcome classes and guard names as the runner's |

What it leaves out or does differently:

| Part | The harness | The engine |
|---|---|---|
| Identification | The exact small-signal response (`reference_model.discrete_frf`) on 0.25 Hz lines over the test band /1.25 to x1.25, every line trusted, coherence 1 | Measured at -20 dB with noise, `max(24, 6 x drives)` records, trust and gate at coherence 0.5 |
| Lag | The identification includes the structure's delays; no service pipeline delay between the law's answer and the drive | The lag (including the service's block) is measured and taken out of the identification; the drive reaches the structure that many frames later |
| Backstop notch | Simplified: each limit channel's ratio to the identification's prediction learned from settled readings (3-update smoothing); the drive scaled at once to put the most loaded channel at its bound; the judged reference scaled by the share the inverse drive allows | The engine's notch with its near-bound and hysteresis rules, release rate and a law's notched reference |
| Units | m/s^2 throughout | The law sees g |
| Stop | The run ends at the stop; no ramp down (`drive_is_zero` null) | A 120 dB/s ramp to zero |
| Not modelled | Law requests (never honoured, as the runner's default), the rotation cap (none, as the default), the law's compute budget, kinematic ratings, events (`events` empty), `rung` in sweeps (refused) | |
