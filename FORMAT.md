# Problem Format 0.2-draft of the Open Benchmark Set

This document defines every file and field of a problem folder in format 0.2-draft, the format of
version 0.2 of the Open Benchmark Set's MIMO Random suite (DOI 10.5281/zenodo.24310742), and the scorecard the
scorer writes. The code is in `code/`:
`reference_model.py` (a copy sits in every problem folder), `scorer.py` and `make_problem.py`.
The reference model and the scorer need NumPy and the Python standard library only.

The format string is still `0.2-draft`: the format may change in a later version of the set. Each
published version of the set is fixed: its files never change, and a change to any of them is a new
version with its own DOI.

## 1. The Folder

One folder holds one draw of one structure: `<structure>_s<seed>`, for example `frame-12x6_s00`.
The structure identifier gives the number of control gauges first, then the number of shakers, and
at stage 2 the size of the change (`realistic` or `strong`, and for the diagnostic problems with one
effect only, `_soft` or `_hard`). Binary files are raw little-endian, C order. Units are SI: volts
at the drives, m/s^2 at the outputs, Pa for stress, spectral densities one-sided per Hz.

| File | Required | What |
|---|---|---|
| `problem.json` | yes | Identity, stage and effects, grid, roles, the file table with SHA-256, the checks made when the problem was made |
| `structure.json` | yes | The modal model; per mode a `level` object when the structure changes with level |
| `geometry.json` | no | Nodes, beam elements, lumped masses, for figures |
| `reference_model.py` | yes | NumPy only: frequency response, stress frequency response, the level rule, a linear time response, a time stepper for any structure |
| `target_banded.json` | yes | The specification: band edges and each control gauge's band value |
| `target_line.c16` | yes | The full spectral matrix at the control gauges, `(lines, controls, controls)`, complex128 |
| `limits.json` | yes | The limits of a fair run |
| `scoring/service_a_psd.f8` | yes | Unit A in service: autospectra at the control gauges, then the held-out gauges, `(channels, lines)` |
| `scoring/service_b_psd.f8` | no | Unit B in service: held-out gauges, then tips. Without it the scorecard has no response-spectrum block |
| `scoring/stress_truth.f8` | no | Per beam element: RMS stress and two damage rates, `(3, elements)` |
| `scoring/stress_modal.f8` | no | Fibre stress per unit modal displacement, `(8 x elements, modes)`. Needed together with `stress_truth` |
| `scoring/modal_monitor_scale.f8` | stage 2 | Monitor output m is this times the acceleration of mode m, `(modes)` |

Every file `problem.json` names is listed in its file table with the file's SHA-256, and every
file of each record is listed in the `MANIFEST.sha256` at its root.

## 2. structure.json

`schema`: `hertzforge-benchmark-structure/0.2-draft`.

| Field | Type, unit | Meaning |
|---|---|---|
| `structure_id`, `seed`, `variant` | string, int, string | Identity |
| `kind` | `modal` or `jointed` | `jointed` when the structure changes with level |
| `units` | object | The units in words |
| `n_modes`, `n_drives`, `n_outputs` | int | Counts |
| `force_per_volt` | N/V | Shaker force per volt, for the drive RMS in newtons |
| `rigid_modes` | int | The first modes are the article on its suspension |
| `drives` | list | Name, node, direction, position (m) |
| `outputs` | list | Name, role (`control`, `held_out`, `monitor`), group (`control`, `gauge`, `tip`, `modal_monitor`), node, direction, position; a monitor names its `mode` |
| `modes[]` | list | `f_hz` (Hz, small signal), `zeta` (small signal), `drive_shape` (per volt, one per drive), `output_shape` (one per output), `reference_amplitude` (m of modal displacement; `jointed` only), `level` (object, below; optional) |
| `level_law` | object or null | The structure-wide rule. Holds `kind` (`jointed`), `decay_cycles` and, as fallback for a mode with no `level`, `delta`, `freq_exponent`, `damping_gain`, `damping_exponent` |
| `level_targets` | object or null | What each mode is meant to do at full level under the reference drive (below) |
| `simulator` | object | How the software's simulator makes the model discrete: `feedthrough`, `residual`, `transport_delay_samples`, `output_delays_s` and `step` (jointed) or `softening`, `noisy_outputs`, `discretization` (modal) |

**`modes[].level`**: `drop`, `drop_exponent`, `rise`, `rise_exponent`, `rise_cap`, `damping_gain`,
`damping_exponent`, `damping_cap`, all numbers. Allowed: `0 <= drop < 1`, `rise >= 0`,
`rise_cap > 0` and finite (a cap below 1 is allowed), `damping_gain >= 0`, `damping_cap > 0` and
finite, exponents above zero. A level object without `damping_cap` means no cap. The reference
model refuses anything else and names the field. With `r = A / reference_amplitude` (A the
envelope, a peak follower of the mode's amplitude that decays by exp(-1) in `decay_cycles` periods
of its small-signal frequency):

    f    = f0 (1 - drop min(1, r)^drop_exponent + rise min(rise_cap, r)^rise_exponent)
    zeta = min(0.9, zeta0 (1 + damping_gain min(damping_cap, r)^damping_exponent))

A drop is softening (a joint); a rise is hardening (a stiffness that grows with displacement). The
highest frequency a mode can reach is `f0 (1 + rise rise_cap^rise_exponent)`
(`reference_model.max_frequency`). The simulator refuses a structure that can reach the Nyquist
frequency, 4,096 Hz at the set's sample rate; every problem of this release stays under it (the
highest is 4,006 Hz), and the generator in `code/generator/` now refuses such a structure when it
is generated (`frequency_bound.py`).

A mode with no `level` follows `level_law` (`drop = delta`, `drop_exponent = freq_exponent`, no
rise). A structure with neither is linear. Every mode of a stage 2 structure in this release
carries its own `level` (rigid modes included).

**`level_targets`**:

| Field | Type | Meaning |
|---|---|---|
| `frequency_change` | list, one per mode, fraction | Signed: negative a drop, positive a rise |
| `damping_factor` | list, one per mode | Damping at full level over small-signal damping |
| `at_full_level` | object or null | The size of change as given to the generator, in words and numbers |
| `envelope_over_rms` | number | The envelope a mode's random motion reaches over its RMS, used when the reference amplitudes were set |
| `what` | string | The words |

## 3. problem.json

`schema`: `hertzforge-benchmark-problem/0.2-draft`.

| Field | Type | Meaning |
|---|---|---|
| `status` | string | The release this file belongs to |
| `set` | object | `name`, `version` (`0.2-draft`), `suite` (`MIMO Random`), `tier` (`Frames`) |
| `evidence_level` | string | `simulator` for every problem of this release |
| `stage` | 1 or 2 | 1: the structure does not change with level. 2: it does |
| `effects` | list of `softening`, `hardening` | The effects present in the structure (empty at stage 1) |
| `structure` | object | Identity, the system's class (`cls`), counts, `file` and `sha256` of `structure.json`, `geometry` and `geometry_sha256` |
| `problems` | object | The two problems of the folder: `<structure>/banded` and `<structure>/line` |
| `grid` | object | `fs_hz`, `nfft`, `band_hz`, `df_hz`, `lo_line`, `n_lines` |
| `roles` | object | How many outputs of each role and group, in words |
| `given_to_the_law`, `kept_for_scoring` | list | What a control law may see, and what only the scorer reads |
| `targets` | file table | `banded` and `line`, with SHA-256 |
| `limits` | object | The limits' identifier, `file` and `sha256` |
| `scorecard` | object | The report card's version, the scorer file, `stress_route` and the starting values of the report card |
| `scoring` | file table | `stress_truth`, `stress_modal` and `service_b_psd` may be absent |
| `baselines` | object | Not included in this release |
| `service_state` | object, stage 2 only | Unit B in service at the fixed point of its level rule: `member` (`f0_hz`, `frequency_change` signed, `damping_factor`, `e_H`, `mid_span_rms_m`) and `article` (`modes`, `frequency_change_p10_p50_p90`, `damping_factor_p10_p50_p90` over the article's other modes, 20 to 2000 Hz). The state the shakers are meant to reproduce |
| `provenance` | object | How the problem was made (below) |

**`provenance`**: `study` (the generator), `study_case` (the generator's case, written shakers x
controls), `study_order`, `service_tier`, `service_truth`, `runner_commit` (the software commit
whose simulator the case was made for), `generator_export` (a label naming the generator's export
the folder was made from; the export itself is not in the release), `generator` (the problem
maker), `export_schema` (the run program's case schema), and `checks`: the frequency response of
the reference model against the generating model (`frf_parity`, `stress_frf_parity`), the
structure object rebuilt from `structure.json` against the generator's
(`runner_structure_equal`, `runner_structure_equal_what`), the targets against the generator's
(`target_banded_equal`, `target_line_equal`), and the equalized step's error against the
continuous model (`equalized_step_residual`).

`scorecard.stress_route`: `drive` at stage 1 (stress and gauges from the played drive through the
structure's own model), `played_modal` at stage 2 (gauges from the played outputs, stress from the
modal monitors' displacement band moments).

## 4. The Scorecard

`schema`: `hertzforge-benchmark-scorecard/0.2.1-draft`, scorer version `0.2.1-draft`. The
report-card block keeps `mimo-metrics-report-card/0.1-provisional`.

- `scorer`: `file`, `version`, `sha256` of the scorer file.
- `problem`: identifier, structure, target, seed, `structure_sha256`, limits, `stage`, `effects`,
  `format`.
- `run`: the case, the law and its parameters, the frequency response mode, the noise seed, the
  software commit, frames per loop.
- `outcome`: completed, or stopped with the check, the loop and the reason; `scored`.
- `scored_on`: `gauges and stress`, or `gauges (the problem has no stress truth)`.
- `report_card` (six lines and the reference measures R1 and R3), `stress`, `ers` (response
  spectra), `card` (the same figures under short names), `drive`, and at stage 2
  `by_drive_route` (the figures with the played drive taken through the small-signal model; not
  the score).
- `structure_state` (stage 2): each mode's frequency change as played, with the 0.1 figures
  (`drop_pct_p10_p50_p90`, `drop_over_target_p50` and the rest, a drop positive) and `signed`
  (`all`, `by_effect`, `modes`; a drop negative, a rise positive), and `service_line`: the state as
  played beside the problem's `service_state` (the member's mode and the softening modes' spread).

The scorecards of the study and of its audit, published as a record of their own (DOI
10.5281/zenodo.24310746), carry three more blocks written by the study's batch
tool: `job` (the case's settings, among them `rig`, the system as shakers x controls), `runner_outcome`
(the run program's own outcome record) and `batch`. In `batch`, the fields that held a local path
hold a plain label instead: `problem` is the problem folder relative to the root of the record that holds
it (`problems/...` in the set, `problems_audit/...` and `problems_audit2/...` in the results), `case`,
`run_dir` and `runner` name the case file, the run folder of kept files and the run program build,
none of which is in the release. `runner_sha256` and `code_commit` identify the run program.

## 5. The Run Result

What the scorer reads from a run: `manifest.json` (grid, outcome), `drive_sdm_played.c16`
(lines, drives, drives; V^2/Hz), `response_psd_played.f8` or `.f4` (lines, outputs) and at stage 2
`response_moments_played.f8` (2, monitors, monitors).

## 6. What Changed From 0.1-draft

| Change | Why |
|---|---|
| A `level` object per mode; `level_law` keeps `decay_cycles` and is the fallback | One structure holds modes that soften and modes that harden |
| A rise term with a cap | Hardening raises a frequency; the cap keeps the highest frequency known so the sample rate can be checked |
| `level_targets.frequency_drop` (positive) becomes `frequency_change` (signed) | A rise cannot be written as a drop |
| `stage` and `effects` in `problem.json` and on the scorecard | A reader of a card sees which effects the structure had |
| Stress truth and service B optional | A problem with no stress truth is scored on its gauges |
| Damping ceiling of 0.9 in `level_state` | The simulator applies it (0.2 files only) |
| `simulate_level`, `discrete_frf`, `max_frequency`, `frequency_change` in the reference model | A structure that changes with level can be run without the simulator |
| `to_runner_structure` writes `level` per mode (case schema `mm-x4-case/2`) | The simulator reads the rule from the case file |

## 7. What Changed in Version 0.2 of the Set

Names and paths only; no number. `RELEASE_NOTES.md` says what changed and gives the checks that
nothing else moved.
