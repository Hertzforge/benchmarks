# Hertzforge Open Benchmark Set: Code

The code of the Hertzforge Open Benchmark Set, MIMO Random suite, version 0.2: the
reference model, the scorer, the problem maker, their tests, and the generator that made the
structures. The set is a group of simulated test problems for multi-shaker random vibration
control: each problem is one structure on its shakers, a target taken from the structure in
service, the limits a run must keep to, and a fixed way of scoring the result. Simulation only.

The problems themselves are not in this repository. They are published on Zenodo:

- the set, version 0.2: https://doi.org/10.5281/zenodo.24310742 (all versions: https://doi.org/10.5281/zenodo.24310741);
- the results of the control law study run on the set and of its independent audit:
  https://doi.org/10.5281/zenodo.24310746.

This repository at tag `v0.2` is identical to the `code/` folder of the set's record.

## What Is Here

| File or folder | What |
|---|---|
| `reference_model.py` | The structure's model: frequency response, stress, the level rule, a time stepper. NumPy only. A copy is in every problem folder |
| `scorer.py` | Scores one run on one problem and writes a scorecard. NumPy only |
| `make_problem.py` | Makes a problem folder from an export of the generator |
| `generator/` | The generator of the frame structures (`obs_gen2.py`), the stager (`stage2.py`), `regenerate.sh`, and the check that refuses a structure the simulator would refuse (`frequency_bound.py`). Needs NumPy, SciPy and cvxpy |
| `tests/`, `generator/tests/` | The tests; `run_tests.sh` runs them |
| `FORMAT.md` | Every file and field of a problem folder and of a scorecard |
| `RELEASE_NOTES.md` | What changed in version 0.2, and the checks that nothing but names and paths changed |

## Getting the Problems

Download the files of the set's record from https://doi.org/10.5281/zenodo.24310742, then in one folder:

```
for f in problems-frame-*.zip code-v0.2.zip; do unzip -q "$f"; done
shasum -a 256 -c MANIFEST.sha256
```

This gives `problems/` (388 folders) and `code/` (this repository).

## Running the Scorer on a Run

```
python code/scorer.py score problems/frame-12x6-realistic_s00 RUN_FOLDER --target line --out scorecard.json
```

`RUN_FOLDER` holds what a run records: `manifest.json`, `drive_sdm_played.c16`,
`response_psd_played.f8` (or `.f4`) and, at stage 2, `response_moments_played.f8` (`FORMAT.md`,
section 5).

## Making the Study's Runs

The study's runs were made with the research runner of the Hertzforge software
(https://github.com/Hertzforge/hertzforge), an example program of its control crate that plays a
control law against the software's simulator. It takes a case file built from a problem folder
(`reference_model.to_runner_structure` gives the structure object the case file holds) and writes
the run folder the scorer reads. The version of the runner that reads the problems of this set
arrives with Hertzforge pull request 324.

## Tests

```
sh run_tests.sh
```

The tests that need no data run as they are, and run on every push and pull request. Tests that
need the set's problems (`OBS_SET`, the folder holding `problems/`), the simulator's player
(`OBS_PLAY`) or run folders (`OBS_DATA`) are skipped when those are not given.

## Licenses

The code is under the MIT license (`LICENSE`): Copyright 2026 Kevin Cross. The documents and the
data set are under CC BY 4.0 (`LICENSE-DATA.md`).

## How to Cite

Cite the set: Cross, Kevin. Hertzforge Open Benchmark Set: MIMO Random Suite, Version 0.2.
Zenodo, 2026-11-17. https://doi.org/10.5281/zenodo.24310742

`CITATION.cff` gives the same in a form citation tools read. The paper that describes the set:
The Open Benchmark Set: Public Test Problems, a Fair Run and a Scorecard for Multi-Shaker Control Laws. Hertzforge Journal, 2026. https://doi.org/10.5281/zenodo.24311105
