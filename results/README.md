# Reporting a Result

A result on the Open Benchmark Set is a control law's scorecards on a released version of a suite,
with what is needed to check them. Accepted results are listed on hertzforge.com/benchmarks/results/,
in the same table as the baselines, with the law, its authors and a link to the article or the code.
Anyone may report one. Hertzforge is not needed.

The rules are those of the paper that defines the set, Section 10: The Open Benchmark Set: Public Test
Problems, a Fair Run and a Scorecard for Multi-Shaker Control Laws, Hertzforge Journal 2026-008,
https://doi.org/10.5281/zenodo.24311105.

## What a Result Is

One control law, at one setting, on one target, with the frequency response either measured once and
kept or measured again during the run. The setting may differ between the two classes of system (more
control gauges than shakers; as many shakers as control gauges), stated in `result.json`. Another
setting of the same law is another result.

Run as many problems of the version as you can. A result on stage 1 alone is listed as such; a result
on both stages shows what the law does when the article changes with level, which is the harder test.

## Step by Step

1. **Run the law.** On problems of a released version, under the limits of a fair run that each
   problem's `limits.json` sets. The law sees the system only by identification, never the structure
   file, the held-out gauges or the stress. Keep each run's folder: `manifest.json`,
   `drive_sdm_played.c16`, `response_psd_played.f8` and, at stage 2, `response_moments_played.f8`
   (`FORMAT.md`, Section 5). Hertzforge's research runner writes these folders; your own code can too.

2. **Score every run** with `scorer.py` from this repository at the version's tag (the same file as
   `code/scorer.py` in the set's record). A scorecard from any other scorer is refused.

   ```
   python scorer.py score PROBLEM_FOLDER RUN_FOLDER --target line --out scorecards/RUN_NAME.json
   ```

3. **Deposit the run records.** Put the run folders in one zip archive, one folder per run, each named
   as its scorecard without `.json`. Deposit it where anyone can download it for good (Zenodo, which
   gives it a DOI, is the usual place) and note its SHA-256.

4. **Make the result folder** in a fork of this repository:

   ```
   results/<suite>-v<version>/<result id>/
     result.json        from TEMPLATE/result.json, filled in
     scorecards/        every scorecard, as the scorer wrote it
     summary.csv        written by results.py, step 5
   ```

   For the MIMO Random suite, version 0.2, the folder is `results/mimo-random-v0.2/<result id>/`.
   The result id is short, lower case, with hyphens, and is the folder's name.

5. **Summarize and check it:**

   ```
   python results.py summarize results/mimo-random-v0.2/my-law
   python results.py check results/mimo-random-v0.2/my-law --set SET_FOLDER --runs RUNS_FOLDER
   ```

   `SET_FOLDER` holds the set's `problems/`; `RUNS_FOLDER` is the unpacked archive of step 3. The
   check scores every run again and requires every number to match.

6. **Open a pull request.** CI checks the result without data. A maintainer then downloads the run
   records, checks the archive's SHA-256, and runs the check of step 5. If every number matches, the
   result is merged and appears on the site at its next build.

## result.json

| Field | What it says |
|---|---|
| `schema` | `hertzforge-benchmark-result/0.1` |
| `id` | The result id, the same as the folder's name |
| `suite`, `set_version` | `MIMO Random`, `0.2` |
| `law.name` | The law's name as it should be listed |
| `law.code_name` | The law's id as the scorecards record it (`run.law`) |
| `law.parameters` | The settings the law was given: `{"all": {...}}`, or `{"more_controls": {...}, "equal": {...}}`. Each scorecard's `run.law_parameters` must equal them |
| `law.frequency_response` | `measured once and kept` or `measured again` |
| `target` | `line` (the full spectral matrix) or `banded` (the banded specification) |
| `authors` | Each with a `name`, and an `orcid` where there is one |
| `link` | The article or the code that describes the law, `https://` |
| `development` | How the law was developed against the set: which problems and seeds it was developed or tuned on, and which it had never seen |
| `run_as_published` | `true`. A problem changed from the published one is a different experiment and is not listed |
| `run_records` | `url` and `sha256` of the archive of step 3 |
| `software` | What ran the law: your own code, or Hertzforge at a named commit and the runner |

## How a Result Is Summarized

`results.py summarize` writes the table the site shows, by the rules the baselines were summarized
by (Hertzforge Journal 2026-009, Section 2), so a reported result and a baseline are read the same
way:

- the result of a case is its completed, scored run; where a case was run twice, once with the
  controller's protections at their defaults and once widened, the widened run where it completed;
- the two classes of system are never pooled; a structure's value in a class is the mean over that
  class's systems;
- the median and the quartiles are over structures;
- the diagnostic problems with one effect are checked but not summarized.

Built from the study's own scorecards, `summarize` gives the published table of the baselines digit
for digit; `tests/test_results.py` holds it to that.

## Licenses

A result folder is published under CC BY 4.0, like the set's data. Say otherwise in the pull request
if the result needs another open license.
