# Release Notes: Version 0.2

Version 0.2 of the MIMO Random suite of the Hertzforge Open Benchmark Set, published
2026-11-17: the set, DOI 10.5281/zenodo.24310742; the results of the control law study and its audit, DOI
10.5281/zenodo.24310746; the code, https://github.com/Hertzforge/benchmarks at tag `v0.2`. It is built by one script from the
set the study and the audit ran on. The paper that describes the set: The Open Benchmark Set: Public Test Problems, a Fair Run and a Scorecard for Multi-Shaker Control Laws. Hertzforge Journal,
2026. https://doi.org/10.5281/zenodo.24311105 Preparing it for release
changed names and paths only: no number in any problem or scorecard changed, and the checks below
show it.

## What This Release Is

- The set's record: `problems/`, the 388 problems of the set, in the format `FORMAT.md` describes;
  `code/`, the reference model, the scorer, the problem maker, the set's generator and their tests
  (the repository's tree at tag `v0.2`).
- The results' record: the scorecards of the study (3,793) and of the audit's two batches (1,761
  and 950), the 240 held-back problems the audit ran on, and the audit's record.

Every path a file records is relative to the root of the record that holds it, or a plain label for something the
release does not carry (the generator's export, a case file, a run folder, the run program's build).
The words a user reads say "structure" for the thing under test and its model, and "system" for the
article on its shakers with its gauges, as the controller sees it. Every hash a problem file or a
scorecard records of another file is the hash of the released file. Kept as they are: the format and
schema strings (`0.2-draft`, `0.2.1-draft`), the run program's case schema names (`mm-x4-case/1`,
`/2`), and the scorecard field and job key `rig` (the data fields keep `rig`; the words say
"system").

## Changes and Omissions

**The generator's missing check.** The generator now refuses a structure one of whose modes can
reach the simulator's bound of 4,096 Hz (`code/generator/frequency_bound.py`, called when a
structure is built). As the study ran it, the generator did not check, and one held-back structure
(structure 24 at the strong size, 4,118.6 Hz) was refused by the simulator at run time. The test
`code/generator/tests/test_frequency_bound.py` refuses structure 24 at the strong size on all five
of its systems and none of the 388 problems (the highest reaches 4,006 Hz).

**`regenerate.sh --audit` is removed.** As the study had it, the option asked the generator for
twenty seeds from 20 (20 to 39, which takes in the tuning seeds and the study's replacement seeds)
and wrote into the study's own folders. The held-back structures were made by the audit's own
scripts instead, with ten seeds and separate folders; the script's comment says how.

**Left out.** The five problems of structure 24 at the strong size: the simulator refused them, they
have no scorecard, and the generator now refuses them. The baselines file the format allows is not
included.

## The Checks That Nothing Else Moved

**Every problem.** All 628 problem folders (the 388 of the set and the 240 held-back ones in the
results' record), released against internal. Each was loaded through the code the study ran and through
the released code, and every array compared byte for byte: the structure's model, every level-rule
array, the small-signal frequency response on the problem's grid, the highest reachable frequency,
the targets, every scoring array and the scorer's channel indices. 24,317 arrays, 1,124,310,056
values: none differs. Every JSON file (3,140) was compared with the internal one, renamed as the release
renames it: 14,414,163 numbers, none differs; only the recorded hashes and the export label differ. Every binary file (3,768 files, 8,490,184,240 bytes) is identical. Every
hash a released `problem.json` records matches its released file.

**Every scorecard.** All 6,504 scorecards of the study (3,793) and of the audit's two batches (1,761
and 950) were scored again by the released scorer, from the kept files of each run and the released
problem, and compared with the internal scorecard: 2,800,634 numbers, largest difference 0, no card
different. Each released scorecard file was also compared with the internal one, renamed as the
release renames it: 3,007,856 numbers, largest difference 0; only the hashes and the four path fields differ.

**A sample of runs made again.** Twelve cases fixed before any result was read (two structures,
seeds 1 and 3; three systems, 6 shakers by 12 control gauges, 6 by 6 and 3 by 12; two laws,
`pinv_closed` at stage 2 at the strong size and `band_average` at stage 1) were run again from the
released problems, on a copy of the study's run program that differs only in reading the
structure under the key `structure`. All twelve default passes stopped at a default check (the
error grew for five loops) and were run again with the checks widened, as in the study: 24 runs.
Every case file built from the released problem equals the study's (the key renamed); every played
drive, response and modal-moment file is identical byte for byte to the study's; every number of
every scorecard (4,869) equals the study's. Only the software commit recorded in the run differs.

**Tests.** `sh code/run_tests.sh`: 32 passed, 5 skipped (the skipped ones need the simulator's
player or the study's run folders, which are not in the release); with those present, 36 passed
and 1 skipped. The released problem maker, given an export of the study's generator, writes a
problem folder identical to the released one in every file but `structure.json`, where the one
difference is the member's words that the export itself carries, and the hash in `problem.json`
that follows from it.

**Names.** The software repository's name gate, run over every text file of the release, finds
nothing. A plain scan finds no absolute or home path, no machine name, no company name and no
internal note name. The generator's own code (`code/generator/`) keeps its own identifiers and file
names, and `code/make_problem.py` names the generator's and the run program's files as they are.

**The published files against the files checked above.** The checks above were made on a first
build of these files under a working name. The published files are the same build with the
version's name, its DOIs and the repository's address filled in: all 14,070 files of the
first build were compared with their counterparts in the two records. 6,299 are identical. The
other 7,771 differ only in: the version's name and DOI (1,261 files: the status line of every
`problem.json`, the header line of every copy of the reference model and of four code files);
hashes recomputed because a file they name changed (all 6,504 scorecards); the audit's problem
folders, now named relative to the results record (2,711 scorecards); the readme, the release notes
and the format document; and the generator's bound test, which now skips its check of the 388
problems where they are not beside the code, as in the repository (2 files). On the published
files, every problem loads to the same arrays (1,124,310,056 values) and every scorecard scored
again gives the same numbers (2,800,634, largest difference 0). The twelve runs were not made
again: no problem's content changed.
