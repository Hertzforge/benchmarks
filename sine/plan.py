"""The MIMO Sine suite's matrix as code: which structures, which laws, which tests. Writes the structure list and the
jobs file the batch reads.

    python sine/plan.py --out <folder> [--problems <problems folder>]    # structures.txt and jobs.json

Blocks (Plan 2026-11-25 sections 2 and 8):
  A  rigs 3x12, 6x12, 12x24, 6x6, 12x12 (shakers x controls; folders frame-12x3, -12x6, -24x12, -6x6, -12x12),
     linear, realistic and strong, the Random study's twenty seeds: 300 structures
  D  rigs 6x12 and 6x6, softening only and hardening only at both tiers, the first ten seeds: 80 structures
  L  the limit block: rigs 6x12 and 6x6, realistic and strong, the first ten seeds, T1 with one limit channel (test L1)
A job is (problem, test, law) for every test the law's manifest accepts and the problem built (T5: block A only, and
only when the problem files say it was built).

Two versions. The Hertzforge version (the runner on the shipped engine, the runs of record) runs the
five shipped laws. The Python version (sine/pyloop/harness.py) runs K3, the followed poles on the M7d fit, admitted
after the screen by Kevin's decision of 2026-11-25 ("Include K3"), on every case the screen allowed (it
takes sweeps: T1, T2, T5 and the limit block's L1, on blocks A, D and L as the problem built them), and `matrix` on
exactly the same cases, so a paired law difference never mixes versions. Each version is its own set of jobs.
"""
import argparse
import json
import os

RANDOM_STUDY_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 33, 12, 13, 14, 15, 34, 17, 35, 19]   # study_seeds.json, in order
RIGS_A = ['12x3', '12x6', '24x12', '6x6', '12x12']        # folder names: controls x drives
RIGS_D = ['12x6', '6x6']
TIERS = [None, 'realistic', 'strong']
LAWS = ['matrix', 'paired', 'ratio', 'steepest_descent', 'secant']
LAW_TESTS = {'matrix': ('T1', 'T2', 'T3', 'T4', 'T5', 'L1'), 'paired': ('T3', 'T4'), 'ratio': ('T3', 'T4'),
             'steepest_descent': ('T4',), 'secant': ('T4',)}
# the Python version: K3 on the sweeps it takes, and matrix on the same cases (never on T3 or T4 here)
PY_LAWS = ['matrix', 'k3']
PY_LAW_TESTS = {'matrix': ('T1', 'T2', 'T5', 'L1'), 'k3': ('T1', 'T2', 'T5', 'L1')}
VERSIONS = {'hertzforge': (LAWS, LAW_TESTS), 'python': (PY_LAWS, PY_LAW_TESTS)}


def sid(rig, tier, seed, effect=None):
    return f'frame-{rig}' + (f'-{tier}' if tier else '') + (f'_{effect}' if effect else '') + f'_s{seed:02d}'


def block_a():
    return [sid(r, t, s) for r in RIGS_A for t in TIERS for s in RANDOM_STUDY_SEEDS]


def block_d():
    return [sid(r, t, s, e) for r in RIGS_D for t in ('realistic', 'strong') for e in ('soft', 'hard') for s in RANDOM_STUDY_SEEDS[:10]]


def limit_structures():
    return [sid(r, t, s) for r in RIGS_D for t in ('realistic', 'strong') for s in RANDOM_STUDY_SEEDS[:10]]


def structures():
    return block_a() + block_d()


def jobs_for(p, problems=None, version='hertzforge'):
    """The jobs of one problem in one version. With `problems`, a test the problem says was not built is left out.
    Returns (jobs, tests left out as not built)."""
    laws, law_tests = VERSIONS[version]
    out, not_built = [], 0
    built = None
    if problems is not None:
        meta = json.load(open(os.path.join(problems, p, 'problem.json')))
        built = {t: v['built'] for t, v in meta['tests'].items()}
    in_a, in_lim = p in set(block_a()), p in set(limit_structures())
    for law in laws:
        for t in law_tests[law]:
            if (t == 'T5' and not in_a) or (t == 'L1' and not in_lim):
                continue
            if built is not None and not built.get(t, False):
                not_built += 1
                continue
            out.append(dict(problem=p, test=t, law=law))
    return out, not_built


def jobs(problems=None, version='hertzforge'):
    """Every (problem, test, law) of the matrix in one version. With `problems`, a test the problem says was not built
    is left out and counted."""
    out, nb = [], 0
    for p in structures():
        j, n = jobs_for(p, problems, version)
        out += j
        nb += n
    return out, nb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--problems', default=None)
    ap.add_argument('--version', default='hertzforge', choices=sorted(VERSIONS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    js, nb = jobs(a.problems, a.version)
    open(os.path.join(a.out, 'structures.txt'), 'w').write(','.join(structures()) + '\n')
    json.dump(js, open(os.path.join(a.out, 'jobs.json'), 'w'), indent=0)
    print(len(structures()), 'structures,', len(js), 'jobs,', nb, 'not built')


if __name__ == '__main__':
    main()
