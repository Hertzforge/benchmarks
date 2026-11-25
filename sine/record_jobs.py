"""The MIMO Sine study's and audit's job lists for sine/batch.py, with each set's case files. Starts nothing.

    python sine/record_jobs.py --problems <study problems> --heldout <held-back problems> --base <jobs folder>

Each set is a folder <base>/<set>/ holding cases/<problem>/<test>__<law>.json (the problem's own case file with the set's
settings) and jobs.json (a list of {problem, test, law}); <base>/sets.json lists every set, its count and the
batch.py command that runs it (`--cases <base>/<set>/cases --jobs <base>/<set>/jobs.json`). The sets:

  study            every job of the matrix (sine/plan.py: the five shipped laws on every test each one's manifest
                   accepts and the problem built), the runs of record's settings: protective "widened"; the reference's
                   phase "specified" for the laws that take one (matrix, steepest_descent, secant) and "auto" for the
                   amplitude-only laws (paired, ratio); the case's drive gain and its ceiling 6 dB over the ideal drive
                   (Plan 2026-11-25 section 10). Everything else is the case's own (its noise seed: one per problem and
                   test, the same for every law).
  study_defaults   the default check (study guide item 9): the systems 12x6 and 6x6, every tier, the study's first two
                   structures (seeds 0 and 1), every job of each, at the engine's protective defaults ("default") and its
                   default 9 V ceiling (the case's ceiling left out, so the runner's default holds), the gain kept.
  study_repeat_<k> the repeat set (item 11), k = 1, 2, 3: twenty cases, the runs of record's settings, each with a
                   noise seed and an identification seed of its own for k (crc32 of "<problem>/<test>/repeat<k>" and
                   ".../identification/repeat<k>"), so the three measure the run-to-run spread the noise and the
                   identification make. The twenty: every job of frame-6x6-realistic_s00 (12) and matrix T1 to T4 and
                   every law's T4 on frame-12x6-strong_s01 (8).
  audit, audit_defaults, audit_repeat_<k>
                   the same on the held-back structures (sine/heldout/gen_heldout.py): every job of the matrix on the ten
                   held-back seeds; the default check on the first two held-back seeds; the repeat set on them.
A candidate law's cases are added after the screen and its Rust build: rerun this with the candidate in plan.py's laws.
"""
import argparse
import json
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan  # noqa: E402

PHASE = {'matrix': 'specified', 'steepest_descent': 'specified', 'secant': 'specified', 'paired': 'auto', 'ratio': 'auto'}
DEFAULT_SYSTEMS = ('12x6', '6x6')
REPEATS = 3


def crc(s):
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def record(case):
    case['protective'] = 'widened'
    case['reference']['phase'] = PHASE[case['law']['id']]
    return case


def defaults(case):
    case['protective'] = 'default'
    case['reference']['phase'] = PHASE[case['law']['id']]
    case['drives'] = {k: v for k, v in case['drives'].items() if k != 'ceiling_vpk'}
    return case


def repeat(k):
    def f(case):
        record(case)
        p, t, _ = case['case_id'].split('__')
        case['noise'] = dict(case['noise'], seed=crc(f'{p}/{t}/repeat{k}'))
        case['identification_seed'] = crc(f'{p}/{t}/identification/repeat{k}')
        return case
    return f


def matrix_jobs(problems):
    out = []
    for p in sorted(os.listdir(problems)):
        if not os.path.exists(os.path.join(problems, p, 'problem.json')):
            continue
        meta = json.load(open(os.path.join(problems, p, 'problem.json')))
        built = {t: v['built'] for t, v in meta['tests'].items()}
        for law in plan.LAWS:
            for t in plan.LAW_TESTS[law]:
                if built.get(t, False) and os.path.exists(os.path.join(problems, p, 'cases', f'{t}__{law}.json')):
                    out.append(dict(problem=p, test=t, law=law))
    return out


def default_jobs(all_jobs, seeds):
    keep = {plan.sid(r, t, s) for r in DEFAULT_SYSTEMS for t in plan.TIERS for s in seeds}
    return [j for j in all_jobs if j['problem'] in keep]


def repeat_jobs(all_jobs, s0, s1):
    a = plan.sid('6x6', 'realistic', s0)
    b = plan.sid('12x6', 'strong', s1)
    pick = [j for j in all_jobs if j['problem'] == a]
    pick += [j for j in all_jobs if j['problem'] == b and (j['test'] in ('T1', 'T2', 'T3', 'T4') if j['law'] == 'matrix' else j['test'] == 'T4')]
    return pick


def write_set(base, name, problems, jobs, change):
    d = os.path.join(base, name)
    for j in jobs:
        src = os.path.join(problems, j['problem'], 'cases', f"{j['test']}__{j['law']}.json")
        case = change(json.load(open(src)))
        out = os.path.join(d, 'cases', j['problem'])
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"{j['test']}__{j['law']}.json"), 'w') as fh:
            json.dump(case, fh, indent=1)
    with open(os.path.join(d, 'jobs.json'), 'w') as fh:
        json.dump(jobs, fh, indent=0)
    return dict(name=name, jobs=len(jobs), structures=len({j['problem'] for j in jobs}), problems=problems,
                cases=os.path.join(d, 'cases'), jobs_file=os.path.join(d, 'jobs.json'))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', required=True)
    ap.add_argument('--heldout', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--runs', default='<runs folder>', help='the --out to write into each batch command')
    a = ap.parse_args(argv)
    problems, heldout, base = (os.path.abspath(x) for x in (a.problems, a.heldout, a.base))
    if os.path.exists(os.path.join(base, 'sets.json')):
        sys.exit(f'{base} holds job lists already; write to a new folder')
    seeds_h = json.load(open(os.path.join(os.path.dirname(heldout), 'heldout_seeds.json')))['seeds']
    study, audit = matrix_jobs(problems), matrix_jobs(heldout)
    sets = [write_set(base, 'study', problems, study, record),
            write_set(base, 'study_defaults', problems, default_jobs(study, plan.RANDOM_STUDY_SEEDS[:2]), defaults)]
    sets += [write_set(base, f'study_repeat_{k}', problems, repeat_jobs(study, *plan.RANDOM_STUDY_SEEDS[:2]), repeat(k))
             for k in range(1, REPEATS + 1)]
    sets += [write_set(base, 'audit', heldout, audit, record),
             write_set(base, 'audit_defaults', heldout, default_jobs(audit, seeds_h[:2]), defaults)]
    sets += [write_set(base, f'audit_repeat_{k}', heldout, repeat_jobs(audit, *seeds_h[:2]), repeat(k))
             for k in range(1, REPEATS + 1)]
    for s in sets:
        s['batch_command'] = ['python', 'sine/batch.py', '--problems', s['problems'], '--out', a.runs, '--name', s['name'],
                              '--runner', '<msnb_run at 382d1990>', '--cases', s['cases'], '--jobs', s['jobs_file'],
                              '--min-free-gb', '50']
        print(f"{s['name']}: {s['jobs']} jobs on {s['structures']} structures")
    json.dump(dict(heldout_seeds=seeds_h, sets=sets), open(os.path.join(base, 'sets.json'), 'w'), indent=1)
    print('TOTAL:', sum(s['jobs'] for s in sets), 'jobs in', len(sets), 'sets')


if __name__ == '__main__':
    sys.exit(main())
