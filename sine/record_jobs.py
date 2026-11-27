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
  study_defaults   the default check: the systems 12x6 and 6x6, every tier, the study's first two
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
  study_py, study_py_defaults, study_py_repeat_<k>, audit_py, audit_py_defaults, audit_py_repeat_<k>
                   the Python version (sine/plan.py's VERSIONS, run by sine/pyloop/harness.py): K3, admitted after the
                   screen (Kevin, 2026-11-25: "Include K3"), and matrix on exactly the same cases (T1, T2, T5, L1). A K3
                   case is the problem's own matrix case with the law set to k3 and nothing else changed (its noise seed
                   and identification seed are matrix's), as the screen built them; the
                   Python matrix case is the Hertzforge set's matrix case, byte for byte. The default check and the
                   repeat sets as above, except that a Python repeat set is every Python job on the two structures (K3
                   takes no T3 or T4). The Python harness builds its law on the exact small-signal response, so its
                   repeats vary the noise seed only (it reads no identification seed).
A further candidate's cases: add it to plan.py's laws of its version and rerun this into a new folder.
"""
import argparse
import json
import os
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan  # noqa: E402

PHASE = {'matrix': 'specified', 'steepest_descent': 'specified', 'secant': 'specified', 'paired': 'auto', 'ratio': 'auto',
         'k3': 'specified'}
# a candidate law has no case file in the problem folders: its case is the problem's matrix case with the law swapped
CANDIDATE_FROM = {'k3': 'matrix'}
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


def source_case(problems, j):
    """The problem's own case file a job's case is made from (a candidate's: the case of the law it is built from)."""
    law = CANDIDATE_FROM.get(j['law'], j['law'])
    return os.path.join(problems, j['problem'], 'cases', f"{j['test']}__{law}.json")


def load_case(problems, j):
    case = json.load(open(source_case(problems, j)))
    if j['law'] in CANDIDATE_FROM:
        frm = CANDIDATE_FROM[j['law']]
        case['law'] = dict(id=j['law'], params={})
        assert case['case_id'].endswith('__' + frm), case['case_id']
        case['case_id'] = case['case_id'][:-len(frm)] + j['law']
    return case


def matrix_jobs(problems, version='hertzforge'):
    laws, law_tests = plan.VERSIONS[version]
    out = []
    for p in sorted(os.listdir(problems)):
        if not os.path.exists(os.path.join(problems, p, 'problem.json')):
            continue
        meta = json.load(open(os.path.join(problems, p, 'problem.json')))
        built = {t: v['built'] for t, v in meta['tests'].items()}
        for law in laws:
            for t in law_tests[law]:
                j = dict(problem=p, test=t, law=law)
                if built.get(t, False) and os.path.exists(source_case(problems, j)):
                    out.append(j)
    return out


def default_jobs(all_jobs, seeds):
    keep = {plan.sid(r, t, s) for r in DEFAULT_SYSTEMS for t in plan.TIERS for s in seeds}
    return [j for j in all_jobs if j['problem'] in keep]


def repeat_jobs(all_jobs, s0, s1, version='hertzforge'):
    a = plan.sid('6x6', 'realistic', s0)
    b = plan.sid('12x6', 'strong', s1)
    pick = [j for j in all_jobs if j['problem'] == a]
    if version == 'python':
        return pick + [j for j in all_jobs if j['problem'] == b]
    pick += [j for j in all_jobs if j['problem'] == b and (j['test'] in ('T1', 'T2', 'T3', 'T4') if j['law'] == 'matrix' else j['test'] == 'T4')]
    return pick


def write_set(base, name, problems, jobs, change):
    d = os.path.join(base, name)
    for j in jobs:
        case = change(load_case(problems, j))
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
    ap.add_argument('--runner', default='<msnb_run>', help='the Hertzforge version\'s runner, for the batch commands')
    ap.add_argument('--harness', default='<sine/pyloop/harness.py>', help='the Python version\'s harness, for the '
                    'batch commands')
    a = ap.parse_args(argv)
    problems, heldout, base = (os.path.abspath(x) for x in (a.problems, a.heldout, a.base))
    if os.path.exists(os.path.join(base, 'sets.json')):
        sys.exit(f'{base} holds job lists already; write to a new folder')
    seeds_h = json.load(open(os.path.join(os.path.dirname(heldout), 'heldout_seeds.json')))['seeds']
    sets = []
    for version, suffix in (('hertzforge', ''), ('python', '_py')):
        study, audit = matrix_jobs(problems, version), matrix_jobs(heldout, version)
        vs = [write_set(base, f'study{suffix}', problems, study, record),
              write_set(base, f'study{suffix}_defaults', problems, default_jobs(study, plan.RANDOM_STUDY_SEEDS[:2]), defaults)]
        vs += [write_set(base, f'study{suffix}_repeat_{k}', problems,
                         repeat_jobs(study, *plan.RANDOM_STUDY_SEEDS[:2], version=version), repeat(k))
               for k in range(1, REPEATS + 1)]
        vs += [write_set(base, f'audit{suffix}', heldout, audit, record),
               write_set(base, f'audit{suffix}_defaults', heldout, default_jobs(audit, seeds_h[:2]), defaults)]
        vs += [write_set(base, f'audit{suffix}_repeat_{k}', heldout, repeat_jobs(audit, *seeds_h[:2], version=version),
                         repeat(k)) for k in range(1, REPEATS + 1)]
        for s in vs:
            s['version'] = version
        sets += vs
    for s in sets:
        runner = a.runner if s['version'] == 'hertzforge' else a.harness
        s['batch_command'] = ['python', 'sine/batch.py', '--problems', s['problems'], '--out', a.runs, '--name', s['name'],
                              '--runner', runner, '--cases', s['cases'], '--jobs', s['jobs_file'],
                              '--workers', '16', '--min-free-gb', '50']
        print(f"{s['name']}: {s['jobs']} jobs on {s['structures']} structures")
    json.dump(dict(heldout_seeds=seeds_h, sets=sets), open(os.path.join(base, 'sets.json'), 'w'), indent=1)
    print('TOTAL:', sum(s['jobs'] for s in sets), 'jobs in', len(sets), 'sets')


if __name__ == '__main__':
    sys.exit(main())
