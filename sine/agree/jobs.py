"""The agreement sample and the timing pilot as job lists for drive.py (the MIMO Sine benchmark's agreement work).

    python3 sine/agree/jobs.py agreement --problems <problems> --base <folder> --runner <msnb_run>
    python3 sine/agree/jobs.py pilot     --problems <problems> --base <folder> --runner <msnb_run>

Each case is the suite's own case file (<problem>/cases/<test>__<law>.json) with the lead's defaults for runs of
record: `protective` "widened", and `reference.phase` "specified" for a law that takes a phase (matrix,
steepest_descent, secant) and "auto" for the amplitude-only laws (paired, ratio). Nothing else changes (the noise
seed is the case's: one per problem and test, the same for every law). Cases go to <base>/cases/<problem>/.

The drive gain (FORMAT_SINE.md section 7): the runner the agreement ran refuses `drives.gain`. Every job here therefore
runs the gain-free equivalent (<base>/cases_gainfree/): no gain, the ceiling in the structure's volts,
min(9 V, gain x 1.99526231 V). Gain and ceiling scale together, and the engine sets its identification from the
ceiling (the probe) and the reference (the measured shot), so the two are the same run wherever gain x 1.995 V is
under 9 V (a gain up to 4.5107); a case over that is listed as changed by the gain. The agreement adds an arm
py_gain: the harness on the case with its gain, to show the equivalence case by case (every case the gain changes,
and every other case but the 1 octave a minute sweeps, T1 and T5, to save their processor time).

agreement: the systems within the engine's limits (frame-6x6, frame-12x6, frame-12x3), linear and realistic, the
  first two structures of each (s00, s01); every test T1 to T5 each law's manifest accepts (sine/plan.py). Four arms,
  each a job list in <base>: hf_ideal (msnb_run --ideal-frf), hf_own (msnb_run), py_exact (the harness, its exact
  small-signal response), py_given (the harness handed hf_own's law_identification.f8, --ident), py_mguard (the
  harness's law on its exact response, its guards on hf_own's measured file, --guard-ident: what the runner does with
  --ideal-frf, so the pair hf_ideal / py_mguard compares the loops on the same law model and the same guards).
pipeline: the agreement redone with the harness's pipeline (its default since 9efb035; the earlier arms ran without
  it): py_mguard_p and py_given_p on the 110 cases the gain does not change (the gain-free case, against hf_own's
  file), py_mguard_pg and py_given_pg on the 22 it changes (the case with its gain, against hf_own_g's file), so the
  sets of record pair them with hf_ideal / hf_own and hf_ideal_g / hf_own_g as the agreement's final sets do. Reads
  <base>/changed_by_gain.json and the case files the agreement set wrote; runs nothing on the runner.
pilot: the Hertzforge version on the systems within the limits, every tier, s00: matrix on every test the problem
  built (T1 to T5, L1), the other laws on T3 and T4; and two block D structures (frame-6x6 and frame-12x6,
  realistic, softening only) on matrix T1. One list, hf_own.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SINE = os.path.dirname(HERE)
sys.path.insert(0, SINE)
import plan  # noqa: E402

PHASE = {'matrix': 'specified', 'steepest_descent': 'specified', 'secant': 'specified', 'paired': 'auto', 'ratio': 'auto'}
IN_LIMITS = ['6x6', '12x6', '12x3']


GAIN_FREE_CAP_V = 9.0


def derive(problems, problem, test, law, base):
    """Returns (the case with its gain, its gain-free equivalent, whether the gain changes the run)."""
    src = os.path.join(problems, problem, 'cases', f'{test}__{law}.json')
    c = json.load(open(src))
    c['protective'] = 'widened'
    c['reference']['phase'] = PHASE[law]
    out = []
    g = c['drives']['gain']
    ceil = g * c['drives']['ceiling_vpk']
    free = json.loads(json.dumps(c))
    free['drives'] = dict(ceiling_vpk=float(f'{min(GAIN_FREE_CAP_V, ceil):.9g}'))
    for sub, cc in (('cases', c), ('cases_gainfree', free)):
        d = os.path.join(base, sub, problem)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f'{test}__{law}.json')
        with open(path, 'w') as fh:
            json.dump(cc, fh, indent=1)
        out.append(path)
    return out[0], out[1], ceil > GAIN_FREE_CAP_V


def agreement_jobs(problems):
    out = []
    for rig in IN_LIMITS:
        for tier in (None, 'realistic'):
            for seed in (0, 1):
                p = plan.sid(rig, tier, seed)
                js, _ = plan.jobs_for(p, problems)
                out += [j for j in js if j['test'] in ('T1', 'T2', 'T3', 'T4', 'T5')]
    return out


def pilot_jobs(problems):
    out = []
    for rig in IN_LIMITS:
        for tier in plan.TIERS:
            p = plan.sid(rig, tier, 0)
            js, _ = plan.jobs_for(p, problems)
            out += [j for j in js if j['law'] == 'matrix' or j['test'] in ('T3', 'T4')]
    for rig in ('6x6', '12x6'):
        out.append(dict(problem=plan.sid(rig, 'realistic', 0, 'soft'), test='T1', law='matrix'))
    return out


def pipeline_jobs(problems, base, harness, python):
    changed = set(json.load(open(os.path.join(base, 'changed_by_gain.json'))))
    arms = {}
    for j in agreement_jobs(problems):
        key = f"{j['problem']}/{j['test']}__{j['law']}"
        g = key in changed
        sub, hf, sfx = ('cases', 'hf_own_g', '_pg') if g else ('cases_gainfree', 'hf_own', '_p')
        case = os.path.join(base, sub, key + '.json')
        own = os.path.join(base, hf, key)
        for arm, flag in (('py_mguard', '--guard-ident'), ('py_given', '--ident')):
            arms.setdefault(arm + sfx, []).append(dict(id=f'{arm}{sfx}:{key}', problem=j['problem'], test=j['test'],
                                                       law=j['law'], cmd=[python, harness, case, flag, own],
                                                       out=os.path.join(base, arm + sfx, key)))
    for arm, lst in arms.items():
        json.dump(lst, open(os.path.join(base, f'jobs_{arm}.json'), 'w'), indent=1)
        print(arm, len(lst), 'jobs')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('set', choices=('agreement', 'pilot', 'pipeline'))
    ap.add_argument('--problems', required=True)
    ap.add_argument('--base', required=True)
    ap.add_argument('--runner', required=True)
    ap.add_argument('--python', default=sys.executable)
    a = ap.parse_args(argv)
    problems, base, runner = (os.path.abspath(x) for x in (a.problems, a.base, a.runner))
    harness = os.path.join(SINE, 'pyloop', 'harness.py')
    os.makedirs(base, exist_ok=True)
    if a.set == 'pipeline':
        return pipeline_jobs(problems, base, harness, a.python)
    js = agreement_jobs(problems) if a.set == 'agreement' else pilot_jobs(problems)
    arms = {}
    changed = []
    for j in js:
        gained, case, ch = derive(problems, j['problem'], j['test'], j['law'], base)
        key = f"{j['problem']}/{j['test']}__{j['law']}"
        if ch:
            changed.append(key)

        def job(arm, cmd):
            arms.setdefault(arm, []).append(dict(id=f'{arm}:{key}', problem=j['problem'], test=j['test'], law=j['law'],
                                                 cmd=cmd, out=os.path.join(base, arm, key)))
        if a.set == 'agreement':
            job('hf_ideal', [runner, case, '--ideal-frf'])
            job('hf_own', [runner, case])
            job('py_exact', [a.python, harness, case])
            job('py_given', [a.python, harness, case, '--ident', os.path.join(base, 'hf_own', key)])
            arms['py_given'][-1]['wait_for'] = os.path.join(base, 'hf_own', key, 'drive.json')
            job('py_mguard', [a.python, harness, case, '--guard-ident', os.path.join(base, 'hf_own', key)])
            arms['py_mguard'][-1]['wait_for'] = os.path.join(base, 'hf_own', key, 'drive.json')
            if ch or j['test'] not in ('T1', 'T5'):     # the 1 octave a minute sweeps only where the gain changes the case
                job('py_gain', [a.python, harness, gained])
        else:
            job('hf_own', [runner, case])
    for arm, lst in arms.items():
        json.dump(lst, open(os.path.join(base, f'jobs_{arm}.json'), 'w'), indent=1)
        print(arm, len(lst), 'jobs')
    json.dump(changed, open(os.path.join(base, 'changed_by_gain.json'), 'w'), indent=1)
    print(len(changed), 'cases the gain changes (gain-free ceiling capped at 9 V)')


if __name__ == '__main__':
    main()
