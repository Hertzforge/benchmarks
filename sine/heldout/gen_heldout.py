"""The MIMO Sine audit's held-back structures: ten seeds no study has read, every cell of the suite's matrix on them.

    export PYTHONDONTWRITEBYTECODE=1
    python sine/heldout/gen_heldout.py seeds   --root <heldout folder> [--jobs 3]   # place each seed's member
    python sine/heldout/gen_heldout.py export  --root <heldout folder> [--jobs 3]   # the Random generator's exports
    python sine/heldout/gen_heldout.py stage   --root <heldout folder> [--jobs 3]   # Random format 0.2 folders + checks
    python sine/heldout/gen_heldout.py sine    --root <heldout folder> [--workers 3] # the MIMO Sine problems and cases
    python sine/heldout/gen_heldout.py plan    --root <heldout folder>              # structures.txt and jobs.json

The rule (fixed before the study ran): seeds 90 to 99; a seed whose member the Random generator cannot place
(generator/obs_gen2.py choose_member gives no candidate) is replaced in order from 100 up, and the dropped seed is
reported and not counted. A structure that a generation check refuses (the export's frequency bound, the stager, the
sine maker's checks) is reported and not counted; it is not replaced.

Every step is the set's own code, unchanged: generator/obs_gen2.py (export, one process per structure),
generator/stage2.py (make_problem.py per export and the stager's checks), sine/make_sine_problem.py make_group (the
same code as the study's problems). This file only chooses the seeds and lists the structures: on the held-back seeds
the matrix is the study's (block A: every rig and tier; block D: 6x12 and 6x6, softening only and hardening only, both
tiers; the limit block: 6x12 and 6x6, realistic and strong; T5 on block A), on all ten seeds.

Nothing here runs a law or reads a reading. The only outputs are generation records.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SINE = os.path.dirname(HERE)
REPO = os.path.dirname(SINE)
GEN = os.path.join(REPO, 'generator')

FIRST, COUNT, REPLACE_FROM = 90, 10, 100
SHAKERS_CONTROLS_A = ((3, 12), (6, 12), (12, 24), (6, 6), (12, 12))   # obs_gen2.RIGS (shakers, controls)
SHAKERS_CONTROLS_D = ((6, 12), (6, 6))                                # obs_gen2.DIAG_RIGS
VARIANTS = ('none', 'realistic', 'strong')
DIAG = ('realistic_soft', 'realistic_hard', 'strong_soft', 'strong_hard')


def stamp():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def exported(root, name):
    """Whether the generator has exported structure `name`: its meta file (in the generator's own folder for structures
    under exports/) and its case file both written."""
    return (bool(glob.glob(os.path.join(root, 'exports', '*', name, 'meta_v2.json'))) and
            os.path.exists(os.path.join(root, 'exports', 'cases_v2', f'{name}_pinv_closed_default_line_hold_divopen.json')))


def env_for(root, threads=None):
    e = dict(os.environ, OBS_GEN_BULK=os.path.join(root, 'exports'), PYTHONDONTWRITEBYTECODE='1')
    if threads:
        for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            e[v] = str(threads)
    return e


def gen(root, *args):
    return subprocess.run([sys.executable, os.path.join(GEN, 'obs_gen2.py'), *args], capture_output=True, text=True,
                          env=env_for(root))


def log(root, line):
    with open(os.path.join(root, 'heldout_generation.log'), 'a') as fh:
        fh.write(f'{stamp()} {line}\n')
    print(line, flush=True)


def seeds_path(root):
    return os.path.join(root, 'heldout_seeds.json')


def load_seeds(root):
    return json.load(open(seeds_path(root)))['seeds']


def choose(root, seed):
    """The generator's own member choice for one seed (cached by the generator in exports/choices_v5/s<seed>.json)."""
    p = gen(root, 'choose', '--seed', str(seed))
    if p.returncode != 0:
        raise RuntimeError(f'seed {seed}: choose failed: {(p.stdout + p.stderr)[-600:]}')
    return json.load(open(os.path.join(root, 'exports', 'choices_v5', f's{seed}.json')))


def cmd_seeds(a):
    """Seeds 90 to 99 in order, each one whose member cannot be placed replaced by the next from 100 up."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(os.path.join(a.root, 'exports'), exist_ok=True)
    log(a.root, f'SEEDS START: {FIRST} to {FIRST + COUNT - 1}, replacements from {REPLACE_FROM}, {a.jobs} at once')
    first = list(range(FIRST, FIRST + COUNT))
    with ThreadPoolExecutor(a.jobs) as ex:
        res = dict(zip(first, ex.map(lambda s: choose(a.root, s), first)))
    out, failed, rep = [], [], REPLACE_FROM
    for sd in first:
        r = res[sd]
        while r['chosen'] is None:
            failed.append(sd)
            log(a.root, f'seed {sd}: no place and tuned frequency keeps the member mode; dropped, not counted, replaced by {rep}')
            sd, rep = rep, rep + 1
            r = choose(a.root, sd)
        ch = r['chosen']
        log(a.root, f"seed {sd}: member placed (place {ch['slot']}, {ch['f_tuned']:.0f} Hz, least share {ch['min_e_H']:.3f}, "
                    f"level flag {r['level_flag']}, widened {r['widened']})")
        out.append(sd)
    res = dict(rule=f'seeds {FIRST} to {FIRST + COUNT - 1}; a seed whose member cannot be placed is replaced in order from '
                    f'{REPLACE_FROM} up; the dropped seed is reported and not counted',
               seeds=out, dropped=failed, made=stamp())
    json.dump(res, open(seeds_path(a.root), 'w'), indent=1)
    log(a.root, f'SEEDS END: {len(out)} seeds {out}, dropped {failed}')


def export_jobs(seeds):
    js = []
    for s in seeds:
        js += [(s, ns, nc, v) for ns, nc in SHAKERS_CONTROLS_A for v in VARIANTS]
        js += [(s, ns, nc, v) for ns, nc in SHAKERS_CONTROLS_D for v in DIAG]
    return js


def cmd_export(a):
    from concurrent.futures import ThreadPoolExecutor
    seeds = load_seeds(a.root)
    scales = {}
    for sd in seeds:
        p = gen(a.root, 'scale', '--seed', str(sd))
        if p.returncode != 0:
            raise SystemExit(f'seed {sd}: scale failed: {(p.stdout + p.stderr)[-600:]}')
        scales[sd] = float(p.stdout.strip().splitlines()[-1])
    js = export_jobs(seeds)
    # the power-weighted table is made once, here, so parallel exports do not race to write it
    p = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, sys.argv[1]); import obs_gen2 as g; g.w_table()', GEN],
                       capture_output=True, text=True, env=env_for(a.root))
    if p.returncode != 0:
        raise SystemExit(f'w_table failed: {p.stderr[-600:]}')
    def done(job):
        s, ns, nc, v = job
        return exported(a.root, f's{s}_{ns}x{nc}_{v}')
    already = [j for j in js if a.resume and done(j)]
    if a.resume:
        log(a.root, f'EXPORT RESUME: {len(already)} of {len(js)} already exported (meta and case written), {len(js) - len(already)} to export')
    all_js, js = js, [j for j in js if j not in already]
    log(a.root, f'EXPORT START: {len(js)} structures of seeds {seeds}, {a.jobs} at once')

    def one(job):
        s, ns, nc, v = job
        p = gen(a.root, 'export', '--seed', str(s), '--ns', str(ns), '--nc', str(nc), '--variant', v, '--scale', repr(scales[s]))
        return job, p.returncode, (p.stdout + p.stderr)[-600:]

    ok = bad = 0
    with ThreadPoolExecutor(a.jobs) as ex:
        for job, rc, tail in ex.map(one, js):
            s, ns, nc, v = job
            if rc == 0:
                ok += 1; log(a.root, f'exported s{s}_{ns}x{nc}_{v}')
            else:
                bad += 1; log(a.root, f'EXPORT REFUSED s{s}_{ns}x{nc}_{v}: ' + ' '.join(tail.split())[-400:])
    if a.resume:
        ok += len(already)
    log(a.root, f'EXPORT END: {ok} exported, {bad} refused, of {len(all_js)}')


def cmd_stage(a):
    seeds = load_seeds(a.root)
    out = os.path.join(a.root, 'random_problems')
    os.makedirs(out, exist_ok=True)
    log(a.root, f'STAGE START: seeds {seeds} into {out}, {a.jobs} at once')
    with open(os.path.join(a.root, 'stage2.out'), 'w') as fh:
        p = subprocess.run([sys.executable, os.path.join(GEN, 'stage2.py'), '--seeds', ','.join(map(str, seeds)), '--out', out,
                            '--jobs', str(a.jobs)], stdout=fh, stderr=subprocess.STDOUT, env=env_for(a.root, threads=1))
    summ = json.load(open(os.path.join(out, 'stage2_summary.json')))
    rows = summ['rows']
    failed = [r for r in rows if 'failed' in r]
    log(a.root, f'STAGE END: exit {p.returncode}, {len(rows) - len(failed)} staged, {len(failed)} failed, of {len(rows)}')


def structures(seeds):
    sys.path.insert(0, SINE)
    import plan
    a_ = [plan.sid(r, t, s) for r in plan.RIGS_A for t in plan.TIERS for s in seeds]
    d_ = [plan.sid(r, t, s, e) for r in plan.RIGS_D for t in ('realistic', 'strong') for e in ('soft', 'hard') for s in seeds]
    lim = [plan.sid(r, t, s) for r in plan.RIGS_D for t in ('realistic', 'strong') for s in seeds]
    return a_, d_, lim


def cmd_sine(a):
    """The study's make_group on the held-back structures, with the held-back limit block and T5 on block A."""
    sys.path.insert(0, SINE)
    import make_sine_problem as msp
    seeds = load_seeds(a.root)
    a_, d_, lim = structures(seeds)
    sids = a_ + d_
    random_root = os.path.join(a.root, 'random_problems')
    out_root = os.path.join(a.root, 'problems')
    os.makedirs(out_root, exist_ok=True)
    present = set(os.listdir(random_root))
    missing = [s for s in sids if s not in present]
    groups = {}
    for s in sids:
        if s in missing:
            continue
        base, sd = s.rsplit('_s', 1)
        groups.setdefault((base.split('-')[1].split('_')[0], int(sd)), []).append(s)
    logf = open(os.path.join(a.root, 'problems.make.log'), 'a', buffering=1)
    logf.write(f'GENERATION START: {len(sids)} structures in {len(groups)} groups, {a.workers} workers, down sweep yes, {stamp()}\n')
    for s in missing:
        logf.write(f"{time.strftime('%H:%M:%S')} REFUSED {s}: its Random folder was not staged\n")
    jobs = [(random_root, out_root, g, set(lim), set(a_), None, True) for _, g in sorted(groups.items())]
    made, refused = 0, len(missing)
    from multiprocessing import get_context
    with get_context('spawn').Pool(a.workers) as pool:
        for lines in pool.imap_unordered(msp.make_group, jobs):
            for sid, status, what in lines:
                made += status == 'made'; refused += status != 'made'
                logf.write(f"{time.strftime('%H:%M:%S')} {status} {sid}: {what}\n")
    end = f'GENERATION END: {made} made, {refused} refused, of {len(sids)}, {stamp()}'
    logf.write(end + '\n')
    log(a.root, end)


def cmd_plan(a):
    """The held-back structure list and the audit's job list in plan.py's form (problem, test, law)."""
    sys.path.insert(0, SINE)
    import plan
    seeds = load_seeds(a.root)
    a_, d_, lim = structures(seeds)
    probs = os.path.join(a.root, 'problems')
    in_a, in_lim = set(a_), set(lim)
    js, nb, absent = [], 0, []
    for p in a_ + d_:
        if not os.path.exists(os.path.join(probs, p, 'problem.json')):
            absent.append(p); continue
        built = {t: v['built'] for t, v in json.load(open(os.path.join(probs, p, 'problem.json')))['tests'].items()}
        for law in plan.LAWS:
            for t in plan.LAW_TESTS[law]:
                if (t == 'T5' and p not in in_a) or (t == 'L1' and p not in in_lim):
                    continue
                if not built.get(t, False):
                    nb += 1; continue
                js.append(dict(problem=p, test=t, law=law))
    open(os.path.join(a.root, 'structures.txt'), 'w').write(','.join(a_ + d_) + '\n')
    json.dump(js, open(os.path.join(a.root, 'jobs.json'), 'w'), indent=0)
    print(len(a_ + d_), 'structures,', len(absent), 'not made,', len(js), 'jobs,', nb, 'not built')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('seeds', 'export', 'stage', 'sine', 'plan'))
    ap.add_argument('--root', required=True)
    ap.add_argument('--jobs', type=int, default=3)
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--resume', action='store_true', help='export: skip the structures already exported')
    a = ap.parse_args()
    a.root = os.path.abspath(a.root)
    os.makedirs(a.root, exist_ok=True)
    dict(seeds=cmd_seeds, export=cmd_export, stage=cmd_stage, sine=cmd_sine, plan=cmd_plan)[a.cmd](a)


if __name__ == '__main__':
    main()
