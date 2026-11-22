"""Run a batch of MIMO Shock runs: a pool of single-threaded workers, one slim JSON a run, resumable.

    python shock/batch.py --jobs jobs.json --problems <problems folder> --out <run folder> --workers 8 --log batch.log

jobs.json: a list of {"problem": folder name, "kind": target kind, "law": law name, "noise": 1e-3, "seed": 11}.
Each run writes <out>/<problem>__<kind>__<law>__n<noise>_s<seed>.json: the settings, the run (every pulse's level,
SRS errors per control and point, state reached, drive RMS and peak, law time), the law's report and the scorer's
card; nothing else (no time histories). A run whose file exists is skipped. The pool stops starting runs when the
free space of the output disk is under --min-free-gb (default 50). The log ends with one line
"BATCH END <name>: <done> done, <failed> failed, <skipped> skipped, <time>" whatever happened.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def code_sha():
    h = hashlib.sha256()
    for f in sorted(os.listdir(HERE)):
        if f.endswith('.py'):
            h.update(f.encode()); h.update(open(os.path.join(HERE, f), 'rb').read())
    h.update(open(os.path.join(os.path.dirname(HERE), 'reference_model.py'), 'rb').read())
    return h.hexdigest()


def run_id(j):
    return f"{j['problem']}__{j['kind']}__{j['law']}__n{j.get('noise', 1e-3):g}_s{j.get('seed', 11)}"


def one(args):
    j, problems, out, sha, impl = args
    import numpy as np  # noqa: F401
    import harness as hz
    import laws
    import shock_scorer as scorer
    path = os.path.join(out, run_id(j) + '.json')
    if os.path.exists(path):
        return run_id(j), 'skipped', 0.0
    t0 = time.time()
    try:
        s, st, tg, meta = hz.load(os.path.join(problems, j['problem']), j['kind'])
        law = laws.LAWS[j['law']]() if impl is None else __import__('laws_rust').LAWS[j['law']]()
        r = hz.run(s, st, law, tg, noise=j.get('noise', 1e-3), seed=j.get('seed', 11))
        card = scorer.score_run(r)
        rec = dict(job=j, code_sha256=sha, seconds=round(time.time() - t0, 2), problem_sha256=meta['structure']['sha256'],
                   target_sha256=meta['targets'][j['kind']]['sha256'], stage=meta['stage'], effects=meta['effects'],
                   cls=meta['structure']['cls'], variant=meta['structure']['variant'], run=r, card=card)
        if impl is not None:
            rec['implementation'] = impl
        tmp = path + '.tmp'
        json.dump(rec, open(tmp, 'w'), default=float)
        os.replace(tmp, path)
        return run_id(j), 'done', time.time() - t0
    except Exception:
        return run_id(j), 'failed: ' + traceback.format_exc()[-400:], time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', required=True); ap.add_argument('--problems', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--log', required=True)
    ap.add_argument('--min-free-gb', type=float, default=50.0); ap.add_argument('--name', default='batch')
    ap.add_argument('--rust', action='store_true',
                    help='followed and followed-hold from the Rust law math (laws_rust.py); the library path, its SHA-256 '
                         'and the law math commit are written to the log and every run file')
    ap.add_argument('--sds', default=os.path.expanduser('~/dev/sds-mshock-followed-wt'), help='the law math worktree')
    a = ap.parse_args()
    jobs = json.load(open(a.jobs))
    os.makedirs(a.out, exist_ok=True)
    sha = code_sha()
    log = open(a.log, 'a', buffering=1)
    impl = None
    if a.rust:
        import subprocess
        import laws_rust
        impl = dict(kind='rust', library=laws_rust.lib_info(),
                    sds_commit=subprocess.run(['git', '-C', a.sds, 'rev-parse', 'HEAD'], capture_output=True,
                                              text=True).stdout.strip(),
                    sds_dirty=bool(subprocess.run(['git', '-C', a.sds, 'status', '--porcelain'], capture_output=True,
                                                  text=True).stdout.strip()))
    log.write(f"BATCH START {a.name}: {len(jobs)} jobs, {a.workers} workers, code {sha[:12]}, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if impl is not None:
        log.write(f"IMPLEMENTATION {json.dumps(impl)}\n")
    done = failed = skipped = 0
    stopped = ''
    from multiprocessing import get_context
    try:
        with get_context('spawn').Pool(a.workers, maxtasksperchild=20) as pool:
            pending = []
            it = iter(jobs)
            for res in pool.imap_unordered(one, ((j, a.problems, a.out, sha, impl) for j in it), chunksize=1):
                rid, status, sec = res
                if status == 'done':
                    done += 1
                elif status == 'skipped':
                    skipped += 1
                else:
                    failed += 1
                log.write(f"{time.strftime('%H:%M:%S')} {status.splitlines()[0] if status else ''} {rid} {sec:.1f}s"
                          + ('' if status in ('done', 'skipped') else ' | ' + status.replace(chr(10), ' / ')) + '\n')
                free = shutil.disk_usage(a.out).free / 1e9
                if free < a.min_free_gb:
                    stopped = f' (stopped: {free:.1f} GB free, under {a.min_free_gb:g})'
                    pool.terminate()
                    break
    except Exception as ex:
        stopped = f' (stopped: {type(ex).__name__}: {ex})'
    log.write(f"BATCH END {a.name}: {done} done, {failed} failed, {skipped} skipped{stopped}, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == '__main__':
    main()
