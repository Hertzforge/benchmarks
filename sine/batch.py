"""Run the MIMO Sine matrix: every job (problem, test, law) through the runner, then the scorer.

    python sine/batch.py --problems <problems folder> --out <runs folder> --name <batch> --runner <msnb_run> \
        [--workers 6] [--only frame-6x6_s00,...] [--tests T1,T4] [--laws matrix] [--min-free-gb 50] [--job-timeout 7200]

The jobs are sine/plan.py's (the laws' manifests against the tests each problem built). A job's run folder is
<out>/<batch>/<problem>/<test>__<law>/: the runner writes run.json (and timing.json) there with
`<runner> <case> --root <problems> --out <run folder>`; the batch adds batch.json (the suite's code hash, the runner's
SHA-256, the case's SHA-256, the runner's exit status, times) and the scorer's card.json, which holds the code hash too.
A job whose run folder already holds card.json is skipped (a batch can be continued). Before a job starts, the free
disk under --out is read; under --min-free-gb the batch starts no more jobs and the rest are skipped.

The log (<out>/<batch>/batch.log) says which job ran, failed or was skipped and why; it gives no outcome of a run and
no count of outcomes (the cards hold them). Its last line is `BATCH END <batch>: N done, F failed, S skipped`.
A job is done when the runner wrote its run folder (exit status 0, 1 or 3) and the scorer wrote a card; failed when
the runner refused the case (2), wrote nothing, timed out, or the scorer failed.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plan  # noqa: E402

MAX_WORKERS = 6


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def code_hash():
    """One SHA-256 over the suite's code: every .py of sine/ (not its tests) and reference_model.py, by name and bytes."""
    h = hashlib.sha256()
    files = sorted(f for f in os.listdir(HERE) if f.endswith('.py')) + ['../reference_model.py']
    for f in files:
        p = os.path.normpath(os.path.join(HERE, f))
        h.update(os.path.basename(p).encode() + b'\0' + open(p, 'rb').read() + b'\0')
    return h.hexdigest()


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def runner_cmd(runner):
    return [sys.executable, runner] if runner.endswith('.py') else [runner]


class Batch:
    def __init__(self, a):
        self.a = a
        self.root = os.path.join(a.out, a.name)
        os.makedirs(self.root, exist_ok=True)
        self.log_fh = open(os.path.join(self.root, 'batch.log'), 'a', buffering=1)
        self.lock = threading.Lock()
        self.stop = None
        self.code = code_hash()
        self.runner_sha = sha256(a.runner)
        self.counts = dict(done=0, failed=0, skipped=0)

    def log(self, line):
        with self.lock:
            self.log_fh.write(f"{time.strftime('%H:%M:%S')} {line}\n")

    def count(self, k):
        with self.lock:
            self.counts[k] += 1

    def job(self, j):
        a = self.a
        cid = f"{j['problem']}__{j['test']}__{j['law']}"
        rd = os.path.join(self.root, j['problem'], f"{j['test']}__{j['law']}")
        if os.path.exists(os.path.join(rd, 'card.json')):
            self.count('skipped'); self.log(f'skipped {cid}: already scored')
            return
        with self.lock:
            if self.stop is None:
                g = free_gb(self.root)
                if g < a.min_free_gb:
                    self.stop = f'free disk {g:.1f} GB under {a.min_free_gb:g} GB'
                    self.log_fh.write(f"{time.strftime('%H:%M:%S')} BATCH STOP {a.name}: {self.stop}; no more jobs start\n")
            stopped = self.stop
        if stopped:
            self.count('skipped'); self.log(f'skipped {cid}: {stopped}')
            return
        case = os.path.join(a.problems, j['problem'], 'cases', f"{j['test']}__{j['law']}.json")
        if os.path.isdir(rd):
            shutil.rmtree(rd)                         # a run folder without a card is a run that did not finish
        os.makedirs(os.path.dirname(rd), exist_ok=True)
        t0 = time.time()
        try:
            p = subprocess.run(runner_cmd(a.runner) + [case, '--root', a.problems, '--out', rd], capture_output=True,
                               text=True, timeout=a.job_timeout)
            code, err = p.returncode, p.stderr[-2000:]
        except subprocess.TimeoutExpired:
            code, err = None, f'timed out after {a.job_timeout:g} s'
        t_run = time.time() - t0
        info = dict(schema='hertzforge-benchmark-sine-batch-job/0.1-draft', batch=a.name, case_id=cid,
                    case_file=f"{j['problem']}/cases/{j['test']}__{j['law']}.json", case_sha256=sha256(case),
                    code_sha256=self.code, runner_sha256=self.runner_sha, runner_exit=code, run_s=round(t_run, 3),
                    started=time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t0)))
        if code not in (0, 1, 3) or not os.path.exists(os.path.join(rd, 'run.json')):
            os.makedirs(rd, exist_ok=True)
            info['failed'] = err.strip().splitlines()[-1] if err and err.strip() else f'runner exit {code}, no run.json'
            json.dump(info, open(os.path.join(rd, 'batch.json'), 'w'), indent=1)
            self.count('failed'); self.log(f"failed {cid}: the runner: {info['failed']}")
            return
        json.dump(info, open(os.path.join(rd, 'batch.json'), 'w'), indent=1)
        q = subprocess.run([sys.executable, os.path.join(HERE, 'sine_scorer.py'), os.path.join(a.problems, j['problem']),
                            rd, '--test', j['test'], '--out', os.path.join(rd, 'card.tmp')], capture_output=True, text=True)
        if q.returncode != 0 or not os.path.exists(os.path.join(rd, 'card.tmp')):
            info['failed'] = 'scorer: ' + (q.stderr.strip().splitlines()[-1] if q.stderr.strip() else f'exit {q.returncode}')
            json.dump(info, open(os.path.join(rd, 'batch.json'), 'w'), indent=1)
            self.count('failed'); self.log(f"failed {cid}: {info['failed']}")
            return
        card = json.load(open(os.path.join(rd, 'card.tmp')))
        card['batch'] = dict(name=a.name, code_sha256=self.code, runner_sha256=self.runner_sha, runner_exit=code)
        json.dump(card, open(os.path.join(rd, 'card.json'), 'w'), indent=1, default=float)
        os.remove(os.path.join(rd, 'card.tmp'))
        self.count('done'); self.log(f'ran {cid} in {t_run:.1f} s')

    def run(self, jobs):
        a = self.a
        self.log_fh.write(f"{time.strftime('%H:%M:%S')} BATCH START {a.name}: {len(jobs)} jobs, {a.workers} workers, code "
                          f"{self.code[:16]}, runner {self.runner_sha[:16]}\n")
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(self.job, jobs))
        c = self.counts
        line = f"BATCH END {a.name}: {c['done']} done, {c['failed']} failed, {c['skipped']} skipped"
        self.log_fh.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        return line


def select(jobs, a):
    only = set(a.only.split(',')) if a.only else None
    tests = set(a.tests.split(',')) if a.tests else None
    laws = set(a.laws.split(',')) if a.laws else None
    return [j for j in jobs if (only is None or j['problem'] in only) and (tests is None or j['test'] in tests)
            and (laws is None or j['law'] in laws)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--problems', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--runner', required=True)
    ap.add_argument('--workers', type=int, default=MAX_WORKERS)
    ap.add_argument('--only', default=None)
    ap.add_argument('--tests', default=None)
    ap.add_argument('--laws', default=None)
    ap.add_argument('--min-free-gb', type=float, default=50.0)
    ap.add_argument('--job-timeout', type=float, default=7200.0)
    a = ap.parse_args(argv)
    a.workers = max(1, min(a.workers, MAX_WORKERS))
    a.problems, a.out, a.runner = (os.path.abspath(x) for x in (a.problems, a.out, a.runner))
    only = a.only.split(',') if a.only else None
    if only:
        jobs = [j for p in only for j in plan.jobs_for(p, a.problems)[0]]
    else:
        jobs = plan.jobs(a.problems)[0]
    jobs = select(jobs, a)
    line = Batch(a).run(jobs)
    print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
