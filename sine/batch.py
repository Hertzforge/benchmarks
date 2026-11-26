"""Run the MIMO Sine matrix: every job (problem, test, law) through the runner, then the scorer.

    python sine/batch.py --problems <problems folder> --out <runs folder> --name <batch> --runner <msnb_run> \
        [--workers 16] [--only frame-6x6_s00,...] [--tests T1,T4] [--laws matrix] [--min-free-gb 50] [--job-timeout 7200] \
        [--cases <case folder>] [--jobs <jobs.json>] [--big-systems 24x12] [--big-at-once 4]

--workers: runs at once, 16 by default (the study's), at most 18 (the Mac mini's 18 cores: the study guide's limit across
every batch running at once). --runner may be the Python version's harness (sine/pyloop/harness.py), run with this
interpreter. Jobs on a --big-systems system (24x12 by default; one 24x12 run held 2.69 GB, MSNB-C-A-7) are spread evenly
through the list and at most --big-at-once of them run at once; the others fill the workers meanwhile. The order and the
limit change no number a run gives (each run is one process on its own case). sine/batch_agent.py starts a set as a
launchd agent at normal priority.

--cases: run <cases>/<problem>/<test>__<law>.json in place of each problem's own cases/<test>__<law>.json (a set's
settings: the runs of record, the default check, a repeat; sine/record_jobs.py writes them). The runner still reads
every other file from --problems. --jobs: the job list (a list of {problem, test, law}) in place of plan.py's matrix.

The jobs are sine/plan.py's (the laws' manifests against the tests each problem built). A job's run folder is
<out>/<batch>/<problem>/<test>__<law>/: the runner writes run.json (and timing.json) there with
`<runner> <case> --root <problems> --out <run folder>`; the batch adds batch.json (the suite's code hash, the runner's
SHA-256, the case's SHA-256, the runner's exit status, times, and the run's own user and system processor seconds
and largest resident set, read from os.wait4) and the scorer's card.json, which holds the code hash too.
A job whose run folder already holds card.json is skipped (a batch can be continued). Before a job starts, the free
disk under --out and on / is read (the smaller); under --min-free-gb the batch starts no more jobs and the rest are skipped.

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

DEFAULT_WORKERS = 16                 # the study's: the launchd agent at normal priority, 16 runs at once
MAX_WORKERS = 18                     # the Mac mini's cores: the study guide's limit across every batch at once
BIG_SYSTEMS = ('24x12',)
BIG_AT_ONCE = 4


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
    """Free disk under `path` and on /, the smaller, GB."""
    return min(shutil.disk_usage(path).free, shutil.disk_usage('/').free) / 1e9


def system_of(problem):
    """The system (controls x drives) of a problem id: frame-24x12-realistic_s03 -> 24x12."""
    return problem.split('-')[1].split('_')[0]


def interleave(jobs, big):
    """The jobs with those for which big(job) is true spread evenly among the others, each group in its own order."""
    b = [j for j in jobs if big(j)]
    o = [j for j in jobs if not big(j)]
    if not b or not o:
        return list(jobs)
    out, k = [], 0
    for i, j in enumerate(o):
        while k < len(b) and (k + 0.5) * len(o) / len(b) <= i:
            out.append(b[k]); k += 1
        out.append(j)
    return out + b[k:]


def run_measured(cmd, timeout):
    """Run cmd; return (exit status or None if it timed out, stderr, the process's own user and system seconds and its
    largest resident set in bytes, from os.wait4). The reading changes nothing the run gives."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    got = {}
    readers = [threading.Thread(target=lambda k=k, f=f: got.__setitem__(k, f.read())) for k, f in (('out', p.stdout),
                                                                                                      ('err', p.stderr))]
    for t in readers:
        t.start()
    deadline, timed_out = time.time() + timeout, False
    while True:
        pid, status, ru = os.wait4(p.pid, os.WNOHANG)
        if pid:
            break
        if time.time() > deadline:
            p.kill()
            pid, status, ru = os.wait4(p.pid, 0)
            timed_out = True
            break
        time.sleep(0.05)
    for t in readers:
        t.join()
    p.returncode = os.waitstatus_to_exitcode(status)
    use = dict(user_s=round(ru.ru_utime, 3), sys_s=round(ru.ru_stime, 3),
               max_rss_bytes=int(ru.ru_maxrss) * (1 if sys.platform == 'darwin' else 1024))
    if timed_out:
        return None, f'timed out after {timeout:g} s', use
    return p.returncode, got.get('err', ''), use


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
        self.big = set(a.big_systems.split(',')) if a.big_systems else set()
        self.big_running = 0
        self.most_big_at_once = 0

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
        case = case_path(a, j)
        if not os.path.exists(case):
            self.count('failed'); self.log(f'failed {cid}: no case file in the set')
            return
        if os.path.isdir(rd):
            shutil.rmtree(rd)                         # a run folder without a card is a run that did not finish
        os.makedirs(os.path.dirname(rd), exist_ok=True)
        t0 = time.time()
        code, err, use = run_measured(runner_cmd(a.runner) + [case, '--root', a.problems, '--out', rd], a.job_timeout)
        err = err[-2000:]
        t_run = time.time() - t0
        info = dict(schema='hertzforge-benchmark-sine-batch-job/0.1-draft', batch=a.name, case_id=cid,
                    case_file=(os.path.relpath(case, os.path.dirname(os.path.dirname(a.cases))) if a.cases
                               else f"{j['problem']}/cases/{j['test']}__{j['law']}.json"), case_sha256=sha256(case),
                    code_sha256=self.code, runner_sha256=self.runner_sha, runner_exit=code, run_s=round(t_run, 3), **use,
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

    def is_big(self, j):
        return system_of(j['problem']) in self.big

    def worker(self, queue, cond):
        """Take the first job in the queue that may start (a big system's only while fewer than big_at_once run)."""
        while True:
            with cond:
                while True:
                    if not queue:
                        return
                    k = next((i for i, j in enumerate(queue)
                              if not self.is_big(j) or self.big_running < self.a.big_at_once), None)
                    if k is not None:
                        break
                    cond.wait()
                j = queue.pop(k)
                big = self.is_big(j)
                if big:
                    self.big_running += 1
                    self.most_big_at_once = max(self.most_big_at_once, self.big_running)
            try:
                self.job(j)
            finally:
                if big:
                    with cond:
                        self.big_running -= 1
                        cond.notify_all()

    def run(self, jobs):
        a = self.a
        jobs = interleave(jobs, self.is_big)
        self.order = list(jobs)
        self.log_fh.write(f"{time.strftime('%H:%M:%S')} BATCH START {a.name}: {len(jobs)} jobs, {a.workers} workers, at most "
                          f"{a.big_at_once} on {','.join(sorted(self.big)) or 'no system'} at once, code {self.code[:16]}, "
                          f"runner {self.runner_sha[:16]}\n")
        queue, cond = list(jobs), threading.Condition()
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for f in [ex.submit(self.worker, queue, cond) for _ in range(a.workers)]:
                f.result()
        c = self.counts
        line = f"BATCH END {a.name}: {c['done']} done, {c['failed']} failed, {c['skipped']} skipped"
        self.log_fh.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        return line


def case_path(a, j):
    root = a.cases if a.cases else None
    if root:
        return os.path.join(root, j['problem'], f"{j['test']}__{j['law']}.json")
    return os.path.join(a.problems, j['problem'], 'cases', f"{j['test']}__{j['law']}.json")


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
    ap.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    ap.add_argument('--only', default=None)
    ap.add_argument('--tests', default=None)
    ap.add_argument('--laws', default=None)
    ap.add_argument('--min-free-gb', type=float, default=50.0)
    ap.add_argument('--job-timeout', type=float, default=7200.0)
    ap.add_argument('--cases', default=None)
    ap.add_argument('--jobs', default=None)
    ap.add_argument('--big-systems', default=','.join(BIG_SYSTEMS))
    ap.add_argument('--big-at-once', type=int, default=BIG_AT_ONCE)
    a = ap.parse_args(argv)
    a.big_at_once = max(1, a.big_at_once)
    a.workers = max(1, min(a.workers, MAX_WORKERS))
    a.problems, a.out, a.runner = (os.path.abspath(x) for x in (a.problems, a.out, a.runner))
    if a.cases:
        a.cases = os.path.abspath(a.cases)
    only = a.only.split(',') if a.only else None
    if a.jobs:
        jobs = [dict(problem=j['problem'], test=j['test'], law=j['law']) for j in json.load(open(a.jobs))]
    elif only:
        jobs = [j for p in only for j in plan.jobs_for(p, a.problems)[0]]
    else:
        jobs = plan.jobs(a.problems)[0]
    jobs = select(jobs, a)
    line = Batch(a).run(jobs)
    print(line)
    return 0


if __name__ == '__main__':
    sys.exit(main())
