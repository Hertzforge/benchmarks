"""Run a list of MIMO Sine runs, at most --workers at once, each under /usr/bin/time -l, and score each run.

    python3 sine/agree/drive.py <jobs.json> --problems <problems folder> [--workers 8] [--min-free-gb 50]

<jobs.json> is a list of {"id", "problem", "test", "cmd": [argv...], "out": <run folder>}; `cmd` is run as given
with `--root <problems> --out <out>` appended (the runner's and the harness's own arguments). For each job the
driver writes, beside the run's own files in <out>:
  drive.json  the exit status, clock seconds, user and system processor seconds, the largest resident set and peak
              memory footprint (bytes), the run folder's file sizes, the start time and the load average at start
  time.txt    /usr/bin/time -l's own lines, as it printed them
  card.json   the suite's scorer's card (sine/sine_scorer.py), when the run wrote run.json
A job whose folder already holds drive.json is skipped (a list can be continued). Before a job starts the free disk
is read; under --min-free-gb no more jobs start. The log (<jobs>.log beside the jobs file) names each job as it
starts and ends; the last line is `DRIVE END: N ran, S skipped, F without a run file`.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SINE = os.path.dirname(HERE)
MAX_WORKERS = 8                      # the package's rule: at most 8 runs at once


def parse_time(txt):
    """/usr/bin/time -l (macOS): the real, user, sys line and the counters by name."""
    out = {}
    for line in txt.splitlines():
        s = line.split()
        if len(s) == 6 and s[1] == 'real' and s[3] == 'user' and s[5] == 'sys':
            out.update(real_s=float(s[0]), user_s=float(s[2]), sys_s=float(s[4]))
        elif len(s) >= 2 and s[0].isdigit():
            out[' '.join(s[1:])] = int(s[0])
    return out


def sizes(folder):
    out = {}
    for dp, _, fs in os.walk(folder):
        for f in fs:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, folder)] = os.path.getsize(p)
    return out


class Driver:
    def __init__(self, a, jobs):
        self.a, self.jobs = a, jobs
        self.lock = threading.Lock()
        self.log_fh = open(os.path.splitext(a.jobs)[0] + '.log', 'a', buffering=1)
        self.n = dict(ran=0, skipped=0, missing=0)
        self.stop = None

    def log(self, s):
        with self.lock:
            self.log_fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {s}\n")

    def job(self, j):
        out = j['out']
        if os.path.exists(os.path.join(out, 'drive.json')):
            with self.lock:
                self.n['skipped'] += 1
            return
        with self.lock:
            if self.stop is None and shutil.disk_usage(os.path.dirname(out) if os.path.isdir(os.path.dirname(out))
                                                       else self.a.problems).free / 1e9 < self.a.min_free_gb:
                self.stop = f'free disk under {self.a.min_free_gb:g} GB'
                self.log_fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} STOP: {self.stop}\n")
            if self.stop:
                self.n['skipped'] += 1
                return
        if os.path.isdir(out):
            shutil.rmtree(out)                 # a folder without drive.json is a run that did not finish
        os.makedirs(os.path.dirname(out), exist_ok=True)
        load = os.getloadavg()
        t0 = time.time()
        self.log(f"start {j['id']}")
        p = subprocess.run(['/usr/bin/time', '-l'] + j['cmd'] + ['--root', self.a.problems, '--out', out],
                           capture_output=True, text=True)
        wall = time.time() - t0
        tl = [x for x in p.stderr.splitlines() if x.startswith(' ') and (' real ' in x or x.strip()[:1].isdigit())]
        tm = parse_time(p.stderr)
        os.makedirs(out, exist_ok=True)
        info = dict(id=j['id'], exit=p.returncode, clock_s=round(wall, 3), real_s=tm.get('real_s'), user_s=tm.get('user_s'),
                    sys_s=tm.get('sys_s'), max_rss_bytes=tm.get('maximum resident set size'),
                    peak_footprint_bytes=tm.get('peak memory footprint'), started=time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(t0)),
                    load_avg_at_start=[round(x, 2) for x in load], stdout=p.stdout[-500:],
                    stderr_tail=[x for x in p.stderr.splitlines() if x not in tl][-5:])
        with open(os.path.join(out, 'time.txt'), 'w') as fh:
            fh.write('\n'.join(tl) + '\n')
        if os.path.exists(os.path.join(out, 'run.json')):
            q = subprocess.run([sys.executable, os.path.join(SINE, 'sine_scorer.py'), os.path.join(self.a.problems, j['problem']),
                                out, '--test', j['test'], '--out', os.path.join(out, 'card.json')], capture_output=True, text=True)
            info['scorer_exit'] = q.returncode
            if q.returncode:
                info['scorer_error'] = q.stderr.strip().splitlines()[-1:] if q.stderr.strip() else []
            with self.lock:
                self.n['ran'] += 1
        else:
            with self.lock:
                self.n['missing'] += 1
        info['files'] = sizes(out)
        json.dump(info, open(os.path.join(out, 'drive.json'), 'w'), indent=1)
        self.log(f"end {j['id']}: exit {p.returncode}, {wall:.1f} s")

    def run(self):
        self.log(f'DRIVE START: {len(self.jobs)} jobs, {self.a.workers} workers')
        with ThreadPoolExecutor(max_workers=self.a.workers) as ex:
            list(ex.map(self.job, self.jobs))
        line = f"DRIVE END: {self.n['ran']} ran, {self.n['skipped']} skipped, {self.n['missing']} without a run file"
        self.log(line)
        return line


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('jobs')
    ap.add_argument('--problems', required=True)
    ap.add_argument('--workers', type=int, default=MAX_WORKERS)
    ap.add_argument('--min-free-gb', type=float, default=50.0)
    a = ap.parse_args(argv)
    a.jobs, a.problems = os.path.abspath(a.jobs), os.path.abspath(a.problems)
    a.workers = max(1, min(a.workers, MAX_WORKERS))
    jobs = json.load(open(a.jobs))
    print(Driver(a, jobs).run())
    return 0


if __name__ == '__main__':
    sys.exit(main())
