"""Start one set of the MIMO Sine study or audit as a launchd agent at normal priority, and see or stop it. Starts
nothing by itself: `start` is the only command that runs a batch.

    python3 sine/batch_agent.py start --sets <sets.json> --set <name> --runner <msnb_run or harness.py> --out <runs>
                                      [--workers 16] [--jobs <a subset of the set's jobs.json>] [--label hf.msnb.<name>]
    python3 sine/batch_agent.py plist ...      the same arguments: write the plist and print it, start nothing
    python3 sine/batch_agent.py status --label <label>
    python3 sine/batch_agent.py stop --label <label>              (launchctl bootout: the batch and every run it started)

Why launchd (study guide, "Run Time"): on the Mac mini a process started from a Claude Code session runs at priority 20
and, with many runners going, gets about half a core; a launchd agent loaded from a plist with ProcessType Interactive
runs at the normal 31. The numbers a run gives do not depend on it. A launchd job may not read a network folder, so
the agent runs sine/batch.py from the folder this file is in, which must be on a local disk (the frozen copy of the
suite); `start` refuses a copy, a runner or an output folder on a network file system (smbfs, nfs, afpfs, webdav).

The agent runs `batch.py` with the set's --problems, --cases and --jobs from sets.json (sine/record_jobs.py), at most
--workers runs at once (16, the study's), at most four 24x12 runs at once (batch.py's --big-at-once), and the free-space
guard: `start` refuses under 50 GB free on /, and batch.py starts no more jobs under 50 GB (--min-free-gb 50). The
numerical libraries are held to one thread a process (one run, one core). The plist and the agent's output go in
<out>/<set>/ beside the batch's own log.
"""
import argparse
import json
import os
import plistlib
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_FREE_GB = 50.0
WORKERS = 16
NETWORK_FS = ('smbfs', 'nfs', 'afpfs', 'webdav', 'cifs')


def fs_type(path, table=None):
    """The file system type of the mount that holds path (the longest mount point above it, from /sbin/mount, or from
    `table`, text in its form)."""
    path = os.path.realpath(path) if table is None else path
    if table is None:
        table = subprocess.run(['/sbin/mount'], capture_output=True, text=True).stdout
    best, kind = '', ''
    for line in table.splitlines():
        m = re.match(r'^.* on (/.*?) \(([^,)]+)', line)
        if not m:
            continue
        point = m[1].rstrip('/') or '/'
        if (path == point or path.startswith(point.rstrip('/') + '/')) and len(point) >= len(best):
            best, kind = point, m[2].strip()
    return kind


def domain():
    return f'gui/{os.getuid()}'


def the_set(sets, name):
    doc = json.load(open(sets))
    for s in doc['sets']:
        if s['name'] == name:
            return s
    sys.exit(f'no set {name} in {sets}')


def program(a, s):
    jobs = os.path.abspath(a.jobs) if a.jobs else s['jobs_file']
    return ['/usr/bin/python3', os.path.join(HERE, 'batch.py'), '--problems', s['problems'], '--out', os.path.abspath(a.out),
            '--name', s['name'], '--runner', os.path.abspath(a.runner), '--cases', s['cases'], '--jobs', jobs,
            '--workers', str(a.workers), '--min-free-gb', f'{MIN_FREE_GB:g}']


def plist(a):
    s = the_set(a.sets, a.set)
    label = a.label or f'hf.msnb.{s["name"]}'
    if not all(c.isalnum() or c in '._' for c in label):
        sys.exit('label: letters, digits, . and _ only')
    folder = os.path.join(os.path.abspath(a.out), s['name'])
    out = os.path.join(folder, label + '.out')
    pl = dict(Label=label, ProgramArguments=program(a, s), ProcessType='Interactive', RunAtLoad=True,
              StandardOutPath=out, StandardErrorPath=out,
              EnvironmentVariables=dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin', PYTHONDONTWRITEBYTECODE='1',
                                        VECLIB_MAXIMUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    return label, folder, pl


def running(label):
    p = subprocess.run(['launchctl', 'print', f'{domain()}/{label}'], capture_output=True, text=True)
    return p.returncode == 0 and 'state = running' in p.stdout


def cmd_plist(a, write=True):
    label, folder, pl = plist(a)
    path = os.path.join(folder, label + '.plist')
    if write:
        os.makedirs(folder, exist_ok=True)
        with open(path, 'wb') as fh:
            plistlib.dump(pl, fh)
    print(json.dumps(dict(label=label, plist=path, program=pl['ProgramArguments']), indent=1))
    return label, path


def cmd_start(a):
    for what, path in (('the suite', HERE), ('the runner', a.runner), ('the output', a.out)):
        kind = fs_type(path if os.path.exists(path) else os.path.dirname(os.path.abspath(path)))
        if kind in NETWORK_FS:
            sys.exit(f'{what} is on a network file system ({kind}), which a launchd job may not read: use a local copy')
    free = shutil.disk_usage('/').free / 1e9
    if free < MIN_FREE_GB:
        sys.exit(f'free disk on / {free:.1f} GB under {MIN_FREE_GB:g} GB: refused')
    if not os.path.exists(a.runner):
        sys.exit(f'no runner {a.runner}')
    label, _, _ = plist(a)
    if running(label):
        sys.exit(f'{label} is running: refused')
    subprocess.run(['launchctl', 'bootout', f'{domain()}/{label}'], capture_output=True)
    label, path = cmd_plist(a)
    p = subprocess.run(['launchctl', 'bootstrap', domain(), path], capture_output=True, text=True)
    print(f'launchctl bootstrap {label}: exit {p.returncode} {p.stderr.strip()}')
    return p.returncode


def cmd_status(a):
    p = subprocess.run(['launchctl', 'print', f'{domain()}/{a.label}'], capture_output=True, text=True)
    keep = [x.strip() for x in p.stdout.splitlines() if x.strip().startswith(('state =', 'pid =', 'last exit code'))]
    print('\n'.join(keep) if p.returncode == 0 else f'{a.label}: not loaded')


def cmd_stop(a):
    p = subprocess.run(['launchctl', 'bootout', f'{domain()}/{a.label}'], capture_output=True, text=True)
    print(f'launchctl bootout {a.label}: exit {p.returncode} {p.stderr.strip()}')


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('start', 'plist'):
        p = sub.add_parser(name)
        p.add_argument('--sets', required=True)
        p.add_argument('--set', required=True)
        p.add_argument('--runner', required=True)
        p.add_argument('--out', required=True)
        p.add_argument('--workers', type=int, default=WORKERS)
        p.add_argument('--jobs', default=None, help='a subset of the set\'s jobs (a list of {problem, test, law})')
        p.add_argument('--label', default=None)
    for name in ('status', 'stop'):
        p = sub.add_parser(name)
        p.add_argument('--label', required=True)
    a = ap.parse_args(argv)
    if a.cmd == 'plist':
        cmd_plist(a)
        return 0
    return dict(start=cmd_start, status=cmd_status, stop=cmd_stop)[a.cmd](a) or 0


if __name__ == '__main__':
    sys.exit(main())
