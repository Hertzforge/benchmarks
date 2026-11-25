"""A stand-in for msnb_run in the batch's tests: `fake_runner.py <case> --root <problems> --out <run folder>`.
Writes a synthetic run (synth.make_run) for the case's problem and test. FAKE_REFUSE: a case id it refuses (exit 2,
nothing written); FAKE_CRASH: one it crashes on after making the folder (exit 101, no run.json)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synth  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('case'); ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
case = json.load(open(a.case))
cid = case['case_id']
if cid == os.environ.get('FAKE_REFUSE'):
    print('case refused: a key unknown', file=sys.stderr)
    sys.exit(2)
os.makedirs(a.out, exist_ok=True)
if cid == os.environ.get('FAKE_CRASH'):
    print('thread panicked', file=sys.stderr)
    sys.exit(101)
problem, test, law = cid.split('__')
synth.make_run(os.path.join(a.root, problem), test, a.out)
sys.exit(0)
