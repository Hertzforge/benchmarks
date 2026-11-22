"""Summarize a batch of MIMO Shock runs: pass rates under each reading per law and cell, and the paired effect sizes
that size a study (structures needed for a paired test at 5 percent, 90 percent power).

    python shock/summarize.py --runs <run folder> --out <folder for the small tables> [--name screen]

Writes <name>_readings.csv (one row per law, frame class, tier and target: runs and the share passing each reading,
the median margin of each), <name>_runs.csv (one row per run: every reading's pass and margin), and
<name>_effects.json (per pair of laws and cell: the paired differences of the run score, their mean and standard
deviation, and the structures a paired test needs). The run score is the last pulse's worst |SRS error| plus how far
below full level it was played (dB): 0 is a perfect run at full level, a run stopped short is charged its distance.
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

Z = 1.959964 + 1.281552          # two-sided 5 percent, 90 percent power


def run_score(rec):
    R = rec['run']['records']
    if not R:
        return 30.0              # declined before any pulse: charged as 30 dB short
    last = R[-1]
    return last['worst_db'] + abs(min(0.0, last['level_db']))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--name', default='screen'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in sorted(glob.glob(os.path.join(a.runs, '*.json'))):
        rec = json.load(open(f))
        c = rec['card']
        j = rec['job']
        rig = j['problem'].split('-')[1].split('_')[0]           # controls x drives, as the structure id writes it
        cls = rig
        cl = c.get('classical') or {}
        rows.append(dict(problem=j['problem'], kind=j['kind'], law=j['law'], cls=cls, tier=rec['variant'], end=c['end'],
                         pulses=c['pulses'], strict=c['strict']['passed'], ends=c['ends']['passed'],
                         ends_margin=c['ends'].get('margin_db'), level=c['level']['passed'],
                         level_mean=c['level'].get('mean_db'), exposure=c['exposure']['passed'],
                         exposure_db=c['exposure']['largest_abs_db'], first_over=c['first_pulse_over_db'],
                         broke=';'.join(f'{k.split("_")[0]}:{v}' for k, v in c['strict']['broke_db_over_limit'].items()),
                         grade=(c['card'] or {}).get('pattern_grade'), classical_band=cl.get('share_in_band_20pct'),
                         score=round(run_score(rec), 3), seconds=rec['seconds']))
    with open(os.path.join(a.out, f'{a.name}_runs.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = defaultdict(list)
    for r in rows:
        for key in ((r['law'], r['cls'], r['tier'], r['kind']), (r['law'], r['cls'], r['tier'], 'all'), (r['law'], 'all', 'all', 'all')):
            groups[key].append(r)
    med = lambda xs: (round(float(np.median([x for x in xs if x is not None])), 3) if any(x is not None for x in xs) else None)
    out = []
    for (law, cls, tier, kind), g in sorted(groups.items()):
        out.append(dict(law=law, cls=cls, tier=tier, kind=kind, runs=len(g),
                        strict=round(np.mean([r['strict'] for r in g]), 3), ends=round(np.mean([r['ends'] for r in g]), 3),
                        level=round(np.mean([r['level'] for r in g]), 3), exposure=round(np.mean([r['exposure'] for r in g]), 3),
                        completed=round(np.mean([r['end'] == 'completed' for r in g]), 3),
                        ends_margin_median=med([r['ends_margin'] for r in g]), exposure_median_db=med([r['exposure_db'] for r in g]),
                        score_median=med([r['score'] for r in g]), seconds_median=med([r['seconds'] for r in g])))
    with open(os.path.join(a.out, f'{a.name}_readings.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    by = defaultdict(dict)
    for r in rows:
        by[(r['cls'], r['tier'])].setdefault(r['law'], {})[(r['problem'], r['kind'])] = r['score']
    eff = []
    for (cls, tier), laws in sorted(by.items()):
        names = sorted(laws)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                keys = sorted(set(laws[x]) & set(laws[y]))
                d = np.array([laws[x][k] - laws[y][k] for k in keys])
                if d.size < 2:
                    continue
                m, s = float(d.mean()), float(d.std(ddof=1))
                need = None if m == 0 else (math.ceil((Z * s / abs(m)) ** 2) if s > 0 else 2)
                eff.append(dict(cls=cls, tier=tier, a=x, b=y, pairs=int(d.size), mean_db=round(m, 3), sd_db=round(s, 3),
                                structure_target_pairs_needed=need))
    json.dump(dict(what=__doc__.strip().splitlines()[0], score='last worst |SRS error| + dB short of full level',
                   effects=eff), open(os.path.join(a.out, f'{a.name}_effects.json'), 'w'), indent=1)
    for r in out:
        if r['kind'] == 'all':
            print(r)


if __name__ == '__main__':
    main()
