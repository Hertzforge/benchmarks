"""The MIMO Sine study's claims, written by one rule from the scorer's per-run readings, so nothing is picked by hand.

    python sine/make_claims.py readings --runs <batch folder> --out readings.csv     # one row a run, from card.json files
    python sine/make_claims.py claims   --readings readings.csv --out claims.json    # every claim the rule gives
    python sine/make_claims.py check    --claims claims.json --readings audit.csv --out verdicts.csv   # the audit's test
    python sine/make_claims.py add      --claims claims.json --text "..."            # a claim in words, stamped with its time

The rule was fixed on 2026-11-25, before any study table existed (brief MIMO_Sine_Benchmark_H_Freeze_Prep.md, Part 2;
in words in 07_Research/MIMO_Sine_Benchmark/H_Claims_Rule.md). The pattern is the three-law study's make_claims.py
(Hertzforge Journal 2026-011).

Readings (the primary three, one value a run, from the scorer's card):
  iec_share      the share of the test, from where full level began, run at full level inside the IEC 60068-2-6 check
                 point band: card amplitude.iec_check_point.share_inside x completion.coverage; 0 for a run the scorer
                 could not score (no judged update). A share: threshold 0.05.
  worst_over_db  the worst over-test on the way through a resonance, dB (card resonances.worst_db); missing for a run
                 the scorer could not score. Threshold 0.2 dB.
  completion     1 for a run that completed, else the share of the test covered at full level (card
                 completion.coverage). A share: threshold 0.05.
A cell is a version (hertzforge: the runner on the shipped engine; python: the suite's harness), a system (controls x
drives), a structure kind (linear, realistic, strong, realistic_soft, ...) and a test. Laws are paired only within a
cell, so never across versions: K3 (Python only) is paired with the Python matrix, never with the Hertzforge one. The
version is the card's run.version (the scorer reads it from the run file's producer).

Claims:
  pair    for every two laws that ran the same test in a cell, the difference (first minus second, same structure) of
          each primary reading over the cell's structures. A claim where the median difference is beyond the threshold
          (> 0.2 dB, > 0.05 in a share), at least 6 in 10 of the cell's paired structures differ by more than the
          threshold on the median's side and at most 2 in 10 on the other side (12 of 20 and at most 4 of 20).
  level   every law in every cell, against each tolerance line the scorer reads (the worst excursion over the line, dB,
          0 or less is inside): "inside" where the median over the cell's structures is 0 or less, else "outside".
          A run the scorer could not score counts as outside. Always a claim.
  effect  the structure's kind, pooled over the laws (naming none): realistic against linear, strong against realistic,
          strong against linear, same system, structure (seed), test and law; the same pair rule over the cell's
          (law, structure) units.
  late    a claim added by hand (add): its text, stamped with the time it was written; not judged by check.
A claim that names a law admitted after the screen (ADMITTED_AFTER_SCREEN: K3) carries admitted_after_screen with the
decision, and its text says so: the admission was post hoc, and the audit's held-back runs test its claims.

The audit's test (check), for each claim, on the held-back structures, the same statistic in the same cell:
  pair, effect: "holds" if the audit's median has the claim's sign, is beyond the threshold, and lies inside the
                study's quartiles; "weaker" if it has the sign and is beyond the threshold but outside the quartiles;
                "does not hold" if it has the opposite sign or is not beyond the threshold.
  level:        "holds" if the audit's median is on the claim's side of the line and inside the study's quartiles;
                "weaker" if on the claim's side outside them; "does not hold" if on the other side.
  A cell with no audit runs is "not tested"; a late claim is "not judged".

Medians and quartiles are numpy's (linear). The output of claims depends only on the readings file and this file, so
two identical inputs give byte-identical outputs; nothing here reads a clock except `add`. NumPy only.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = 'hertzforge-benchmark-sine-claims/0.1-draft'
THRESHOLD = {'iec_share': 0.05, 'worst_over_db': 0.2, 'completion': 0.05}
READINGS = tuple(THRESHOLD)
UNITS = {'iec_share': 'share', 'worst_over_db': 'dB', 'completion': 'share'}
WORDS = {'iec_share': 'share of the test inside the IEC check-point band at full level',
         'worst_over_db': 'worst over-test through a resonance', 'completion': 'completion'}
# the tolerance lines the scorer reads (sine_model.TOL); a level claim is made against each
LINES = ('iec_check_point', 'iec_reference_point', 'mil_composite', 'mil_average_individual', 'mil_maxi_individual')
LAW_ORDER = ('matrix', 'paired', 'ratio', 'steepest_descent', 'secant')
VERSIONS = ('hertzforge', 'python')
ADMITTED_AFTER_SCREEN = {'k3': 'admitted to the study after the screen, by Kevin\'s decision of 2026-11-25 ("Include K3"; '
                               'DECISIONS.md, G_Screen.md): post hoc, its claims tested on the held-back structures'}
EFFECTS = (('realistic', 'linear'), ('strong', 'realistic'), ('strong', 'linear'))
SID = re.compile(r'^frame-(?P<system>\d+x\d+)(?:-(?P<tier>realistic|strong))?(?:_(?P<effect>soft|hard))?_s(?P<seed>\d+)$')
COLUMNS = (['version', 'structure', 'system', 'kind', 'seed', 'test', 'law', 'outcome', 'scored'] + list(READINGS)
           + [f'excursion_{k}' for k in LINES])


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def parse_sid(sid):
    m = SID.match(sid)
    if not m:
        raise ValueError(f'not a structure id: {sid}')
    kind = (m['tier'] or 'linear') + (f"_{m['effect']}" if m['effect'] else '')
    return m['system'], kind, int(m['seed'])


# ---------------------------------------------------------------------------------------------- readings from cards
def reading_row(card):
    """One run's row from its card (the scorer's output)."""
    sid, test, law = card['run']['case_id'].split('__')
    system, kind, seed = parse_sid(sid)
    version = card['run'].get('version')
    if version not in VERSIONS:
        raise ValueError(f"{card['run']['case_id']}: the card does not say which version wrote the run ({version!r})")
    comp = card['completion']
    scored = bool(card.get('scored'))
    cov = float(comp.get('coverage') or 0.0)
    done = comp.get('outcome') == 'completed'
    row = dict(version=version, structure=sid, system=system, kind=kind, seed=seed, test=test, law=law, outcome=comp.get('outcome'),
               scored=int(scored), completion=1.0 if done else cov)
    if scored:
        row['iec_share'] = float(card['amplitude']['iec_check_point']['share_inside']) * (1.0 if done else cov)
        w = card['resonances'].get('worst_db')
        row['worst_over_db'] = '' if w is None else float(w)
        for k in LINES:
            row[f'excursion_{k}'] = float(card['amplitude'][k]['worst_excursion_db'])
    else:
        row['iec_share'] = 0.0
        row['worst_over_db'] = ''
        for k in LINES:
            row[f'excursion_{k}'] = 'unscored'
    return row


def cmd_readings(a):
    rows, seen = [], set()
    for d, _, files in sorted(os.walk(a.runs)):
        if 'card.json' in files:
            card = json.load(open(os.path.join(d, 'card.json')))
            r = reading_row(card)
            key = (r['version'], r['structure'], r['test'], r['law'])
            if key in seen:
                sys.exit(f'two cards for {key}: the readings come from one batch of record')
            seen.add(key)
            rows.append(r)
    rows.sort(key=lambda r: (r['version'], r['structure'], r['test'], r['law']))
    write_csv(a.out, rows)
    print(f'{len(rows)} runs read into {a.out}')


def fmt(v):
    if isinstance(v, float):
        return repr(v)
    return v


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator='\n')
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, '')) for k in COLUMNS})


def num(v):
    if v in ('', None):
        return math.nan
    if v == 'unscored':
        return math.inf
    return float(v)


def load_readings(path):
    out = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            r = dict(r)
            if r.get('version') not in VERSIONS:
                sys.exit(f"a reading without a version ({r.get('version')!r}): {r['structure']} {r['test']} {r['law']}")
            r['seed'] = int(r['seed'])
            for k in READINGS:
                r[k] = num(r[k])
            for k in LINES:
                r[f'excursion_{k}'] = num(r[f'excursion_{k}'])
            out.append(r)
    keys = [(r['version'], r['structure'], r['test'], r['law']) for r in out]
    if len(set(keys)) != len(keys):
        sys.exit('a run appears twice in the readings')
    return out


# ---------------------------------------------------------------------------------------------- statistics
def q(v, p):
    """numpy's linear percentile; an infinite value (an unscored run against a line) stays infinite."""
    x = float(np.percentile(np.asarray(v, float), p))
    return math.inf if math.isnan(x) else x


def pair_stats(d, thr):
    """d: the paired differences of a cell (NaN where a reading is missing). Returns the statistic and whether the rule
    gives a claim. Counts are against every paired structure of the cell, a missing reading on neither side."""
    d = np.asarray(d, float)
    n = int(d.size)
    v = d[~np.isnan(d)]
    if v.size == 0:
        return None
    med = float(np.median(v))
    side = 'lower' if med < 0 else 'higher'
    lower, higher = int(np.sum(v < -thr)), int(np.sum(v > thr))
    mine, other = (lower, higher) if med < 0 else (higher, lower)
    claim = abs(med) > thr and 5 * mine >= 3 * n and 5 * other <= n
    return dict(n=n, n_valid=int(v.size), median=med, q1=q(v, 25), q3=q(v, 75), lower=lower, higher=higher, side=side,
                claim=bool(claim))


def law_rank(law):
    return (LAW_ORDER.index(law), law) if law in LAW_ORDER else (len(LAW_ORDER), law)


def index(rows):
    by = {}
    for r in rows:
        by[(r['version'], r['system'], r['kind'], r['test'], r['law'], r['seed'])] = r
    return by


def cells(rows):
    out = {}
    for r in rows:
        out.setdefault((r['version'], r['system'], r['kind'], r['test']), {}).setdefault(r['law'], set()).add(r['seed'])
    return out


def r6(x):
    if x is None or (isinstance(x, float) and (math.isinf(x) or math.isnan(x))):
        return None
    return round(float(x), 6)


def clean(st):
    return {k: (r6(v) if isinstance(v, float) else v) for k, v in st.items() if k != 'claim'}


def pair_values(by, version, system, kind, test, a, b, seeds, reading):
    return [by[(version, system, kind, test, a, s)][reading] - by[(version, system, kind, test, b, s)][reading] for s in seeds]


def effect_values(by, version, system, test, hi, lo, units, reading):
    return [by[(version, system, hi, test, law, s)][reading] - by[(version, system, lo, test, law, s)][reading]
            for law, s in units]


def admitted(c):
    """Mark a claim that names a law admitted after the screen."""
    late = [ADMITTED_AFTER_SCREEN[x] for x in (c.get('law'), c.get('against')) if x in ADMITTED_AFTER_SCREEN]
    if late:
        c['admitted_after_screen'] = late[0]
    return c


def the_claims(rows):
    by = index(rows)
    cs = cells(rows)
    claims = []
    for (version, system, kind, test), laws in sorted(cs.items()):
        names = sorted(laws, key=law_rank)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                seeds = sorted(laws[a] & laws[b])
                if not seeds:
                    continue
                for reading in READINGS:
                    st = pair_stats(pair_values(by, version, system, kind, test, a, b, seeds, reading), THRESHOLD[reading])
                    if st and st['claim']:
                        claims.append(admitted(dict(kind='pair', version=version, system=system, structure_kind=kind,
                                                    test=test, law=a, against=b, reading=reading, side=st['side'],
                                                    threshold=THRESHOLD[reading], study=clean(st))))
        for law in names:
            seeds = sorted(laws[law])
            for line in LINES:
                v = np.array([by[(version, system, kind, test, law, s)][f'excursion_{line}'] for s in seeds], float)
                med = q(v, 50)
                cell = [by[(version, system, kind, test, law, s)] for s in seeds]
                claims.append(admitted(dict(kind='level', version=version, system=system, structure_kind=kind, test=test,
                                            law=law, line=line, side='inside' if med <= 0 else 'outside',
                                            study=dict(n=len(seeds), median=r6(med), q1=r6(q(v, 25)), q3=r6(q(v, 75)),
                                                       unscored=int(np.sum(np.isinf(v))),
                                                       median_iec_share=r6(q([r['iec_share'] for r in cell], 50)),
                                                       median_completion=r6(q([r['completion'] for r in cell], 50))))))
    # the structure's kind, pooled over the laws of one version
    systems_tests = sorted({(v, s, t) for (v, s, k, t) in cs})
    for version, system, test in systems_tests:
        for hi, lo in EFFECTS:
            A, B = cs.get((version, system, hi, test)), cs.get((version, system, lo, test))
            if not A or not B:
                continue
            units = sorted((law, s) for law in set(A) & set(B) for s in A[law] & B[law])
            if not units:
                continue
            for reading in READINGS:
                st = pair_stats(effect_values(by, version, system, test, hi, lo, units, reading), THRESHOLD[reading])
                if st and st['claim']:
                    claims.append(dict(kind='effect', version=version, system=system, test=test, structure_kind=hi,
                                       against=lo, reading=reading, side=st['side'], threshold=THRESHOLD[reading],
                                       study=dict(clean(st), laws=len({u[0] for u in units}))))
    for k, c in enumerate(claims, 1):
        c['id'] = f'R{k:04d}'
        c['text'] = text(c)
    return claims


def text(c):
    t = _text(c)
    if c['version'] == 'python':
        t = 'Python version. ' + t
    if c.get('admitted_after_screen'):
        t += ' (K3 was admitted to the study after the screen.)'
    return t


def _text(c):
    s = c['study']
    where = f"system {c['system']}, {c['structure_kind']} structures, {c['test']}"
    if c['kind'] == 'pair':
        u = UNITS[c['reading']]
        return (f"On {where}: {c['law']} has a {c['side']} {WORDS[c['reading']]} than {c['against']} (study: median "
                f"difference {s['median']:+.3f} {u}, quartiles {s['q1']:+.3f} to {s['q3']:+.3f}; {s['lower']} lower and "
                f"{s['higher']} higher by more than {c['threshold']:g} {u}, of {s['n']} structures).")
    if c['kind'] == 'level':
        med = 'unscored' if s['median'] is None else f"{s['median']:+.3f} dB"
        return (f"On {where}: {c['law']}'s median run is {c['side']} the {c['line']} line (study: median worst excursion "
                f"{med} over {s['n']} structures, {s['unscored']} unscored).")
    u = UNITS[c['reading']]
    return (f"On system {c['system']}, {c['test']}, pooled over {s['laws']} laws: {c['structure_kind']} structures have a "
            f"{c['side']} {WORDS[c['reading']]} than {c['against']} (study: median difference {s['median']:+.3f} {u}, "
            f"{s['lower']} lower and {s['higher']} higher of {s['n']} law-structure pairs).")


def cmd_claims(a):
    if os.path.exists(a.out):
        sys.exit(f'{a.out} exists: claims are never rewritten; write to a new file')
    rows = load_readings(a.readings)
    claims = the_claims(rows)
    doc = dict(schema=SCHEMA, rule=dict(file='sine/make_claims.py', sha256=sha256(os.path.abspath(__file__)),
                                        thresholds=THRESHOLD, words='07_Research/MIMO_Sine_Benchmark/H_Claims_Rule.md'),
               readings=dict(file=os.path.basename(a.readings), sha256=sha256(a.readings), runs=len(rows)),
               claims=claims, late=[])
    with open(a.out, 'w') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write('\n')
    k = {}
    for c in claims:
        k[c['kind']] = k.get(c['kind'], 0) + 1
    print(f'{len(claims)} claims from {len(rows)} runs:', ', '.join(f'{x} {n}' for x, n in sorted(k.items())))


def cmd_add(a):
    doc = json.load(open(a.claims))
    stamp = a.stamp or time.strftime('%Y-%m-%d %H:%M:%S %Z')
    nid = len(doc['late']) + 1
    doc['late'].append(dict(id=f'L{nid:04d}', kind='late', text=a.text, written=stamp,
                            note='added by hand, after the rule; marked with the time it was written'))
    with open(a.claims, 'w') as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write('\n')
    print(f'late claim L{nid:04d} added at {stamp}')


# ---------------------------------------------------------------------------------------------- the audit's test
def verdict(c, rows):
    by = index(rows)
    cs = cells(rows)
    s = c['study']
    v_ = c['version']
    if c['kind'] in ('pair', 'level'):
        laws = cs.get((v_, c['system'], c['structure_kind'], c['test']), {})
    if c['kind'] == 'pair':
        seeds = sorted(laws.get(c['law'], set()) & laws.get(c['against'], set()))
        if not seeds:
            return None, 0, 'not tested'
        st = pair_stats(pair_values(by, v_, c['system'], c['structure_kind'], c['test'], c['law'], c['against'], seeds,
                                    c['reading']), c['threshold'])
    elif c['kind'] == 'effect':
        A, B = cs.get((v_, c['system'], c['structure_kind'], c['test'])), cs.get((v_, c['system'], c['against'], c['test']))
        units = sorted((law, x) for law in set(A or {}) & set(B or {}) for x in A[law] & B[law]) if A and B else []
        if not units:
            return None, 0, 'not tested'
        st = pair_stats(effect_values(by, v_, c['system'], c['test'], c['structure_kind'], c['against'], units,
                                      c['reading']), c['threshold'])
    else:
        seeds = sorted(laws.get(c['law'], set()))
        if not seeds:
            return None, 0, 'not tested'
        v = np.array([by[(v_, c['system'], c['structure_kind'], c['test'], c['law'], x)][f"excursion_{c['line']}"]
                      for x in seeds], float)
        med = q(v, 50)
        side = 'inside' if med <= 0 else 'outside'
        if side != c['side']:
            return med, len(seeds), 'does not hold'
        lo = -math.inf if s['q1'] is None else s['q1']
        hi = math.inf if s['q3'] is None else s['q3']
        return med, len(seeds), 'holds' if lo <= med <= hi else 'weaker'
    if st is None:
        return None, 0, 'not tested'
    med = st['median']
    sign_ok = (med < 0) == (c['side'] == 'lower') and med != 0
    if not sign_ok or abs(med) <= c['threshold']:
        return med, st['n'], 'does not hold'
    return med, st['n'], 'holds' if s['q1'] <= med <= s['q3'] else 'weaker'


def cmd_check(a):
    doc = json.load(open(a.claims))
    rows = load_readings(a.readings)
    out = []
    for c in doc['claims']:
        med, n, v = verdict(c, rows)
        out.append(dict(id=c['id'], kind=c['kind'], verdict=v, audit_median=r6(med) if med is not None else '', audit_n=n,
                        study_median=c['study']['median'] if c['study']['median'] is not None else '',
                        study_q1=c['study']['q1'] if c['study']['q1'] is not None else '',
                        study_q3=c['study']['q3'] if c['study']['q3'] is not None else '', text=c['text']))
    for c in doc['late']:
        out.append(dict(id=c['id'], kind='late', verdict='not judged', audit_median='', audit_n=0, study_median='',
                        study_q1='', study_q3='', text=c['text']))
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]) if out else ['id'], lineterminator='\n')
        w.writeheader()
        w.writerows(out)
    k = {}
    for r in out:
        k[r['verdict']] = k.get(r['verdict'], 0) + 1
    print(f'{len(out)} claims checked:', ', '.join(f'{x} {n}' for x, n in sorted(k.items())))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('readings'); p.add_argument('--runs', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('claims'); p.add_argument('--readings', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('check'); p.add_argument('--claims', required=True); p.add_argument('--readings', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('add'); p.add_argument('--claims', required=True); p.add_argument('--text', required=True)
    p.add_argument('--stamp', default=None)
    a = ap.parse_args(argv)
    dict(readings=cmd_readings, claims=cmd_claims, check=cmd_check, add=cmd_add)[a.cmd](a)
    return 0


if __name__ == '__main__':
    sys.exit(main())
