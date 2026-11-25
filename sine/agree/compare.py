"""Compare two MIMO Sine runs of the same case by what the scorer reads: their cards, reading by reading, against
the agreement criteria (07_Research/MIMO_Sine_Benchmark/D_Agreement.md section 3, fixed before any run).

    python3 sine/agree/compare.py pair <run folder A> <run folder B>             # one pair, printed
    python3 sine/agree/compare.py set <base> <arm A> <arm B> --out <report.json>  # every case two arms share
    python3 sine/agree/compare.py break <run folder> <new folder> --problems <p> --how control_1db|outcome

Each run folder holds the scorer's card.json (drive.py writes it). `pair` and `set` say, for each pair:
  identical        the two cards are equal in every key compared or reported (the producer's own keys left out)
  outcome_agrees   class, guard, scored equal; where an ended run stopped within the criteria
  readings_agree   every reading in the criteria within its tolerance
and list each reading beyond its tolerance, and the readings outside the criteria that differ (reported only).
`break` writes a copy of a run built to break the comparison (every control raised 1 dB over the judged updates, or
the outcome changed) and scores it, so the script can be shown to fail on it.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SINE = os.path.dirname(HERE)

SKIP = re.compile(r'^(schema|scorer|run|batch|problem|practice\.departures)(\.|$)|\.(what|why|clause|status|window|state|'
                  r'composite|between_points|analogy|coverage_what|held_from|effort_what|message|tolerances?|rate_tolerance|'
                  r'duration_tolerance)(\.|$)|\.(clause|status|what)$')
DB = 'db'
# (pattern, kind, tolerance): section 3 of D_Agreement.md
RULES = [
    (r'^completion\.(outcome|guard)$', 'equal', None),
    (r'^scored$', 'equal', None),
    (r'^completion\.at_level_db$', 'abs', 0.5),
    (r'^amplitude\.error_db\.mean$', 'abs', 0.2),
    (r'^practice\.line_1_level\.mean_error_db$', 'abs', 0.2),
    (r'^practice\.line_2_pattern\.controls\.(summed_rms_db|per_channel_rms_db)$', 'abs', 0.2),
    (r'^practice\.line_5_declared\.law_error_rms_db$', 'abs', 0.2),
    (r'^amplitude\.error_db\.p95_abs$', 'abs', 0.5),
    (r'^practice\.line_2_pattern\.held_out\.(summed_rms_db|per_channel_rms_db)$', 'abs', 0.5),
    (r'^practice\.line_3_over_test\.worst_band_db$', 'abs', 0.5),
    (r'^drive\.effort_db_p(50|95)$', 'abs', 0.5),
    (r'^stress\.error_db_p10_p50_p90\.\d+$', 'abs', 0.5),
    (r'^amplitude\.error_db\.worst_(over|under)$', 'abs', 1.0),
    (r'^amplitude\.[a-z0-9_]+\.(worst_excursion_db|worst_error_db)$', 'abs', 1.0),
    (r'^resonances\.worst_db$', 'abs', 1.0),
    (r'^resonances\.modes\.\d+\.worst_control_db$', 'abs', 1.0),
    (r'^stress\.worst_(over|under)_db$', 'abs', 1.0),
    (r'^drive\.effort_db_max$', 'abs', 1.0),
    (r'^drive\.(norm_max_v|largest_drive_v)$', DB, 1.0),
    (r'^amplitude\.[a-z0-9_]+\.(share_inside|share_inside_every_control)$', 'abs', 0.02),
    (r'^completion\.(coverage|ladder_share)$', 'abs', 0.02),
    (r'^stress\.share_within_3_db$', 'abs', 0.02),
    (r'^practice\.line_3_over_test\.share_of_bands_over_alarm$', 'abs', 0.02),
    (r'^practice\.line_4_under_test\.share_[a-z0-9_]+$', 'abs', 0.02),
    (r'^phase\.mean_deg$', 'deg', 1.5),
    (r'^phase\.p95_abs_deg$', 'abs', 3.0),
    (r'^phase\.worst_deg$', 'deg', 10.0),
    (r'^resonances\.count_over_db\.[0-9.]+$', 'abs', 1.0),
    (r'^completion\.judged_updates$', 'count', None),
    (r'^frequency\.rate_realized_oct_per_min$', 'rel', 0.005),
    (r'^frequency\.(duration_s|full_level_s)$', 'abs', 0.5),
    (r'^frequency\.largest_frequency_error_pct$', 'abs', 0.05),
    (r'^state\.as_tested\.(median_abs_change|largest_drop|largest_rise)$', 'abs', 0.002),
]
RULES = [(re.compile(p), k, t) for p, k, t in RULES]


def flatten(x, pre='', out=None):
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            flatten(v, f'{pre}.{k}' if pre else str(k), out)
    elif isinstance(x, list) and x and all(isinstance(v, (dict, list)) for v in x):
        for i, v in enumerate(x):
            flatten(v, f'{pre}.{i}', out)
    elif isinstance(x, list) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x) and x:
        for i, v in enumerate(x):
            out[f'{pre}.{i}'] = v
    else:
        out[pre] = x
    return out


def rule_for(path):
    for rx, kind, tol in RULES:
        if rx.match(path):
            return kind, tol
    return None


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v is not None


def within(kind, tol, a, b):
    """(within, size of the difference) for one reading."""
    if kind == 'equal':
        return a == b, None if a == b else f'{a!r} vs {b!r}'
    if a is None and b is None:
        return True, 0.0
    if not (_num(a) and _num(b)):
        return a == b, None if a == b else f'{a!r} vs {b!r}'
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b, None if a == b else f'{a!r} vs {b!r}'
    if kind == 'abs':
        d = abs(a - b)
        return d <= tol + 1e-12, d
    if kind == 'rel':
        d = abs(a - b) / max(abs(a), abs(b), 1e-300)
        return d <= tol, d
    if kind == DB:
        if a <= 0 or b <= 0:
            return a == b, None
        d = abs(20 * math.log10(a / b))
        return d <= tol, d
    if kind == 'deg':
        d = abs((a - b + 180.0) % 360.0 - 180.0)
        return d <= tol, d
    if kind == 'count':
        d = abs(a - b)
        return d <= max(3, 0.02 * max(a, b)), d
    raise ValueError(kind)


def where_ended(ca, cb, kind):
    """The criteria for where an ended run stopped. Returns (agrees, note)."""
    a, b = ca.get('completion') or {}, cb.get('completion') or {}
    if a.get('outcome') == 'completed' and b.get('outcome') == 'completed':
        return True, None
    fa, fb, ta, tb = a.get('at_hz'), b.get('at_hz'), a.get('at_s'), b.get('at_s')
    if None in (fa, fb, ta, tb):
        return (fa, ta) == (fb, tb), f'at_hz {fa} vs {fb}, at_s {ta} vs {tb}'
    if kind == 'sweep':
        d = abs(math.log2(fa / fb))
        return d <= 1.0 / 24.0 + 1e-12, f'{d * 48:.2f}/48 octave ({fa:.4g} vs {fb:.4g} Hz)'
    if kind == 'stepped':
        d = abs(fa / fb - 1.0)
        return d <= 0.01, f'{100 * d:.3g} % ({fa:.4g} vs {fb:.4g} Hz)'
    d = abs(ta - tb)
    return d <= 1.0, f'{d:.3g} s ({ta:.4g} vs {tb:.4g} s)'


def test_kind(card):
    tid = card['problem']['test']
    return {'T1': 'sweep', 'T2': 'sweep', 'T5': 'sweep', 'L1': 'sweep', 'T3': 'stepped', 'T4': 'dwell'}[tid]


def compare_cards(ca, cb):
    fa, fb = flatten(ca), flatten(cb)
    keys = sorted(set(fa) | set(fb))
    beyond, reported, identical = [], [], True
    for k in keys:
        if SKIP.search(k):
            continue
        va, vb = fa.get(k, '<absent>'), fb.get(k, '<absent>')
        if va != vb:
            identical = False
        r = rule_for(k)
        if r is None:
            if va != vb:
                reported.append(dict(key=k, a=va, b=vb))
            continue
        if (va == '<absent>') != (vb == '<absent>'):
            beyond.append(dict(key=k, a=va, b=vb, diff='present in one only', tol=r[1]))
            continue
        ok, d = within(r[0], r[1], va, vb)
        if not ok:
            beyond.append(dict(key=k, a=va, b=vb, diff=d, tol=r[1]))
    kind = test_kind(ca)
    ended_ok, ended_note = where_ended(ca, cb, kind)
    oc = {'completion.outcome', 'completion.guard', 'scored', 'completion.at_level_db'}
    outcome_ok = ended_ok and not any(x['key'] in oc for x in beyond)
    readings_ok = not any(x['key'] not in oc for x in beyond)
    return dict(identical=identical, outcome_agrees=outcome_ok, readings_agree=readings_ok, where_ended=ended_note,
                outcome=[(ca.get('completion') or {}).get('outcome'), (cb.get('completion') or {}).get('outcome')],
                guard=[(ca.get('completion') or {}).get('guard'), (cb.get('completion') or {}).get('guard')],
                beyond=beyond, reported_outside_criteria=reported)


def load_card(run):
    return json.load(open(os.path.join(run, 'card.json')))


def cmd_pair(a):
    r = compare_cards(load_card(a.a), load_card(a.b))
    for k in ('identical', 'outcome_agrees', 'readings_agree', 'outcome', 'guard', 'where_ended'):
        print(f'{k}: {r[k]}')
    print(f"beyond tolerance: {len(r['beyond'])}")
    for x in r['beyond']:
        print('  ', x)
    print(f"outside the criteria and different: {len(r['reported_outside_criteria'])}")
    for x in r['reported_outside_criteria'][: a.show]:
        print('  ', x)


def cmd_set(a):
    jobs = json.load(open(os.path.join(a.base, f'jobs_{a.arm_a}.json')))
    out = []
    for j in jobs:
        key = os.path.relpath(j['out'], os.path.join(a.base, a.arm_a))
        ra, rb = os.path.join(a.base, a.arm_a, key), os.path.join(a.base, a.arm_b, key)
        row = dict(case=key, problem=j['problem'], test=j['test'], law=j['law'])
        if not (os.path.exists(os.path.join(ra, 'card.json')) and os.path.exists(os.path.join(rb, 'card.json'))):
            row['missing'] = [x for x, r in (('a', ra), ('b', rb)) if not os.path.exists(os.path.join(r, 'card.json'))]
        else:
            row.update(compare_cards(load_card(ra), load_card(rb)))
        out.append(row)
    rep = dict(arm_a=a.arm_a, arm_b=a.arm_b, pairs=len(out), missing=sum('missing' in r for r in out),
               identical=sum(r.get('identical', False) for r in out),
               outcome_agrees=sum(r.get('outcome_agrees', False) for r in out),
               readings_agree=sum(r.get('readings_agree', False) for r in out),
               both_agree=sum(r.get('outcome_agrees', False) and r.get('readings_agree', False) for r in out), rows=out)
    json.dump(rep, open(a.out, 'w'), indent=1, default=str)
    print({k: v for k, v in rep.items() if k != 'rows'})


def cmd_break(a):
    if os.path.exists(a.new):
        shutil.rmtree(a.new)
    os.makedirs(a.new)
    run = json.load(open(os.path.join(a.run, 'run.json')))
    h = run['history']
    if a.how == 'control_1db':
        g = 10 ** (1.0 / 20.0)
        for i, j in enumerate(h['judged']):
            if j:
                h['control_re'][i] = [x * g for x in h['control_re'][i]]
                h['control_im'][i] = [x * g for x in h['control_im'][i]]
                if 'bin_error_db_min' in h:
                    h['bin_error_db_min'][i] = [None if x is None else x + 1.0 for x in h['bin_error_db_min'][i]]
                    h['bin_error_db_max'][i] = [None if x is None else x + 1.0 for x in h['bin_error_db_max'][i]]
    else:
        o = run['outcome']
        if o['class'] == 'completed':
            o.update({'class': 'aborted', 'engine_outcome': 'control_abort', 'guard': 'control_abort',
                      'frequency_hz': h['frequency_hz'][-1], 'time_s': h['time_s'][-1], 'level_db': h['level_db'][-1]})
        else:
            o.update({'class': 'completed', 'engine_outcome': 'completed', 'guard': None, 'frequency_hz': None,
                      'time_s': None, 'level_db': None})
    json.dump(run, open(os.path.join(a.new, 'run.json'), 'w'), separators=(',', ':'))
    prob = os.path.join(a.problems, run['case_id'].split('__')[0])
    q = subprocess.run([sys.executable, os.path.join(SINE, 'sine_scorer.py'), prob, a.new, '--out',
                        os.path.join(a.new, 'card.json')], capture_output=True, text=True)
    print('scorer exit', q.returncode, q.stderr[-300:])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pair'); p.add_argument('a'); p.add_argument('b'); p.add_argument('--show', type=int, default=20)
    s = sub.add_parser('set'); s.add_argument('base'); s.add_argument('arm_a'); s.add_argument('arm_b'); s.add_argument('--out', required=True)
    b = sub.add_parser('break'); b.add_argument('run'); b.add_argument('new'); b.add_argument('--problems', required=True)
    b.add_argument('--how', choices=('control_1db', 'outcome'), required=True)
    a = ap.parse_args(argv)
    {'pair': cmd_pair, 'set': cmd_set, 'break': cmd_break}[a.cmd](a)


if __name__ == '__main__':
    main()
