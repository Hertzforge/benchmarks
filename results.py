"""Reported results on the Hertzforge Open Benchmark Set: check one, and summarize it the way the baselines are.

One file. Needs numpy and the Python standard library, nothing else.

    python results.py check RESULT_FOLDER [--set SET_FOLDER] [--runs RUNS_FOLDER]
    python results.py summarize RESULT_FOLDER [--out summary.csv]
    python results.py check-all [RESULTS_ROOT]

A RESULT_FOLDER (results/<suite>-v<version>/<result id>/, see results/README.md) holds result.json, which says
what was run and by whom, scorecards/, one scorecard per run as scorer.py wrote it, and summary.csv, which
`summarize` writes and `check` holds to.

check      Without data: result.json is complete; every scorecard was written by this repository's scorer.py
           (version and SHA-256), on this suite and version, for the law, target and frequency response the
           result declares; no case is scored twice; summary.csv is what `summarize` writes.
           With --set (the folder holding the set's problems/): every scorecard's structure hash matches the
           published problem.
           With --runs (the unpacked run records the result names): every run is scored again with scorer.py
           and every number of the new scorecard must equal the submitted one.
summarize  The summary table of the result, in the columns of the baselines' table (study_scores.csv of
           Hertzforge Journal 2026-009), by the same rules:
             - the result of a case is its completed, scored run; a case run twice, once with the
               controller's protections at their defaults and once with them widened, takes the widened run
               where it completed, else the default one;
             - systems with as many shakers as control gauges and systems with more control gauges than
               shakers are never pooled; a structure's value in a class is the mean over that class's systems;
             - the median and quartiles (numpy, linear) are over structures.
           Only stage 1 and the two stage 2 sizes are summarized; the diagnostic problems with one effect
           (_soft, _hard) are checked but not summarized.
check-all  `check` without data on every result under RESULTS_ROOT (default results/). This is what CI runs.

Copyright 2026 Kevin Cross. MIT license.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_SCHEMA = 'hertzforge-benchmark-result/0.1'
SCORECARD_PREFIX = 'hertzforge-benchmark-scorecard/'
SUMMARY_COLUMNS = ['rigs', 'structure', 'law', 'structures', 'control_db', 'held_out_db', 'level_db', 'worst_over_test_db',
                   'n', 'stress_db', 'stress_q1', 'stress_q3', 'stress_bias_db']
LINES = {   # the scorecard's lines the summary carries, as the baselines' study reads them
    'control_db': lambda d: d['card']['c_b3_rms'],
    'held_out_db': lambda d: d['card']['h_b3_rms'],
    'level_db': lambda d: d['card']['level_mean'],
    'worst_over_test_db': lambda d: d['card']['worst_over_b3'],
    'stress_db': lambda d: d['stress']['rms'],
    'stress_bias_db': lambda d: d['stress']['bias'],
}
VARIANTS = ('none', 'realistic', 'strong')
CLASSES = ('more_controls', 'equal')
STRUCTURE = re.compile(r'^frame-(\d+)x(\d+)(?:-(realistic|strong)(?:_(soft|hard))?)?$')
REQUIRED = {
    'schema': str, 'id': str, 'suite': str, 'set_version': str, 'law': dict, 'target': str, 'authors': list,
    'link': str, 'development': str, 'run_as_published': bool, 'run_records': dict, 'software': str,
}
LAW_REQUIRED = {'name': str, 'code_name': str, 'parameters': dict, 'frequency_response': str}
FREQUENCY_RESPONSE = {'measured once and kept': 'Hold', 'measured again': 'Update'}


class Fail(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def scorer_identity():
    """The version and SHA-256 of the scorer.py beside this file: the only scorer a result may use."""
    path = os.path.join(HERE, 'scorer.py')
    m = re.search(r"^SCORER_VERSION = '([^']+)'", open(path).read(), re.M)
    return m.group(1), sha256(path)


def parse_structure(sid):
    """frame-12x6-realistic -> (rig '6x12', class, variant 'realistic', effect None). Control gauges come first in the id."""
    m = STRUCTURE.match(sid or '')
    if not m:
        raise Fail(f'structure "{sid}" is not a Frames structure identifier')
    controls, shakers, size, effect = int(m.group(1)), int(m.group(2)), m.group(3) or 'none', m.group(4)
    return f'{shakers}x{controls}', ('equal' if controls == shakers else 'more_controls'), size, effect


def load_result(folder):
    path = os.path.join(folder, 'result.json')
    if not os.path.isfile(path):
        raise Fail(f'{folder}: no result.json')
    r = json.load(open(path))
    for key, kind in REQUIRED.items():
        if not isinstance(r.get(key), kind):
            raise Fail(f'result.json: "{key}" is missing or not a {kind.__name__}')
    for key, kind in LAW_REQUIRED.items():
        if not isinstance(r['law'].get(key), kind):
            raise Fail(f'result.json: "law.{key}" is missing or not a {kind.__name__}')
    if r['schema'] != RESULT_SCHEMA:
        raise Fail(f'result.json: schema is "{r["schema"]}", expected "{RESULT_SCHEMA}"')
    if r['id'] != os.path.basename(os.path.normpath(folder)):
        raise Fail(f'result.json: id "{r["id"]}" is not the folder\'s name')
    if r['target'] not in ('line', 'banded'):
        raise Fail('result.json: target must be "line" or "banded"')
    if r['law']['frequency_response'] not in FREQUENCY_RESPONSE:
        raise Fail(f'result.json: law.frequency_response must be one of {", ".join(FREQUENCY_RESPONSE)}')
    if not r['authors'] or not all(isinstance(a, dict) and a.get('name') for a in r['authors']):
        raise Fail('result.json: every author needs a name')
    for key in ('url', 'sha256'):
        if not isinstance(r['run_records'].get(key), str) or not r['run_records'][key]:
            raise Fail(f'result.json: run_records.{key} is required (the archive of the run folders, and its SHA-256)')
    if not r['run_as_published']:
        raise Fail('result.json: a problem changed from the published one is a different experiment and is not listed')
    if not r['link'].startswith('https://'):
        raise Fail('result.json: link must be an https address of the article or the code')
    params = r['law']['parameters']
    if set(params) - {'all', 'more_controls', 'equal'}:
        raise Fail('result.json: law.parameters is keyed by "all", or by "more_controls" and "equal"')
    return r


def expected_parameters(result, cls):
    p = result['law']['parameters']
    return p.get(cls, p.get('all', {}))


def load_cards(folder):
    cards_dir = os.path.join(folder, 'scorecards')
    if not os.path.isdir(cards_dir):
        raise Fail(f'{folder}: no scorecards/ folder')
    names = sorted(n for n in os.listdir(cards_dir) if n.endswith('.json'))
    if not names:
        raise Fail(f'{cards_dir}: no scorecards')
    return [(n[:-5], json.load(open(os.path.join(cards_dir, n)))) for n in names]


def check_card(name, d, result, scorer):
    where = f'scorecards/{name}.json'
    if not str(d.get('schema', '')).startswith(SCORECARD_PREFIX):
        raise Fail(f'{where}: not a scorecard')
    version, digest = scorer
    if (d.get('scorer') or {}).get('version') != version or (d.get('scorer') or {}).get('sha256') != digest:
        raise Fail(f'{where}: not written by this repository\'s scorer.py (version {version}, SHA-256 {digest[:12]}...)')
    s = d.get('set') or {}
    if s.get('suite') != result['suite'] or not str(s.get('version', '')).startswith(result['set_version']):
        raise Fail(f'{where}: suite "{s.get("suite")}" version "{s.get("version")}", not the result\'s {result["suite"]} {result["set_version"]}')
    pb, run = d.get('problem') or {}, d.get('run') or {}
    if pb.get('target') != result['target']:
        raise Fail(f'{where}: target "{pb.get("target")}", the result declares "{result["target"]}"')
    if run.get('law') != result['law']['code_name']:
        raise Fail(f'{where}: law "{run.get("law")}", the result declares "{result["law"]["code_name"]}"')
    rig, cls, variant, effect = parse_structure(pb.get('structure'))
    if (run.get('law_parameters') or {}) != expected_parameters(result, cls):
        raise Fail(f'{where}: law parameters {run.get("law_parameters")} are not the ones the result declares for {cls}')
    mode = FREQUENCY_RESPONSE[result['law']['frequency_response']]
    if not str(run.get('model') or '').startswith(mode):
        raise Fail(f'{where}: frequency response "{run.get("model")}", the result declares "{result["law"]["frequency_response"]}"')
    if not isinstance(pb.get('seed'), int):
        raise Fail(f'{where}: no seed')
    return rig, cls, variant, effect


def pick_runs(cards, result, scorer):
    """The result of each case: its completed, scored run; widened protections first where both were run."""
    cases = {}
    for name, d in cards:
        rig, cls, variant, effect = check_card(name, d, result, scorer)
        key = (d['problem']['structure'], d['problem']['seed'])
        slot = 'opened' if (d.get('outcome') or {}).get('protections') == 'opened' else 'default'
        entry = cases.setdefault(key, {'rig': rig, 'cls': cls, 'variant': variant, 'effect': effect, 'seed': d['problem']['seed']})
        if slot in entry:
            raise Fail(f'scorecards/{name}.json: {key[0]} seed {key[1]} is scored twice with the same protections')
        entry[slot] = d
    picked = []
    for entry in cases.values():
        for slot in ('opened', 'default'):
            d = entry.get(slot)
            if d is not None and (d.get('outcome') or {}).get('completed') and d.get('scored'):
                picked.append({**{k: entry[k] for k in ('rig', 'cls', 'variant', 'effect', 'seed')}, 'card': d})
                break
    return picked, len(cases)


def summary_rows(result, picked):
    rows = []
    law = result['law']['code_name']
    for cls in CLASSES:
        for variant in VARIANTS:
            per = {}
            for p in picked:
                if p['cls'] == cls and p['variant'] == variant and p['effect'] is None:
                    per.setdefault(p['seed'], []).append(p)
            if not per:
                continue
            row = [cls, variant, law, 'all']
            for line, get in LINES.items():
                v = np.array([float(np.mean([get(p['card']) for p in runs])) for runs in per.values()])
                if line == 'stress_db':
                    row += [int(v.size), f'{np.median(v):.2f}', f'{np.percentile(v, 25):.2f}', f'{np.percentile(v, 75):.2f}']
                else:
                    row += [f'{np.median(v):.2f}']
            rows.append(row)
    return rows


def summary_text(result, picked):
    out = io.StringIO()
    w = csv.writer(out, lineterminator='\n')
    w.writerow(SUMMARY_COLUMNS)
    w.writerows(summary_rows(result, picked))
    return out.getvalue()


def find_problem(set_folder, structure, seed):
    name = f'{structure}_s{seed:02d}'
    for sub in sorted(os.listdir(set_folder)):
        cand = os.path.join(set_folder, sub, name)
        if sub.startswith('problems') and os.path.isfile(os.path.join(cand, 'problem.json')):
            return cand
    raise Fail(f'{name}: not found under {set_folder}/problems*/')


def numbers_equal(a, b, path=''):
    """Every number in a equals the one in b; returns the first path that differs, or None."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b:
                return f'{path}.{k} (missing)'
            bad = numbers_equal(a[k], b[k], f'{path}.{k}')
            if bad:
                return bad
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f'{path} (length)'
        for i, (x, y) in enumerate(zip(a, b)):
            bad = numbers_equal(x, y, f'{path}[{i}]')
            if bad:
                return bad
        return None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            return f'{path} ({a} against {b})'
        return None
    return None if a == b else f'{path} ({a!r} against {b!r})'


def check(folder, set_folder=None, runs_folder=None, quiet=False):
    say = (lambda *a: None) if quiet else print
    result = load_result(folder)
    scorer = scorer_identity()
    cards = load_cards(folder)
    picked, n_cases = pick_runs(cards, result, scorer)
    if not picked:
        raise Fail(f'{folder}: no case has a completed, scored run')
    say(f'{result["id"]}: {len(cards)} scorecards, {n_cases} cases, {len(picked)} with a result')
    summary_path = os.path.join(folder, 'summary.csv')
    if not os.path.isfile(summary_path):
        raise Fail(f'{folder}: no summary.csv; write it with: python results.py summarize {folder}')
    if open(summary_path).read() != summary_text(result, picked):
        raise Fail(f'{folder}: summary.csv is not what the scorecards give; write it again with: python results.py summarize {folder}')
    stages = sorted({p['variant'] for p in picked})
    if stages == ['none']:
        say('  stage 1 only: listed as such')
    if set_folder:
        for name, d in cards:
            pb = d['problem']
            prob = json.load(open(os.path.join(find_problem(set_folder, pb['structure'], pb['seed']), 'problem.json')))
            if prob['structure']['sha256'] != pb['structure_sha256']:
                raise Fail(f'scorecards/{name}.json: the structure hash is not the published problem\'s')
        say(f'  every structure hash matches the set at {set_folder}')
    if runs_folder:
        sys.path.insert(0, HERE)
        import scorer as sc
        for name, d in cards:
            run = os.path.join(runs_folder, name)
            if not os.path.isdir(run):
                raise Fail(f'{run}: no run folder for scorecards/{name}.json')
            pb = d['problem']
            fresh = sc.score(find_problem(set_folder, pb['structure'], pb['seed']), run, pb['target'])
            bad = numbers_equal(json.loads(json.dumps(fresh)), d)
            if bad:
                raise Fail(f'scorecards/{name}.json: scored again, {bad.lstrip(".")} differs')
        say(f'  every run scored again: every number matches')
    say('  passed')
    return result, picked


def main(argv=None):
    ap = argparse.ArgumentParser(description='Check and summarize a reported result on the Open Benchmark Set.')
    ap.add_argument('cmd', choices=('check', 'summarize', 'check-all'))
    ap.add_argument('folder', nargs='?')
    ap.add_argument('--set', default=os.environ.get('OBS_SET'), help='the folder holding the set\'s problems/')
    ap.add_argument('--runs', default=None, help='the unpacked run records, one folder per scorecard')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    try:
        if a.cmd == 'check-all':
            root = a.folder or os.path.join(HERE, 'results')
            found = 0
            for suite in sorted(os.listdir(root)):
                sdir = os.path.join(root, suite)
                if not os.path.isdir(sdir) or suite == 'TEMPLATE':
                    continue
                for rid in sorted(os.listdir(sdir)):
                    if os.path.isfile(os.path.join(sdir, rid, 'result.json')):
                        check(os.path.join(sdir, rid))
                        found += 1
            print(f'{found} reported results checked')
            return 0
        if not a.folder:
            ap.error('a result folder is required')
        if a.runs and not a.set:
            ap.error('--runs needs --set, the folder holding the problems')
        if a.cmd == 'check':
            check(a.folder, a.set, a.runs)
            return 0
        result = load_result(a.folder)
        picked, _ = pick_runs(load_cards(a.folder), result, scorer_identity())
        out = a.out or os.path.join(a.folder, 'summary.csv')
        with open(out, 'w', newline='') as fh:
            fh.write(summary_text(result, picked))
        print(f'wrote {out}')
        return 0
    except Fail as e:
        print(f'results: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
