"""Tests of results.py: checking and summarizing a reported result.

The unit tests build small results from made-up scorecards. The data test rebuilds the baselines' summary
table, law by law, from the study's own scorecards, and needs two things not in this repository:
  OBS_RESULTS        the unpacked results record of the study and its audit, version 0.2
                     (DOI 10.5281/zenodo.24310746), which holds cards/main
  OBS_STUDY_SCORES   study_scores.csv from the data of Hertzforge Journal 2026-009
It is skipped when either is not given.
"""
import csv
import json
import os
import shutil
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import results as rs  # noqa: E402

VERSION, DIGEST = rs.scorer_identity()


def result_json(rid, **over):
    r = {
        'schema': rs.RESULT_SCHEMA,
        'id': rid,
        'suite': 'MIMO Random',
        'set_version': '0.2',
        'law': {'name': 'A test law', 'code_name': 'test_law', 'parameters': {'all': {}}, 'frequency_response': 'measured once and kept'},
        'target': 'line',
        'authors': [{'name': 'A. Author'}],
        'link': 'https://example.org/code',
        'development': 'Developed on structures 30 and 31; never run on the others before this result.',
        'run_as_published': True,
        'run_records': {'url': 'https://example.org/runs.zip', 'sha256': '0' * 64},
        'software': 'The author\'s own code.',
    }
    r.update(over)
    return r


def card(structure, seed, value, protections='opened', completed=True, law='test_law', params=None, target='line', model='Hold',
         digest=DIGEST):
    return {
        'schema': 'hertzforge-benchmark-scorecard/0.2.1-draft',
        'scorer': {'file': 'scorer.py', 'version': VERSION, 'sha256': digest},
        'set': {'name': 'Hertzforge Open Benchmark Set', 'version': '0.2-draft', 'suite': 'MIMO Random', 'tier': 'Frames'},
        'problem': {'id': f'{structure}/{target}', 'structure': structure, 'target': target, 'seed': seed, 'structure_sha256': 'x'},
        'run': {'law': law, 'law_parameters': params or {}, 'model': model},
        'outcome': {'completed': completed, 'protections': protections},
        'scored': completed,
        'card': {'c_b3_rms': value, 'h_b3_rms': value + 1, 'level_mean': -value / 10, 'worst_over_b3': 10 * value},
        'stress': {'rms': value + 2, 'bias': value / 2},
    }


def make(tmp_path, rid='test-law', cards=(), **over):
    folder = tmp_path / rid
    (folder / 'scorecards').mkdir(parents=True)
    (folder / 'result.json').write_text(json.dumps(result_json(rid, **over)))
    for i, c in enumerate(cards):
        (folder / 'scorecards' / f'run{i:03d}.json').write_text(json.dumps(c))
    return str(folder)


def summarize(folder):
    assert rs.main(['summarize', folder]) == 0
    return list(csv.DictReader(open(os.path.join(folder, 'summary.csv'))))


def test_parse_structure():
    assert rs.parse_structure('frame-12x6') == ('6x12', 'more_controls', 'none', None)
    assert rs.parse_structure('frame-6x6-strong') == ('6x6', 'equal', 'strong', None)
    assert rs.parse_structure('frame-24x12-realistic_soft') == ('12x24', 'more_controls', 'realistic', 'soft')
    with pytest.raises(rs.Fail):
        rs.parse_structure('modal-4x4')


def test_summary_means_over_systems_then_median_over_structures(tmp_path):
    cards = [card('frame-12x6', 0, 2.0), card('frame-12x3', 0, 4.0),   # structure 0: mean 3.0
             card('frame-12x6', 1, 5.0), card('frame-12x3', 1, 7.0),   # structure 1: mean 6.0
             card('frame-6x6', 0, 1.0)]                                 # square, its own row
    folder = make(tmp_path, cards=cards)
    rows = summarize(folder)
    more = next(r for r in rows if r['rigs'] == 'more_controls')
    assert more['control_db'] == '4.50' and more['n'] == '2'
    assert more['stress_db'] == '6.50' and more['stress_q1'] == '5.75' and more['stress_q3'] == '7.25'
    square = next(r for r in rows if r['rigs'] == 'equal')
    assert square['control_db'] == '1.00' and square['structure'] == 'none'
    assert rs.check(folder, quiet=True)


def test_widened_run_wins_and_stopped_runs_are_not_results(tmp_path):
    cards = [card('frame-12x6', 0, 9.0, protections='default'), card('frame-12x6', 0, 3.0, protections='opened'),
             card('frame-12x6', 1, 4.0, protections='default'), card('frame-12x6', 1, 8.0, protections='opened', completed=False),
             card('frame-12x6', 2, 6.0, protections='opened', completed=False)]
    rows = summarize(make(tmp_path, cards=cards))
    assert rows[0]['n'] == '2' and rows[0]['control_db'] == '3.50'


def test_diagnostic_problems_are_checked_not_summarized(tmp_path):
    rows = summarize(make(tmp_path, cards=[card('frame-12x6-strong', 0, 2.0), card('frame-12x6-strong_soft', 0, 9.0)]))
    assert [(r['structure'], r['control_db']) for r in rows] == [('strong', '2.00')]


@pytest.mark.parametrize('bad, message', [
    (dict(law='other_law'), 'law "other_law"'),
    (dict(target='banded'), 'target "banded"'),
    (dict(params={'x': 1}), 'law parameters'),
    (dict(model='Update { blend: 0.5 }'), 'frequency response'),
    (dict(digest='f' * 64), "this repository's scorer.py"),
])
def test_a_scorecard_that_does_not_match_the_result_is_refused(tmp_path, bad, message):
    folder = make(tmp_path, cards=[card('frame-12x6', 0, 2.0, **bad)])
    with pytest.raises(rs.Fail, match=message):
        rs.check(folder, quiet=True)


def test_per_class_parameters(tmp_path):
    params = {'more_controls': {'amount': 0.1}, 'equal': {'amount': 0.01}}
    law = dict(result_json('x')['law'], parameters=params)
    folder = make(tmp_path, law=law, cards=[card('frame-12x6', 0, 2.0, params={'amount': 0.1}), card('frame-6x6', 0, 2.0, params={'amount': 0.01})])
    summarize(folder)
    assert rs.check(folder, quiet=True)
    bad = make(tmp_path, rid='bad', law=law, cards=[card('frame-6x6', 0, 2.0, params={'amount': 0.1})])
    with pytest.raises(rs.Fail, match='parameters'):
        rs.check(bad, quiet=True)


def test_twice_with_the_same_protections_is_refused(tmp_path):
    folder = make(tmp_path, cards=[card('frame-12x6', 0, 2.0), card('frame-12x6', 0, 3.0)])
    with pytest.raises(rs.Fail, match='scored twice'):
        rs.check(folder, quiet=True)


def test_summary_must_match_the_scorecards(tmp_path):
    folder = make(tmp_path, cards=[card('frame-12x6', 0, 2.0)])
    with pytest.raises(rs.Fail, match='no summary.csv'):
        rs.check(folder, quiet=True)
    summarize(folder)
    with open(os.path.join(folder, 'summary.csv'), 'a') as fh:
        fh.write('tampered\n')
    with pytest.raises(rs.Fail, match='not what the scorecards give'):
        rs.check(folder, quiet=True)


@pytest.mark.parametrize('field, value, message', [
    ('id', 'elsewhere', 'folder'),
    ('run_as_published', False, 'different experiment'),
    ('link', 'http://example.org', 'https'),
    ('run_records', {'url': 'https://example.org/x.zip'}, 'run_records.sha256'),
    ('authors', [{}], 'name'),
])
def test_result_json_is_complete(tmp_path, field, value, message):
    folder = make(tmp_path, cards=[card('frame-12x6', 0, 2.0)], **{field: value})
    with pytest.raises(rs.Fail, match=message):
        rs.check(folder, quiet=True)


def test_the_template_is_a_valid_result_json():
    t = json.load(open(os.path.join(HERE, '..', 'results', 'TEMPLATE', 'result.json')))
    for key, kind in rs.REQUIRED.items():
        assert isinstance(t[key], kind), key
    assert t['schema'] == rs.RESULT_SCHEMA


def test_check_all_on_the_repository():
    assert rs.main(['check-all']) == 0


STUDY = os.environ.get('OBS_RESULTS')
SCORES = os.environ.get('OBS_STUDY_SCORES')


@pytest.mark.skipif(not (STUDY and SCORES), reason='needs OBS_RESULTS and OBS_STUDY_SCORES')
@pytest.mark.parametrize('law', ['pinv_closed', 'pinv_open', 'shape_svd', 'buzz_match', 'min_drive', 'band_average'])
def test_the_baselines_summary_is_rebuilt_from_the_studys_scorecards(tmp_path, law):
    """Each law at its defaults, as a reported result: summarize gives the published table's rows, digit for digit."""
    folder = tmp_path / law
    (folder / 'scorecards').mkdir(parents=True)
    target = 'banded' if law == 'band_average' else 'line'
    main = os.path.join(STUDY, 'cards', 'main')
    for name in sorted(os.listdir(main)):
        d = json.load(open(os.path.join(main, name)))
        j = d['job']
        if j['law'] == law and j['setting'] == 'default' and j['frf'] == 'hold':
            shutil.copy(os.path.join(main, name), folder / 'scorecards' / name)
    r = result_json(law, target=target, law=dict(result_json('x')['law'], code_name=law))
    (folder / 'result.json').write_text(json.dumps(r))
    got = [row for row in summarize(str(folder))]
    want = [row for row in csv.DictReader(open(SCORES)) if row['law'] == law and row['structures'] == 'all']
    key = lambda row: (row['rigs'], row['structure'])
    assert sorted(got, key=key) == sorted(want, key=key)
    assert rs.check(str(folder), quiet=True)
