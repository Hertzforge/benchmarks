"""The scorer's service line (version 0.2.1, the study part 3): structure_state.service_line sets the state as played
beside the problem's service_state. Run with run_tests.sh.
"""
import importlib.util
import math
import os
import sys
import types

import numpy as np
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import scorer as bs

BEFORE = os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'runs/V/scorer.before_V3.py')
PROB = os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'problems/frame-12x6-realistic_s00')
RUN = os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'runs/V/sim8/s0_6x12_realistic_pinv_closed_default_line_hold_divopen_full')


def stub(service=True):
    lv = lambda rise: dict(drop=0.0 if rise else 0.02, drop_exponent=1.0, rise=rise, rise_exponent=2.0, rise_cap=1.0,
                           damping_gain=0.5, damping_exponent=1.0, damping_cap=1.0)
    modes = [dict(level=lv(0.0)), dict(level=lv(0.004)), dict(level=lv(0.0)), dict(level=lv(0.0))]
    ss = dict(unit='B', member=dict(f0_hz=99.0, frequency_change=0.005, damping_factor=1.2),
              article=dict(modes=3, frequency_change_p10_p50_p90=[-0.03, -0.02, -0.01], damping_factor_p10_p50_p90=[1.2, 1.5, 1.8]))
    return types.SimpleNamespace(meta=dict(service_state=ss) if service else {}, structure=dict(modes=modes),
                                 fn=np.array([50.0, 100.0, 200.0, 400.0]), zeta=np.array([0.01, 0.01, 0.02, 0.02]))


def test_service_line_numbers():
    """Member: the mode with the largest rise, played change 100 (fn/f0 - 1) against the service change; softening: the
    resolved modes with a negative target, percentiles against the service article's. Catches a wrong mode, a sign
    flip, or percent against fraction."""
    pr = stub()
    fn = np.array([49.0, 101.0, 196.0, 396.0]); ze = np.array([0.012, 0.0125, 0.03, 0.03])
    tc = np.array([-0.02, 0.006, -0.02, -0.01]); ok = np.array([True, True, True, False])
    sl = bs.service_line(pr, fn, ze, tc, ok)
    m = sl['member']
    assert m['mode'] == 1 and m['resolved']
    assert m['change_pct_played'] == pytest.approx(1.0) and m['change_pct_service'] == pytest.approx(0.5)
    assert m['difference_pct_points'] == pytest.approx(0.5) and m['zfac_played'] == pytest.approx(1.25)
    assert m['zfac_difference'] == pytest.approx(0.05) and m['change_pct_target'] == pytest.approx(0.6)
    s = sl['softening']
    assert s['modes_played'] == 2                                         # mode 3 not resolved, mode 1 is the member
    assert s['change_pct_played_p10_p50_p90'][1] == pytest.approx(-2.0)   # -2 and -2 percent
    assert s['change_pct_service_p10_p50_p90'] == pytest.approx([-3.0, -2.0, -1.0])
    assert s['difference_pct_points_p10_p50_p90'][1] == pytest.approx(0.0)


def test_no_service_state_is_none():
    assert bs.service_line(stub(False), np.ones(4), np.ones(4), np.zeros(4), np.ones(4, bool)) is None


def _leaves(x, path=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _leaves(v, f'{path}.{k}' if path else k)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _leaves(v, f'{path}[{i}]')
    else:
        yield path, x


@pytest.mark.skipif(not (os.path.exists(BEFORE) and os.path.isdir(PROB) and os.path.exists(os.path.join(RUN, 'manifest.json'))), reason='staged problem, run or the 0.2 copy absent')
def test_staged_problem_old_numbers_unchanged():
    """On a staged level-dependent problem and its simulator run, every value of the 0.2 scorecard is in the 0.2.1 one,
    exactly, and the service line is there with a resolved member."""
    spec = importlib.util.spec_from_file_location('bm_scorer_02', BEFORE)
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    a, b = old.score(PROB, RUN), bs.score(PROB, RUN)
    new = dict(_leaves(b))
    for k, v in _leaves(a):
        if k.startswith('schema') or k.startswith('scorer'):
            continue
        w = new[k]
        assert (isinstance(v, float) and math.isnan(v) and math.isnan(w)) or v == w, k
    sl = b['structure_state']['service_line']
    assert sl['member']['resolved'] and sl['softening']['modes_played'] > 10
