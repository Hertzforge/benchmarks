"""The MIMO Shock suite draft: the model is linear when the level rule is off, the SRS, the pulses, the scorer's
readings on hand-made runs. Needs the Random set's problems for the structure test (OBS_SET, else skipped)."""
import copy
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'shock'))
import shock_model as sm  # noqa: E402
import shock_scorer as scorer  # noqa: E402

RANDOM = os.environ.get('OBS_SET', os.path.expanduser('~/dev/obs-data.noindex/problems'))
ONE = os.path.join(RANDOM, 'frame-6x6-realistic_s00', 'structure.json')


@pytest.mark.skipif(not os.path.exists(ONE), reason='the Random set is not here')
def test_linear_when_the_rule_is_off():
    st = sm.rm.load_structure(ONE)
    raw = copy.deepcopy(st.raw); raw['level_law'] = None
    for m in raw['modes']:
        m.pop('level', None)
    lin = sm.rm.Structure(raw)
    rng = np.random.default_rng(1)
    d = np.zeros((lin.n_drives, sm.N)); d[:, 2000:2400] = rng.standard_normal((lin.n_drives, 400))
    y1, _ = sm.play(lin, d)
    y2, _ = sm.play(lin, 3.0 * d)
    assert np.max(np.abs(y2 - 3.0 * y1)) <= 1e-9 * np.max(np.abs(y2))        # homogeneous
    # the from-rest play is the linear convolution with the stepped model's response
    f2 = np.fft.rfftfreq(2 * sm.N, 1 / sm.FS)
    H2 = sm.rm.discrete_frf(lin, f2, sm.FS, rows=sm.controls(lin))
    yl = np.fft.irfft(np.einsum('lcd,dl->cl', H2, np.fft.rfft(d, 2 * sm.N, axis=1)), n=2 * sm.N, axis=1)[:, :sm.N]
    assert np.linalg.norm(y1 - yl) / np.linalg.norm(yl) < 0.02
    # and the structure with its rule is not linear at full level, but is at a small one
    yn, _ = sm.play(st, 1e-6 * d)
    assert np.max(np.abs(yn - 1e-6 * y1)) <= 1e-3 * np.max(np.abs(1e-6 * y1))     # the rule acts in proportion to r


def test_srs_numpy_equals_scipy_and_half_sine():
    x = sm.classical('half_sine')[0][None]
    a = sm.srs(x, sm.FS, sm.NAT3)
    import builtins
    real = builtins.__import__
    def no_scipy(name, *k, **kw):
        if name.startswith('scipy'):
            raise ImportError
        return real(name, *k, **kw)
    builtins.__import__ = no_scipy
    try:
        b = sm.srs(x, sm.FS, sm.NAT3)
    finally:
        builtins.__import__ = real
    assert np.max(np.abs(a - b)) <= 1e-9 * a.max()
    assert abs(a[0, -1] - 1.0) < 0.05                    # at 2 kHz an 11 ms pulse's SRS is its peak


@pytest.mark.parametrize('shape', ['half_sine', 'sawtooth'])
def test_classical_compensated(shape):
    w, ideal, _ = sm.classical(shape)
    v = np.cumsum(w) / sm.FS
    assert abs(v[-1]) < 1e-9 and abs(np.sum(v) / sm.FS) < 1e-9
    assert abs(ideal.max() - 1.0) < 0.02                  # the peak to within one sample


def run_of(records, end='completed'):
    return dict(problem='p', kind='srs', law='x', end=end, records=records)


def rec(i, level, e, kind='pulse'):
    e = np.asarray(e, float)
    return dict(index=i, kind=kind, level_db=level, over_db=float(e.max()), worst_db=float(np.abs(e).max()),
                abs_db=level + float(e.max()), over12_db=float(e.max()), errors_db=e.tolist(), drive_rms=1.0)


def test_readings_clean_run():
    ok = [[0.5, -0.5], [1.0, -1.0]]
    R = [rec(0, -12, ok), rec(1, -6, ok), rec(2, 0, ok), rec(3, 0, ok), rec(4, 0, ok)]
    c = scorer.score_run(run_of(R))
    assert c['strict']['passed'] and c['ends']['passed'] and c['level']['passed'] and c['exposure']['passed']
    assert c['card']['pattern_grade'] == 'A' and abs(c['ends']['margin_db'] - 0.5) < 1e-9
    assert c['ends_pm3']['passed'] and abs(c['ends_pm3']['margin_db'] - 2.0) < 1e-9


def test_readings_disagree_on_purpose():
    # every point 2 dB over and one point 4 dB over at full level: strict and ends fail, the Level reading passes
    # on its limit line and fails on its mean; exposure fails by 1 dB
    e = [[2.0, 2.0], [2.0, 4.0]]
    R = [rec(0, -12, [[0, 0], [0, 0]]), rec(1, 0, e), rec(2, 0, e), rec(3, 0, e)]
    c = scorer.score_run(run_of(R, end='not converged'))
    assert not c['strict']['passed'] and abs(c['strict']['broke_db_over_limit']['3_f1_within_3_over'] - 1.0) < 1e-9
    assert not c['ends']['passed'] and c['ends']['controls_out'] == 1
    assert c['level']['broke'] == ['mean'] and abs(c['level']['mean_db'] - 2.5) < 1e-9
    assert not c['exposure']['passed'] and abs(c['exposure']['margin_db'] + 1.0) < 1e-9
    # a run that over-tests at a low level but ends clean: ends and Level pass, strict and exposure do not
    R = [rec(0, -12, [[16.0, 0], [0, 0]]), rec(1, 0, [[0, 0], [0, 0]]), rec(2, 0, [[0, 0], [0, 0]]), rec(3, 0, [[0, 0], [0, 0]])]
    c = scorer.score_run(run_of(R))
    assert c['ends']['passed'] and c['level']['passed'] and not c['strict']['passed'] and not c['exposure']['passed']
    assert c['exposure']['where']['level_db'] == -12


def test_no_full_level():
    c = scorer.score_run(run_of([rec(0, -12, [[5.0]])], end='not converged'))
    assert not c['ends']['passed'] and not c['level']['passed'] and c['card'] is None


def test_516_srs_tolerance():
    # Method 516.8: -1.5/+3 dB on at least 90 % of the points, the rest within -3/+6 dB
    e = np.zeros((1, 20))
    e[0, :2] = 5.0                                        # 10 % of the points at +5: inside, by 1 dB (the outer bound)
    assert sm.within_516(e) and abs(sm.excess_516(e)[0] + 1.0) < 1e-12
    e[0, 2] = 5.0                                         # 15 %: the point at the 90 % share is 2 dB over +3
    assert not sm.within_516(e) and abs(sm.excess_516(e)[0] - 2.0) < 1e-12
    e = np.zeros((1, 20)); e[0, 0] = 7.0                  # one point over +6: out by 1 dB
    assert abs(sm.excess_516(e)[0] - 1.0) < 1e-12
    assert not sm.within_516(np.full((1, 20), -2.0))      # asymmetric: -2 dB everywhere is out (the old +/-3 passed it)
    assert sm.within_516(np.full((1, 20), 2.9))
    assert sm.NAT516.size == 93 and abs(sm.NAT516[0] - 10.0) < 1e-9 and abs(sm.NAT516[-1] - 1995.262) < 1e-3


def test_effective_duration():
    # Method 516.8 Annex A 1.3.2: the minimum time holding every magnitude above APk / CF, CF 3 unless given.
    # Peak 5 at sample 150 (threshold 5/3); 2.0 over 100 to 199 is above it; -1.7 at 50 is above it in magnitude;
    # 1.5 at 300 is under it (it was above draft 0.2's 10 %).
    x = np.zeros((1, 8192)); x[0, 100:200] = 2.0; x[0, 150] = 5.0; x[0, 50] = -1.7; x[0, 300] = 1.5
    assert abs(sm.effective_duration(x, fs=1000.0)[0] - 0.150) < 1e-12        # samples 50 to 199
    assert abs(sm.effective_duration(x, fs=1000.0, crest_factor=10.0)[0] - 0.251) < 1e-12   # 50 to 300
    assert sm.TE_CREST_FACTOR == 3.0


def test_duration_reading():
    ok = [[0.5, -0.5], [1.0, -1.0]]
    R = [rec(0, -12, ok), rec(1, 0, ok), rec(2, 0, ok), rec(3, 0, ok)]
    R[-1]['te_ratio'] = [1.0, 1.25]
    c = scorer.score_run(run_of(R))
    d = c['duration']
    assert not d['passed'] and d['share_te_0p8_1p2'] == 0.5 and abs(d['te_ratio_median'] - 1.125) < 1e-12
    R[-1]['te_ratio'] = [0.85, 1.15]
    assert scorer.score_run(run_of(R))['duration']['passed']
    c = scorer.score_run(run_of([rec(0, -12, [[5.0]])], end='not converged'))
    assert c['duration'] == dict(passed=False, why='no full-level pulse')


@pytest.mark.parametrize('shape', ['half_sine', 'sawtooth'])
def test_classical_reading_of_the_pulse_itself(shape):
    import harness as hz
    w, _, _ = sm.classical(shape)
    y = np.tile(w, (2, 1))
    r = hz.classical_band(y, y, dict(kind=shape, scale=1.0), 1.0)
    assert max(r['band_pct']) == 0 and max(map(abs, r['duration_pct'])) == 0 and max(map(abs, r['velocity_pct'])) == 0
    assert max(r['compensation_pct']) < 20.0              # the pulse's own compensation is inside the baseline band
    if shape == 'sawtooth':
        assert 7.5 < r['post_slope_pct'][0] < 10.0        # the 10 % TD return reaches 0.2 A at 8 % TD
