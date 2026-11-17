"""Tests of damping_cap in reference_model (the study, 2026-11-14; FORMAT.md change note): a mode's damping
stops growing at r = damping_cap. A level object without the field keeps its old meaning (no cap).

Run with run_tests.sh. Each test states what it would catch; every one fails on the reference model before the cap
was added (checked against $OBS_DATA/runs/V/reference_model.before_V2.py).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reference_model as rm
from test_stepper import FS, level, structure

PLAY = os.environ.get('OBS_PLAY', 'obs_play')


def capped(gain=2.0, cap=1.0, ge=1.0, drop=0.04, zeta=0.01, f=100.0, aref=1e-3, with_cap=True, decay_cycles=5.0):
    lv = level(drop=drop, gain=gain, ze=ge)
    if with_cap:
        lv['damping_cap'] = cap
    return structure([dict(f_hz=f, zeta=zeta, drive_shape=[1.0], output_shape=[1.0], reference_amplitude=aref, level=lv)], 1, 1,
                     law=dict(kind='jointed', decay_cycles=decay_cycles))


def test_level_state_damping_flat_past_cap():
    """level_state: zeta0 (1 + gain min(cap, r)^exp), flat past the cap, a cap below 1 included. Catches the cap not
    applied, applied to the frequency instead, or applied after the exponent."""
    st = capped(gain=2.0, cap=0.5, ge=2.0)
    for r, want in [(0.25, 1 + 2 * 0.0625), (0.5, 1.5), (0.9, 1.5), (4.0, 1.5)]:
        assert rm.level_state(st, r)[1][0] == pytest.approx(0.01 * want, rel=1e-14)
    assert rm.level_state(st, 4.0)[0][0] == pytest.approx(96.0, rel=1e-14)       # the drop is untouched


def test_no_cap_is_the_old_rule():
    """A level object without damping_cap gives, bit for bit, zeta0 (1 + gain r^exp) at every r, as before the field
    existed. Catches a default cap of 1 (or any finite value) slipped in for old files."""
    st = capped(with_cap=False)
    r = np.array([0.0, 0.3, 1.0, 2.5, 7.0])
    for x in r:
        assert rm.level_state(st, x)[1][0] == 0.01 * (1.0 + 2.0 * x ** 1.0)
    assert np.isinf(st.rules['damping_cap'][0])


@pytest.mark.parametrize('bad', [0.0, -1.0, float('inf'), float('nan')])
def test_cap_validation(bad):
    """A damping_cap that is not positive and finite is refused, the field named."""
    with pytest.raises(ValueError, match='damping_cap'):
        capped(cap=bad)


def test_runner_structure_carries_cap():
    """to_runner_structure writes damping_cap when the structure file has it, leaves it out when the file has none, and
    from_runner_structure reads it back to the same rule. Catches the cap dropped on the way to the runner."""
    raw = capped(cap=1.5).raw
    raw['simulator'].update(output_delays_s=[])
    st = rm.Structure(raw)
    p = rm.to_runner_structure(st)
    assert p['modes'][0]['level']['damping_cap'] == 1.5
    back = rm.from_runner_structure(p)
    assert back.rules['damping_cap'][0] == 1.5
    raw0 = capped(with_cap=False).raw
    assert 'damping_cap' not in rm.to_runner_structure(rm.Structure(raw0))['modes'][0]['level']


def test_free_decay_damping_flat_past_cap():
    """simulate_level, free decay from r = 3 after one impulse, gain 4 and cap 1: while r is between 1.3 and 2.7 the
    envelope decays at the capped damping 5 zeta0 (within 2 percent), not at zeta0 (1 + 4 r). decay_cycles 1, so the
    peak follower (which can fall no faster than a damping of 1 / (2 pi decay_cycles)) follows the motion. Catches a
    stepper that ignores the cap."""
    st = capped(gain=4.0, cap=1.0, drop=0.0, decay_cycles=1.0)
    n = int(2 * FS)
    d = np.zeros((n, 1))
    d[100, 0] = 3.0 * 1e-3 * 2 * np.pi * 100.0 * FS
    run = rm.simulate_level(st, d, FS, keep_modes=True)
    r = run.env[:, 0] / 1e-3
    t = np.arange(n) / FS
    sel = (r > 1.3) & (r < 2.7) & (t > 0.02)
    slope = np.polyfit(t[sel], np.log(r[sel]), 1)[0]
    zeta = -slope / (2 * np.pi * 100.0)
    assert abs(zeta / 0.05 - 1) < 0.02, zeta


@pytest.mark.skipif(not os.path.exists(PLAY), reason='the simulator player (obs_play) is not built')
@pytest.mark.parametrize('step', [None, dict(kind='equalized', band_hz=2000.0, half_taps=2)])
def test_cap_matches_simulator_samples(step):
    """Two modes with a damping gain and a cap that binds (r up to about 3), noise through the software simulator and
    through simulate_level: every sample within 1e-9 of the largest, both steps."""
    modes = []
    for f, cap in [(150.0, 1.0), (700.0, 0.6)]:
        lv = dict(level(drop=0.03, gain=2.0), damping_cap=cap)
        modes.append(dict(f_hz=f, zeta=0.015, drive_shape=[1.0], output_shape=[1.0, 0.5], reference_amplitude=1.0, level=lv))
    sim = {} if step is None else dict(step=step)
    st = structure(modes, 2, 1, law=dict(kind='jointed', decay_cycles=5.0), sim=sim)
    n = 8000
    d = np.random.default_rng(2).standard_normal((n, 1))
    lin = rm.simulate_level(st, d, FS, keep_modes=True)
    gain = 3.0 / np.abs(lin.modal).max(0).min()
    st.raw['modes'][0]['reference_amplitude'] = 1.0
    case_structure = dict(kind='jointed', n_outputs=2, n_drives=1, joint=dict(delta=0.0, damping_gain=0.0, decay_cycles=5.0),
                 modes=[dict(m, drive_shape=m['drive_shape']) for m in st.raw['modes']],
                 feedthrough=0.0, residual=[], transport_delay_samples=0, output_delays_s=[])
    if step is not None:
        case_structure['step'] = step
    with tempfile.TemporaryDirectory() as tmp:
        pj, dj, oj = (os.path.join(tmp, x) for x in ('p.json', 'd.f8', 'o.f8'))
        json.dump(case_structure, open(pj, 'w'))
        np.ascontiguousarray(gain * d, '<f8').tofile(dj)
        subprocess.run([PLAY, pj, '--drive', dj, '--out', oj], check=True, capture_output=True)
        ys = np.fromfile(oj, '<f8').reshape(n, 2)
    yn = rm.simulate_level(st, gain * d, FS)
    assert yn.peak_ratio(st).min() > 1.0
    rel = np.abs(ys - yn.outputs).max() / np.abs(ys).max()
    assert rel < 1e-9, rel
