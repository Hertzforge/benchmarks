"""Tests of the equalized step in reference_model (the study, 2026-11-14): simulate_level and discrete_frf with a
structure whose simulator block carries step {"kind": "equalized", "band_hz", "half_taps"}.

Run with run_tests.sh. Each test states what it would catch. The last test plays the same drive through the software
simulator (the obs_play example, research/obs-runner) when its binary is present, and is skipped otherwise.
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
from test_stepper import FS, level, steady_tones, structure

EQ = dict(kind='equalized', band_hz=2000.0, half_taps=2)
PLAY = os.environ.get('OBS_PLAY', 'obs_play')


def three_mode(step=None, extras=False, rules=False):
    rng = np.random.default_rng(3)
    modes = []
    for k, (f, z) in enumerate([(180.0, 0.03), (410.0, 0.02), (1505.0, 0.015)]):
        m = dict(f_hz=f, zeta=z, drive_shape=list(rng.standard_normal(2)), output_shape=list(rng.standard_normal(3)),
                 reference_amplitude=1e-3)
        if rules:
            m['level'] = [level(drop=0.05, gain=1.0), level(rise=0.08), level(drop=0.02, rise=0.03, gain=0.5)][k]
        modes.append(m)
    sim = dict(feedthrough=0.01, residual=list(np.linspace(-0.2, 0.3, 6)), transport_delay_samples=3,
               output_delays_s=[0.0, 1.4 / FS, 7.25 / FS]) if extras else {}
    if step is not None:
        sim['step'] = step
    return structure(modes, 3, 2, law=dict(kind='jointed', decay_cycles=5.0) if rules else None, sim=sim)


def test_taps_undo_the_hold():
    """The FIR is symmetric, and its zero-phase response times sinc^2(w/2) is one within 0.05 dB up to the band. Catches
    a wrong weight, a wrong grid, or taps not centred."""
    c = rm.equalizer_taps(2000.0 / FS, 2)
    assert c.size == 5 and np.allclose(c, c[::-1], rtol=0, atol=0)
    w = 2 * np.pi * np.linspace(1.0, 2000.0, 500) / FS
    G = c[2] + 2 * sum(c[2 + k] * np.cos(k * w) for k in (1, 2))
    err = np.abs(20 * np.log10(G * (np.sin(w / 2) / (w / 2)) ** 2))
    assert err.max() < 0.05, err.max()


def test_equalized_linear_equals_discrete_frf():
    """A linear structure with the equalized step, stepped by simulate_level, gives at steady state the discrete_frf
    of the same step, with residual, feedthrough, transport delay and fractional output delays. Catches the FIR on the
    wrong part (direct or restoring), a delay of the wrong length on the residual or feedthrough, or a frf that does
    not include the step."""
    st = three_mode(EQ, extras=True)
    freqs = [200.0, 1497.0]
    Y, U = steady_tones(st, freqs, 0.5)
    H = rm.discrete_frf(st, freqs, FS)
    for j, f in enumerate(freqs):
        err = np.abs(Y[:, j] / U[j] - H[j, :, j]) / np.abs(H[j, :, j])
        assert err.max() < 1e-8, err


def test_equalized_matches_continuous_with_two_sample_delay():
    """With every mode inside the band, the equalized step's small-signal response is the continuous model times
    exp(-j w 2 dt) within 0.01 dB (median) and 0.15 dB (largest) from 20 to 2000 Hz, and within 1 degree of that
    delay; the hold departs by more than 0.3 dB near 1.5 kHz. Catches an equalizer that does nothing, or a delay
    that is not two samples."""
    f = np.linspace(20.0, 2000.0, 1981)
    st_eq, st_h = three_mode(EQ), three_mode(None)
    Hc = rm.frf(st_eq, f)
    a = rm.discrete_frf(st_eq, f, FS) / (Hc * np.exp(-2j * np.pi * f / FS * 2)[:, None, None])
    db = np.abs(20 * np.log10(np.abs(a)))
    assert np.median(db) < 0.01 and db.max() < 0.15, (np.median(db), db.max())
    assert np.degrees(np.abs(np.angle(a))).max() < 1.0
    near = np.abs(f - 1505.0) < 5
    hold = np.abs(20 * np.log10(np.abs(rm.discrete_frf(st_h, f, FS) / Hc)))
    assert hold[near].max() > 0.3


def test_hold_named_or_absent_is_the_same():
    """A structure whose step is {"kind": "triangle_hold"} steps to the same bits as one with no step. Catches the old
    step changed by the new code."""
    d = np.random.default_rng(0).standard_normal((4000, 2)) * 50.0
    a = rm.simulate_level(three_mode(None, extras=True, rules=True), d, FS).outputs
    b = rm.simulate_level(three_mode(dict(kind='triangle_hold'), extras=True, rules=True), d, FS).outputs
    assert np.array_equal(a, b)


def test_unknown_step_refused():
    with pytest.raises(ValueError, match='step kind'):
        rm.simulate_level(three_mode(dict(kind='zoh')), np.zeros((10, 2)), FS)


@pytest.mark.skipif(not os.path.exists(PLAY), reason='the simulator player (obs_play) is not built')
@pytest.mark.parametrize('step', [None, EQ])
def test_matches_simulator_samples(step):
    """The same drive through the software simulator and through simulate_level, with rules, delays, residual and
    feedthrough, at a level that moves every mode: every output sample within 1e-9 of the largest. Catches any
    departure of the restatement from the simulator, on either step."""
    st = three_mode(step, extras=True, rules=True)
    n = 6000
    d = np.random.default_rng(1).standard_normal((n, 2))
    lin = rm.simulate_level(three_mode(step, extras=True), d, FS, keep_modes=True)
    d = d * (1.5e-3 / np.abs(lin.modal).max())                 # peak modal displacement about 1.5 reference amplitudes
    raw = st.raw
    case_structure = dict(kind='jointed', n_outputs=raw['n_outputs'], n_drives=raw['n_drives'], joint=dict(delta=0.0, damping_gain=0.0, decay_cycles=5.0),
                 modes=[dict(f_hz=m['f_hz'], zeta=m['zeta'], drive_shape=m['drive_shape'], output_shape=m['output_shape'],
                             reference_amplitude=m['reference_amplitude'], level=dict(m['level'], damping_cap=m['level'].get('damping_cap', rm.NO_CAP))) for m in raw['modes']],
                 **{k: raw['simulator'][k] for k in ('feedthrough', 'residual', 'transport_delay_samples', 'output_delays_s')})
    if step is not None:
        case_structure['step'] = step
    with tempfile.TemporaryDirectory() as tmp:
        pj, dj, oj = (os.path.join(tmp, x) for x in ('p.json', 'd.f8', 'o.f8'))
        json.dump(case_structure, open(pj, 'w'))
        np.ascontiguousarray(d, '<f8').tofile(dj)
        subprocess.run([PLAY, pj, '--drive', dj, '--out', oj], check=True, capture_output=True)
        ys = np.fromfile(oj, '<f8').reshape(n, 3)
    yn = rm.simulate_level(st, d, FS)
    assert yn.peak_ratio(st).max() > 1.0
    rel = np.abs(ys - yn.outputs).max() / np.abs(ys).max()
    assert rel < 1e-9, rel
