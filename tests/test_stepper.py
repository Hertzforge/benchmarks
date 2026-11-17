"""Closed-form tests of reference_model.simulate_level, the NumPy stepper for a structure that changes with level.

Run with run_tests.sh. Each test states what it would catch.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reference_model as rm

FS = 8192.0


def structure(modes, n_out, n_drv, law=None, sim=None):
    """A small structure.json object. modes: dicts with f_hz, zeta, drive_shape, output_shape, and optionally
    reference_amplitude and level."""
    raw = dict(schema=rm.SCHEMA_STRUCTURE, kind='jointed' if (law or any('level' in m for m in modes)) else 'modal',
               n_outputs=n_out, n_drives=n_drv, outputs=[dict(name=f'o{i}', role='control') for i in range(n_out)],
               drives=[dict(name=f'd{j}') for j in range(n_drv)], level_law=law, modes=modes,
               simulator={**dict(feedthrough=0.0, residual=[], transport_delay_samples=0, output_delays_s=[]), **(sim or {})})
    return rm.Structure(raw)


def level(drop=0.0, rise=0.0, cap=2.0, de=1.0, re=2.0, gain=0.0, ze=1.0):
    return dict(drop=drop, drop_exponent=de, rise=rise, rise_exponent=re, rise_cap=cap, damping_gain=gain, damping_exponent=ze)


def three_mode(with_rule, sim=None):
    rng = np.random.default_rng(3)
    modes = []
    for k, (f, z) in enumerate([(180.0, 0.03), (410.0, 0.02), (905.0, 0.04)]):
        m = dict(f_hz=f, zeta=z, drive_shape=list(rng.standard_normal(2)), output_shape=list(rng.standard_normal(3)),
                 reference_amplitude=1e-3)
        if with_rule:
            m['level'] = [level(drop=0.05, gain=1.0), level(rise=0.08), level(drop=0.02, rise=0.03, gain=0.5)][k]
        modes.append(m)
    return structure(modes, 3, 2, law=dict(kind='jointed', decay_cycles=5.0) if with_rule else None, sim=sim)


def tone_fit(x, f, fs):
    """Complex amplitude X of x(t) = Re(X exp(j 2 pi f t)) + c by least squares (t = sample index / fs)."""
    t = np.arange(x.size) / fs
    A = np.stack([np.cos(2 * np.pi * f * t), -np.sin(2 * np.pi * f * t), np.ones_like(t)], 1)
    c = np.linalg.lstsq(A, x, rcond=None)[0]
    return c[0] + 1j * c[1]


def steady_tones(st, freqs, amp, seconds=4.0, keep=1.0, sim_rows=None):
    """Drive every drive with a sum of tones (one per drive, its own frequency), return the fitted complex output
    amplitude per (output, drive tone) over the last `keep` seconds, and the drive phasors."""
    n = int(seconds * FS)
    t = np.arange(n) / FS
    rng = np.random.default_rng(1)
    ph = rng.uniform(0, 2 * np.pi, len(freqs))
    drive = np.stack([amp * np.cos(2 * np.pi * f * t + p) for f, p in zip(freqs, ph)], 1)
    y = rm.simulate_level(st, drive, FS).outputs
    k = int(keep * FS)
    Y = np.array([[tone_fit(y[-k:, i], f, FS) * np.exp(-2j * np.pi * f * (n - k) / FS) for f in freqs] for i in range(st.n_outputs)])
    U = amp * np.exp(1j * ph)
    return Y, U


def test_linear_equals_discrete_frf():
    """A linear structure stepped by simulate_level is the discrete model the simulator steps: its steady response
    to tones equals discrete_frf, with residual, feedthrough, transport delay and fractional output delays.
    Catches a wrong step matrix, a wrong ramp particular solution, a wrong delay or a shape transposed."""
    sim = dict(feedthrough=0.01, residual=list(np.linspace(-0.2, 0.3, 6)), transport_delay_samples=3,
               output_delays_s=[0.0, 1.4 / FS, 7.25 / FS])
    st = three_mode(False, sim)
    freqs = [200.0, 397.0]
    Y, U = steady_tones(st, freqs, 0.5)
    H = rm.discrete_frf(st, freqs, FS)
    for j, f in enumerate(freqs):
        err = np.abs(Y[:, j] / U[j] - H[j, :, j]) / np.abs(H[j, :, j])
        assert err.max() < 1e-8, err


def test_discrete_frf_close_to_continuous():
    """The stepped model is the continuous modal model to the triangle hold's error: below a tenth of the
    sample rate the FRF agrees within 1 percent of its largest value (the hold's own error is about (w dt)^2 / 12, 3
    percent at 800 Hz). Catches a stepper whose discrete model is not the structure."""
    st = three_mode(False)
    f = np.linspace(20, 800, 400)
    Hd, Hc = rm.discrete_frf(st, f, FS), rm.frf(st, f)
    rel = np.abs(Hd - Hc).max() / np.abs(Hc).max()
    assert rel < 1e-2, rel


def test_low_level_is_linear():
    """At a low level (envelope a millionth of the reference amplitude) a structure with drops and rises responds as
    its linear self: relative difference under 1e-5. Catches a rule applied with the wrong scale or at r = 0."""
    st, lin = three_mode(True), three_mode(False)
    n = int(1.0 * FS)
    d = np.random.default_rng(0).standard_normal((n, 2))
    gain = 1e-6 * 1e-3 / np.abs(rm.simulate_level(st, d, FS, keep_modes=True).modal).max()
    a = rm.simulate_level(st, d * gain, FS)
    b = rm.simulate_level(lin, d * gain, FS)
    assert a.peak_ratio(st).max() < 2e-6
    rel = np.abs(a.outputs - b.outputs).max() / np.abs(b.outputs).max()
    assert rel < 1e-5, rel


def frozen_frequency(st_mode, H_meas, tone, fs, lo, hi):
    """The small-signal frequency of a one-mode structure whose frozen discrete response at the tone matches H_meas
    (its damping held): a grid of 0.001 Hz steps, then the best point. Returns (frequency, relative misfit)."""
    def H(fm):
        raw = dict(st_mode.raw)
        m = dict(raw['modes'][0], f_hz=float(fm)); m.pop('level', None)
        raw.update(modes=[m], level_law=None, kind='modal')
        return rm.discrete_frf(rm.Structure(raw), [tone], fs)[0, 0, 0]
    grid = np.arange(lo, hi, 0.001)
    c = np.array([abs(H(g) - H_meas) for g in grid])
    return float(grid[c.argmin()]), float(c.min() / abs(H_meas))


def held_tone(st, r_aim, off, sweep_s=8.0, hold_s=3.0):
    """Sweep a tone up from 0.85 f0 to `off` times the rule's frequency at r_aim, at the force that holds the mode at
    r_aim there (closed form: the frozen mode's displacement per unit force at the tone), then hold it. The sweep up
    puts a hardening mode on its upper branch. Returns (tone, force, Y the output phasor over the last second, the
    envelope over the last second)."""
    W = 2 * np.pi
    aref = st.reference_amplitude[0]
    fr, zr = rm.level_state(st, r_aim)
    tone = fr[0] * off
    force = r_aim * aref * abs(W ** 2 * (fr[0] ** 2 - tone ** 2) + 2j * zr[0] * W ** 2 * fr[0] * tone)
    n = int((sweep_s + hold_s) * FS)
    t = np.arange(n) / FS
    f0 = st.f_hz[0]
    finst = np.where(t < sweep_s, 0.85 * f0 + (tone - 0.85 * f0) * t / sweep_s, tone)
    phase = W * np.cumsum(finst) / FS
    run = rm.simulate_level(st, (np.minimum(1.0, t) * force * np.cos(phase))[:, None], FS, keep_modes=True)
    k = int(FS)
    A = np.stack([np.cos(phase[-k:]), -np.sin(phase[-k:]), np.ones(k)], 1)
    c = np.linalg.lstsq(A, run.outputs[-k:, 0], rcond=None)[0]
    return tone, force, c[0] + 1j * c[1], run.env[-k:, 0]


@pytest.mark.parametrize('kind', ['hardening', 'hardening_strong', 'both', 'softening', 'capped'])
def test_tone_shows_rule_frequency(kind):
    """A tone held just under a mode's shifted resonance, at the force that the closed form says holds the mode at a
    known amplitude ratio r_aim. Checked: (1) the envelope settles at r_aim within 1 percent; (2) the frequency the
    steady response shows (found by matching the frozen discrete response, independent of the stepper's internals)
    lies between the rule's frequency at the smallest and the largest envelope over the last second, and within
    6e-4 of f0 of the rule at the mean envelope; (3) where the rule is flat (a drop at r >= 1, a rise past its cap)
    the match is exact to 1e-6. The envelope ripples by about half a percent at twice the tone (the amplitude
    sqrt(q^2 + (v/w)^2) is constant only at w), so below the cap a hardening mode sits a few 1e-4 above the
    mean-envelope rule; that is the simulator's method, not an error. Catches a rise with the wrong sign, exponent or
    cap, an envelope on the wrong amplitude, or a frequency update that lags."""
    f0, z0, aref = 100.0, 0.02, 1e-3
    lv, r_aim, flat = dict(hardening=(level(rise=0.10, cap=2.0), 1.0, False),
                           hardening_strong=(level(rise=0.10, cap=2.0), 1.5, False),
                           both=(level(drop=0.03, rise=0.08, cap=2.0), 1.2, False),
                           softening=(level(drop=0.06, de=1.0), 1.5, True),
                           capped=(level(rise=0.05, cap=1.5), 3.0, True))[kind]
    st = structure([dict(f_hz=f0, zeta=z0, drive_shape=[1.0], output_shape=[1.0], reference_amplitude=aref, level=lv)], 1, 1,
                   law=dict(kind='jointed', decay_cycles=5.0))
    off = 1.005 if kind == 'softening' else 0.995       # on the stable side: above a softened resonance, below a hardened one
    tone, force, Y, env = held_tone(st, r_aim, off)
    r = env / aref
    assert abs(r.mean() / r_aim - 1) < 0.01, (kind, r.mean())
    f_rule = np.array([rm.level_state(st, x)[0][0] for x in (r.min(), r.mean(), r.max())])
    f_meas, misfit = frozen_frequency(st, Y / force, tone, FS, 0.85 * f0, 1.3 * f0)
    if flat:
        assert abs(f_meas - f_rule[1]) / f0 < 1e-5 and misfit < 1e-6, (kind, f_meas, f_rule, misfit)
    else:
        assert f_rule[0] - 0.002 <= f_meas <= f_rule[2] + 0.002, (kind, f_meas, f_rule)
        assert abs(f_meas - f_rule[1]) / f0 < 6e-4, (kind, f_meas, f_rule)
    ch = f_rule[1] / f0 - 1
    assert (ch > 0) == (kind != 'softening')
    if kind == 'capped':
        assert abs(ch - 0.05 * 1.5 ** 2) < 1e-12


def test_off_resonance_tone_hardening_low_branch():
    """Started from rest at the force of the upper branch, a tone just under a hardening resonance settles on the
    low branch (the jump of a hardening oscillator): the envelope stays well under r_aim. Recorded so a reader of
    the stepper knows it shows the jump; a stepper with no rise would not."""
    st = structure([dict(f_hz=100.0, zeta=0.02, drive_shape=[1.0], output_shape=[1.0], reference_amplitude=1e-3,
                         level=level(rise=0.10, cap=2.0))], 1, 1, law=dict(kind='jointed', decay_cycles=5.0))
    W = 2 * np.pi
    fr, zr = rm.level_state(st, 1.0)
    tone = 0.995 * fr[0]
    force = 1e-3 * abs(W ** 2 * (fr[0] ** 2 - tone ** 2) + 2j * zr[0] * W ** 2 * fr[0] * tone)
    t = np.arange(int(6 * FS)) / FS
    run = rm.simulate_level(st, (np.minimum(1.0, t / 2) * force * np.cos(W * tone * t))[:, None], FS, keep_modes=True)
    assert run.env[-int(FS):, 0].mean() / 1e-3 < 0.6


def test_damping_ceiling_and_refusals():
    """The damping ceiling of 0.9 holds in the stepper, and a structure that can rise past the Nyquist frequency or
    carries a bad field is refused with the field named."""
    st = structure([dict(f_hz=100.0, zeta=0.2, drive_shape=[1.0], output_shape=[1.0], reference_amplitude=1e-6,
                         level=level(gain=50.0))], 1, 1, law=dict(kind='jointed', decay_cycles=3.0))
    f, z = rm.level_state(st, 10.0)
    assert z[0] == rm.MAX_DAMPING
    run = rm.simulate_level(st, np.ones((2000, 1)), FS)
    assert np.isfinite(run.outputs).all()
    hi = structure([dict(f_hz=3000.0, zeta=0.02, drive_shape=[1.0], output_shape=[1.0], reference_amplitude=1.0,
                         level=level(rise=0.5, cap=2.0))], 1, 1, law=dict(kind='jointed', decay_cycles=3.0))
    assert rm.max_frequency(hi)[0] == pytest.approx(3000.0 * 3.0)
    with pytest.raises(ValueError, match='Nyquist'):
        rm.simulate_level(hi, np.zeros((10, 1)), FS)
    for bad, word in [(dict(drop=1.0), 'drop'), (dict(rise=-0.1), 'rise'), (dict(cap=0.0), 'rise_cap'), (dict(cap=-1.0), 'rise_cap'), (dict(re=0.0), 'rise_exponent')]:
        with pytest.raises(ValueError, match=word):
            structure([dict(f_hz=100.0, zeta=0.02, drive_shape=[1.0], output_shape=[1.0], reference_amplitude=1.0,
                            level=level(**bad))], 1, 1, law=dict(kind='jointed', decay_cycles=3.0))


def test_fallback_rule_and_mixed():
    """A mode without a level object follows level_law (the 0.1 rule); a mode with one follows its own."""
    modes = [dict(f_hz=100.0, zeta=0.02, drive_shape=[1.0], output_shape=[1.0], reference_amplitude=1.0),
             dict(f_hz=200.0, zeta=0.02, drive_shape=[1.0], output_shape=[1.0], reference_amplitude=1.0, level=level(rise=0.1))]
    st = structure(modes, 1, 1, law=dict(kind='jointed', delta=0.05, freq_exponent=1.0, damping_gain=1.0, damping_exponent=1.0, decay_cycles=3.0))
    f, z = rm.level_state(st, 1.0)
    assert f[0] == pytest.approx(95.0) and z[0] == pytest.approx(0.04)
    assert f[1] == pytest.approx(220.0) and z[1] == pytest.approx(0.02)
    ch = rm.frequency_change(st, 1.0)
    assert ch[0] < 0 < ch[1]


def test_rise_cap_below_one():
    """A cap below 1 is allowed (spec change, 2026-11-14): the rise stops growing at r = rise_cap even though the drop
    term keeps growing to r = 1. Checked in the rule, in max_frequency, and in the stepper with a tone held past the cap."""
    lv = level(drop=0.02, rise=0.10, cap=0.6, re=2.0)
    st = structure([dict(f_hz=100.0, zeta=0.02, drive_shape=[1.0], output_shape=[1.0], reference_amplitude=1e-3, level=lv)], 1, 1,
                   law=dict(kind='jointed', decay_cycles=5.0))
    for r, want in [(0.3, 1 - 0.02 * 0.3 + 0.1 * 0.09), (0.6, 1 - 0.012 + 0.036), (0.8, 1 - 0.016 + 0.036), (2.0, 1 - 0.02 + 0.036)]:
        assert rm.level_state(st, r)[0][0] == pytest.approx(100.0 * want, rel=1e-14)
    assert rm.max_frequency(st)[0] == pytest.approx(100.0 * (1 + 0.1 * 0.36))
    # held past both the cap and r = 1 the rule is flat, so the steady response shows its frequency exactly
    tone, force, Y, env = held_tone(st, 1.5, 0.995)
    r = env / 1e-3
    assert abs(r.mean() / 1.5 - 1) < 0.01
    f_meas, misfit = frozen_frequency(st, Y / force, tone, FS, 85.0, 130.0)
    assert abs(f_meas - 100.0 * (1 - 0.02 + 0.036)) < 1e-3 and misfit < 1e-6, (f_meas, misfit)
