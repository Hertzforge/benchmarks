"""Synthetic runs in the runner's msnb-run/1 layout (FORMAT_RUN.md), built from a sine problem, to pass or to fail each
reading of the scorer. Not a simulation: the controls are the reference times a chosen error, the drive the ideal drive
times a chosen scale, the modes where the test puts them."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sine_model as sm  # noqa: E402
import sine_scorer as sc  # noqa: E402


def _ri(z):
    return [[float(x) for x in row] for row in np.real(z)], [[float(x) for x in row] for row in np.imag(z)]


def make_run(problem, test_id, out, gain_db=None, phase_deg=0.0, drive_scale=1.0, rate_factor=1.0, stop_fraction=None,
             dwell_offset=0.0, mode_shift=0.0, bin_bump_db=None, limit_over_db=None, outcome_class='completed',
             all_unjudged=False, tick=0.01):
    """Write `out`/run.json. gain_db: None or a function f -> dB (scalar or per control) applied to every control.
    bin_bump_db: (f_hz, dB): one sweep bin whose largest error is that much, its point untouched."""
    pr = sc.Problem(problem)
    test = pr.tests[test_id]
    st = pr.st
    lad = [-12.0, -6.0]
    if test['kind'] == 'sweep':
        rate = test['rate_oct_per_min'] * rate_factor
        lo, hi = test['band_hz']
        n_oct = np.log2(hi / lo)
        k = np.arange(int(np.floor(48 * n_oct)) + 1)
        x = k / 48.0
        f = lo * 2 ** x if test['direction'] == 'up' else hi * 2 ** (-x)
        t = 60.0 * x / rate
        f0 = f[0]
    elif test['kind'] == 'stepped':
        f, t, t0 = [], [], 0.0
        for fs in test['frequencies_hz']:
            n = int(round(test['step_s'] / tick))
            f += [fs] * n
            t += list(t0 + tick * np.arange(1, n + 1))
            t0 += n * tick
        f, t = np.array(f), np.array(t)
        f0 = f[0]
    else:
        n = int(round(test['seconds'] / tick))
        f = np.full(n, test['dwell_hz'] * (1 + dwell_offset))
        t = tick * np.arange(1, n + 1)
        f0 = f[0]
    if stop_fraction is not None:
        keep = max(2, int(stop_fraction * f.size))
        f, t = f[:keep], t[:keep]
    # the ladder's rungs before the judged level, at the first frequency, 2 s each, every 0.5 s
    fr = np.full(8, f0)
    tr = np.arange(1, 9) * 0.5
    lv = np.repeat(lad, 4)
    f_all = np.concatenate([fr, f])
    t_all = np.concatenate([tr, 4.0 + t])
    lvl = np.concatenate([lv, np.zeros(f.size)])
    judged = np.concatenate([np.zeros(8, bool), np.full(f.size, not all_unjudged)])
    R = pr.reference_at(f_all) * 10 ** (lvl / 20)[:, None]
    g = np.zeros((f_all.size, R.shape[1]))
    if gain_db is not None:
        g = g + np.asarray([gain_db(x) for x in f_all], float).reshape(f_all.size, -1)
    C = R * 10 ** (g / 20) * np.exp(1j * np.radians(phase_deg))
    D = pr.ideal_at(f_all)[0] * 10 ** (lvl / 20)[:, None] * drive_scale
    h = dict(decimation='1/48 octave' if test['kind'] == 'sweep' else 'every update', points=int(f_all.size),
             index=list(range(f_all.size)), time_s=[float(x) for x in t_all], frequency_hz=[float(x) for x in f_all],
             level_db=[float(x) for x in lvl], judged=[bool(x) for x in judged], settling=[False] * f_all.size,
             miss=[None] * f_all.size)
    h['control_re'], h['control_im'] = _ri(C)
    h['reference_re'], h['reference_im'] = _ri(R)
    h['drive_re'], h['drive_im'] = _ri(D)
    h['error_db'] = [[float(x) for x in row] for row in g]
    if test['kind'] == 'sweep':
        h['bin_updates'] = [1] * f_all.size
        h['bin_judged'] = [int(x) for x in judged]
        bmin = g.copy()
        bmax = g.copy()
        if bin_bump_db is not None:
            p = int(np.argmin(np.abs(np.log(f_all / bin_bump_db[0])) + 1e9 * (~judged)))
            bmax[p, 0] = bin_bump_db[1]
        h['bin_error_db_min'] = [[float(x) for x in row] for row in bmin]
        h['bin_error_db_max'] = [[float(x) for x in row] for row in bmax]
    if test.get('limit'):
        L = test['limit']
        Yl = np.einsum('lgd,ld->lg', sm.frf(st, f_all, [L['output']]), D)
        if limit_over_db is not None:
            Yl = Yl / np.abs(Yl) * L['bound_mps2_pk'] * 10 ** (limit_over_db / 20)
        h['limit_re'], h['limit_im'] = _ri(Yl)
        h['limit_bound'] = [[L['bound_mps2_pk']] for _ in f_all]
    lo, hi = (test['band_hz'] if test['kind'] != 'dwell' else (test['dwell_hz'],) * 2)
    fol = np.flatnonzero((st.f_hz >= lo / 1.25) & (st.f_hz <= hi * 1.25))
    tm = dict(tracked=st.rules is not None, modes=[dict(index=int(i), f0_hz=float(st.f_hz[i])) for i in fol],
              frequency_hz=[[float(st.f_hz[i] * (1 + mode_shift)) for i in fol] for _ in f_all], resolution_samples=256)
    if st.rules is None:
        tm = dict(tracked=False, modes=[], frequency_hz=[], resolution_samples=256)
    stopped = outcome_class != 'completed'
    run = dict(schema='msnb-run/1', case_id=f"{pr.meta['structure']['id']}__{test_id}__matrix", case_sha256='0' * 64,
               runner_sha256='1' * 64, code_commit='synthetic', options=dict(ideal_frf=False, record=False),
               outcome=dict(**{'class': outcome_class}, engine_outcome='completed' if not stopped else 'control_abort',
                            guard=None if not stopped else 'control_abort', channel=None if not stopped else 'c01',
                            frequency_hz=None if not stopped else float(f_all[-1]), time_s=None if not stopped else float(t_all[-1]),
                            level_db=None if not stopped else 0.0, message=None),
               history=h, true_modes=tm,
               summary=dict(updates=int(f_all.size), records_dropped=0, judged_updates=int(judged.sum()),
                            largest_drive_vpk=float(np.abs(D).max()), over_ceiling_samples=0, drive_is_zero=True, engaged_at_s=0.0))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'run.json'), 'w') as fh:
        json.dump(run, fh)
    return os.path.join(out, 'run.json')
