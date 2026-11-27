"""Hold the Python laws to the shipped Rust laws' parity vectors (hertzforge-mimo-sine-law-math/tests/data/*_parity.json
of the Hertzforge repository): the linear 4 by 4, its quasi-steady loop, guards off, the identification the vectors
were made with (the small-signal response with a fixed 2 % scatter on a 0.25 Hz grid).

    python3 sine/pyloop/parity.py <folder holding matrix_parity.json, laws_parity.json, amplitude_parity.json>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pyloop import structures as S, tone as T                       # noqa: E402
from pyloop.ident import Identification                             # noqa: E402
from pyloop.interface import Setup                                  # noqa: E402
from pyloop.controller import Config, Guards, run                   # noqa: E402
from pyloop.laws_baselines import BASELINES                         # noqa: E402
from pyloop.paired_ref import PairedRef                             # noqa: E402
from pyloop.matrix_ref import RefMatrix                             # noqa: E402
from pyloop.matrix_shipped import ShippedMatrix                     # noqa: E402

LAWS = {'steepest_descent': (BASELINES['steepest_descent'], {}),
        'secant_factor': (BASELINES['secant_complex_gain'], dict(carry='factor')),
        'secant_plain': (BASELINES['secant_complex_gain'], dict(carry='plain')),
        'ratio_reference': (BASELINES['ratio_per_control'], dict(pairing='reference')),
        'ratio_drive': (BASELINES['ratio_per_control'], dict(pairing='drive')),
        'paired': (PairedRef, {})}


def scatter(L, n, m):
    k = np.arange(L)[:, None, None]; i = np.arange(n)[None, :, None]; j = np.arange(m)[None, None, :]
    return 1.0 + 0.02 * np.sin(1.7 * k + 0.9 * i + 2.3 * j) + 0.02j * np.cos(1.3 * k + 0.7 * i + 0.4 * j)


def build(case, dt_c=0.01, fs=16384.0):
    band = tuple(case['band'])
    st = S.ModalStructure(name=case['name'], f_hz=np.array(case['f_hz']), zeta=np.array(case['zeta']),
                          drive_shape=np.array(case['drive_shape']), output_shape=np.array(case['output_shape']),
                          roles=['control'] * np.array(case['output_shape']).shape[1])
    grid = np.arange(np.ceil(band[0] / 0.25), np.floor(band[1] / 0.25) + 1) * 0.25
    H = st.frf(grid) * scatter(grid.size, st.n, st.m)
    idn = Identification(grid, H, np.zeros((grid.size, 0, st.m), complex), np.ones((grid.size, st.n)),
                         np.zeros((grid.size, 0)), np.ones(grid.size, bool), {})
    n, m = st.n, st.m
    setup = Setup(n=n, m=m, p=0, dt_c=round(dt_c * fs) / fs, fs=fs, band=band, drive_ceiling=np.full(m, 1e6), ident=idn,
                  ref_grid=np.ones((grid.size, n), complex), phase_specified=True, limit_grid=np.zeros((grid.size, 0)),
                  alarm_db=3.0, abort_db=12.0, shape='square' if n == m else ('tall' if n > m else 'wide'),
                  tracking=dict(mode='proportional', pct=10.0, max_hz=20.0), identification_level_db=-20.0)
    return st, setup


def go(st, setup, seg, cls, prm):
    g = Guards(change_caps=False, ceiling_on=False, notch=False, divergence_on=False, abort_db=None, limit_abort_db=None,
               first_drive_db=300.0)
    cfg = Config(fs=setup.fs, dt_c=setup.dt_c, tracking=setup.tracking, fast=True, guards=g)
    return run(st, T.Tone([seg], setup.fs), cls, prm, setup, cfg, lambda f: np.ones(st.n, complex), record_ratio=False)


def cx(v):
    a = np.asarray(v, float)
    return a[:, 0] + 1j * a[:, 1]


def compare(res, vec):
    a = res.arrays
    worst = 0.0
    if len(a['f']) != vec['updates']:
        return np.inf, len(a['f'])
    for j, i in enumerate(vec['index']):
        for key in ('D', 'C'):
            want = cx(vec[key][j])
            got = np.asarray(a[key][i]).ravel()
            worst = max(worst, float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-300)))
    return worst, len(a['f'])


def check_all(folder):
    """Each law and schedule: the largest relative difference of the drive and the controls at the vectors' updates."""
    mp = json.load(open(os.path.join(folder, 'matrix_parity.json')))
    lp = json.load(open(os.path.join(folder, 'laws_parity.json')))
    ap = json.load(open(os.path.join(folder, 'amplitude_parity.json')))
    out = {}
    st, setup = build(mp['lin4x4'])
    sched = {'dwell': T.dwell(69.0, 2.0), 'steps': T.steps(40.0, 56.0, 12.0, 0.4)}
    for law, (cls, prm) in LAWS.items():
        for kind, seg in sched.items():
            out[f'{law}/{kind}'] = compare(go(st, setup, seg, cls, prm), lp[f'{law}/{kind}'])
    msched = {'lin4x4': {'dwell': T.dwell(69.0, 1.5), 'steps': T.steps(40.0, 56.0, 12.0, 0.4), 'sweep': T.sweep(60.0, 85.0, 1.0)},
              'lin6x3': {'sweep': T.sweep(40.0, 70.0, 1.0)}, 'lin3x6': {'sweep': T.sweep(60.0, 90.0, 1.0)}}
    for name, runs in msched.items():
        st, setup = build(mp[name])
        for kind, seg in runs.items():
            out[f'matrix/{name}/{kind}'] = compare(go(st, setup, seg, RefMatrix, {}), mp[name]['runs'][kind])
            out[f'matrix_shipped/{name}/{kind}'] = compare(go(st, setup, seg, ShippedMatrix, {}), mp[name]['runs'][kind])
    st, setup = build(mp['lin4x4'])
    setup.phase_specified = False
    for fp in ('in_phase', 'measured'):
        for kind, seg in (('dwell', T.dwell(69.0, 1.5)), ('sweep', T.sweep(60.0, 85.0, 1.0))):
            out[f'matrix_amplitude/{kind}/{fp}'] = compare(go(st, setup, seg, RefMatrix, dict(free_phase=fp)), ap[f'{kind}/{fp}'])
            out[f'matrix_shipped_amplitude/{kind}/{fp}'] = compare(go(st, setup, seg, ShippedMatrix, dict(free_phase=fp)),
                                                                   ap[f'{kind}/{fp}'])
    return out


if __name__ == '__main__':
    res = check_all(sys.argv[1])
    for k, (w, n) in res.items():
        print(f'{k}: {n} updates, worst relative difference {w:.2e}')
