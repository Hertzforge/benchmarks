"""The suite's Python closed-loop harness: one MIMO Sine case (msnb-case/1) run through `reference_model` with the
structure's own level rule, a Python law, and the shipped engine's schedule and guards as far as they are restated
here, written as the runner's run file (msnb-run/1), so `sine_scorer.py` reads either. No Hertzforge build is needed.

    python3 sine/pyloop/harness.py <case.json> --root <problems folder> --out <run folder>

Exit status as the runner's: 0 completed, 1 another end (a guard, the law, the time limit), 2 refused (nothing
written). FORMAT_SINE.md, "The Python Harness", says what of the shipped engine is modelled and what is left out.

The loop, one update every tick (the case's tick_s, rounded to whole samples):
  - the drive ramps linearly over the tick from the last drive to the drive in force (the engine's start_ramps), the
    tone is phase-continuous and its frequency is held through the tick (the engine sets it once per update);
  - `reference_model.simulate_level` steps the structure (controls and limit channels only, the modes and their level
    rule the structure's own), continued from the last tick; white measurement noise of the case's rms is added;
  - the engine's tone tracker reads each channel: four cascaded one-pole sections on the heterodyned signal, alpha =
    B_n / (0.15625 fs), B_n 10 % of the frequency and at most 100 Hz (hertzforge-dsp tone.rs, its defaults);
  - then the update, in the engine's order (hertzforge-mimo-sine core.rs `update`): read the trackers; carry the level
    (the drive in force scaled with each move of the level, by at most the change caps per update, a settle window
    opened with every move); judge (control abort band, abort_ticks updates in a row, not in a settle window; a limit
    channel over its abort band at once); step the schedule (schedule.rs: the ladder runs on the test's clock, a dwell's
    and a step's clock and the ladder's stop in a settle window, a sweep's does not); at a new step the law engages
    again with the first drive's bound; otherwise the law's tick, the change caps against the drive as played, the
    divergence stop, the ceiling per drive, the drive abort.
The settle window is the engine's: 2 x 1.3 / B_n + 3 x the identification's group delay (largest over elements above a
tenth of the line's largest, five-line smoothing, the largest within +-2 B_n).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
SINE = os.path.dirname(HERE)
sys.path.insert(0, SINE)
sys.path.insert(0, os.path.dirname(SINE))
import reference_model as rm                                        # noqa: E402
import case_schema                                                  # noqa: E402
from pyloop.ident import Identification                             # noqa: E402
from pyloop.interface import Setup, TickIn, reg_pinv                 # noqa: E402
from pyloop.laws_baselines import BASELINES                         # noqa: E402
from pyloop.paired_ref import PairedRef                             # noqa: E402
from pyloop.matrix_shipped import ShippedMatrix                     # noqa: E402

RUN_SCHEMA = 'msnb-run/1'
REFERENCE_SCHEMA = 'msnb-reference/1'
TWO_PI = 2.0 * np.pi

# the shipped laws by their manifest id: (class, parameters at the manifest's defaults, amplitude_only)
LAWS = {
    'matrix': (ShippedMatrix, {}, True),
    'paired': (PairedRef, {}, True),
    'ratio': (BASELINES['ratio_per_control'], dict(pairing='reference'), True),
    'steepest_descent': (BASELINES['steepest_descent'], {}, False),
    'secant': (BASELINES['secant_complex_gain'], dict(carry='factor'), False),
}

# the engine's defaults the runner leaves in force (FORMAT_RUN.md, "protective"; hertzforge-mimo-sine)
TRACKER_POLES = 4
NOISE_GAIN_PER_ALPHA = 720.0 / (36.0 * 128.0)
TRACKER_DELAY_FACTOR = TRACKER_POLES * NOISE_GAIN_PER_ALPHA          # 0.625: the loop-speed rule's tracker delay x B_n
SETTLE_WIDTHS = 1.3
SETTLE_TRACKER_TIMES = 2.0
SETTLE_STRUCTURE_TIMES = 3.0
FIRST_DRIVE_MARGIN_DB = 6.0
FIRST_DRIVE_UNTRUSTED_FRACTION = 0.05
FIRST_DRIVE_EPS = 1e-3
MAX_STEP_DB = 1.0
MAX_RISE_DB_PER_S = 240.0
LAW_MISSES = 3
TRACKING_PCT, TRACKING_MAX_HZ = 10.0, 100.0
IDENT_LINE_HZ = 0.25
NOTCH_SMOOTHING = 3.0                       # updates (the research testbed's; the engine's notch is not restated)
NOTCH_A = 1.0 - np.exp(-1.0 / NOTCH_SMOOTHING)
IDENT_BAND = 1.25
PROTECTIVE = {
    'default': dict(alarm_db=1.5, abort_db=3.0, abort_ticks=3, limit_alarm_db=1.5, limit_abort_db=3.0, divergence_s=1.0),
    'widened': dict(alarm_db=1.5, abort_db=20.0, abort_ticks=100, limit_alarm_db=1.5, limit_abort_db=20.0, divergence_s=60.0),
}


class Refused(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def tracking_bw(f):
    return min(f * TRACKING_PCT / 100.0, TRACKING_MAX_HZ)


def loglog(x, xp, yp):
    return np.exp(np.interp(np.log(x), np.log(xp), np.log(yp)))


# ---------------------------------------------------------------------------------------------- the case
class Case:
    def __init__(self, path, root):
        self.path = path
        raw = open(path, 'rb').read()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        c = json.loads(raw)
        bad = case_schema.check(c)
        if bad:
            raise Refused('the case does not meet msnb-case/1: ' + '; '.join(bad[:3]))
        self.raw = c
        sfile = os.path.join(root, c['structure']['file'])
        if not os.path.exists(sfile) or sha256(sfile) != c['structure']['sha256']:
            raise Refused(f"the structure {c['structure']['file']} is missing or not the case's (SHA-256)")
        self.st = rm.load_structure(sfile)
        names = [o['name'] for o in self.st.outputs]
        for k in c['controls'] + [L['output'] for L in c.get('limits', [])]:
            if k not in names:
                raise Refused(f'no output {k} in the structure')
        self.controls = [names.index(k) for k in c['controls']]
        self.limit_names = [L['output'] for L in c.get('limits', [])]
        self.limits = [names.index(k) for k in self.limit_names]
        self.limit_bounds = [(np.asarray(L['bound']['frequency_hz'], float), np.asarray(L['bound']['amplitude_m_s2_pk'], float))
                             for L in c.get('limits', [])]
        self.drive_names = [d['name'] if isinstance(d, dict) else str(d) for d in self.st.drives]
        law = c['law']
        if law['id'] not in LAWS:
            raise Refused(f"no Python version of the law {law['id']}")
        if law.get('params'):
            raise Refused('law parameters other than the manifest defaults are not modelled by the harness')
        self.law_id = law['id']
        self.law_cls, self.law_params, amp_only = LAWS[law['id']]
        ph = c['reference'].get('phase', 'auto')
        self.phase_specified = (not amp_only) if ph == 'auto' else ph == 'specified'
        rfile = os.path.join(root, c['reference']['file'])
        if not os.path.exists(rfile) or sha256(rfile) != c['reference']['sha256']:
            raise Refused(f"the reference {c['reference']['file']} is missing or not the case's (SHA-256)")
        ref = json.load(open(rfile))
        if ref.get('schema') != REFERENCE_SCHEMA or ref.get('controls') != c['controls']:
            raise Refused('the reference file is not msnb-reference/1 for these controls')
        self.ref_f = np.asarray(ref['frequency_hz'], float)
        self.ref_a = np.asarray(ref['amplitude_m_s2_pk'], float) * float(c.get('full_level_scale', 1.0))
        self.ref_ph = np.unwrap(np.radians(np.asarray(ref['phase_deg'], float)), axis=0)
        self.fs = float(c.get('fs_hz', 8192.0))
        self.tick_n = int(round(float(c.get('tick_s', 0.01)) * self.fs))
        self.dt = self.tick_n / self.fs
        t = c['test']
        self.test = t
        lad = c.get('ladder') or {}
        self.levels = [float(x) for x in lad.get('levels_db', [0.0])]
        rung = lad.get('rung') or ({'sweeps': 1} if t['type'] == 'sweep' else {'seconds': 2.0})
        if 'sweeps' in rung:
            raise Refused('a rung held for whole sweeps is not modelled by the harness')
        self.rung_s = float(rung['seconds'])
        drv = c.get('drives') or {}
        self.ceiling = float(drv.get('ceiling_vpk', 9.0))
        self.abort_vpk = float(drv.get('abort_vpk', 10.0))
        # the amplifier (owner, 2026-11-25): structure input per engine volt, applied where the drive is played; the
        # law, the ceilings and the run file's drives are in engine volts
        self.gain = float(drv.get('gain', 1.0))
        noise = c.get('noise') or {}
        self.noise_rms = float(noise.get('rms_m_s2', 0.0))
        self.noise_seed = int(noise.get('seed', 1))
        self.protective_name = c.get('protective', 'default')
        self.protective = dict(PROTECTIVE[self.protective_name])
        if self.protective_name == 'widened':
            self.abort_vpk = 10.0
        if t['type'] == 'dwell':
            self.ident_band = (t['freq_hz'] / IDENT_BAND, t['freq_hz'] * IDENT_BAND)
            self.band = (t['freq_hz'], t['freq_hz'])
        else:
            self.ident_band = (t['band_hz'][0] / IDENT_BAND, t['band_hz'][1] * IDENT_BAND)
            self.band = tuple(t['band_hz'])
        rungs = (len(self.levels) - 1) * self.rung_s
        self.max_test_s = float(c['max_test_s']) if c.get('max_test_s') else 1.5 * (self.test_length_s() + rungs) + 60.0

    def test_length_s(self):
        t = self.test
        if t['type'] == 'sweep':
            return 60.0 * np.log2(t['band_hz'][1] / t['band_hz'][0]) / t['rate_oct_per_min']
        if t['type'] == 'stepped':
            return len(step_grid(t)) * t['step_s']
        return t['dwell_s']

    def reference_at(self, f):
        """The full-level reference at f (controls,): amplitude in log-log between lines, phase linear in log
        frequency, held beyond the file's ends."""
        x = np.log(np.clip(f, self.ref_f[0], self.ref_f[-1]))
        lf = np.log(self.ref_f)
        k = int(np.clip(np.searchsorted(lf, x), 1, lf.size - 1))
        w = (x - lf[k - 1]) / (lf[k] - lf[k - 1])
        a = np.exp((1 - w) * np.log(self.ref_a[k - 1]) + w * np.log(self.ref_a[k]))
        if not self.phase_specified:
            return a.astype(complex)
        p = (1 - w) * self.ref_ph[k - 1] + w * self.ref_ph[k]
        return a * np.exp(1j * p)

    def bounds_at(self, f):
        return np.array([float(loglog(f, fb, ab)) for fb, ab in self.limit_bounds])


def step_grid(t):
    """The stepped sine's log grid (the engine's StepSine, log spacing): from the band's low end, steps_per_octave a
    octave, up to the high end."""
    lo, hi = t['band_hz']
    n = int(np.floor(np.log2(hi / lo) * t['steps_per_octave'] + 1e-9)) + 1
    return lo * 2.0 ** (np.arange(n) / t['steps_per_octave'])


# ---------------------------------------------------------------------------------------------- the identification
def identify(case):
    """The exact small-signal response of the stepped structure on the identification's 0.25 Hz lines over its band
    (the test band /1.25 to x1.25): in place of the engine's measured one, every line trusted, coherence 1."""
    lo, hi = case.ident_band
    g = np.arange(np.ceil(lo / IDENT_LINE_HZ), np.floor(hi / IDENT_LINE_HZ) + 1) * IDENT_LINE_HZ
    g = g[(g > 0) & (g < 0.5 * case.fs)]
    with np.errstate(all='ignore'):
        # per engine volt: the structure's response times the amplifier's gain
        H = case.gain * rm.discrete_frf(case.st, g, case.fs, rows=case.controls)
        T = case.gain * rm.discrete_frf(case.st, g, case.fs, rows=case.limits) if case.limits else np.zeros((g.size, 0, case.st.n_drives), complex)
    # C order: the laws' grid smoothing (carried.smooth_grid) reshapes in place and needs it
    H, T = np.ascontiguousarray(H), np.ascontiguousarray(T)
    if not (np.all(np.isfinite(H)) and np.all(np.isfinite(T))):
        raise Refused('the small-signal response is not finite on the identification\'s lines')
    return Identification(g, H, T, np.ones((g.size, H.shape[1])), np.ones((g.size, T.shape[1])), np.ones(g.size, bool),
                          dict(what='exact small-signal response (reference_model.discrete_frf)'))


def identification_from_run(case, run_dir):
    """The identification an engine run built its law on (the runner's `law_identification.f8`, FORMAT_RUN.md), for
    this harness. The file is in the engine's convention: the identification's lag is taken out of the tracked
    values, so a structure's response is turned by exp(j w turn / fs), `turn` the run's
    `identification.ideal_vs_measured.delay_turn_samples` (lag less the frames ahead less the service's simulator
    block). This loop has no lag, so the file is turned back by exp(-j w turn / fs). Coherence and the trusted flags
    are taken as written."""
    run = json.load(open(os.path.join(run_dir, 'run.json')))
    ig = run['identification']
    lg = ig['law_given']
    n, p, m = len(case.controls), len(case.limits), case.st.n_drives
    if (lg['controls'], lg['limits'], lg['drives']) != (case.raw['controls'], case.limit_names, case.drive_names):
        raise Refused('the identification file is not for these controls, limit channels and drives')
    if run['structure']['sha256'] != case.raw['structure']['sha256']:
        raise Refused('the identification file is not of this structure')
    if float(run['structure']['fs_hz']) != case.fs:
        raise Refused('the identification file was made at another sample rate')
    turn = (ig.get('ideal_vs_measured') or {}).get('delay_turn_samples')
    if turn is None:
        raise Refused('the run does not record the identification\'s delay turn')
    path = os.path.join(run_dir, lg['file'])
    if sha256(path) != lg['sha256']:
        raise Refused('the identification file does not match its SHA-256 in the run file')
    cols = 1 + 2 * n * m + 2 * p * m + n + p + 1
    a = np.fromfile(path, dtype='<f8')
    if a.size != cols * int(lg['lines']):
        raise Refused('the identification file is not the size its run file states')
    a = a.reshape(int(lg['lines']), cols)
    g = a[:, 0].copy()
    o = 1
    H = (a[:, o:o + n * m] + 1j * a[:, o + n * m:o + 2 * n * m]).reshape(g.size, n, m)
    o += 2 * n * m
    T = (a[:, o:o + p * m] + 1j * a[:, o + p * m:o + 2 * p * m]).reshape(g.size, p, m)
    o += 2 * p * m
    coh = a[:, o:o + n].copy()
    o += n
    coh_l = a[:, o:o + p].copy()
    o += p
    trusted = a[:, o] > 0.5
    back = np.exp(-1j * TWO_PI * g * float(turn) / case.fs)[:, None, None]
    H, T = np.ascontiguousarray(H * back), np.ascontiguousarray(T * back)
    return Identification(g, H, T, coh, coh_l, trusted,
                          dict(what='the engine run\'s law_identification.f8, turned back by its delay turn',
                               source=ig.get('source'), file_sha256=lg['sha256'], delay_turn_samples=float(turn)))


def make_setup(case, idn):
    n, m, p = len(case.controls), case.st.n_drives, len(case.limits)
    g = idn.grid_hz
    ref = np.array([case.reference_at(f) for f in g])
    lim = np.array([case.bounds_at(f) for f in g]) if p else np.zeros((g.size, 0))
    s = Setup(n=n, m=m, p=p, dt_c=case.dt, fs=case.fs, band=case.band, drive_ceiling=np.full(m, case.ceiling), ident=idn,
              ref_grid=ref, phase_specified=case.phase_specified, limit_grid=lim, alarm_db=case.protective['alarm_db'],
              abort_db=case.protective['abort_db'], shape='square' if n == m else ('tall' if n > m else 'wide'),
              tracking=dict(mode='proportional', pct=TRACKING_PCT, max_hz=TRACKING_MAX_HZ), identification_level_db=float('nan'))
    # the loop delays the engine gives its laws (LawSetup.loop_delay): its tracker's delay factor; no transport beyond
    # what the identification shows (here the identification holds the structure's delays and there is no pipeline)
    s.tracker_delay_factor = TRACKER_DELAY_FACTOR
    s.transport_delay_s = 0.0
    return s


# ---------------------------------------------------------------------------------------------- the schedule
class Schedule:
    """The engine's schedule, as restated: the test plan and the ladder side by side on one clock."""

    def __init__(self, case):
        self.case = case
        t = case.test
        self.kind = t['type']
        self.levels = case.levels
        self.rung = 0
        self.rung_t = 0.0
        self.settle_left = 0.0
        self.done = False
        if self.kind == 'sweep':
            lo, hi = t['band_hz']
            self.up = t.get('direction', 'up') == 'up'
            self.f0 = lo if self.up else hi
            self.total = np.log2(hi / lo)
            self.rate = t['rate_oct_per_min']
            self.pos = 0.0
        elif self.kind == 'stepped':
            self.grid = step_grid(t)
            self.k = 0
            self.step_t = 0.0
        else:
            self.dwell_t = 0.0

    def frequency(self):
        if self.kind == 'sweep':
            return self.f0 * 2.0 ** (self.pos if self.up else -self.pos)
        if self.kind == 'stepped':
            return float(self.grid[min(self.k, self.grid.size - 1)])
        return float(self.case.test['freq_hz'])

    def level_db(self):
        return self.levels[self.rung]

    def settle_for(self, s):
        self.settle_left = max(self.settle_left, s)

    def settling(self):
        return self.settle_left > 0.0

    def sweep_rate(self):
        if self.kind != 'sweep' or self.done:
            return 0.0, 0.0
        octm = self.rate if self.up else -self.rate
        return self.frequency() * np.log(2.0) * octm / 60.0, octm

    def step(self, dt):
        """One update: returns (frequency for the next tick, jumped, level_changed)."""
        settling = self.settle_left > 0.0
        self.settle_left = max(0.0, self.settle_left - dt)
        before = self.frequency()
        if self.kind == 'sweep':
            self.pos = min(self.total, self.pos + dt * self.rate / 60.0)
            if self.pos >= self.total - 1e-12:
                self.done = True
        elif self.kind == 'stepped':
            if not settling:
                self.step_t += dt
                if self.step_t >= self.case.test['step_s'] - 1e-9:
                    self.step_t = 0.0
                    self.k += 1
                    if self.k >= self.grid.size:
                        self.done = True
        else:
            if not settling:
                self.dwell_t += dt
                if self.dwell_t >= self.case.test['dwell_s'] - 1e-9:
                    self.done = True
        after = self.frequency()
        jumped = self.kind != 'sweep' and after != before
        lev0 = self.rung
        if not settling and self.rung + 1 < len(self.levels):
            self.rung_t += dt
            if self.rung_t >= self.case.rung_s - 1e-9:
                self.rung += 1
                self.rung_t = 0.0
        return after, jumped, self.rung != lev0


# ---------------------------------------------------------------------------------------------- the station side
class Station:
    """The structure (simulate_level, continued tick to tick), the noise and the trackers."""

    def __init__(self, case):
        self.case = case
        self.rows = case.controls + case.limits
        self.state = None
        self.rng = np.random.default_rng(case.noise_seed)
        self.trk = np.zeros((TRACKER_POLES, 2, len(self.rows)))    # (pole, i/q, channel)
        self.phi = 0.0
        self.env = np.zeros(case.st.n_modes)
        self.alpha = None

    def set_bandwidth(self, f):
        self.alpha = float(np.clip(tracking_bw(f) / (NOISE_GAIN_PER_ALPHA * self.case.fs), 1e-12, 0.5))

    def play(self, D0, D1, f):
        """One tick at f with the drive ramping from D0 to D1; returns (C, L, drive samples)."""
        N = self.case.tick_n
        fs = self.case.fs
        w = (np.arange(1, N + 1) / N)[:, None]
        Dn = D0[None, :] * (1.0 - w) + D1[None, :] * w
        phi = self.phi + TWO_PI * f * np.arange(1, N + 1) / fs
        self.phi = float(np.mod(phi[-1], TWO_PI))
        c, s = np.cos(phi), np.sin(phi)
        d = Dn.real * c[:, None] - Dn.imag * s[:, None]                # engine volts
        r = rm.simulate_level(self.case.st, d * self.case.gain, fs, rows=self.rows, state=self.state)
        self.state = r.state
        self.env = r.envelope if self.case.st.rules is not None else self.env
        y = r.outputs
        if self.case.noise_rms > 0:
            y = y + self.rng.normal(size=y.shape) * self.case.noise_rms
        a = self.alpha
        b, av = [a], [1.0, -(1.0 - a)]
        xs = [2.0 * y * c[:, None], -2.0 * y * s[:, None]]        # in phase, quadrature: C = i + j q
        for k in range(TRACKER_POLES):                             # each section x_k += a (x_(k-1) - x_k), per sample
            for q in (0, 1):
                xs[q], _ = lfilter(b, av, xs[q], axis=0, zi=((1.0 - a) * self.trk[k, q])[None, :])
                self.trk[k, q] = xs[q][-1]
        z = self.trk[-1, 0] + 1j * self.trk[-1, 1]
        n = len(self.case.controls)
        return z[:n], z[n:], d

    def mode_frequencies(self):
        st = self.case.st
        if st.rules is None:
            return st.f_hz.copy()
        return rm.level_state(st, self.env / st.reference_amplitude)[0]


# ---------------------------------------------------------------------------------------------- the loop
def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))


def run_case(case, ident=None):
    t_wall = time.perf_counter()
    idn = identify(case) if ident is None else ident
    setup = make_setup(case, idn)
    law = case.law_cls(setup, **case.law_params)
    sched = Schedule(case)
    sta = Station(case)
    pr = case.protective
    n, m, p = setup.n, setup.m, setup.p
    dt = case.dt
    ceiling = np.full(m, case.ceiling)

    def settle_s(f):
        b = tracking_bw(f)
        return SETTLE_TRACKER_TIMES * SETTLE_WIDTHS / b + SETTLE_STRUCTURE_TIMES * idn.group_delay_at(f, 2.0 * b)

    rec = []
    state = dict(level_db=sched.level_db(), target_db=sched.level_db(), engage_settle=True, misses=0, cap_run=0.0)
    outside = np.zeros(n, int)
    D_force = np.zeros(m, complex)
    C = np.zeros(n, complex)
    Lm = np.zeros(p, complex)
    end = dict(cls='completed', engine='completed', guard=None, channel=None, f=None, t=None, level=None, message=None)
    over_ceiling_samples = 0
    largest = 0.0

    def level():
        return 10.0 ** (state['level_db'] / 20.0)

    notch = dict(kappa=np.full(p, np.nan), depth=1.0, scale=1.0)

    def notch_observe():
        # the ratio of each limit channel's reading to the identification's prediction, from settled readings only,
        # smoothed over NOTCH_SMOOTHING updates (simplified: see FORMAT_SINE.md, "The Python Harness")
        T0 = idn.T0_at(f)
        pf = np.abs(T0 @ D_force)
        fresh = pf > 1e-12
        smp = np.where(fresh, np.abs(Lm) / np.maximum(pf, 1e-300), np.nan)
        kap = notch['kappa']
        seed = fresh & ~np.isfinite(kap)
        kap = np.where(seed, smp, kap)
        upd = fresh & ~seed
        notch['kappa'] = np.where(upd, kap + NOTCH_A * (smp - kap), kap)

    def kappa():
        return np.where(np.isfinite(notch['kappa']), notch['kappa'], 1.0)

    def notch_depth(fq, Rf):
        # the share of the reference the limit channels allow (the engine's notch_allowed), at most 1
        dd = reg_pinv(idn.H0_at(fq), FIRST_DRIVE_EPS) @ Rf
        pred = kappa() * np.abs(idn.T0_at(fq) @ dd)
        b = case.bounds_at(fq)
        with np.errstate(divide='ignore', invalid='ignore'):
            al = np.where(pred > 0, b / pred, np.inf)
        return float(min(1.0, al.min())) if al.size else 1.0

    def finish_guards(d, fq):
        if p:                                       # the backstop notch: the most loaded limit channel at its bound
            pred = kappa() * np.abs(idn.T0_at(fq) @ d)
            b = case.bounds_at(fq)
            with np.errstate(divide='ignore', invalid='ignore'):
                sc = np.where(pred > 0, b / pred, np.inf)
            notch['scale'] = float(min(1.0, sc.min()))
            d = d * notch['scale']
        a = np.abs(d)                               # a drive over its ceiling scales the whole vector
        if np.any(a > ceiling):
            d = d * float(np.min(np.where(a > 0, ceiling / np.maximum(a, 1e-300), np.inf)))
        return d

    def drive_abort(d):
        a = np.abs(d)
        if np.any(a > case.abort_vpk):
            return int(np.argmax(a))
        return None

    def call_law(engage, k, f, hold, settling):
        R = level() * case.reference_at(f)
        # the shipped laws aim at the reference times the notch scale (law.rs aimed_reference); the shipped matrix
        # takes the notch itself, the testbed's laws are handed the aimed reference
        Rl = R if isinstance(law, ShippedMatrix) else R * notch['depth']
        lim = case.bounds_at(f) if p else np.zeros(0)
        hz, octm = (0.0, 0.0) if engage else sched.sweep_rate()
        x = TickIn(k=k, dt=0.0 if engage else dt, f=f, sweep_hz_s=hz, sweep_oct_min=octm, level=level(), R=Rl.copy(),
                   C=C.copy(), L=Lm.copy(), D_in_force=D_force.copy(), hold=hold, notch_scale=notch['depth'], limit_bounds=lim,
                   settle_open=bool(settling))
        try:
            out = law.engage(x) if engage else law.tick(x)
            D = np.asarray(out.D)
            if D.shape != (m,):
                return None, 'shape', R
            if not np.all(np.isfinite(D)):
                return None, 'nonfinite', R
            return D.astype(complex), None, R
        except Exception as exc:                                             # a failed call is a miss
            return None, f'exception: {type(exc).__name__}', R

    def first_drive(k, f):
        nonlocal D_force
        sched.settle_for(settle_s(f))
        D, why, R = call_law(True, k, f, 'engage', True)
        d = D if D is not None else D_force.copy()
        if idn.trust_at(f):
            bound = np.linalg.norm(reg_pinv(idn.H0_at(f), FIRST_DRIVE_EPS) @ R) * 10 ** (FIRST_DRIVE_MARGIN_DB / 20.0)
        else:
            bound = FIRST_DRIVE_UNTRUSTED_FRACTION * np.linalg.norm(ceiling) * level()
        nd = np.linalg.norm(d)
        if nd > bound:
            d = d * (bound / nd)
        state['engage_settle'] = True
        return finish_guards(d, f), why, R

    def record(k, f, R, D, settling, why, err):
        top = abs(state['level_db'] - case.levels[-1]) < 1e-9
        rec.append(dict(index=k, t=k * dt, f=f, level=state['level_db'], judged=bool(top and not settling and why is None),
                        settling=bool(settling), miss=why, C=C.copy(), R=R.copy(), D=D.copy(), err=err, L=Lm.copy(),
                        Lb=case.bounds_at(f) if p else np.zeros(0), modes=sta.mode_frequencies()))

    # ---- engage
    k = 0
    f = sched.frequency()
    sta.set_bandwidth(f)
    D_prev = np.zeros(m, complex)
    D_force, why, R = first_drive(k, f)
    if why:
        state['misses'] += 1
    record(k, f, R, D_force, True, why, np.full(n, np.nan))
    largest = max(largest, float(np.abs(D_force).max()))
    while True:
        # ---- play the tick
        C, Lm, d = sta.play(D_prev, D_force, f)
        over_ceiling_samples += int(np.sum(np.abs(d) > ceiling[None, :] * (1 + 1e-12)))
        k += 1
        t_now = k * dt
        played = D_force.copy()
        # ---- the level carry
        if state['level_db'] != state['target_db']:
            up = min(MAX_STEP_DB, MAX_RISE_DB_PER_S * dt)
            want = state['target_db'] - state['level_db']
            stp = float(np.clip(want, -MAX_STEP_DB, up))
            before = level()
            state['level_db'] = state['target_db'] if stp == want else state['level_db'] + stp
            D_force = D_force * (level() / before)
            sched.settle_for(settle_s(f))
            state['engage_settle'] = True
        R = level() * case.reference_at(f)
        if p:
            notch['depth'] = notch_depth(f, R)
            if not sched.settling():
                notch_observe()
        RJ = R * notch['depth']                     # the controls are judged against the notched reference
        # ---- judge
        settling = sched.settling()
        err = _db(C) - _db(RJ)
        trip = None
        if settling:
            outside[:] = 0
        else:
            out = np.abs(err) > pr['abort_db']
            outside[:] = np.where(out, outside + 1, 0)
            if np.any(outside >= pr['abort_ticks']):
                c = int(np.argmax(outside))
                trip = ('control_abort', case.raw['controls'][c])
        if trip is None and p:
            over = _db(Lm) - _db(case.bounds_at(f))
            if np.any(over > pr['limit_abort_db']):
                trip = ('limit_abort', case.limit_names[int(np.argmax(over))])
        if trip is not None:
            end.update(cls='stopped_by_protective_default' if case.protective_name == 'default' else 'aborted',
                       engine=trip[0], guard=trip[0], channel=trip[1], f=f, t=t_now, level=state['level_db'])
            record(k, f, R, D_force, settling, None, err)
            break
        if t_now > case.max_test_s:
            end.update(cls='runner_stopped', engine='stopped', f=f, t=t_now, level=state['level_db'],
                       message=f'the time limit, {case.max_test_s:g} s of test')
            record(k, f, R, D_force, settling, None, err)
            break
        # ---- the schedule
        f_next, jumped, lev_changed = sched.step(dt)
        if lev_changed:
            state['target_db'] = sched.level_db()
            sched.settle_for(settle_s(f_next))
        if jumped and not sched.done:
            f = f_next
            sta.set_bandwidth(f)
            state['cap_run'] = 0.0
            D_prev = D_force.copy()
            D_force, why, R = first_drive(k, f)
            largest = max(largest, float(np.abs(D_force).max()))
            record(k, f, R, D_force, True, why, np.full(n, np.nan))
            continue
        if not sched.settling():
            state['engage_settle'] = False
        hold = 'engage' if state['engage_settle'] else ('notch' if (notch['depth'] < 1.0 or notch['scale'] < 1.0) else 'free')
        settling_now = sched.settling()
        D, why, _ = call_law(False, k, f, hold, settling_now)
        capped = False
        if D is None:
            state['misses'] += 1
            D = D_force.copy()
            if state['misses'] >= LAW_MISSES:
                end.update(cls='law_fault', engine='law_fault', f=f, t=t_now, level=state['level_db'],
                           message=f'{LAW_MISSES} law misses in a row ({why})')
        else:
            state['misses'] = 0
            a0, a1 = np.abs(played), np.abs(D)
            upc = min(MAX_STEP_DB, MAX_RISE_DB_PER_S * dt)
            act = a0 > 1e-12 * max(1.0, case.ceiling)
            with np.errstate(divide='ignore', invalid='ignore'):
                s_db = np.where(act & (a1 > 0), _db(a1) - _db(a0), np.where(act, -np.inf, 0.0))
            hi = act & (s_db > upc + 1e-9)
            lo = act & (s_db < -(MAX_STEP_DB + 1e-9))
            if np.any(hi | lo):
                capped = True
                ph = np.where(a1 > 0, D / np.maximum(a1, 1e-300), played / np.maximum(a0, 1e-300))
                D = np.where(hi, a0 * 10 ** (upc / 20.0) * ph, D)
                D = np.where(lo, a0 * 10 ** (-MAX_STEP_DB / 20.0) * ph, D)
        state['cap_run'] = state['cap_run'] + dt if capped else 0.0
        if state['cap_run'] >= pr['divergence_s'] - 1e-9 and end['cls'] == 'completed':
            end.update(cls='stopped_by_protective_default' if case.protective_name == 'default' else 'aborted',
                       engine='divergence', guard='divergence', f=f, t=t_now, level=state['level_db'])
        D = finish_guards(D, f)
        j = drive_abort(D)
        if j is not None and end['cls'] == 'completed':
            end.update(cls='stopped_by_protective_default' if case.protective_name == 'default' else 'aborted',
                       engine='drive_abort', guard='drive_abort', channel=case.drive_names[j], f=f, t=t_now,
                       level=state['level_db'])
        largest = max(largest, float(np.abs(D).max()))
        record(k, f, R, D, settling_now, why, err)
        if end['cls'] != 'completed':
            break
        if sched.done:
            break
        D_prev, D_force = D_force, D
        f = f_next
        sta.set_bandwidth(f)
    return rec, end, dict(over_ceiling_samples=over_ceiling_samples, largest=largest, wall_s=time.perf_counter() - t_wall,
                          ident=idn)


# ---------------------------------------------------------------------------------------------- the run file
def _ri(z):
    z = np.asarray(z)
    return [[float(x) for x in row] for row in z.real], [[float(x) for x in row] for row in z.imag]


def _rows(a):
    return [[None if not np.isfinite(x) else float(x) for x in row] for row in np.asarray(a, float)]


def decimate(case, rec):
    """The runner's history points: a sweep's first update of each 1/48-octave bin (from the band's low edge) at each
    level, with the bin's counts and error extremes over its judged updates; every update otherwise. The last update
    is always a point."""
    if case.test['type'] != 'sweep':
        return list(range(len(rec))), None
    lo = case.test['band_hz'][0]
    keys = [(int(np.floor(48.0 * np.log2(r['f'] / lo) + 1e-9)), r['level']) for r in rec]
    pts, groups = [], []
    for i, kk in enumerate(keys):
        if i == 0 or kk != keys[i - 1]:
            pts.append(i)
            groups.append([i])
        else:
            groups[-1].append(i)
    if pts[-1] != len(rec) - 1:
        last = groups[-1].pop()
        pts.append(last)
        groups.append([last])
    return pts, groups


def write_run(case, rec, end, info, out, case_path, runner_sha):
    pts, groups = decimate(case, rec)
    P = [rec[i] for i in pts]
    n = len(case.controls)
    h = dict(decimation='1/48 octave' if groups is not None else 'every update', points=len(P),
             index=[r['index'] for r in P], time_s=[float(r['t']) for r in P], frequency_hz=[float(r['f']) for r in P],
             level_db=[float(r['level']) for r in P], judged=[r['judged'] for r in P], settling=[r['settling'] for r in P],
             miss=[r['miss'] for r in P])
    h['control_re'], h['control_im'] = _ri([r['C'] for r in P])
    h['reference_re'], h['reference_im'] = _ri([r['R'] for r in P])
    h['error_db'] = _rows([r['err'] for r in P])
    h['drive_re'], h['drive_im'] = _ri([r['D'] for r in P])
    if case.limits:
        h['limit_re'], h['limit_im'] = _ri([r['L'] for r in P])
        h['limit_bound'] = _rows([r['Lb'] for r in P])
    if groups is not None:
        h['bin_updates'] = [len(g) for g in groups]
        h['bin_judged'] = [int(sum(rec[i]['judged'] for i in g)) for g in groups]
        mn, mx = [], []
        for g in groups:
            e = np.array([rec[i]['err'] for i in g if rec[i]['judged']]).reshape(-1, n)
            mn.append([None] * n if e.size == 0 else [float(x) for x in e.min(0)])
            mx.append([None] * n if e.size == 0 else [float(x) for x in e.max(0)])
        h['bin_error_db_min'], h['bin_error_db_max'] = mn, mx
    st = case.st
    lo, hi = case.ident_band
    fol = np.flatnonzero((st.f_hz >= lo) & (st.f_hz <= hi)) if st.rules is not None else np.zeros(0, int)
    tm = dict(tracked=bool(st.rules is not None), modes=[dict(index=int(i), f0_hz=float(st.f_hz[i])) for i in fol],
              frequency_hz=[[float(r['modes'][i]) for i in fol] for r in P] if fol.size else [],
              resolution_samples=case.tick_n)
    judged = sum(r['judged'] for r in rec)
    run = dict(schema=RUN_SCHEMA, case_id=case.raw['case_id'], case_sha256=case.sha256, runner_sha256=runner_sha,
               code_commit=None, runner='the MIMO Sine suite\'s Python harness (sine/pyloop/harness.py)',
               options=dict(ideal_frf=info['ident'].level.get('source') in (None, 'ideal_frf'), record=False,
                            harness='python'),
               structure=dict(file=case.raw['structure']['file'], sha256=case.raw['structure']['sha256'],
                              controls=case.raw['controls'], limits=case.limit_names, drives=case.drive_names,
                              n_modes=int(st.n_modes), fs_hz=case.fs, tick_s=case.dt, tick_frames=case.tick_n),
               law=dict(id=case.law_id, params={}, python=case.law_cls.__name__, python_params=case.law_params,
                        phase_specified=case.phase_specified),
               settings=dict(protective=case.protective_name, **case.protective, max_step_db=MAX_STEP_DB,
                             max_rise_db_per_s=MAX_RISE_DB_PER_S, rotation_cap_deg=None, law_misses=LAW_MISSES,
                             ceiling_vpk=case.ceiling, abort_vpk=case.abort_vpk, drive_gain=case.gain, ladder_db=case.levels,
                             rung=dict(seconds=case.rung_s), noise_rms_m_s2=case.noise_rms, noise_seed=case.noise_seed,
                             max_test_s=case.max_test_s),
               identification=dict(source='exact_small_signal' if info['ident'].level.get('source') is None else 'engine_file',
                                   engine_source=info['ident'].level.get('source'),
                                   file_sha256=info['ident'].level.get('file_sha256'),
                                   delay_turn_samples=info['ident'].level.get('delay_turn_samples'),
                                   grid_hz=[float(info['ident'].grid_hz[0]), float(info['ident'].grid_hz[-1])],
                                   lines=int(info['ident'].grid_hz.size), line_hz=IDENT_LINE_HZ),
               outcome={'class': end['cls'], 'engine_outcome': end['engine'], 'guard': end['guard'], 'channel': end['channel'],
                        'frequency_hz': end['f'], 'time_s': end['t'], 'level_db': end['level'], 'message': end['message']},
               events=[], history=h, true_modes=tm,
               summary=dict(updates=len(rec), records_dropped=0, judged_updates=int(judged),
                            largest_drive_vpk=float(info['largest']), over_ceiling_samples=int(info['over_ceiling_samples']),
                            drive_is_zero=None, engaged_at_s=0.0))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'run.json'), 'w') as fh:
        json.dump(run, fh, separators=(',', ':'))
    with open(os.path.join(out, 'timing.json'), 'w') as fh:
        json.dump(dict(run_s=round(info['wall_s'], 3)), fh)
    return run


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('case')
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ident', default=None, help='an engine run folder (msnb_run): build the law on its '
                    'law_identification.f8 in place of the exact small-signal response')
    a = ap.parse_args(argv)
    try:
        case = Case(os.path.abspath(a.case), os.path.abspath(a.root))
        idn = identification_from_run(case, os.path.abspath(a.ident)) if a.ident else None
    except Refused as ex:
        print(f'REFUSED: {ex}', file=sys.stderr)
        return 2
    rec, end, info = run_case(case, idn)
    write_run(case, rec, end, info, a.out, a.case, sha256(os.path.abspath(__file__)))
    print(f"{case.raw['case_id']}: {end['cls']} ({end['engine']}), {len(rec)} updates, {info['wall_s']:.1f} s")
    return 0 if end['cls'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
