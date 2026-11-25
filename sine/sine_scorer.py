"""Scorer for the MIMO Sine problems of the Open Benchmark Set (draft 0.1). Several readings, never one verdict.

    python sine/sine_scorer.py <sine problem folder> <run file> [--test T1] [--out card.json]

The run is the runner's `msnb-run/1` folder or its run.json (the runner's FORMAT_RUN.md; FORMAT_SINE.md section 7
lists the fields this scorer reads; `read_run` is the one place that reads them). NumPy only.

Readings (FORMAT_SINE.md section 8):
  amplitude    each control's amplitude against the reference along the whole judged run, against each standard's
               band (IEC 60068-2-6 4.1.4.1 and 4.1.4.2; MIL-STD-810H Method 514.8 Table 514.8-III): share inside,
               worst excursion and where; composites as the standards define them
  phase        each control's phase against the reference: reported, not judged (no standard read sets one)
  resonances   the worst over-test on the way through each resonance band, controls and held-out gauges
  practice     the lines of the proposed practice (Hertzforge Journal 2026-001, section 5) read for a tone, with each
               departure from the article's statement written in the card
  frequency    the realized sweep rate, the frequency of a dwell or a step, the duration, against the standards
  completion   the outcome and why the run ended; the share of the test covered at full level
  drive        the drive vector's norm and the largest drive voltage; the effort over the ideal drive
  state        at stage 2: how far the resonances moved as tested, beside the article in service
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sine_model as sm  # noqa: E402
rm = sm.rm

SCORER_VERSION = '0.1-draft'
CARD_SCHEMA = 'hertzforge-benchmark-sine-card/0.1-draft'
LEVEL_DB = 1.5                 # practice line 1 (2026-001 P4)
GRADES = (('A', 3.0), ('B', 4.5), ('C', 6.0))      # line 2 (P5)
ALARM_DB = 6.0                 # line 3 (P6)
UNDER_DB = -3.0                # line 4 (P7)
EXACT_FIT_DB = 1.0             # line 6: the Random scorer's flag value (provisional there too)
FULL_LEVEL_DB = -0.05          # an update at or above this level is at full level
RES_MIN_OCT = 1.0 / 96.0       # a resonance window is at least half the runner's 1/48-octave recording step each side
OVER_COUNTS_DB = (1.5, 3.0, 6.0)

DEPARTURES = [
    'line 1 and 2 are read on tone amplitudes (dB of the amplitude, 20 log10), not on spectral densities in bands; '
    'the 1/3-octave bands of line 2 hold the judged updates whose frequency falls in them',
    'the held-out gauges\' target is their small-signal response to the ideal (least-squares) drive at full level, not '
    'a unit in service: a sine specification has no service measurement',
    'the held-out gauges as tested are estimated, not measured: the run\'s drive at each point through the modal model '
    'with the followed modes at the run\'s true_modes frequencies (the runner records only the controls and limit channels)',
    'stress: the article\'s peak fibre stress per element as tested (drive route through the modal model, the modes in '
    'the state the run reports, small signal where it reports none) against the same under the ideal drive at the '
    'same frequencies; the article in service is not a sine state, so the ideal drive stands in for it',
    'line 4\'s coverage margin is not computed (no level is raised after a sine run); the share under is reported',
    'line 6\'s condition is of the small-signal response at the controls over the reference grid',
]


def db20(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        return 20.0 * np.log10(np.abs(a) / np.abs(b))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def grade(v):
    for name, lim in GRADES:
        if v <= lim:
            return name
    return 'below C'


# ---------------------------------------------------------------------------------------------- the problem
class Problem:
    def __init__(self, folder):
        self.folder = folder
        self.meta = json.load(open(os.path.join(folder, 'problem.json')))
        self.st = rm.load_structure(folder)
        F = self.meta['files']
        self.f_grid = self.array(F['reference_hz'])
        self.amp = self.array(F['reference_amplitude'])
        self.phase = self.array(F['reference_phase'])
        self.d_ideal = self.array(F['ideal_drive'])
        self.y_ideal_h = self.array(F['ideal_held_out'])
        self.stress_modal = self.array(F['stress_modal']) if 'stress_modal' in F else None
        self.tests = {t: json.load(open(os.path.join(folder, v['file']))) for t, v in self.meta['tests'].items()}
        self.i_c, self.i_h = sm.controls(self.st), sm.gauges(self.st)
        self.names_c = [self.st.outputs[i]['name'] for i in self.i_c]
        self.names_h = [self.st.outputs[i]['name'] for i in self.i_h]
        self.spec = self.meta['full_level']['spec_amplitude_mps2_pk']

    def array(self, e):
        a = np.fromfile(os.path.join(self.folder, e['file']), e['dtype']).reshape(e['shape'])
        return a

    def reference_at(self, f):
        return sm.interp_reference(self.f_grid, self.amp, self.phase, np.asarray(f, float))

    def ideal_at(self, f):
        """The ideal (least-squares) drive and the held-out gauges' response to it at full level, computed at the
        frequencies f by the model the files were made with (the files hold the same on the reference grid; computing
        them avoids interpolating across a sharp resonance between grid points)."""
        f = np.asarray(f, float)
        uf, inv = np.unique(f, return_inverse=True)
        with np.errstate(all='ignore'):
            d = sm.ideal(self.st, uf, self.spec)[0]
            y = np.einsum('lgd,ld->lg', sm.frf(self.st, uf, self.i_h), d) if self.i_h.size else None
        return d[inv], (None if y is None else y[inv])

    def complex_at(self, arr, f):
        lf, lq = np.log(self.f_grid), np.log(np.clip(f, self.f_grid[0], self.f_grid[-1]))
        out = np.empty((len(lq), arr.shape[1]), complex)
        for j in range(arr.shape[1]):
            out[:, j] = np.interp(lq, lf, arr[:, j].real) + 1j * np.interp(lq, lf, arr[:, j].imag)
        return out


# ---------------------------------------------------------------------------------------------- the run
class Run:
    """The arrays of a run file's history, shapes (points, channels), complex where the file gives re and im."""


def _cx(h, key):
    if h.get(key + '_re') is None:
        return None
    return np.asarray(h[key + '_re'], float) + 1j * np.asarray(h[key + '_im'], float)


def _num(v):
    """A list that may hold nulls, as floats with NaN for null."""
    if v is None:
        return None
    return np.array([[np.nan if x is None else x for x in row] if isinstance(row, list) else (np.nan if row is None else row)
                     for row in v], float)


def read_run(path, st=None):
    """Read a run (`msnb-run/1`, the runner's FORMAT_RUN.md): the run folder or its run.json. The one place the run's
    layout is read. With the structure `st`, every mode's state along the run is filled in: the modes the runner
    followed at the frequencies it reports (true_modes), their damping from the level rule at that frequency
    (sine_model.zeta_from_frequency), every other mode small signal."""
    if os.path.isdir(path):
        path = os.path.join(path, 'run.json')
    raw = json.load(open(path))
    h = raw['history']
    r = Run()
    r.path, r.raw = path, raw
    r.outcome = raw.get('outcome') or {}
    r.case_id = raw.get('case_id', '')
    r.t = np.asarray(h['time_s'], float)
    r.f = np.asarray(h['frequency_hz'], float)
    r.level_db = np.asarray(h['level_db'], float)
    r.judged = np.asarray(h['judged'], bool) if h.get('judged') is not None else np.ones(r.t.size, bool)
    r.settling = np.asarray(h['settling'], bool) if h.get('settling') is not None else np.zeros(r.t.size, bool)
    r.C = _cx(h, 'control')
    r.Rrun = _cx(h, 'reference')
    r.D = _cx(h, 'drive')
    r.L = _cx(h, 'limit')
    r.Lb = _num(h.get('limit_bound'))
    r.bin_min = _num(h.get('bin_error_db_min'))
    r.bin_max = _num(h.get('bin_error_db_max'))
    r.mode_hz = r.mode_zeta = None
    r.modes_followed = []
    tm = raw.get('true_modes') or {}
    if st is not None:
        fn = np.broadcast_to(st.f_hz, (r.t.size, st.n_modes)).copy()
        ze = np.broadcast_to(st.zeta, (r.t.size, st.n_modes)).copy()
        if tm.get('tracked') and tm.get('modes'):
            F = np.asarray(tm['frequency_hz'], float)
            for k, m in enumerate(tm['modes']):
                i = int(m['index'])
                fn[:, i] = F[:, k]
                ze[:, i] = sm.zeta_from_frequency(st, i, F[:, k])
                r.modes_followed.append(i)
        r.mode_hz, r.mode_zeta = fn, ze
    return r


def estimated_response(st, run, rows):
    """The response at `rows` (output indices) to the run's drive at each point, through the modal model with every
    mode in the state of read_run: an estimate (the runner records the controls and limit channels only)."""
    out = np.empty((run.t.size, len(rows)), complex)
    for p in range(run.t.size):
        with np.errstate(all='ignore'):
            H = rm.frf(st, np.array([run.f[p]]), rows=rows, fn=run.mode_hz[p], zeta=run.mode_zeta[p])[0]
        out[p] = H @ run.D[p]
    return out


# ---------------------------------------------------------------------------------------------- judged updates
def judged_mask(pr, run, test):
    """The runner's judged flag (outside the engine's settle window, the law's answer used, at the last level), at full
    level, and for a step or a dwell outside the suite's own settle window (the test's settle seconds from the first
    full-level update of the step or the dwell)."""
    full = run.level_db >= FULL_LEVEL_DB
    ok = full & run.judged
    if test['kind'] == 'sweep':
        return ok
    settle_s = test['settle']['seconds']
    newstep = np.concatenate([[True], np.abs(np.diff(run.f)) > 1e-6 * run.f[1:]])
    step = np.cumsum(newstep)
    for k in np.unique(step):
        idx = np.flatnonzero((step == k) & full)
        if idx.size:
            ok[idx] &= (run.t[idx] - run.t[idx[0]]) >= settle_s
    return ok


def weights(run, mask, kind):
    """Each judged point's share of the judged run: for a sweep each point stands for its 1/48-octave bin (equal
    shares where the run gives bins, else octaves between points); for a step or a dwell, time."""
    idx = np.flatnonzero(mask)
    w = np.zeros(run.t.size)
    if idx.size == 0:
        return w
    if idx.size == 1 or (kind == 'sweep' and run.bin_max is not None):
        w[idx] = 1.0 / idx.size
        return w
    x = np.log2(run.f[idx]) if kind == 'sweep' else run.t[idx]
    g = np.abs(np.gradient(x))
    if g.sum() <= 0:
        g = np.ones(idx.size)
    w[idx] = g / g.sum()
    return w


# ---------------------------------------------------------------------------------------------- readings
def amplitude_reading(pr, f, C, R, w, names, emin=None, emax=None):
    """emin, emax: each control's smallest and largest error over the judged updates its point stands for (a sweep's
    1/48-octave bin, dB, NaN where none), so the worst excursion is found between the points too."""
    ratio = np.abs(C) / np.abs(R)
    e_pt = 20 * np.log10(ratio)
    lo_e = e_pt if emin is None else np.fmin(e_pt, emin)
    hi_e = e_pt if emax is None else np.fmax(e_pt, emax)
    out = {}
    for key, tol in sm.TOL.items():
        lo, hi = sm.band_at(tol, f)
        if 'all' in tol and key != 'iec_check_point':           # composites: the average of the control amplitudes
            comp = np.mean(np.abs(C), axis=1) / np.mean(np.abs(R), axis=1)
            exc = np.maximum(20 * np.log10(comp / hi), 20 * np.log10(lo / comp))
            inside = exc <= 0
            k = int(np.argmax(exc))
            out[key] = dict(share_inside=float(np.sum(w * inside)), worst_excursion_db=float(exc[k]), at_hz=float(f[k]),
                            worst_error_db=float(20 * np.log10(comp[k])), composite='average of the control amplitudes '
                            'over the average of the reference amplitudes', clause=tol['clause'], status=tol['status'])
        else:
            up = hi_e - 20 * np.log10(hi)[:, None]
            dn = 20 * np.log10(lo)[:, None] - lo_e
            exc = np.maximum(up, dn)
            inside = exc <= 0
            k, c = np.unravel_index(int(np.argmax(exc)), exc.shape)
            worst_err = float(hi_e[k, c] if up[k, c] >= dn[k, c] else lo_e[k, c])
            per = (w[:, None] * inside).sum(0)
            out[key] = dict(share_inside=float(np.sum(w[:, None] * inside) / exc.shape[1]),
                            share_inside_every_control=float(np.sum(w * inside.all(1))),
                            share_inside_per_control={n: float(v) for n, v in zip(names, per)},
                            worst_excursion_db=float(exc[k, c]), at_hz=float(f[k]), at_control=names[c],
                            worst_error_db=worst_err, clause=tol['clause'], status=tol['status'])
    e = e_pt
    out['error_db'] = dict(mean=float(np.sum(w[:, None] * e) / e.shape[1]), p95_abs=float(np.percentile(np.abs(e), 95)),
                           worst_over=float(hi_e.max()), worst_under=float(lo_e.min()),
                           between_points='the run\'s bin extremes' if emax is not None else 'none given')
    return out


def phase_reading(f, C, R, w, names):
    psi = np.degrees(np.angle(C / R))
    a = np.abs(psi)
    k, c = np.unravel_index(int(np.argmax(a)), a.shape)
    return dict(judged=False, why='no standard read sets a phase tolerance for several exciters (IEC 60068-2-6 4.1.1 and '
                'MIL-STD-810H 514.8 4.2.2.2 say only that fixing points move substantially in phase)',
                p95_abs_deg=float(np.percentile(a, 95)), mean_deg=float(np.sum(w[:, None] * psi) / psi.shape[1]),
                worst_deg=float(psi[k, c]), at_hz=float(f[k]), at_control=names[c],
                share_inside_10_deg_analogy=float(np.sum(w[:, None] * (a <= sm.PHASE_ANALOGY_DEG)) / a.shape[1]),
                analogy='MIL-STD-810H Method 527.2 Annex E 5.2, random cross-spectra (+/-10 degrees), by analogy only')


def resonance_reading(pr, run, mask, E_c, E_h):
    """The worst over-test on the way through each resonance band: each in-band article mode, its half-power band
    (as tested when the run reports the modes' state, else small signal), at least RES_MIN_OCT each side."""
    st = pr.st
    am = np.flatnonzero(sm.article_modes(st))
    idx = np.flatnonzero(mask)
    rows = []
    for m in am:
        fm = run.mode_hz[idx, m] if run.mode_hz is not None else np.full(idx.size, st.f_hz[m])
        zm = run.mode_zeta[idx, m] if run.mode_zeta is not None else np.full(idx.size, st.zeta[m])
        half = np.maximum(zm / np.log(2.0), RES_MIN_OCT)
        inb = np.abs(np.log2(run.f[idx] / fm)) <= half
        if not inb.any():
            rows.append(dict(mode=int(m), f0_hz=float(st.f_hz[m]), judged_updates=0))
            continue
        ec = E_c[inb]
        eh = E_h[inb] if E_h is not None else None
        kc = np.unravel_index(int(np.argmax(ec)), ec.shape)
        worst_c = float(ec[kc])
        row = dict(mode=int(m), f0_hz=float(st.f_hz[m]), judged_updates=int(inb.sum()), worst_control_db=worst_c,
                   worst_control=pr.names_c[kc[1]], worst_control_hz=float(run.f[idx][inb][kc[0]]))
        if eh is not None:
            kh = np.unravel_index(int(np.argmax(eh)), eh.shape)
            row.update(worst_held_out_db=float(eh[kh]), worst_held_out=pr.names_h[kh[1]])
        row['worst_db'] = max(worst_c, row.get('worst_held_out_db', -np.inf))
        rows.append(row)
    seen = [r for r in rows if r['judged_updates']]
    worst = max(seen, key=lambda r: r['worst_db']) if seen else None
    return dict(resonances=len(rows), judged=len(seen), worst_db=None if worst is None else worst['worst_db'],
                worst_mode=None if worst is None else worst['mode'],
                count_over_db={f'{x:g}': int(sum(r['worst_db'] > x for r in seen)) for x in OVER_COUNTS_DB},
                window='each article mode\'s half-power band, as tested where the run reports the modes\' state, at least '
                       '1/96 octave each side', modes=rows,
                state=('as tested: the modes the runner followed at its true_modes frequencies, the rest small signal'
                       if run.modes_followed else 'small signal (the run follows no mode)'))


def third_octave_bands(f):
    return np.round(3 * np.log2(np.asarray(f) / 1000.0)).astype(int)


def practice_reading(pr, run, mask, w, C, R, Hh, Yh):
    """The 2026-001 report card lines for a tone (the departures are in DEPARTURES)."""
    f = run.f[mask]
    b = third_octave_bands(f)
    ub = np.unique(b)
    wm = w[mask]

    def band_db(num, den):                    # (bands,) dB of band-averaged power
        out = []
        for k in ub:
            s = b == k
            out.append(10 * np.log10(np.sum(wm[s] * num[s]) / np.sum(wm[s] * den[s])))
        return np.array(out)
    E = {}
    for tag, X, T in (('c', C, R), ('h', Hh, Yh)):
        if X is None:
            continue
        per = np.array([band_db(np.abs(X[:, i]) ** 2, np.abs(T[:, i]) ** 2) for i in range(X.shape[1])])   # (ch, bands)
        summed = band_db(np.sum(np.abs(X) ** 2, 1), np.sum(np.abs(T) ** 2, 1))
        E[tag] = dict(per=per, summed=summed)
    allb = np.concatenate([E[k]['per'] for k in E], 0)
    card = dict(
        line_1_level=dict(mean_error_db=float(allb.mean()), starting_value_db=LEVEL_DB, within=bool(abs(allb.mean()) <= LEVEL_DB),
                          what='mean dB error over controls and held-out gauges and 1/3-octave bands'),
        line_2_pattern={('held_out' if k == 'h' else 'controls'): dict(
            summed_rms_db=float(np.sqrt(np.mean(E[k]['summed'] ** 2))), per_channel_rms_db=float(np.sqrt(np.mean(E[k]['per'] ** 2))),
            grade=grade(float(np.sqrt(np.mean(E[k]['summed'] ** 2))))) for k in E},
        line_3_over_test=dict(worst_band_db=float(allb.max()), share_of_bands_over_alarm=float(np.mean(allb > ALARM_DB)),
                              alarm_db=ALARM_DB, alarm=bool(allb.max() > ALARM_DB), what='controls and held-out gauges, any band'),
    )
    if 'h' in E:
        eh = E['h']['per']
        card['line_4_under_test'] = dict(share_of_held_out_bands_more_than_3_db_under=float(np.mean(eh < UNDER_DB)),
                                         held_out_band_error_p10_db=float(np.percentile(eh, 10)),
                                         what='reported; the coverage margin is not computed')
    dd = pr.meta['reference']['declared_deviation']
    card['line_5_declared'] = dict(declared_deviation_rms_db=dd['amplitude_db_rms'], declared_deviation_min_db=dd['amplitude_db_min'],
                                   declared_deviation_max_db=dd['amplitude_db_max'],
                                   law_error_rms_db=float(np.sqrt(np.mean(E['c']['per'] ** 2))),
                                   what='the reference against the specification (the problem\'s record), and the controls '
                                        'against the reference in 1/3-octave bands')
    nd, nc = pr.st.n_drives, pr.i_c.size
    cres = float(np.sqrt(np.mean(E['c']['per'] ** 2)))
    cond = pr.meta['reference']['condition']
    card['line_6_rig'] = dict(drives=int(nd), controls=int(nc), condition_median=cond['median'], condition_p95=cond['p95'],
                              condition_max=cond['max'], control_residual_rms_db=cres,
                              exact_fit_flag=bool(nd >= nc and cres < EXACT_FIT_DB), exact_fit_flag_under_db=EXACT_FIT_DB)
    card['departures'] = DEPARTURES
    return card


def stress_reading(pr, run, mask):
    """Peak fibre stress per element as tested against the ideal drive's, over the judged updates."""
    if pr.stress_modal is None:
        return None
    st = pr.st
    idx = np.flatnonzero(mask)
    f = run.f[idx]
    w = 2 * np.pi * f[:, None]

    def q_of(D, fn, ze):
        wn = 2 * np.pi * fn
        h = 1.0 / (wn ** 2 - w ** 2 + 2j * ze * wn * w)              # (updates, modes)
        with np.errstate(all='ignore'):
            return h * (D @ st.drive_shape.T)
    fn = run.mode_hz[idx] if run.mode_hz is not None else np.broadcast_to(st.f_hz, (idx.size, st.n_modes))
    ze = run.mode_zeta[idx] if run.mode_zeta is not None else np.broadcast_to(st.zeta, (idx.size, st.n_modes))
    q_run = q_of(run.D[idx], fn, ze)
    q_id = q_of(pr.ideal_at(f)[0], np.broadcast_to(st.f_hz, (idx.size, st.n_modes)),
                np.broadcast_to(st.zeta, (idx.size, st.n_modes)))
    S = pr.stress_modal                                                 # (8 x elements, modes)
    ne = S.shape[0] // 8
    with np.errstate(all='ignore'):
        s_run = np.abs(q_run @ S.T).reshape(idx.size, ne, 8).max(2).max(0)
        s_id = np.abs(q_id @ S.T).reshape(idx.size, ne, 8).max(2).max(0)
    e = db20(s_run, s_id)
    return dict(elements=int(ne), error_db_p10_p50_p90=[float(x) for x in np.percentile(e, [10, 50, 90])],
                share_within_3_db=float(np.mean(np.abs(e) <= 3)), share_over_6_db=float(np.mean(e > 6)),
                worst_over_db=float(e.max()), worst_under_db=float(e.min()),
                state=('as tested: the followed modes at the run\'s true_modes frequencies, damping from the level rule there, '
                       'the rest small signal; through the run\'s drive' if run.modes_followed else 'small signal, through the run\'s drive'),
                what='per element, the largest peak fibre stress over the judged updates against the same under the ideal drive')


def frequency_reading(pr, run, mask, test, w):
    f, t = run.f[mask], run.t[mask]
    out = {}
    if test['kind'] == 'sweep' and f.size > 2:
        rate = 60.0 * np.abs(np.diff(np.log2(f))) / np.maximum(np.diff(t), 1e-12)
        want = test['rate_oct_per_min']
        ok = np.abs(rate / want - 1) <= sm.RATE_TOL['fraction']
        oct_ = np.abs(np.diff(np.log2(f)))
        span_t = t[-1] - t[0]
        sched = 60.0 * abs(np.log2(f[-1] / f[0])) / want
        out = dict(rate_stated_oct_per_min=want, rate_realized_oct_per_min=float(60.0 * abs(np.log2(f[-1] / f[0])) / max(span_t, 1e-12)),
                   share_of_sweep_within_rate_tolerance=float(np.sum(oct_ * ok) / max(oct_.sum(), 1e-300)),
                   rate_tolerance=sm.RATE_TOL, duration_s=float(span_t), scheduled_s=float(sched),
                   duration_within_5_pct=bool(abs(span_t / sched - 1) <= sm.DURATION_TOL['fraction']),
                   duration_tolerance=sm.DURATION_TOL)
    elif test['kind'] == 'dwell' and f.size:
        err = np.abs(f / test['dwell_hz'] - 1)
        # the engine's dwell clock stops in its settle window: the dwell is the full-level time from where that window
        # closed (the run's settling flag) to the end
        full = (run.level_db >= FULL_LEVEL_DB) & ~run.settling
        held = float(run.t[full][-1] - run.t[full][0]) if full.sum() > 1 else 0.0
        out = dict(largest_frequency_error_pct=float(100 * err.max()), within_iec_fixed_2_pct=bool(err.max() <= 0.02),
                   within_mil_0_1_pct=bool(err.max() <= 0.001), full_level_s=held, scheduled_s=test['seconds'],
                   held_from='the first full-level update outside the engine\'s settle window',
                   duration_within_5_pct=bool(abs(held / test['seconds'] - 1) <= sm.DURATION_TOL['fraction']),
                   tolerances=[sm.FREQ_TOL['iec_fixed'], sm.FREQ_TOL['mil'], sm.DURATION_TOL])
    elif test['kind'] == 'stepped' and f.size:
        lad = np.asarray(test['frequencies_hz'])
        near = lad[np.argmin(np.abs(np.log(f[:, None] / lad[None, :])), axis=1)]
        err = np.abs(f / near - 1)
        out = dict(largest_frequency_error_pct=float(100 * err.max()), within_iec_fixed_2_pct=bool(err.max() <= 0.02),
                   within_mil_0_1_pct=bool(err.max() <= 0.001), tolerances=[sm.FREQ_TOL['iec_fixed'], sm.FREQ_TOL['mil']])
    return out


def completion_reading(pr, run, mask, test):
    o = run.outcome
    full = run.level_db >= FULL_LEVEL_DB
    band = test['band_hz']
    # the engine runs the ladder's lower rungs on the test's own clock (its schedule.rs), so the start of a sweep or a
    # stepped test is run below full level: coverage is counted from where full level began to the test's far end, and
    # the share run below full level first is reported as ladder_share
    ladder_share = 0.0
    if test['kind'] == 'sweep':
        far = band[1] if test.get('direction', 'up') == 'up' else band[0]
        near = band[0] if far == band[1] else band[1]
        n_band = np.log2(band[1] / band[0])
        if full.sum() > 1:
            f0 = run.f[np.flatnonzero(full)[0]]
            span = abs(np.log2(far / f0))
            cov = float(min(1.0, abs(np.log2(run.f[full].max() / run.f[full].min())) / span)) if span > 0 else 0.0
            ladder_share = float(abs(np.log2(f0 / near)) / n_band)
        else:
            cov = 0.0
        what = 'octaves at full level over the octaves from where full level began to the band\'s far end'
    elif test['kind'] == 'stepped':
        lad = np.asarray(test['frequencies_hz'])
        hit = np.zeros(lad.size, bool)
        first = lad.size
        if full.any():
            first = int(np.argmin(np.abs(np.log(run.f[np.flatnonzero(full)[0]] / lad))))
        if mask.any():
            k = np.argmin(np.abs(np.log(run.f[mask][:, None] / lad[None, :])), axis=1)
            hit[np.unique(k)] = True
        cov = float(hit[first:].mean()) if first < lad.size else 0.0
        ladder_share = float(first / lad.size)
        what = 'steps with a judged update over the steps from the first at full level to the last'
    else:
        tj = float(run.t[mask][-1] - run.t[mask][0]) if mask.sum() > 1 else 0.0
        jw = test['seconds'] - test['settle']['seconds']
        cov = float(min(1.0, tj / jw)) if jw > 0 else 0.0
        what = 'judged seconds over the dwell\'s seconds after its settle window'
    return dict(outcome=o.get('class'), engine_outcome=o.get('engine_outcome'), guard=o.get('guard'), channel=o.get('channel'),
                at_hz=o.get('frequency_hz'), at_s=o.get('time_s'), at_level_db=o.get('level_db'), message=o.get('message'),
                coverage=cov, coverage_what=what, ladder_share=ladder_share, judged_updates=int(mask.sum()), updates=int(run.t.size),
                run_seconds=float(run.t[-1] - run.t[0]) if run.t.size > 1 else 0.0)


def drive_reading(pr, run, mask):
    D = run.D
    nrm = np.linalg.norm(D, axis=1)
    a = np.abs(D)
    k, j = np.unravel_index(int(np.argmax(a)), a.shape)
    summ = run.raw.get('summary') or {}
    out = dict(norm_max_v=float(nrm.max()), largest_drive_v=float(a[k, j]), largest_drive=int(j), largest_at_hz=float(run.f[k]),
               largest_drive_vpk_any_sample=summ.get('largest_drive_vpk'), over_ceiling_samples=summ.get('over_ceiling_samples'),
               engine_ceiling_v=sm.ENGINE_MAX_DRIVE_VPK,
               what='norm and largest drive over the history points; largest_drive_vpk_any_sample is the runner\'s summary')
    if mask.any():
        di = pr.ideal_at(run.f[mask])[0]
        eff = db20(nrm[mask], np.linalg.norm(di, axis=1))
        out.update(effort_db_p50=float(np.median(eff)), effort_db_p95=float(np.percentile(eff, 95)), effort_db_max=float(eff.max()),
                   effort_what='20 log10 of the drive norm over the ideal drive\'s norm, judged updates')
    return out


def state_reading(pr, run):
    st = pr.st
    if st.rules is None:
        return None
    sv = pr.meta.get('service_state') or {}
    art = sv.get('article') or {}
    lt = st.raw.get('level_targets') or {}
    am = np.flatnonzero(sm.article_modes(st))
    out = dict(in_service=dict(article_change_p10_p50_p90=art.get('frequency_change_p10_p50_p90'),
                               member=sv.get('member')),
               design_full_level_median_abs_change=float(np.median(np.abs(np.asarray(lt.get('frequency_change', [0.0]))[am])))
               if lt.get('frequency_change') else None,
               quasi_steady_at_full_level=pr.meta['checks'].get('state', {}).get('full_level'))
    am = np.array([m for m in am if m in set(run.modes_followed)], int)
    if am.size == 0:
        out['as_tested'] = None
        out['why'] = 'the run follows no article mode (true_modes)'
        return out
    ch = run.mode_hz[:, am] / st.f_hz[am] - 1.0
    k = np.argmax(np.abs(ch), axis=0)
    signed = ch[k, np.arange(am.size)]
    out['as_tested'] = dict(change_p10_p50_p90=[float(x) for x in np.percentile(signed, [10, 50, 90])],
                            median_abs_change=float(np.median(np.abs(signed))), largest_drop=float(signed.min()),
                            largest_rise=float(signed.max()), modes=int(am.size),
                            what='each article mode\'s largest frequency change during the run, signed')
    return out


# ---------------------------------------------------------------------------------------------- the card
def _subset(run, mask):
    r = Run()
    r.t, r.f, r.D = run.t[mask], run.f[mask], run.D[mask]
    r.mode_hz, r.mode_zeta = run.mode_hz[mask], run.mode_zeta[mask]
    return r


def score(problem_folder, run_path, test_id=None):
    pr = Problem(problem_folder)
    run = read_run(run_path, pr.st)
    run_path = run.path
    if test_id is None:
        test_id = run.case_id.split('__')[1]
    test = pr.tests[test_id]
    mask = judged_mask(pr, run, test)
    card = dict(schema=CARD_SCHEMA, scorer=dict(file='sine_scorer.py', version=SCORER_VERSION, sha256=sha256(os.path.abspath(__file__))),
                problem=dict(id=pr.meta['structure']['id'], seed=pr.meta['structure']['seed'], stage=pr.meta['stage'],
                             effects=pr.meta['effects'], structure_sha256=pr.meta['structure']['sha256'], test=test_id),
                run=dict(case_id=run.case_id, case_sha256=run.raw.get('case_sha256'), code_commit=run.raw.get('code_commit'),
                         runner_sha256=run.raw.get('runner_sha256'), run_sha256=sha256(run_path)))
    card['completion'] = completion_reading(pr, run, mask, test)
    card['drive'] = drive_reading(pr, run, mask)
    if not mask.any():
        card['scored'] = False
        card['why'] = 'no judged update (the run never held full level outside a settle window)'
        return card
    card['scored'] = True
    w = weights(run, mask, test['kind'])
    f = run.f[mask]
    R = pr.reference_at(f)
    C = run.C[mask]
    if run.Rrun is not None:
        card['run_reference_vs_problem_db_max'] = float(np.max(np.abs(db20(run.Rrun[mask], R))))
    wm = w[mask]
    card['amplitude'] = amplitude_reading(pr, f, C, R, wm, pr.names_c,
                                          None if run.bin_min is None else run.bin_min[mask],
                                          None if run.bin_max is None else run.bin_max[mask])
    card['phase'] = phase_reading(f, C, R, wm, pr.names_c)
    Yh = pr.ideal_at(f)[1]
    Hh = estimated_response(pr.st, _subset(run, mask), list(pr.i_h)) if pr.i_h.size else None
    E_c = db20(C, R)
    E_h = db20(Hh, Yh) if Hh is not None else None
    card['resonances'] = resonance_reading(pr, run, mask, E_c, E_h)
    card['practice'] = practice_reading(pr, run, mask, w, C, R, Hh, Yh)
    card['stress'] = stress_reading(pr, run, mask)
    card['frequency'] = frequency_reading(pr, run, mask, test, w)
    card['state'] = state_reading(pr, run)
    if run.L is not None and test.get('limit'):
        lim = np.abs(run.L[:, 0])
        bound = test['limit']['bound_mps2_pk']
        over = db20(lim[mask], bound)
        if run.Lb is not None:
            card['limit_bound_in_run_vs_test_db_max'] = float(np.nanmax(np.abs(db20(run.Lb[mask, 0], bound))))
        card['limit'] = dict(bound_mps2_pk=bound, worst_over_bound_db=float(over.max()),
                             share_over_bound=float(np.sum(wm * (over > 0))), share_over_half_db=float(np.sum(wm * (over > 0.5))),
                             what='the limit channel against its bound, judged updates (requirements M7: worst within +0.5 dB)')
    return card


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('problem'); ap.add_argument('run'); ap.add_argument('--test', default=None); ap.add_argument('--out', default=None)
    a = ap.parse_args()
    card = score(a.problem, a.run, a.test)
    txt = json.dumps(card, indent=1, default=float)
    if a.out:
        open(a.out, 'w').write(txt)
    else:
        print(txt)


if __name__ == '__main__':
    main()
