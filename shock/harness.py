"""Run a control law through a MIMO Shock problem's ladder: the shock record's harness (mshock_gym.harness, campaign 6)
restated on the shock suite's structures. NumPy only (the laws may need more).

    setup = load(problem_folder, 'srs')
    result = run(setup, law, noise=1e-3, seed=11)

The ladder is the record's engine ladder (EngineLadder below, ported line for line from the record's harness, engine mode): a
level is left only on a passing pulse (draft 0.2: every control's SRS inside Method 516.8's tolerance, -1.5/+3 dB on
at least 90 % of its points and every point within -3/+6 dB; draft 0.1 had every point within -3/+3 dB); six pulses at a level
without a pass end the run "not converged"; full level needs three passes; before a step up the first pulse's
over-test at the next level is predicted from the first pulses so far and the step halved, down to 0.75 dB, while
the prediction is over 3 dB. A law's probes (levels below the first ladder level) are the first rungs of that
table, as the record's harness had them, and are recorded as probes. Every pulse is played from rest through the structure's level rule
(shock_model.play). Measurement noise: Gaussian, `noise` times the full-level reference's RMS, on every measured
record, from numpy.random.default_rng([seed, 1]).
"""
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shock_model as sm  # noqa: E402

TOL = 3.0                       # the predicted over-test a step is halved against (the upper inner bound, +3 dB)
MIN_STEP_DB = 0.75


@dataclass
class Setup:
    """What a law is given, as the record's `Setup`: fs, n, nc, nd, band_lo, band_hi, reference (nc, n) full level,
    natural_hz, q, points (nc, nf) full level, frf (n/2 + 1, nc, nd). The harness keeps the structure apart."""
    fs: float
    n: int
    nc: int
    nd: int
    band_lo: float
    band_hi: float
    reference: np.ndarray
    natural_hz: np.ndarray
    q: float
    points: np.ndarray
    frf: np.ndarray
    fine_natural_hz: np.ndarray = None
    fine_points: np.ndarray = None
    info: dict = field(default_factory=dict)

    @property
    def lines(self):
        return self.n // 2 + 1

    @property
    def reference_rms(self):
        return float(np.sqrt(np.mean(self.reference ** 2)))

    def drive_mask(self):
        f = np.arange(self.lines) * self.fs / self.n
        return (f >= self.band_lo) & (f <= self.band_hi)


def load(folder, kind):
    """(Setup, structure, target record, problem record) of one problem: `folder` and target kind."""
    meta = json.load(open(os.path.join(folder, 'problem.json')))
    st = sm.rm.load_structure(os.path.join(folder, 'structure.json'))
    tg = json.load(open(os.path.join(folder, f'target_{kind}.json')))
    ref = np.fromfile(os.path.join(folder, tg['reference']['file']), '<f8').reshape(tg['reference']['shape'])
    H = sm.frf(st)
    s = Setup(fs=sm.FS, n=sm.N, nc=ref.shape[0], nd=st.n_drives, band_lo=tg['band_hz'][0], band_hi=tg['band_hz'][1],
              reference=ref, natural_hz=np.array(tg['natural_hz']), q=tg['q'], points=np.array(tg['points']), frf=H,
              fine_natural_hz=np.array(tg['fine_natural_hz']), fine_points=np.array(tg['fine_points']),
              info=dict(problem=meta['structure']['id'] + f"_s{meta['structure']['seed']:02d}", kind=kind))
    return s, st, tg, meta


class EngineLadder:
    """The record's EngineLadder (mshock_gym.harness), line for line."""

    def __init__(self, table, per_level=6, full_pulses=3, tol=TOL):
        self.table = list(table)
        self.level = self.table[0]
        self.per_level, self.full_pulses, self.tol = per_level, max(1, full_pulses), tol
        self.passes = self.without = 0
        self.first_pending = True
        self.firsts = []
        self.notes = []

    def predict(self, lvl):
        if not self.firsts:
            return None
        if len(self.firsts) == 1:
            return self.firsts[0][1]
        (l0, e0), (l1, e1) = self.firsts[-2], self.firsts[-1]
        slope = (e1 - e0) / (l1 - l0) if abs(l1 - l0) > 1e-12 else 0.0
        return e1 + slope * (lvl - l1)

    def on_judged(self, passed, over):
        if self.first_pending:
            self.firsts.append((self.level, over))
            self.first_pending = False
        if not passed:
            self.without += 1
            return 'not converged' if self.without >= self.per_level else 'held'
        self.without = 0
        self.passes += 1
        if self.level >= -1e-9:
            return 'completed' if self.passes >= self.full_pulses else 'held'
        planned = next((x for x in self.table if x > self.level + 1e-9), 0.0)
        step = planned - self.level
        pp = self.predict(planned)
        if pp is not None and pp > self.tol:
            while step / 2 >= MIN_STEP_DB - 1e-9 and (self.predict(self.level + step) or 0.0) > self.tol:
                step /= 2
            self.notes.append(f'step from {self.level:+.2f} halved to {self.level + step:+.2f} (predicted {pp:+.2f} at {planned:+.2f})')
        self.level += step
        if abs(self.level) < 1e-9:
            self.level = 0.0
        self.passes = 0
        self.first_pending = True
        return 'advanced'


@dataclass
class Pulse:
    index: int
    level_db: float
    next_level_db: float
    drives: np.ndarray
    responses: np.ndarray
    errors_db: np.ndarray


class Declined(Exception):
    """A law's refusal to fire (nothing is played)."""


class Law:
    name = 'law'

    def ladder(self, table):
        return list(table)

    def start(self, setup):
        pass

    def seed(self, level_db):
        raise NotImplementedError

    def iterate(self, pulse):
        raise NotImplementedError

    def next_level(self, history, planned):
        return planned

    def report(self):
        return {}


def classical_band(y, ref, tg, scale):
    """Method 516.8's classical reading of each control's response (rows, samples) at `scale`, against the target
    waveform (the projected compensated pulse, `ref` at full level), A the specification's peak at the level:
      band_pct       the largest departure from the target over the pulse window widened by 20 % each side, % of A
      duration_pct   the pulse's width above 10 % of its peak against the target's, % (TD within +/-10 %)
      velocity_pct   the velocity change over the pulse window against the target's, % (within +/-20 %)
      compensation_pct  the largest magnitude before and after the widened window over the compensation pulses'
                     span (COMP_FACTOR x TD each side), % of A (inside the +/-20 % baseline band)
      post_slope_pct (sawtooth) the time from the terminal peak to inside 20 % of A, % of TD (under 10 %)"""
    if tg['kind'] == 'srs':
        return None
    _, _, (a0, a1) = sm.classical(tg['kind'])
    s = tg['scale'] * scale
    t = np.arange(y.shape[1]) / sm.FS
    w = a1 - a0
    win = (t >= a0 - 0.2 * w) & (t <= a1 + 0.2 * w)
    dev = np.max(np.abs(y[:, win] - scale * ref[:, win]), axis=1) / s * 100.0
    m = (t >= a0) & (t <= a1)
    v = np.sum(y[:, m], axis=1)
    v0 = scale * np.sum(ref[:, m], axis=1)
    tc = sm.COMP_FACTOR * sm.T_CLASSICAL
    comp = ((t >= a0 - tc) & (t < a0 - 0.2 * w)) | ((t > a1 + 0.2 * w) & (t <= a1 + tc))
    cmp_pct = np.max(np.abs(y[:, comp]), axis=1) / s * 100.0

    def width(x):
        k0 = np.flatnonzero(win)[0] + int(np.argmax(x[win]))
        thr = 0.1 * x[k0]
        lo = k0
        while lo > 0 and x[lo - 1] > thr:
            lo -= 1
        hi = k0
        while hi < x.size - 1 and x[hi + 1] > thr:
            hi += 1
        return (hi - lo + 1) / sm.FS, k0

    dur, post = [], []
    for i in range(y.shape[0]):
        wy, ky = width(y[i])
        wr, _ = width(scale * ref[i])
        dur.append(100 * (wy / wr - 1))
        if tg['kind'] == 'sawtooth':
            k = ky
            while k < y.shape[1] - 1 and y[i, k] > 0.2 * s:
                k += 1
            post.append(100 * (k - ky) / sm.FS / sm.T_CLASSICAL)
    out = dict(band_pct=[round(float(x), 2) for x in dev], velocity_pct=[round(float(x), 2) for x in 100 * (v / v0 - 1)],
               duration_pct=[round(float(x), 2) for x in dur], compensation_pct=[round(float(x), 2) for x in cmp_pct])
    if post:
        out['post_slope_pct'] = [round(float(x), 2) for x in post]
    return out


def run(setup, st, law, tg=None, noise=1e-3, seed=11, table=(-12.0, -9.0, -6.0, -3.0, -1.5, 0.0), per_level=6,
        max_pulses=60, first_level=-12.0, keep_modes=False):
    """One run. Returns a dict: records (one per pulse: kind, level, over, worst, absolute, the SRS errors (nc, nf)
    rounded to 0.01 dB, the twelfth-octave over, drive RMS and peak, law seconds, the state reached: median and
    largest peak ratio and the signed frequency change at the peak by effect), end, ladder notes, law report."""
    sigma = noise * setup.reference_rms
    rng = np.random.default_rng([seed, 1])
    lad = law.ladder(list(table))
    eng = EngineLadder(lad, per_level)          # the record's: the law's probes are the first rungs of the engine's table
    am = sm.article_modes(st)
    recs, history = [], []
    # Effective duration (SRS targets): each control's TE against its target waveform's, both by Method 516.8
    # Annex A 1.3.2 with the target's crest factor (3 when it gives none)
    te_cf = float((tg or {}).get('crest_factor') or sm.TE_CREST_FACTOR)
    te_target = (sm.effective_duration(setup.reference, crest_factor=te_cf)
                 if tg is not None and tg['kind'] == 'srs' else None)

    def measure(d):
        y, lr = sm.play(st, d)
        if sigma > 0:
            y = y + sigma * rng.standard_normal(y.shape)
        return y, lr

    def record(kind, level, d, y, lr, law_s):
        sc = 10 ** (level / 20)
        e = sm.error_db(sm.srs(y, setup.fs, setup.natural_hz, setup.q, sc), setup.points)
        same = setup.fine_natural_hz.shape == setup.natural_hz.shape and np.array_equal(setup.fine_natural_hz, setup.natural_hz)
        e12 = e if same else sm.error_db(sm.srs(y, setup.fs, setup.fine_natural_hz, setup.q, sc), setup.fine_points)
        r = dict(index=len(recs), kind=kind, level_db=float(level), over_db=float(e.max()), worst_db=float(np.abs(e).max()),
                 abs_db=float(level + e.max()), over12_db=float(e12.max()), excess516_db=float(sm.excess_516(e).max()),
                 errors_db=np.round(e, 2).tolist(),
                 drive_rms=float(np.sqrt(np.mean(d ** 2)) / sc), drive_peak=float(np.abs(d).max()), law_s=law_s)
        if st.rules is not None:
            rr = lr.peak_envelope / st.reference_amplitude
            ch = sm.rm.frequency_change(st, rr)
            r['state'] = dict(peak_ratio_p50=float(np.median(rr[am])), peak_ratio_max=float(rr[am].max()),
                              largest_drop=float(ch[am].min()), largest_rise=float(ch[am].max()))
        if tg is not None and tg['kind'] != 'srs':
            r['classical'] = classical_band(y, setup.reference, tg, sc)
        if te_target is not None:
            te = sm.effective_duration(y, crest_factor=te_cf)
            r['te_s'] = np.round(te, 6).tolist()
            r['te_ratio'] = np.round(te / te_target, 4).tolist()
        recs.append(r)
        return e

    end = 'completed'
    try:
        law.start(setup)
        t0 = time.perf_counter(); d = law.seed(lad[0]); law_s = time.perf_counter() - t0
        k, level = 0, lad[0]
        while True:
            if not np.all(np.isfinite(d)):
                end = 'law fault: drives not finite'; break
            y, lr = measure(d)
            if not np.all(np.isfinite(y)):
                end = 'divergence: responses not finite'; break
            kind = 'probe' if level < first_level - 1e-9 else 'pulse'
            e = record(kind, level, d, y, lr, law_s)
            history.append(recs[-1])
            tick = eng.on_judged(sm.within_516(e), float(e.max()))
            planned = eng.level if tick in ('held', 'advanced') else None
            if tick == 'not converged':
                end = 'not converged'
            nxt = law.next_level(history, planned)
            if nxt is None and planned is not None:
                end = 'stopped by the law'
            last = nxt is None or len(history) >= max_pulses
            if len(history) >= max_pulses and nxt is not None:
                end = 'pulse limit'
            p = Pulse(index=k, level_db=level, next_level_db=level if last else nxt, drives=d, responses=y, errors_db=e)
            t0 = time.perf_counter(); d = law.iterate(p); law_s = time.perf_counter() - t0
            if last:
                break
            k, level = k + 1, nxt
    except Declined as ex:
        end = f'declined: {ex}'
    except Exception as ex:  # a law fault ends the run, as the engine's would
        end = f'law fault: {type(ex).__name__}: {ex}'
    try:
        rep = law.report()
    except Exception as ex:
        rep = {'error': str(ex)}
    return dict(problem=setup.info.get('problem'), kind=setup.info.get('kind'), law=law.name, noise=noise, seed=seed,
                ladder=lad, first_level_db=first_level, records=recs, end=end, ladder_notes=eng.notes, law_report=rep,
                **({} if te_target is None else dict(te_target_s=np.round(te_target, 6).tolist(), te_crest_factor=te_cf)))
