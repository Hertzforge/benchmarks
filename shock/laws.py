"""The MIMO Shock laws of the screen, in Python, against the shock suite's harness (harness.Law).

Baselines (the product's starter laws, ported from their Rust sources, hertzforge-mimo-shock-law-math):
  inverse   D+ = D + g(f) G (R - Y), G the regularized inverse (1e-3), the per-line watchdog (9 lines, cut x0.5 on
            5 % growth, restore x1.2) (inverse.rs, InverseParams::default)
  shape     per-control SRS-ratio gains on the lines (log-f interpolation between points, shape's weight table,
            6 dB step limit) and delay alignment (relax 0.5, bound 2 ms), the identified model frozen
            (shape.rs, safety off, carry Keep; the record's numpy port e3law/port.py ShapePy, line for line)
  paired    each control paired with one drive, the pair's own path inverted, per-line magnitude ratio smoothed
            over 15 lines, weight 0.8, 6 dB step limit (paired.rs, PairedParams::diagonal)
Campaign 6 candidates (ported from the record's reference implementations, mshock-research):
  followed       shape on round 4's followed softening family: probes at -40 and -34 dB, per-octave drops fitted
                 from the pulses (coarse -5 to 40 % by 1 %, fine +/-1 % by 0.1 %), carried up as a bounded power of
                 the amplitude, margin 1.5, ceiling 12 %, the family inverse over members 0, 1/2, 1 of the drop
                 (e3law/port.py FollowedProvider + ShapePy; the warp, the family inverse and the fit, which the
                 record reached through its Rust bridge, are restated here in NumPy: see Soft)
  followed-hold  followed with no ceiling, the probe check (scale a probe down when its prediction plus the
                 measured prediction error carried in amplitude is over +3 dB), the resolved power (gate 0.5 %),
                 and the hold (a repeat pulse inside 3 dB at every point plays the same drive)
                 (audit_e3/ae3.py E3Var(freeze=True, dead_band_db=3.0) over e3law/law.py E3Law)
  srs-design     for more controls than drives: per control a smooth log gain and a bounded delay at each SRS point,
                 the drive the regularized inverse of those references inside a support window, the parameters found
                 by gradient descent on the predicted from-rest SRS error; declines before the first pulse when its
                 own prediction misses by more than 3 dB; between pulses the measured-minus-predicted error is kept
                 as a bias and the design re-optimized; ends the run after a pulse more than 3 dB over; square
                 frames are handed to followed-hold (f2/law_f2.py, simplified: see SrsDesign)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shock_model as sm  # noqa: E402
from harness import Law, Declined  # noqa: E402

regularized_inverse = sm.regularized_inverse
WEIGHTS = [(-12.0, 0.5), (-9.0, 0.72), (-6.0, 0.82), (-3.0, 0.85), (0.0, 0.85)]


def weight_at(level):
    t = WEIGHTS
    if level <= t[0][0]:
        return t[0][1]
    for (l0, w0), (l1, w1) in zip(t[:-1], t[1:]):
        if level <= l1:
            return w0 + (level - l0) / (l1 - l0) * (w1 - w0)
    return t[-1][1]


def to_lines_log(at_hz, values, lines, df):
    la = np.log(np.maximum(at_hz, 1e-12))
    lf = np.log(np.maximum(np.arange(lines) * df, 1.0))
    return np.interp(lf, la, values)


def apply(M, X):
    return np.einsum('lrc,cl->rl', M, X)


def estimate_delay(y, r, fs):
    n = y.size
    m = 2 * n
    c = np.fft.irfft(np.fft.rfft(y, m) * np.conj(np.fft.rfft(r, m)), m)
    k = int(np.argmax(np.abs(c)))
    if abs(c[k]) <= 0:
        return 0.0
    sg = np.sign(c[k])
    ym, y0, yp = c[(k - 1) % m] * sg, c[k] * sg, c[(k + 1) % m] * sg
    den = ym - 2 * y0 + yp
    frac = float(np.clip(0.5 * (ym - yp) / den, -0.5, 0.5)) if den < 0 else 0.0
    lag = k if k < n else k - m
    return (lag + frac) / fs


def srs_err(y, s, level):
    return sm.error_db(sm.srs(y, s.fs, s.natural_hz, s.q, 10 ** (level / 20)), s.points)


# ------------------------------------------------------------------------------------------------ inverse, paired
class Inverse(Law):
    name = 'inverse'

    def start(self, s):
        self.s, self.mask = s, s.drive_mask()
        self.G = regularized_inverse(s.frf, self.mask)
        self.R = np.fft.rfft(s.reference, axis=1)
        self.D = apply(self.G, self.R)
        self.g = np.ones(s.lines)
        self.prev = None

    def seed(self, level):
        return np.fft.irfft(self.D, n=self.s.n, axis=1) * 10 ** (level / 20)

    def iterate(self, p):
        sc = 10 ** (p.level_db / 20)
        Y = np.fft.rfft(p.responses, axis=1) / sc
        D = np.fft.rfft(p.drives, axis=1) / sc
        E = self.R - Y
        e = np.convolve(np.sum(np.abs(E) ** 2, axis=0), np.ones(9) / 9, mode='same')
        if self.prev is not None:
            grow = e > 1.05 * self.prev
            self.g = np.where(grow, 0.5 * self.g, np.minimum(1.0, 1.2 * self.g))
        self.prev = e
        self.D = D + self.g[None] * apply(self.G, E)
        self.D[:, ~self.mask] = 0
        return np.fft.irfft(self.D, n=self.s.n, axis=1) * 10 ** (p.next_level_db / 20)


class Paired(Law):
    name = 'paired'

    def start(self, s):
        self.s, self.mask = s, s.drive_mask()
        self.pairs = list(range(min(s.nc, s.nd)))
        self.R = np.fft.rfft(s.reference, axis=1)
        self.D = np.zeros((s.nd, s.lines), complex)
        for c, d in enumerate(self.pairs):
            h = s.frf[:, c, d]
            ok = self.mask & (np.abs(h) > 0)
            self.D[d, ok] = self.R[c, ok] / h[ok]

    def seed(self, level):
        return np.fft.irfft(self.D, n=self.s.n, axis=1) * 10 ** (level / 20)

    def iterate(self, p):
        sc = 10 ** (p.level_db / 20)
        Y = np.fft.rfft(p.responses, axis=1) / sc
        D = np.fft.rfft(p.drives, axis=1) / sc
        lim = 6.0 / 20 * np.log(10)
        k = np.ones(15) / 15
        for c, d in enumerate(self.pairs):
            want = np.convolve(np.abs(self.R[c]) ** 2, k, mode='same')
            got = np.convolve(np.abs(Y[c]) ** 2, k, mode='same')
            ok = self.mask & (want > 0) & (got > 0)
            step = np.clip(0.5 * 0.8 * np.log(want[ok] / got[ok]), -lim, lim)
            self.D[d, ok] = D[d, ok] * np.exp(step)
        return np.fft.irfft(self.D, n=self.s.n, axis=1) * 10 ** (p.next_level_db / 20)


# ------------------------------------------------------------------------------------------------ the softening family
class Soft:
    """The followed family's fixed pieces in NumPy (the record reached them through its Rust bridge: warp.rs,
    softening.rs, first_pulse.rs). Knots at third octaves from 40 Hz to the band's top; bands of one octave from
    the lowest knot; a member lowers band b's resonances by frac * drops[b] (the delay-free response read at
    f / (1 - d(f)), d linear in log f between knots, the row's bulk delay put back); the family inverse is the
    weighted least-squares inverse over the members (regularization 1e-3 of the largest member's largest singular
    value) and the model handed to shape is its equivalent model G^H (G G^H)^-1. Departures from the Rust warp: no
    decay parameters, no secant refinement, linear interpolation between lines."""

    def __init__(self, s, base=None):
        self.s = s
        self.base = s.frf if base is None else base
        self.mask = s.drive_mask()
        lo = 40.0
        kn = lo * 2.0 ** (np.arange(0, 40) / 3.0)
        self.knots = kn[kn <= s.band_hi * 2 ** (1 / 6)]
        self.f = np.arange(s.lines) * s.fs / s.n
        hi = self.knots[-1]
        self.nb = max(1, int(np.floor(np.log2(hi / lo))) + 1)
        band_of = lambda f: np.minimum(np.floor(np.maximum(np.log2(f / lo), 0)).astype(int), self.nb - 1)
        self.knot_band = band_of(self.knots)
        lb = np.full(s.lines, -1)
        on = self.mask & (self.f > lo) & (self.f < hi)
        lb[on] = band_of(self.f[on])
        self.line_band = lb
        # each row's bulk delay from its impulse response's peak
        h = np.fft.irfft(self.base * self.mask[:, None, None], n=s.n, axis=0)
        en = np.sum(h ** 2, axis=2)                                              # (n, nc)
        k = np.argmax(en[: s.n // 2], axis=0)
        self.delay = k / s.fs
        self.rot = np.exp(1j * 2 * np.pi * self.f[:, None] * self.delay[None])  # (lines, nc)
        self.flat = self.base * self.rot[:, :, None]

    def member(self, drops, frac=1.0):
        x = -frac * np.asarray(drops)[self.knot_band]
        lf = np.log(np.maximum(self.f, 1.0))
        d = np.interp(lf, np.log(self.knots), x)                                 # 1/a - 1 per line
        src = self.f / np.maximum(1.0 + d, 1e-3)
        pos = np.clip(src * self.s.n / self.s.fs, 0, self.s.lines - 1.000001)
        i = np.floor(pos).astype(int); w = (pos - i)[:, None, None]
        Hn = self.flat[i] * (1 - w) + self.flat[i + 1] * w
        H = Hn / self.rot[:, :, None]
        return np.where(self.mask[:, None, None], H, self.base)

    def family(self, drops, members=(0.0, 0.5, 1.0), lam=1e-3):
        Ms = [self.member(drops, m) for m in members]
        nd = self.s.nd
        A = sum(np.conj(np.swapaxes(H, 1, 2)) @ H for H in Ms) / len(Ms)
        B = sum(np.conj(np.swapaxes(H, 1, 2)) for H in Ms) / len(Ms)
        smax = np.max([np.linalg.norm(H, ord=2, axis=(1, 2)) for H in Ms], axis=0)
        mu = np.maximum(lam * smax, 1e-12 * smax)
        A = A + (mu ** 2)[:, None, None] * np.eye(nd)[None]
        ok = self.mask & (smax > 0)
        G = np.zeros((self.s.lines, nd, self.s.nc), complex)
        G[ok] = np.linalg.solve(A[ok], B[ok])
        M = self.base.copy()
        GG = G[ok] @ np.conj(np.swapaxes(G[ok], 1, 2))
        M[ok] = np.conj(np.swapaxes(G[ok], 1, 2)) @ np.linalg.inv(GG)
        return M

    def _sums(self, H, P):
        out = np.zeros((self.nb, 2))
        on = self.line_band >= 0
        for D, Y in P:
            r = Y[:, on] - np.einsum('lcd,dl->cl', H[on], D[:, on])
            np.add.at(out[:, 0], self.line_band[on], np.sum(np.abs(r) ** 2, axis=0))
            np.add.at(out[:, 1], self.line_band[on], np.sum(np.abs(Y[:, on]) ** 2, axis=0))
        return out

    def fit(self, drives_list, responses_list, levels):
        P, sc = [], []
        for d, y, l in zip(drives_list, responses_list, levels):
            s = 10 ** (l / 20)
            P.append((np.fft.rfft(d, axis=1) / s, np.fft.rfft(y, axis=1) / s)); sc.append(s)
        rel = lambda S: np.where(S[:, 1] > 0, np.sqrt(S[:, 0] / np.where(S[:, 1] > 0, S[:, 1], 1)), np.nan)
        base = self._sums(self.member(np.zeros(self.nb), 0.0), P)
        r0 = rel(base)
        best = np.column_stack([r0, np.zeros(self.nb)])
        def consider(drops):
            r = rel(self._sums(self.member(drops, 1.0), P))
            b = r < best[:, 0]
            best[b, 0], best[b, 1] = r[b], drops[b]
        for d in np.arange(-0.05, 0.40 + 1e-9, 0.01):
            consider(np.full(self.nb, d))
        centre = best[:, 1].copy()
        for i in range(-10, 11):
            consider(centre + i * 0.001)
        on = self.line_band >= 0
        R = np.fft.rfft(self.s.reference, axis=1)
        ref_e = np.zeros(self.nb)
        np.add.at(ref_e, self.line_band[on], np.sum(np.abs(R[:, on]) ** 2, axis=0))
        amp = np.zeros(self.nb)
        for (D, Y), s in zip(P, sc):
            ye = np.zeros(self.nb)
            np.add.at(ye, self.line_band[on], np.sum(np.abs(Y[:, on]) ** 2, axis=0))
            amp += np.where(ref_e > 0, s * np.sqrt(ye / np.where(ref_e > 0, ref_e, 1)), 0) / len(P)
        e_max = base[:, 1].max()
        sig = best[:, 0] <= 0.9 * r0
        drops = np.where((base[:, 1] < 1e-4 * e_max) | ~sig, 0.0, best[:, 1])
        return dict(drops=drops, raw=best[:, 1].copy(), amplitude=amp, r0=r0, residual=best[:, 0].copy())


class FollowedProvider:
    """e3law/port.py FollowedProvider, line for line (power 'measured' or 'resolved' with its gate)."""

    def __init__(self, soft, base, ceiling=0.12, margin=1.5, p_max=1.0, probe_levels=2, members=(0.0, 0.5, 1.0),
                 narrow=1.0, power='measured', gate=0.005):
        self.power, self.gate = power, gate
        self.soft, self.base = soft, base
        self.ceiling = np.inf if ceiling is None else ceiling
        self.margin, self.p_max, self.probe_levels = margin, p_max, probe_levels
        self.members, self.narrow = tuple(members), narrow
        self.nb = soft.nb
        self.assumed = np.zeros(self.nb)
        self.model = base.copy()
        self.levels, self.pulses, self.log = [], [], []
        self.set_once = False

    def observe(self, level, drives, responses):
        new = (not self.levels) or abs(self.levels[-1][0] - level) > 1e-9
        if new:
            self.pulses = []
        self.pulses.append((drives, responses, level))
        fit = self.soft.fit([p[0] for p in self.pulses], [p[1] for p in self.pulses], [p[2] for p in self.pulses])
        if new:
            self.levels.append((level, fit))
        else:
            self.levels[-1] = (level, fit)

    def predict(self, next_level):
        at = 10 ** (next_level / 20)
        last = self.levels[-1][1]
        prev = self.levels[-2][1] if len(self.levels) >= 2 else None
        out = np.zeros(self.nb)
        for b in range(self.nb):
            a2, d2 = last['amplitude'][b], last['drops'][b]
            p = min(self.p_max, 1.0)
            if prev is not None and self.power in ('measured', 'resolved'):
                a1, d1 = prev['amplitude'][b], prev['drops'][b]
                if d1 > 0 and d2 > 0 and a1 > 0 and abs(a2 / a1 - 1) > 1e-6:
                    pm = np.log(d2 / d1) / np.log(a2 / a1)
                    if self.power == 'resolved' and min(d1, d2) >= self.gate:
                        h = 0.5 * 0.001
                        pm += (h / d1 + h / d2) / abs(np.log(a2 / a1))
                    p = float(np.clip(pm, 0.0, self.p_max))
            if d2 <= 0 or not np.isfinite(a2) or a2 <= 0:
                d = 0.0
            elif at <= a2:
                d = d2
            else:
                d = d2 * (at / a2) ** p
            out[b] = d
        return out

    def target(self, next_level):
        return np.clip(self.margin * self.predict(next_level), 0.0, self.ceiling)

    def set(self, level, t, narrow):
        new = np.maximum(t, narrow * self.assumed)
        if not np.any(np.abs(new - self.assumed) > 1e-4):
            return False
        self.assumed = new
        self.log.append((level, [round(float(x), 4) for x in new]))
        self.model = self.soft.family(self.assumed, self.members)
        return True

    def model_for(self, next_level):
        if not self.levels:
            return
        level = self.levels[-1][0]
        step_up = next_level > level + 1e-9
        if len(self.levels) < self.probe_levels or (not step_up and not self.set_once):
            return
        if step_up:
            n = self.narrow if self.set_once else 0.0
            self.set(next_level, self.target(next_level), n)
            self.set_once = True


class FixedProvider:
    def __init__(self, base):
        self.model = base

    def observe(self, *a):
        pass

    def model_for(self, *a):
        pass


class ShapePy:
    """`shape` with a model provider, safety off, carry Keep (e3law/port.py, line for line). hold: the record's
    FreezeShape (a repeat pulse with every point inside hold_db keeps the drive)."""

    def __init__(self, s, provider, lambda_rel=1e-3, max_step_db=6.0, delay_relax=0.5, delay_bound_s=2e-3, hold_db=None):
        self.s, self.p = s, provider
        self.mask = s.drive_mask()
        self.lam, self.lim, self.relax, self.bound, self.hold = lambda_rel, max_step_db, delay_relax, delay_bound_s, hold_db
        self.R = np.fft.rfft(s.reference, axis=1)
        self.om = 2 * np.pi * np.arange(s.lines) * s.fs / s.n
        self.gains = np.ones((s.nc, s.lines))
        self.delays = np.zeros(s.nc)
        self.holds = 0
        self.set_model(provider.model)

    def set_model(self, M):
        if getattr(self, 'model', None) is M:
            return
        self.model = M
        self.inv = regularized_inverse(M, self.mask, self.lam)

    def shaped(self):
        return self.R * np.exp(-1j * self.om[None, :] * self.delays[:, None]) * self.gains

    def drive_spec(self):
        return apply(self.inv, self.shaped())

    def seed(self, level):
        return np.fft.irfft(self.drive_spec(), n=self.s.n, axis=1) * 10 ** (level / 20)

    def ratios(self, responses, level):
        S = sm.srs(responses, self.s.fs, self.s.natural_hz, self.s.q, 10 ** (level / 20))
        T = self.s.points
        with np.errstate(divide='ignore', invalid='ignore'):
            e = np.where((S > 0) & (T > 0), 20 * np.log10(T / S), 0.0)
        db = np.clip(weight_at(level) * e, -self.lim, self.lim)
        df = self.s.fs / self.s.n
        return np.vstack([10 ** (to_lines_log(self.s.natural_hz, r, self.s.lines, df) / 20) for r in db]), -e

    def update_gains_delays(self, ratios, responses, level):
        s = 10 ** (level / 20)
        self.gains = self.gains * ratios
        if self.relax > 0:
            sh_t = np.fft.irfft(self.shaped(), n=self.s.n, axis=1)
            for c in range(self.s.nc):
                self.delays[c] += self.relax * estimate_delay(responses[c] / s, sh_t[c], self.s.fs)
            self.delays = np.clip(self.delays - self.delays[0], -self.bound, self.bound)

    def iterate(self, drives, responses, level, next_level):
        r, err = self.ratios(responses, level)
        self.p.observe(level, drives, responses)
        self.p.model_for(next_level)
        self.set_model(self.p.model)
        repeat = abs(next_level - level) < 1e-9
        if self.hold is not None and repeat and np.max(np.abs(err)) <= self.hold:
            self.holds += 1
        else:
            self.update_gains_delays(r, responses, level)
        return np.fft.irfft(self.drive_spec(), n=self.s.n, axis=1) * 10 ** (next_level / 20)


class Shape(Law):
    name = 'shape'

    def start(self, s):
        self.law = ShapePy(s, FixedProvider(s.frf.copy()))

    def seed(self, level):
        return self.law.seed(level)

    def iterate(self, p):
        return self.law.iterate(p.drives, p.responses, p.level_db, p.next_level_db)


class Followed(Law):
    name = 'followed'
    probes = (-40.0, -34.0)

    def ladder(self, table):
        return list(self.probes) + list(table)

    def start(self, s):
        self.soft = Soft(s)
        self.prov = FollowedProvider(self.soft, s.frf.copy(), ceiling=0.12)
        self.law = ShapePy(s, self.prov)

    def seed(self, level):
        return self.law.seed(level)

    def iterate(self, p):
        return self.law.iterate(p.drives, p.responses, p.level_db, p.next_level_db)

    def report(self):
        return dict(assumed_log=self.prov.log)


def margin_at(eps, a_from, a_to, p_max=1.0):
    if eps <= 0:
        return 0.0
    return 10 * np.log10(1 + (10 ** (eps / 10) - 1) * (a_to / a_from) ** (2 * p_max))


class FollowedHold(Law):
    """E3Var(freeze=True, dead_band_db=3.0): no ceiling, power 'resolved', the probe check, the hold."""
    name = 'followed-hold'
    probes = (-40.0, -34.0)

    def ladder(self, table):
        return list(self.probes) + list(table)

    def start(self, s):
        self.s = s
        self.soft = Soft(s)
        self.prov = FollowedProvider(self.soft, s.frf.copy(), ceiling=None, power='resolved', probe_levels=len(self.probes))
        self.law = ShapePy(s, self.prov, hold_db=3.0)
        self.H_pred = s.frf
        self.prev, self.k_db, self.eps, self.checks = None, 0.0, [], []
        self._pending = None

    def _pred_over(self, Y):
        y = np.fft.irfft(Y, n=self.s.n, axis=1)
        return float(np.max(sm.error_db(sm.srs(y, self.s.fs, self.s.natural_hz, self.s.q), self.s.points)))

    def _scale_for(self, over_pred, limit, amp):
        def need(k):
            a = amp * 10 ** (k / 20)
            m = max((margin_at(e, aj, a) for aj, e in self.eps), default=0.0)
            return over_pred + k + m - limit
        if need(0.0) <= 0:
            return 0.0
        lo, hi = -40.0, 0.0
        if need(lo) > 0:
            return lo
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if need(mid) > 0 else (mid, hi)
        return lo

    def _play(self, Dn, level):
        k = 0.0
        self._pending = None
        if level in self.probes:
            Y = apply(self.H_pred, Dn) if self.prev is None else self.prev[1] + apply(self.H_pred, Dn - self.prev[0])
            op = self._pred_over(Y)
            k = self._scale_for(op, 3.0, 10 ** (level / 20) * 10 ** (op / 20))
            self._pending = (level, k, op + k)
        self.k_db = k
        return np.fft.irfft(Dn, n=self.s.n, axis=1) * 10 ** ((level + k) / 20)

    def seed(self, level):
        return self._play(self.law.drive_spec(), level)

    def iterate(self, p):
        eff = p.level_db + self.k_db
        if self._pending is not None:
            lv, k, pred = self._pending
            meas = float(np.max(p.errors_db))
            self.checks.append((lv, round(k, 3), round(pred, 3), round(meas, 3)))
            self.eps.append((10 ** (eff / 20) * 10 ** ((pred - k) / 20), max(meas - pred, 0.0)))
        s_eff = 10 ** (eff / 20)
        self.prev = (np.fft.rfft(p.drives, axis=1) / s_eff, np.fft.rfft(p.responses, axis=1) / s_eff)
        self.law.iterate(p.drives, p.responses, eff, p.next_level_db)
        return self._play(self.law.drive_spec(), p.next_level_db)

    def report(self):
        return dict(assumed_log=self.prov.log, probe_checks=self.checks, holds=self.law.holds)


# ------------------------------------------------------------------------------------------------ srs-design
class SrsDesign(Law):
    """Simplified port of the record's F2Law (f2/law_f2.py). Kept: per control a log gain and a delay (within 2 ms)
    at each SRS point, linear in log f between points, a free phase offset; the drive the regularized inverse of
    the shaped references inside the support window (ramp up over n/64, one to 3n/4, ramp down to zero at 7n/8,
    zero after); the response predicted FROM REST by the identified model (linear convolution); the record's
    non-circular SRS of the prediction; the decline before the first pulse (predicted worst point beyond 3 dB);
    the bias update between pulses blended by shape's weight table; the stop after a pulse more than 3 dB over; the
    hand-off of square frames to followed-hold. Left out: the robustness term (first-order spread over a model set)
    and its not-robust report; the nested G -> A -> B start; the objective is a soft maximum of |error| (dB) by Adam
    instead of the exact maximax by Danskin's rule. Needs torch."""
    name = 'srs-design'

    def __init__(self, steps_seed=300, steps_update=60):
        self.steps_seed, self.steps_update = steps_seed, steps_update
        self.inner = None

    def ladder(self, table):
        return list(table)

    def start(self, s):
        self.s = s
        if s.nc <= s.nd:
            self.inner = FollowedHold()
            self.inner.start(s)
            self.ladder_note = 'square: handed to followed-hold'
            return
        import torch
        torch.set_num_threads(1)
        self.t = torch
        n, f = s.n, np.arange(s.lines) * s.fs / s.n
        mask = s.drive_mask()
        G = regularized_inverse(s.frf, mask)
        self.HG = torch.tensor(np.einsum('lcd,ldk->lck', s.frf, G))          # model response to a shaped reference
        self.G = G
        self.R = torch.tensor(np.fft.rfft(s.reference, axis=1))
        lf = np.log(np.maximum(f, 1.0)); lp = np.log(s.natural_hz)
        W = np.zeros((s.lines, lp.size))                                     # interpolation weights, holds the ends
        for i, x in enumerate(lf):
            j = np.searchsorted(lp, x)
            if j <= 0:
                W[i, 0] = 1
            elif j >= lp.size:
                W[i, -1] = 1
            else:
                a = (x - lp[j - 1]) / (lp[j] - lp[j - 1]); W[i, j - 1], W[i, j] = 1 - a, a
        self.W = torch.tensor(W)
        self.om = torch.tensor(2 * np.pi * f)
        t = np.arange(n)
        w = np.ones(n); r = n // 64
        w[:r] = 0.5 - 0.5 * np.cos(np.pi * t[:r] / r)
        a, b = 3 * n // 4, 7 * n // 8
        w[a:b] = 0.5 + 0.5 * np.cos(np.pi * (t[a:b] - a) / (b - a)); w[b:] = 0
        self.win = torch.tensor(w)
        # the model's impulse response at the controls per drive and the SRS oscillators, on 2n for linear convolution
        h = np.fft.irfft(s.frf, n=n, axis=0)                                  # (n, nc, nd)
        self.Hlin = torch.tensor(np.fft.rfft(h, n=2 * n, axis=0))             # (n+1, nc, nd)
        imp = np.zeros((s.natural_hz.size, n)); imp[:, 0] = 1
        o = np.vstack([self._osc(fn) for fn in s.natural_hz])
        self.O = torch.tensor(np.fft.rfft(o, n=2 * n, axis=1))               # (nf, n+1)
        self.T = torch.tensor(20 * np.log10(s.points))
        nc, nf = s.nc, s.natural_hz.size
        self.theta = torch.zeros(nc, 2 * nf + 1, dtype=torch.float64, requires_grad=True)
        self.bias = torch.zeros(nc, nf, dtype=torch.float64)
        self.declined_pred = None
        self._opt(self.steps_seed, torch.zeros(nc, nf, dtype=torch.float64))
        pred = self._pred_db().detach()
        worst = float(torch.max(torch.abs(pred - self.T)))
        self.seed_worst = worst
        if worst > 3.0:
            raise Declined(f'predicted worst point {worst:+.2f} dB beyond 3 dB')

    def _osc(self, fn):
        from scipy.signal import lfilter
        zeta = 1 / (2 * self.s.q); dt = 1 / self.s.fs
        wn = 2 * np.pi * fn; wd = wn * np.sqrt(1 - zeta ** 2); e = np.exp(-zeta * wn * dt); kk = wd * dt
        c, s_ = e * np.cos(kk), e * np.sin(kk); sp = s_ / kk
        x = np.zeros(self.s.n); x[0] = 1
        return lfilter([1 - sp, 2 * (sp - c), e * e - sp], [1, -2 * c, e * e], x)

    def _drive(self):
        t = self.t
        nf = self.s.natural_hz.size
        g = self.theta[:, :nf] @ self.W.T                                     # (nc, lines) log gain
        tau = 2e-3 * t.tanh(self.theta[:, nf:2 * nf]) @ self.W.T              # (nc, lines) delay
        ph = self.om[None] * tau + self.theta[:, 2 * nf:]
        X = self.R * t.exp(g) * t.exp(-1j * ph)
        D = t.einsum('ldc,cl->dl', t.tensor(self.G), X)
        return t.fft.irfft(D, n=self.s.n, dim=1) * self.win[None]           # (nd, n)

    def _pred_db(self, d=None):
        t = self.t
        d = self._drive() if d is None else d
        Dd = t.fft.rfft(d, n=2 * self.s.n, dim=1)                            # (nd, n+1)
        Y = t.einsum('lcd,dl->cl', self.Hlin, Dd)
        y = t.fft.irfft(Y, n=2 * self.s.n, dim=1)[:, : self.s.n]
        Yf = t.fft.rfft(y, n=2 * self.s.n, dim=1)
        z = t.fft.irfft(Yf[:, None, :] * self.O[None], n=2 * self.s.n, dim=2)[:, :, : self.s.n]
        S = t.amax(t.abs(z), dim=2)
        return 20 * t.log10(S + 1e-30)

    def _opt(self, steps, offset):
        t = self.t
        opt = t.optim.Adam([self.theta], lr=0.03)
        for _ in range(steps):
            opt.zero_grad()
            e = self._pred_db() + offset - self.T
            loss = 0.25 * t.logsumexp(t.abs(e).flatten() / 0.25, 0) + 0.1 * t.mean(e ** 2)
            loss.backward()
            opt.step()

    def seed(self, level):
        if self.inner is not None:
            return self.inner.seed(level)
        with self.t.no_grad():
            return self._drive().numpy() * 10 ** (level / 20)

    def next_level(self, history, planned):
        if self.inner is not None:
            return self.inner.next_level(history, planned)
        return None if history and history[-1]['over_db'] > 3.0 else planned

    def iterate(self, p):
        if self.inner is not None:
            return self.inner.iterate(p)
        t = self.t
        with t.no_grad():
            pred = self._pred_db()
        meas = t.tensor(p.errors_db) + self.T
        self.bias = self.bias + weight_at(p.level_db) * ((meas - pred) - self.bias)
        self._opt(self.steps_update, self.bias)
        with t.no_grad():
            return self._drive().numpy() * 10 ** (p.next_level_db / 20)

    def report(self):
        if self.inner is not None:
            return dict(handed_to='followed-hold', **self.inner.report())
        return dict(seed_predicted_worst_db=round(self.seed_worst, 3))


LAWS = {'inverse': Inverse, 'shape': Shape, 'paired': Paired, 'followed': Followed, 'followed-hold': FollowedHold,
        'srs-design': SrsDesign}
