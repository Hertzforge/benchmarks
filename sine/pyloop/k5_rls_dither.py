"""K5, recursive identification with a cycling dither, for a dwell: the research testbed's law
(the msine-research testbed, drafts/D2/k5_rls_dither.py, package D2; catalog `Candidate_Laws.md` K5) ported to the suite's
law interface (Package F of the MIMO Sine benchmark). The testbed's text follows; two changes, both no-ops on a
testbed setup (no `tracker_delay_factor`, no `transport_delay_s`):
  - imports from the suite's pyloop (`interface`, `laws_baselines`);
  - the period's settle time reads the tracker's delay as the suite's laws do (`laws_baselines.tracker_delay`: the
    setup's factor over the bandwidth, the testbed's 2 / (2 pi B) by default) and adds the setup's transport delay,
    as Package B made the baselines' settle time read the engine's loop delays.
The port is checked against the testbed's own class on testbed cases (07_Research/MIMO_Sine_Benchmark/
experiments/F/port_check.py).

K5, recursive identification with a cycling dither, for a dwell (catalog entry K5, components G1
and G6; package D2 draft).

At a fixed frequency the law estimates the response matrix from differences of the drive and the
response over a dither period, with forgetting, and corrects through the estimate. Each period one
drive (in rotation, sign alternating) carries an extra offset, so the differences span every drive
direction. dither_on='response' (default): the offset is sized so the estimated response to it is
`dither` times the reference norm; 'drive': `dither` times that drive's amplitude (first built; at
a resonance the drives are large and partly cancel, so 1 percent of a drive gave a response
change of several percent of the reference and 2.9 dB of standing error, 3 percent 12.9 dB).

    base drive:   B_{k+1} = B_k + g Zhat (R_k - C_k)        (no step inside the dead band, G6)
    drive:        D_{k+1} = B_{k+1} + d_j,   d_j = +-dither |B_i| e_i, i = j mod m, j the period
    every period: dD = D_now - D_mark, dC = C_now - C_mark (both read at the period's end, after the
                  response to the last change has settled for `period` updates)
                  Hhat += (dC - Hhat dD) dD^H P / (lam + dD^H P dD)
                  P = (P - P dD dD^H P / (lam + dD^H P dD)) / lam
    Zhat = regularized pseudo-inverse of Hhat; Hhat starts at the identification Hhat0(f).

The difference is the tangent of the response at the operating point (what a correction step needs),
not the secant (catalog section 1, fact 4). Gain by the testbed's bandwidth rule. Wind-up: B is
rebased on the drive in force minus the dither offset whenever the two differ.
Period: the larger of `period` and the settle time (tracking filter plus the identification's
group delay at f) in updates.
"""
from __future__ import annotations

import numpy as np

from .interface import Law, TickOut, reg_pinv
from .laws_baselines import loop_gain, tracker_delay


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))


class RLSDither(Law):
    name = 'k5_rls_dither'
    version = '0.1'
    values = ('err_db', 'g', 'model_step', 'held')

    def __init__(self, setup, dither=0.01, lam=0.98, period=None, deadband_db=None, dither_in_deadband=True,
                 g=0.5, beta=0.5, eps=1e-3, p0=1e2, min_dD=1e-9, dither_on='response'):
        super().__init__(setup, dither=dither, lam=lam, period=period, deadband_db=deadband_db,
                         dither_in_deadband=dither_in_deadband, g=g, beta=beta, eps=eps, p0=p0, dither_on=dither_on)
        self.dither, self.lam, self.period0 = dither, lam, period
        self.db, self.dib = deadband_db, dither_in_deadband
        self.g, self.beta, self.eps, self.p0, self.min_dD = g, beta, eps, p0, min_dD
        self.H = None
        self.dither_on = dither_on

    def _period(self, f):
        s = self.setup
        tf = tracker_delay(s, f)
        ts = s.ident.group_delay_at(f) + getattr(s, 'transport_delay_s', 0.0)
        n = int(np.ceil((tf + ts) / s.dt_c))
        return max(n, self.period0 or 0, 2)

    def engage(self, x):
        self.f = x.f
        self.H = self.setup.ident.H0_at(x.f).astype(complex)
        m = self.setup.m
        scale = np.linalg.norm(reg_pinv(self.H, self.eps) @ x.R)
        self.P = np.eye(m, dtype=complex) * self.p0 * max(scale, 1e-12) ** -2 * 1e-4
        self.B = reg_pinv(self.H, self.eps) @ x.R
        self.d = np.zeros(m, complex)
        self.j = 0
        self.kp = 0
        self.T = self._period(x.f)
        self.mark = None
        self.D_last = self.B.copy()
        self.mstep = 0.0
        return TickOut(self.B.copy(), values=dict(err_db=0.0, g=0.0, model_step=0.0, held=0.0))

    def tick(self, x):
        m = self.setup.m
        if abs(x.f - self.f) > 1e-9 * x.f:          # a dwell law: a new frequency restarts the estimate
            self.f = x.f
            self.H = self.setup.ident.H0_at(x.f).astype(complex)
            self.T = self._period(x.f)
            self.mark = None
        if np.linalg.norm(x.D_in_force - self.D_last) > 1e-9 * max(1e-12, np.linalg.norm(self.D_last)):
            self.B = x.D_in_force - self.d                                     # rebase
        # ---- the period's end: identification step, next dither
        self.kp += 1
        if self.kp >= self.T and x.hold != 'engage':
            self.kp = 0
            if self.mark is not None:
                dD = x.D_in_force - self.mark[0]
                dC = x.C - self.mark[1]
                if np.linalg.norm(dD) > self.min_dD * max(1e-12, np.linalg.norm(x.D_in_force)):
                    Pd = self.P @ dD
                    den = self.lam + np.vdot(dD, Pd).real
                    K = Pd.conj() / den                                         # row gain
                    Hn = self.H + np.outer(dC - self.H @ dD, K)
                    self.mstep = float(np.linalg.norm(Hn - self.H) / max(np.linalg.norm(self.H), 1e-300))
                    self.H = Hn
                    self.P = (self.P - np.outer(Pd, Pd.conj()) / den) / self.lam
            self.mark = (x.D_in_force.copy(), x.C.copy())
            i = self.j % m
            sgn = 1.0 if (self.j // m) % 2 == 0 else -1.0
            self.j += 1
            err = float(np.max(np.abs(_db(x.C) - _db(x.R))))
            hold = self.db is not None and err < self.db
            self.d = np.zeros(m, complex)
            if self.dither > 0 and (self.dib or not hold):
                if self.dither_on == 'response':      # sized so the estimated response to it is dither ||R||
                    self.d[i] = sgn * self.dither * np.linalg.norm(x.R) / max(np.linalg.norm(self.H[:, i]), 1e-300)
                else:                                 # a fraction of the drive's own amplitude
                    self.d[i] = sgn * self.dither * abs(self.B[i]) * np.exp(1j * np.angle(self.B[i]))
        # ---- the correction step (G6: none inside the dead band)
        err = float(np.max(np.abs(_db(x.C) - _db(x.R))))
        held = self.db is not None and err < self.db
        g = 0.0 if (x.hold == 'engage' or held) else loop_gain(self.setup, x.f, self.g, 'bandwidth', self.beta)
        self.B = self.B + g * (reg_pinv(self.H, self.eps) @ (x.R - x.C))
        D = self.B + self.d
        self.D_last = D
        return TickOut(D, values=dict(err_db=err, g=float(g), model_step=self.mstep, held=float(held)))

    def describe(self):
        return dict(name=self.name, version=self.version,
                    summary='Dwell law: recursive least squares with forgetting on drive and response differences over a '
                            'cycling one-drive dither; correction through the estimate; optional dead band (K5, G6).',
                    parameters={'dither': ('dither offset, fraction of the drive amplitude', 0.01, '[0, 0.2]', 'run'),
                                'lam': ('forgetting factor per period', 0.98, '(0.8, 1]', 'run'),
                                'period': ('least period in updates', 'settle time', '>= 2', 'fixed'),
                                'deadband_db': ('G6 dead band: no correction step inside it, dB', None, '>= 0', 'run'),
                                'dither_in_deadband': ('keep identifying while held', True, 'bool', 'run'),
                                'g': ('gain ceiling before the bandwidth rule', 0.5, '(0, 1]', 'fixed'),
                                'beta': ('bandwidth rule factor', 0.5, '(0, 1]', 'fixed'),
                                'eps': ('pseudo-inverse regularization', 1e-3, '', 'fixed'),
                                'p0': ('initial covariance scale', 1e2, '>0', 'fixed'),
                    'dither_on': ("what the dither is a fraction of: 'response' or 'drive'", 'response', '', 'fixed')},
                    shapes=('square', 'tall', 'wide'), reads=('f', 'R', 'C', 'D_in_force', 'hold'))
