"""K1. Carried inverse with a loop-speed rule (catalog entry K1;
component G1).

The error is integrated at the controls, not at the drives, and sent through the model at the
frequency the next drive will play at:

    u_{k+1} = u_k + g_k (R_k - C_k) / l_k                      (integrated error per unit level)
    D_{k+1} = Z(f_{k+1}) (R_{k+1} + l_{k+1} u_{k+1})           (the drive carried along the model)

Z(f) = Hhat(f)^+ (regularized pseudo-inverse of the model; here the frozen low-level identification),
f_{k+1} = f_k + fdot dt_c the frequency of the next interval, l the scheduled level. With an exact
model and zero error the drive follows Z(f) R(f) through the sweep with no standing error (LD-2).

Gain: the testbed's bandwidth rule, g <= beta dt_c / (tau_f + tau_s + dt_c) (the same rule as the
baseline P1, so that a difference between the two is the carrying and not the gain).

No wind-up: when the drive in force is not the drive the law asked for (a change cap, the ceiling,
the backstop notch, a refused answer), u is rebased on the drive in force,
u = (Hhat(f_k) D_in_force - R_k) / l_k, so the integrator holds what is actually being played.

A level change: the integrated correction is held per unit level (u), so a level step scales the
correction with the reference (a model error proportional to level carries; a fixed offset does not).
Option scale_with_level=False holds it in absolute units instead.

The loop has the same steady-state eigenvalues as P1 (I - g Hhat^+ H): carrying does not change where
it is stable, only the standing error in a sweep.
"""
from __future__ import annotations

import numpy as np

from .interface import Law, TickOut, reg_pinv
from .laws_baselines import loop_gain


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))


def smooth_grid(g, H, s):
    """Local quadratic least squares of the identification over +-s lines (each element, real and
    imaginary parts): the identification's line-to-line estimation scatter goes straight into a
    carried drive, where the published law's integrator averaged it (a probe of the research round)."""
    L = g.size
    out = np.empty_like(H)
    Hf = H.reshape(L, -1)
    of = out.reshape(L, -1)
    for k in range(L):
        a, b = max(0, k - s), min(L, k + s + 1)
        x = (g[a:b] - g[k]) / max(g[1] - g[0], 1e-12)
        V = np.column_stack([np.ones_like(x), x, x * x]) if b - a >= 3 else np.ones((b - a, 1))
        c, *_ = np.linalg.lstsq(V, Hf[a:b], rcond=None)
        of[k] = c[0]
    return out


class CarriedInverse(Law):
    name = 'k1_carried_inverse'
    version = '0.1'
    values = ('err_db', 'g', 'rebased', 'u_norm', 'leaking')

    def __init__(self, setup, g=0.5, eps=1e-3, gain_rule='bandwidth', beta=0.5, rebase_tol=1e-6,
                 scale_with_level=True, lookahead=True, model=None, watch=False, watch_n=10, watch_rise_db=1.0,
                 watch_floor_db=3.0, leak=0.05, watch_release=20, smooth_lines=0, form='transport', w=1.0):
        super().__init__(setup, g=g, eps=eps, gain_rule=gain_rule, beta=beta, rebase_tol=rebase_tol,
                         scale_with_level=scale_with_level, lookahead=lookahead)
        self.g, self.eps, self.rule, self.beta = g, eps, gain_rule, beta
        self.rebase_tol, self.swl, self.look = rebase_tol, scale_with_level, lookahead
        self.model = model                      # callable f -> (n, m); default the identification
        self.form, self.w = form, w
        self.smooth = int(smooth_lines)
        self._Hs = smooth_grid(setup.ident.grid_hz, setup.ident.H0, self.smooth) if self.smooth > 0 else None
        self.u = np.zeros(setup.n, complex)
        self.D_asked = None
        # G3, divergence watch with switch-on leakage (catalog G3): off unless asked
        self.watch, self.wn, self.wrise, self.wfloor = watch, int(watch_n), watch_rise_db, watch_floor_db
        self.leak, self.wrel = leak, int(watch_release)
        self.ehist = []
        self.leaking = 0
        self.watch_fired = 0

    # the model the correction goes through (K2, K2f, K3 replace these two)
    def H(self, f):
        if self.model is not None:
            return self.model(f)
        if self._Hs is not None:
            return self.setup.ident._interp(self._Hs, f)
        return self.setup.ident.H0_at(f)

    def Z(self, f):
        return reg_pinv(self.H(f), self.eps)

    def gain(self, f):
        return loop_gain(self.setup, f, self.g, self.rule, self.beta)

    def describe(self):
        return dict(name=self.name, version=self.version,
                    summary='Carried inverse: the integrated control error is sent through the model at the next '
                            'frequency, so the drive follows the model through a sweep (catalog K1).',
                    params=dict(g=dict(default=0.5, range=(0.0, 1.0), live=True, doc='upper bound on the gain per update'),
                                eps=dict(default=1e-3, range=(0.0, 0.1), live=False, doc='regularization of the inverse'),
                                gain_rule=dict(default='bandwidth', range=('bandwidth', 'fixed'), live=False),
                                beta=dict(default=0.5, range=(0.05, 1.0), live=True, doc='loop-speed rule factor'),
                                scale_with_level=dict(default=True, range=(True, False), live=False),
                                lookahead=dict(default=True, range=(True, False), live=False,
                                           doc='False reads the model at the present frequency, not the next interval (study only)')),
                    shapes=('square', 'tall', 'wide'),
                    reads=('R', 'C', 'f', 'sweep_hz_s', 'dt', 'level', 'D_in_force', 'hold',
                           'setup.ident.H0_at', 'setup.ident.group_delay_at', 'setup.tracking'))

    def _lvl(self, x):
        return max(x.level, 1e-6) if self.swl else 1.0

    def _f_next(self, x):
        if not self.look:
            return x.f
        lo, hi = self.setup.ident.grid_hz[0], self.setup.ident.grid_hz[-1]
        return float(np.clip(x.f + x.sweep_hz_s * self.setup.dt_c, lo, hi))

    def engage(self, x):
        self.u[:] = 0.0
        D = self.Z(x.f) @ x.R
        self.D_asked = D.copy()
        return TickOut(D)

    def tick(self, x):
        lv = self._lvl(x)
        rebased = 0.0
        if self.D_asked is not None:
            dif = np.linalg.norm(x.D_in_force - self.D_asked)
            if dif > self.rebase_tol * max(np.linalg.norm(self.D_asked), 1e-30):
                self.u = (self.H(x.f) @ x.D_in_force - x.R) / lv          # rebase on the drive in force
                rebased = 1.0
        g = 0.0 if x.hold == 'engage' else self.gain(x.f)
        E = x.R - x.C
        if self.watch and x.hold != 'engage':
            e = float(np.max(np.abs(_db(x.C) - _db(x.R))))
            self.ehist.append(e)
            if len(self.ehist) > self.wn:
                self.ehist.pop(0)
                rising = e > self.wfloor and e - self.ehist[0] > self.wrise and \
                    all(b >= a - 0.05 for a, b in zip(self.ehist[:-1], self.ehist[1:]))
                if rising:
                    if not self.leaking:
                        self.watch_fired += 1
                    self.leaking = self.wrel
                elif self.leaking:
                    self.leaking -= 1
            if self.leaking:                       # leak the correction toward the model's open-loop drive
                self.u = (1.0 - self.leak) * self.u
                g = 0.5 * g
        self.u = self.u + g * E / lv
        fn = self._f_next(x)
        Rn = self.setup.ref_at(fn) * x.level if self.look else x.R
        if self.form == 'transport':
            # LD-2 amendment form: the integrator stays at the drives (as P1); the drive is moved by the
            # model's own change between this frequency and the next, Dtilde = D + w (Z_{k+1} - Z_k) H_k D,
            # plus the reference's change through the model. A model error common to Z and H cancels.
            Hk = self.H(x.f)
            Zk, Zn = self.Z(x.f), self.Z(fn)
            D0 = x.D_in_force
            D = D0 + self.w * ((Zn - Zk) @ (Hk @ D0)) + Zn @ (Rn - x.R) + g * (Zk @ E)
            self.D_asked = D.copy()
            return TickOut(D, values=dict(err_db=float(np.max(np.abs(_db(x.C) - _db(x.R)))), g=float(g),
                                          rebased=0.0, u_norm=0.0, leaking=float(bool(self.leaking))))
        D = self.Z(fn) @ (Rn + lv * self.u)
        self.D_asked = D.copy()
        return TickOut(D, values=dict(err_db=float(np.max(np.abs(_db(x.C) - _db(x.R)))), g=float(g),
                                      rebased=rebased, u_norm=float(np.linalg.norm(self.u)),
                                      leaking=float(bool(self.leaking))))
