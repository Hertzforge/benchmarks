"""The published baseline laws behind the testbed's interface.

Built from package L1's report (Prior_Art_Published_Control_2026-11-14.md), cited by section
number only. Shared notation (Notation.md): E = R - C, Z = regularized pseudo-inverse of the model.
Each law uses the drive in force as played (after the controller's guards) as D_k.
"""
from __future__ import annotations

import numpy as np

from .interface import Law, TickOut, reg_pinv


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))


def loop_gain(setup, f, g, rule, beta):
    """The per-update gain. 'fixed': g as published (the source gives g in (0, 1), 0.4 to 0.6
    recommended, and no update interval). 'bandwidth' (testbed default): at most

        g <= beta dt_c / (tau_f + tau_s(f) + dt_c),

    tau_f = 2 / (2 pi B(f)) the tracking filter's group delay (two poles), tau_s(f) the structure's
    own response time read from the identification as the group delay of Hhat0 (largest over the
    elements above a tenth of the line's largest; about 1 / (zeta w) at a resonance), taken as the
    largest within f +- 2 B, because a light mode whose free response falls inside the filter's
    band limits the loop even off its resonance (package L2, D-2 item 4; MSN-T-09 found the
    local value alone let the linear twin diverge in a stepped sine), dt_c for the update's own
    delay. It keeps the integrating loop's crossover (g / dt_c rad/s) at beta over the
    total lag: Ratz's rules (compression proportional to f / Q, B T_L >> 1; L1 section 9) read for
    a matrix loop. Without it, g = 0.5 per 10 ms update oscillates on the development structures
    (MSN-T-03)."""
    if rule == 'fixed':
        return g
    B = setup.tracking_bw(f)
    tf = tracker_delay(setup, f)
    ts = setup.ident.group_delay_at(f, span_hz=2.0 * B)
    tt = getattr(setup, 'transport_delay_s', 0.0)
    return min(g, beta * setup.dt_c / (tf + ts + tt + setup.dt_c))


def tracker_delay(setup, f):
    """The tracking filter's delay, s: tracker_delay_factor / B, the factor 1 / pi for the testbed's two one-pole stages
    (the default, as the shipped law math's DEFAULT_TRACKER_DELAY_FACTOR) or the setup's own (the shipped engine gives
    0.625 for its four sections). Added for the suite's harness; with the default it is the testbed's 2 / (2 pi B)."""
    return getattr(setup, 'tracker_delay_factor', 1.0 / np.pi) / setup.tracking_bw(f)


class FrozenInverse(Law):
    """L1 section 3.1: D_{k+1} = D_k + g Z0(f) (R - C_k), Z0 the regularized pseudo-inverse of
    Hhat0 interpolated to the present frequency. engage: D = Z0 R (the prediction). While the
    controller's engage settle window is open (hold 'engage': the filters are still rising) the law
    holds the drive in force (testbed choice)."""
    name = 'frozen_inverse'
    values = ('err_db', 'step', 'g')

    def __init__(self, setup, g=0.5, eps=1e-3, gain_rule='bandwidth', beta=0.5):
        super().__init__(setup, g=g, eps=eps, gain_rule=gain_rule, beta=beta)
        self.g, self.eps, self.rule, self.beta = g, eps, gain_rule, beta

    def Z(self, f):
        return reg_pinv(self.setup.ident.H0_at(f), self.eps)

    def engage(self, x):
        return TickOut(self.Z(x.f) @ x.R)

    def tick(self, x):
        g = loop_gain(self.setup, x.f, self.g, self.rule, self.beta)
        E = x.R - x.C
        S = (0.0 if x.hold == 'engage' else g) * (self.Z(x.f) @ E)
        return TickOut(x.D_in_force + S, values=dict(err_db=float(np.max(np.abs(_db(x.C) - _db(x.R)))),
                                                    step=float(np.linalg.norm(S)), g=float(g)))


class SteepestDescent(Law):
    """L1 section 3.3 (its Eq. 22): D_{k+1} = D_k + (g / sigma_max^2) Hhat0^H (R - C_k)."""
    name = 'steepest_descent'
    values = ('err_db',)

    def __init__(self, setup, g=0.5, eps=1e-3, gain_rule='bandwidth', beta=0.5):
        super().__init__(setup, g=g, eps=eps, gain_rule=gain_rule, beta=beta)
        self.g, self.eps, self.rule, self.beta = g, eps, gain_rule, beta

    def engage(self, x):
        return TickOut(reg_pinv(self.setup.ident.H0_at(x.f), self.eps) @ x.R)

    def tick(self, x):
        H = self.setup.ident.H0_at(x.f)
        smax = np.linalg.norm(H, 2)
        E = x.R - x.C
        g = 0.0 if x.hold == 'engage' else loop_gain(self.setup, x.f, self.g, self.rule, self.beta)
        return TickOut(x.D_in_force + (g / smax ** 2) * (H.conj().T @ E),
                       values=dict(err_db=float(np.max(np.abs(_db(x.C) - _db(x.R))))))


class SecantComplexGain(Law):
    """L1 section 3.2 (the 1990 multi-exciter sine patent, listed as expired): a two-step cycle.

        learn:   Dhat_k = D_k + c_k Z_k E_k,            measure Chat_k
        gain:    a_k = c_k (Chat_k - C_k)^H E_k / ||Chat_k - C_k||^2
        control: S_k = a_k Z_k E_k, D_{k+1} = D_k + S_k, measure C_{k+1}
        model:   Z_{k+1} = Z_k + (S_k - Z_k y_k) S_k^H Z_k / (S_k^H Z_k y_k),   y_k = C_{k+1} - C_k

    with <x, v> = v^H x (L1 section 3.2, Derived item 1). Each step waits `settle` updates with the
    drive held, so the tracking filters and the structure settle before the response is read (the
    patent's cycle measures a settled response; the wait, settle_factor (default 5) times the
    filter's group delay plus the model's group delay, is the testbed's choice). c_k starts at c0; the patent
    lets c approach 1 "as confidence ... as measured by the value of a_n" increases and gives no
    formula. The testbed's rule: c doubles (to at most c_max) when |a - 1| < 0.3 and halves (to at
    least c0) otherwise; a = 1 is what a correct model gives. When ||E|| <= tol ||R|| (default
    0.1 dB) the cycle holds instead of probing (testbed choice: a probe on a noise-sized error
    gives a noise gain and a noise secant).

    A level change (the shipped law since hertzforge c54b9677, law-math `secant.rs` `follow_level`): the controller
    scales the drive in force by the level's ratio and tells the law the new level; the law scales the readings and
    drives it holds (the base reading, its drive and error, the last control step's reading and drive) by the same
    ratio and restarts its settle wait (the larger of the wait left and a full one), so its probe, gain and secant pair
    are read against the drive as it is now. `follow_level=False` leaves it out (the testbed's law, as at a6327de).

    carry: how Z moves from one sweep frequency to the next, which the patent does not say.
      'factor' (testbed default): Z(f) = M Z0(f), the secant updates acting on the drive-side factor
               M (m by m, starting at I): Z_{k+1} = (I + u S^H) Z_k gives M_{k+1} = (I + u S^H) M_k.
      'plain':  Z is carried unchanged, starting from Z0 at the engage frequency (L1 section 10's
               proposal).
    """
    name = 'secant_complex_gain'
    values = ('a_re', 'a_im', 'c', 'phase', 'err_db', 'secant_skipped')

    def __init__(self, setup, c0=0.1, c_max=1.0, carry='factor', settle=None, eps=1e-3, a_band=0.3, tol=0.0116,
                 settle_factor=5.0, follow_level=True):
        super().__init__(setup, c0=c0, c_max=c_max, carry=carry, settle=settle, eps=eps, tol=tol,
                         settle_factor=settle_factor, follow_level=follow_level)
        self.follow_level, self.level, self.base = follow_level, 0.0, None
        self.tol, self.settle_factor = tol, settle_factor
        self.c0, self.c_max, self.carry, self.eps, self.a_band = c0, c_max, carry, eps, a_band
        self.settle = settle
        self.c = c0
        self.M = np.eye(setup.m, dtype=complex)
        self.Zp = None
        self.phase = 'learn'
        self.wait = 0
        self.a = 1.0 + 0j
        self.skipped = 0

    def Z(self, f):
        if self.carry == 'plain':
            return self.Zp
        return self.M @ reg_pinv(self.setup.ident.H0_at(f), self.eps)

    def _settle_n(self, f):
        if self.settle is not None:
            return int(self.settle)
        tf = tracker_delay(self.setup, f)
        ts = self.setup.ident.group_delay_at(f) + getattr(self.setup, 'transport_delay_s', 0.0)
        return max(1, int(np.ceil(self.settle_factor * (tf + ts) / self.setup.dt_c)))

    def _out(self, D, x):
        return TickOut(D, values=dict(a_re=float(self.a.real), a_im=float(self.a.imag), c=float(self.c),
                                      phase=0.0 if self.phase == 'await_base' else 1.0,
                                      err_db=float(np.max(np.abs(_db(x.C) - _db(x.R)))) if np.any(x.C) else 0.0,
                                      secant_skipped=float(self.skipped)))

    def _follow_level(self, x):
        if not self.follow_level or not (x.level > 0.0 and self.level > 0.0) or x.level == self.level:
            return
        r = x.level / self.level
        self.level = x.level
        if self.base is not None:
            self.base = tuple(v * r for v in self.base)
        if self.prev is not None:
            c0, d0, zs = self.prev
            self.prev = (c0 * r, d0 * r, zs)
        self.wait = max(self.wait, self._settle_n(x.f))

    def engage(self, x):
        self.level = x.level
        Z0 = reg_pinv(self.setup.ident.H0_at(x.f), self.eps)
        self.Zp = Z0.copy()
        self.prev = None                      # (C before the control step, D before it, Z used) for the secant
        self.phase = 'await_base'
        self.wait = self._settle_n(x.f)
        return self._out(Z0 @ x.R, x)

    def tick(self, x):
        self._follow_level(x)
        if self.wait > 0:                     # hold the drive while the filters settle
            self.wait -= 1
            return self._out(x.D_in_force, x)
        if self.phase == 'await_base':
            Ck, Dk = x.C.copy(), x.D_in_force.copy()
            if self.prev is not None:          # the secant update from the last control step
                C0, D0, Zs = self.prev
                self._secant(Dk - D0, Ck - C0, Zs)
            self.base = (Ck, Dk, x.R - Ck)
            if np.linalg.norm(self.base[2]) <= self.tol * np.linalg.norm(x.R):
                self.prev = None                 # converged: no probe (a probe on a noise-sized error gives a noise gain)
                self.wait = self._settle_n(x.f)
                return self._out(Dk, x)
            self.Zk = self.Z(x.f)
            self.phase = 'await_probe'
            self.wait = self._settle_n(x.f)
            return self._out(Dk + self.c * (self.Zk @ self.base[2]), x)
        # await_probe: the response to the probe as played gives the least-squares step length
        Ck, Dk, Ek = self.base
        P = x.D_in_force - Dk                  # the probe as played (a cap may have shortened it)
        dC = x.C - Ck
        den = np.vdot(dC, dC).real
        beta = np.vdot(dC, Ek) / den if den > 0 else 0.0 + 0j      # (Chat - C)^H E / ||Chat - C||^2
        self.a = beta * self.c
        ok = abs(self.a - 1.0) < self.a_band
        self.c = min(self.c_max, 2.0 * self.c) if ok else max(self.c0, 0.5 * self.c)
        self.prev = (Ck, Dk, self.Zk)
        self.phase = 'await_base'
        self.wait = self._settle_n(x.f)
        return self._out(Dk + beta * P, x)

    def _secant(self, S, y, Zk):
        """Broyden's minimum-change update of the inverse model (L1 section 3.2, EQU1)."""
        Zy = Zk @ y
        den = np.vdot(S, Zy)                   # S^H Z y
        if not np.isfinite(den) or abs(den) <= 1e-9 * np.linalg.norm(S) * np.linalg.norm(Zy) or np.linalg.norm(S) == 0:
            self.skipped += 1
            return
        u = (S - Zy) / den
        G = np.eye(self.setup.m) + np.outer(u, S.conj())
        if self.carry == 'plain':
            self.Zp = G @ self.Zp
        else:
            self.M = G @ self.M


class RatioPerControl(Law):
    """L1 section 3.5: the published amplitude-only ratio law (its Eq. 15) on a matrix feedforward,
    with the ratio read as response over reference, E_q = |C_i| / r_i (L1 explains why the printed
    definition reads inverted). The source pairs each drive with its control point (collocated). The
    testbed's default reading, `pairing='reference'`, applies the ratio of control i to that
    control's commanded reference, D = Z0(f) diag(sigma) R, which needs no collocation and is the
    same law where H is diagonal; `pairing='drive'` is the source's form, D = diag(sigma) Z0(f) R
    (square setups only). Each sigma_i carries the ratio updates along the sweep:

        sigma_i <- sigma_i (K + 1/E_q) / (K + E_q)   if E_q <= 1
        sigma_i <- sigma_i (K + 1) / (K + E_q)       if E_q > 1

    Carrying sigma as a factor on the frozen feedforward (rather than on the drive itself) is the
    testbed's reading for a continuous sweep; the source updates the drive amplitude at each step.
    The ratio update is applied once every `stride` updates (default: about three group delays of
    the tracking filter, so each ratio is read settled, as in the source's stepped sweep)."""
    name = 'ratio_per_control'
    values = ('sigma_min', 'sigma_max', 'err_db')

    def __init__(self, setup, K=1.0, eps=1e-3, stride=None, pairing='reference'):
        super().__init__(setup, K=K, eps=eps, stride=stride, pairing=pairing)
        if pairing == 'drive' and setup.shape != 'square':
            raise ValueError('drive pairing (drive i with control i) needs a square setup')
        self.K, self.eps, self.stride, self.pairing = K, eps, stride, pairing
        self.sigma = np.ones(setup.n if pairing == 'reference' else setup.m)
        self.count = 0

    def _D(self, x):
        Z = reg_pinv(self.setup.ident.H0_at(x.f), self.eps)
        if self.pairing == 'reference':
            return Z @ (self.sigma * x.R)
        return self.sigma * (Z @ x.R)

    def engage(self, x):
        return TickOut(self._D(x))

    def _stride(self, f):
        if self.stride is not None:
            return int(self.stride)
        tau = 1.0 / (2.0 * np.pi * self.setup.tracking_bw(f))
        return max(1, int(np.ceil(3.0 * 2.0 * tau / self.setup.dt_c)))

    def tick(self, x):
        self.count += 1
        if x.hold == 'engage' or self.count < self._stride(x.f):
            return TickOut(self._D(x), values=dict(sigma_min=float(self.sigma.min()), sigma_max=float(self.sigma.max()),
                                                   err_db=float(np.max(np.abs(_db(x.C) - _db(x.R))))))
        self.count = 0
        Eq = np.abs(x.C) / np.maximum(np.abs(x.R), 1e-300)
        Eq = np.maximum(Eq, 1e-6)
        K = self.K
        fac = np.where(Eq <= 1.0, (K + 1.0 / Eq) / (K + Eq), (K + 1.0) / (K + Eq))
        self.sigma = self.sigma * fac
        return TickOut(self._D(x), values=dict(sigma_min=float(self.sigma.min()), sigma_max=float(self.sigma.max()),
                                               err_db=float(np.max(np.abs(_db(x.C) - _db(x.R))))))


class ServoPerControl(Law):
    """An amplitude servo in dB on each control, the multi-shaker form of the software's single-
    shaker servo (control-law-spec section 10, `servo_blend`): each control i carries its own level
    x_i (dB), moved each update by a velocity-form PI on its error e_i = 20 log10(r_i / |C_i|),
    x_i += kp (e_i - e_i,prev) + clip(ki dt e_i, +-rate dt), the total clamped to +-max_step_db. The
    drive is D = Z0(f) diag(10^(x / 20)) R: magnitudes from the servo, drive phases (and the shape
    across frequency) from the frozen inverse. Square setups only. kp defaults to 0 here: the
    single-shaker default of 0.3 made the per-control loops oscillate at a dwell on the
    development 4 by 4 (MSN-T-03)."""
    name = 'servo_per_control'
    values = ('x_min_db', 'x_max_db', 'err_db')

    def __init__(self, setup, kp=0.0, ki=2.0, rate_db_s=20.0, max_step_db=1.0, eps=1e-3):
        super().__init__(setup, kp=kp, ki=ki, rate_db_s=rate_db_s, max_step_db=max_step_db, eps=eps)
        if setup.shape != 'square':
            raise ValueError('servo_per_control: square setups only')
        self.kp, self.ki, self.rate, self.mx, self.eps = kp, ki, rate_db_s, max_step_db, eps
        self.x = np.zeros(setup.n)
        self.e_prev = None

    def _D(self, x):
        return reg_pinv(self.setup.ident.H0_at(x.f), self.eps) @ (10 ** (self.x / 20.0) * x.R)

    def engage(self, x):
        return TickOut(self._D(x))

    def tick(self, x):
        e = _db(x.R) - _db(x.C)
        if x.hold in ('notch', 'engage'):
            self.e_prev = e
            return TickOut(self._D(x), values=dict(x_min_db=float(self.x.min()), x_max_db=float(self.x.max()),
                                                   err_db=float(np.max(np.abs(e)))))
        de = np.zeros_like(e) if self.e_prev is None else e - self.e_prev
        d = self.kp * de + np.clip(self.ki * x.dt * e, -self.rate * x.dt, self.rate * x.dt)
        self.x += np.clip(d, -self.mx, self.mx)
        self.e_prev = e
        return TickOut(self._D(x), values=dict(x_min_db=float(self.x.min()), x_max_db=float(self.x.max()),
                                               err_db=float(np.max(np.abs(e)))))


BASELINES = dict(frozen_inverse=FrozenInverse, steepest_descent=SteepestDescent,
                 secant_complex_gain=SecantComplexGain, ratio_per_control=RatioPerControl,
                 servo_per_control=ServoPerControl)
