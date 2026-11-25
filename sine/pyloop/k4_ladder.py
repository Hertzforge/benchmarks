"""K4, the level ladder: the research testbed's law (the msine-research testbed, drafts/D2/k4_ladder.py, package D2; catalog
`Candidate_Laws.md` K4) ported to the suite's law interface (Package F of the MIMO Sine benchmark). The testbed's
text follows. Changes, each a no-op on a testbed setup with `truth` given:
  - imports from the suite's pyloop;
  - the model: the testbed's draft refuses to build without `truth` (study only). Here, without `truth`, the model is
    the law's own modal fit of the identification, chosen by `modal_fit` ('current': `modal.ModalForm.fit`, the fit
    the research round has; the modal-fit package's fit is plugged in under another name when it reports). The
    fitted residues are made rank one (the dominant singular pair, as the testbed's own residue check did:
    drafts/D2/k4_failures.py `shapes_from_ident`), and the model keeps the identification's remainder (smoothed, as
    the carried laws' modal form does) so it equals the identification at low level: `FittedModalModel`;
  - the fit's pairing of a reading with an earlier drive reads the tracker's delay and the transport delay as the
    suite's laws do (`laws_baselines.tracker_delay` + the setup's `transport_delay_s`; the testbed's 2 / (2 pi B)
    by default).
Passes: a pass is a run of updates at one scheduled level. Under the suite's ladder of 2 s rungs at the start of the
test a pass covers a few seconds of the band; the law's passes over the modes need a ladder of whole sweeps
(the harness's `ladder.rung.sweeps`). The port is checked against the testbed's own class on testbed cases
(07_Research/MIMO_Sine_Benchmark/experiments/F/port_check.py).

K4, the level ladder (catalog entry K4, components G1 and G8; package D2 draft).

Low-level passes come before the full-level pass. After each pass the law fits, from the pass's
own record (operating drives and tracked controls along the sweep), each mode's frequency, damping
and its own response amplitude at that level; it fits how frequency and damping go with the mode's
own amplitude; it predicts each mode's amplitude and poles at the next level; the next pass runs
through the predicted modal model in the simplest carried loop:

    v_{k+1} = v_k + g (R_k - C_k),     D_{k+1} = Z(f_{k+1}) (R_{k+1} + v_{k+1}),

Z the regularized pseudo-inverse of the modal model at the predicted poles, g by the testbed's
bandwidth rule (G1 as the testbed applies it to the published law), held at zero inside the engage
settle window. When the drive in force is not the drive the law asked for (cap, ceiling, notch,
hold), v is rebased so that Z (R + v) equals the drive in force (no wind-up).

A new pass is detected when the scheduled level changes, or at engage of a run when the instance
already holds a pass (the instance may be reused across runs).

Model: modal, from shapes and poles. STUDY ONLY: `truth` hands the law the structure's true
low-level shapes and poles (dict f_hz, zeta, Phi (modes, n) control shapes, Gam (modes, m) drive
shapes). Without `truth` the law has no model and refuses to build.

Fit after a pass (Derived, report section K4): with the other modes subtracted through the model,
the projection of the control vector on mode r's control shape gives y = -w^2 u / den, u the
mode's modal force Gam_r D and den = wn^2 - w^2 + 2 j zeta wn w; so each update near the mode
gives wn^2 = Re(den) + w^2 and zeta = Im(den) / (2 wn w), with the mode's own amplitude |y| / w^2.
Drives and frequency are taken a delay earlier (tracking filter 2 / (2 pi B) plus the mode's own
1 / (zeta w)) to undo the swept lag. Points within fit_frac of the pass's peak |y| are averaged
(weights |y|^2) into one (a, f, zeta) point per mode per pass.

Dependence (a choice, stated): f / f0 - 1 = c a^p and zeta / zeta0 - 1 = d a^q, through the
low-level anchor (a = 0). p and q from `forms` by least squares when two or more pass points exist,
else p_default and q_default.
"""
from __future__ import annotations

import numpy as np

from .interface import Law, TickOut, reg_pinv
from .laws_baselines import loop_gain, tracker_delay

TWO_PI = 2.0 * np.pi


def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))


def truth_of(st):
    """STUDY ONLY: the true low-level shapes and poles of a ModalStructure, for the `truth` parameter."""
    c = st.rows('control')
    return dict(f_hz=st.f_hz.copy(), zeta=st.zeta.copy(), Phi=st.output_shape[:, c].copy(),
                Gam=st.drive_shape.copy())


class ModalModel:
    def __init__(self, truth):
        self.f0 = np.asarray(truth['f_hz'], float)
        self.z0 = np.asarray(truth['zeta'], float)
        self.Phi = np.asarray(truth['Phi'])
        self.Gam = np.asarray(truth['Gam'])

    def h(self, f, fn, zeta):
        w = TWO_PI * f
        wn = TWO_PI * np.asarray(fn)
        return -(w * w) / (wn * wn - w * w + 2j * np.asarray(zeta) * wn * w)

    def H(self, f, fn, zeta, skip=None):
        h = self.h(f, fn, zeta)
        if skip is not None:
            h = h.copy(); h[skip] = 0.0
        return np.einsum('rn,r,rm->nm', self.Phi, h, self.Gam)


class FittedModalModel(ModalModel):
    """K4's model from a fitted modal form (`modal.ModalForm`): its poles, each residue A_r made rank one
    (A_r ~ Phi_r Gam_r^T, the dominant singular pair), and the remainder B(f) = Hhat0(f) - sum_r Phi_r h_r(f0, z0)
    Gam_r^T on the smoothed identification, so the model equals the identification at the low-level poles."""

    def __init__(self, mf):
        self.mf = mf
        self.f0, self.z0 = mf.f0.copy(), mf.z0.copy()
        n, m = mf.A.shape[1], mf.A.shape[2]
        self.Phi = np.zeros((mf.M, n), complex)
        self.Gam = np.zeros((mf.M, m), complex)
        for r in range(mf.M):
            U, s, Vh = np.linalg.svd(mf.A[r])
            self.Phi[r] = U[:, 0] * np.sqrt(s[0])
            self.Gam[r] = Vh[0] * np.sqrt(s[0])

    def base(self, f):
        return self.mf.H0_at(f) - super().H(f, self.f0, self.z0)

    def H(self, f, fn, zeta, skip=None):
        return self.base(f) + super().H(f, fn, zeta, skip)


def fitted_form(setup, modal_fit='current', smooth_base=8):
    """The law's own modal form of the identification (modal.modal_form by the named fit), its remainder smoothed as
    the carried laws' is."""
    from .modal import modal_form
    mf = modal_form(setup, fit=modal_fit)
    if smooth_base:
        mf.smooth_base(int(smooth_base))
    return mf


class LevelLadder(Law):
    name = 'k4_level_ladder'
    version = '0.1'
    values = ('err_db', 'g', 'pass', 'shift_bw_max')

    def __init__(self, setup, truth=None, g=0.5, beta=0.5, eps=1e-3, predict=True, forms=(0.5, 1.0, 2.0),
                 p_default=1.0, q_default=1.0, comp_delay=True, fit_bw=3.0, fit_frac=0.5, step_bw=0.5, tol_db=3.0,
                 min_points=5, fit_iters=2, amp_rule='scale', modal_fit='current'):
        super().__init__(setup, truth='study-only' if truth is not None else None, g=g, beta=beta, eps=eps,
                         predict=predict, forms=forms, p_default=p_default, q_default=q_default,
                         comp_delay=comp_delay, fit_bw=fit_bw, fit_frac=fit_frac, step_bw=step_bw, tol_db=tol_db,
                         modal_fit=None if truth is not None else modal_fit)
        if truth is not None:
            self.M = ModalModel(truth)
        else:
            self.M = FittedModalModel(fitted_form(setup, modal_fit))
        self.g, self.beta, self.eps = g, beta, eps
        self.predict, self.forms = predict, tuple(forms)
        self.p_default, self.q_default = p_default, q_default
        self.comp_delay, self.fit_bw, self.fit_frac = comp_delay, fit_bw, fit_frac
        self.step_bw, self.tol_db, self.min_points = step_bw, tol_db, min_points
        self.fit_iters, self.amp_rule = fit_iters, amp_rule
        lo, hi = setup.band
        self.lad = np.where((self.M.f0 > lo * 0.9) & (self.M.f0 < hi * 1.1))[0]   # modes the ladder follows
        self.fn, self.z = self.M.f0.copy(), self.M.z0.copy()                   # the model in use
        self.points = {int(r): [] for r in self.lad}                          # r -> [(level, a, f, z)]
        self.dep = {}                                                         # r -> (c, p, d, q)
        self.passes = []                                                      # one record per finished pass
        self.hist = None
        self.level_cur = None
        self.v = np.zeros(setup.n, complex)
        self.D_last = None
        self.fit_log = []

    # ------------------------------------------------------------------ model
    def Z(self, f, fn=None, z=None):
        return reg_pinv(self.M.H(f, self.fn if fn is None else fn, self.z if z is None else z), self.eps)

    def poles_at(self, a, dep=None):
        """Poles of every followed mode at own amplitude a (modes,), from the dependence fit."""
        dep = self.dep if dep is None else dep
        fn, z = self.M.f0.copy(), self.M.z0.copy()
        for r in self.lad:
            if r in dep:
                c, p, d, q = dep[r]
                fn[r] = self.M.f0[r] * max(0.2, 1.0 + c * a[r] ** p)
                z[r] = self.M.z0[r] * max(0.3, 1.0 + d * a[r] ** q)
        return fn, z

    def predict_amplitudes(self, level, dep=None, iters=30, npts=41):
        """Each followed mode's own amplitude at `level`. amp_rule 'scale' (default): the last
        measured amplitude times the level ratio (the controls are held, so near its resonance a
        mode's own amplitude goes with the level; an assumption that fails with the control at the
        base). 'fixed_point': the peak amplitude when the controls are held at the reference
        through the model at the poles that amplitude implies (found unstable when the dependence
        is extrapolated far; kept for the record). Returns (a, fn, z)."""
        a = np.zeros(self.M.f0.size)
        for r in self.lad:
            if self.points[r]:
                lv, ar = self.points[r][-1][0], self.points[r][-1][1]
                a[r] = ar * level / lv
        fn, z = self.poles_at(a, dep)
        if self.amp_rule == 'scale':
            return a, fn, z
        for it in range(iters):
            a_new = a.copy()
            for r in self.lad:
                bw = 2 * z[r] * fn[r]
                fg = np.linspace(fn[r] - 3 * bw, fn[r] + 3 * bw, npts)
                best = 0.0
                for f in fg:
                    D = reg_pinv(self.M.H(f, fn, z), self.eps) @ (level * self.setup.ref_at(f))
                    q = (self.M.Gam[r] @ D) * self.M.h(f, fn[r], z[r]) / -(TWO_PI * f) ** 2
                    best = max(best, abs(q))
                a_new[r] = best
            if np.max(np.abs(a_new - a) / np.maximum(a_new, 1e-300)) < 1e-4:
                a = a_new
                break
            a = 0.5 * a + 0.5 * a_new
            fn, z = self.poles_at(a, dep)
        fn, z = self.poles_at(a, dep)
        return a, fn, z

    # ------------------------------------------------------------------ the fit after a pass
    def _fit_pass(self):
        h = self.hist
        if h is None or len(h['f']) < self.min_points:
            return
        f = np.asarray(h['f']); C = np.asarray(h['C']); D = np.asarray(h['D'])
        lev = self.level_cur
        rec = dict(level=lev, modes={})
        fn_used, z_used = self.fn.copy(), self.z.copy()
        dt = self.setup.dt_c
        for it in range(self.fit_iters):
            est = self._fit_modes(f, C, D, fn_used, z_used, dt)
            for r, v in est.items():
                if v is not None:
                    fn_used[r], z_used[r] = v['f'], v['z']
        for r, v in est.items():
            rec['modes'][r] = v
            if v is not None:
                self.points[r].append((lev, v['a'], v['f'], v['z']))
        self.passes.append(rec)
        self._fit_dependence()

    def _fit_modes(self, f, C, D, fn_used, z_used, dt):
        out = {}
        rec = dict(modes=out)
        for r in self.lad:
            bw = 2 * z_used[r] * fn_used[r]
            near = np.where(np.abs(f - fn_used[r]) < self.fit_bw * bw)[0]
            if near.size < self.min_points:
                rec['modes'][int(r)] = None
                continue
            ys, us, fs_, dQ, ws_ = [], [], [], [], []
            phi = self.M.Phi[r]
            Lf = 0
            for k in near:
                if self.comp_delay:                     # the tracking filter's lag: pair C_k with the drive tau_f earlier
                    Lf = int(round((tracker_delay(self.setup, f[k]) + getattr(self.setup, 'transport_delay_s', 0.0))
                                   / dt))
                if k - Lf < 0:
                    continue
                Dk, fk = D[k - Lf], f[k - Lf]
                res = C[k] - self.M.H(fk, fn_used, z_used, skip=r) @ Dk
                ys.append((phi.conj() @ res) / (phi.conj() @ phi)); us.append(self.M.Gam[r] @ Dk); fs_.append(fk)
            if len(ys) < self.min_points:
                rec['modes'][int(r)] = None
                continue
            ys, us, fs_ = np.asarray(ys), np.asarray(us), np.asarray(fs_)
            w_all = TWO_PI * fs_
            Q = -ys / w_all ** 2                        # the mode's complex envelope (displacement)
            if self.comp_delay and Q.size > 2:          # swept envelope (Derived): den Q = U - (2 j w + 2 zeta wn) dQ/dt - j dw/dt Q
                dQdt = np.gradient(Q, dt)
                dwdt = np.gradient(w_all, dt)
                wn_u = TWO_PI * fn_used[r]
                us = us - (2j * w_all + 2 * z_used[r] * wn_u) * dQdt - 1j * dwdt * Q
            sel = np.abs(ys) >= self.fit_frac * np.abs(ys).max()
            if sel.sum() < 3:
                rec['modes'][int(r)] = None
                continue
            w = TWO_PI * fs_[sel]
            den = -(w ** 2) * us[sel] / ys[sel]
            wn2 = den.real + w ** 2
            ok = wn2 > 0
            if ok.sum() < 3:
                rec['modes'][int(r)] = None
                continue
            wn = np.sqrt(wn2[ok]); ww = np.abs(ys[sel][ok]) ** 2
            fnr = float(np.sum(ww * wn) / ww.sum() / TWO_PI)
            zr = float(np.sum(ww * den.imag[ok] / (2 * wn * w[ok])) / ww.sum())
            ar = float(np.sum(ww * np.abs(ys[sel][ok]) / w[ok] ** 2) / ww.sum())
            zr = max(zr, 0.2 * self.M.z0[r])
            out[int(r)] = dict(a=ar, f=fnr, z=zr, n=int(ok.sum()))
        return out

    @staticmethod
    def _fit_power(a, d, forms, default):
        if a.size == 0:
            return 0.0, default
        if a.size < 3:
            p = default
            return float(d[0] / a[0] ** p), p
        best = None
        for p in forms:
            x = a ** p
            c = float(np.sum(x * d) / np.sum(x * x))
            e = float(np.sum((d - c * x) ** 2))
            if best is None or e < best[0] - 1e-18:
                best = (e, c, p)
        return best[1], best[2]

    def _fit_dependence(self, forms=None):
        forms = forms or self.forms
        dep = {}
        for r in self.lad:
            pts = self.points[int(r)]
            if not pts:
                continue
            a = np.array([p[1] for p in pts]); fr = np.array([p[2] for p in pts]); zr = np.array([p[3] for p in pts])
            c, p = self._fit_power(a, fr / self.M.f0[r] - 1.0, forms, self.p_default)
            d, q = self._fit_power(a, zr / self.M.z0[r] - 1.0, forms, self.q_default)
            dep[r] = (c, p, d, q)
        self.dep = dep
        return dep

    def alternatives(self):
        """Dependence fits with each fixed exponent pair (p = q from forms): the spread G8 judges."""
        out = []
        for p in self.forms:
            dep = {}
            for r in self.lad:
                pts = self.points[int(r)]
                if not pts:
                    continue
                a = np.array([x[1] for x in pts]); fr = np.array([x[2] for x in pts]); zr = np.array([x[3] for x in pts])
                c, _ = self._fit_power(a, fr / self.M.f0[r] - 1.0, (p,), p)
                d, _ = self._fit_power(a, zr / self.M.z0[r] - 1.0, (p,), p)
                dep[r] = (c, p, d, p)
            out.append((p, dep))
        return out

    # ------------------------------------------------------------------ step rule and G8
    def shift_bw(self, fn_pred):
        """Per followed mode: predicted pole minus the last measured pole (or the low-level pole),
        in low-level half-power bandwidths f0 / Q0 = 2 zeta0 f0."""
        out = {}
        for r in self.lad:
            last = self.points[int(r)][-1][2] if self.points[int(r)] else self.M.f0[r]
            out[int(r)] = float((fn_pred[r] - last) / (2 * self.M.z0[r] * self.M.f0[r]))
        return out

    def next_level_db(self, last_db, cands=None):
        """Step rule (L3 N4): the largest candidate level whose predicted shift from the last
        measured pole stays within step_bw low-level bandwidths for every followed mode."""
        cands = cands if cands is not None else np.arange(last_db + 1.0, 0.0 + 1e-9, 1.0)
        best = None
        for c in cands:
            _, fn, _ = self.predict_amplitudes(10 ** (c / 20.0))
            s = self.shift_bw(fn)
            if max(abs(v) for v in s.values()) <= self.step_bw:
                best = float(c)
            else:
                break
        return best if best is not None else float(cands[0])

    def g8(self, level, f_grid, use='predicted'):
        """G8: predict every control along the sweep at `level` when the pass runs through the model
        `use` ('predicted' or 'frozen' low-level), against each plausible model (the alternatives).
        Returns (decline, worst_db, at_hz, per-alternative worst)."""
        if use == 'predicted':
            _, fu, zu = self.predict_amplitudes(level)
        else:
            fu, zu = self.M.f0, self.M.z0
        worst, at, per = -np.inf, None, []
        alts = [(None, self.dep)] + self.alternatives()
        for p, dep in alts:
            _, fa, za = self.predict_amplitudes(level, dep)
            wm, wa = -np.inf, None
            for f in f_grid:
                R = level * self.setup.ref_at(f)
                Cp = self.M.H(f, fa, za) @ (reg_pinv(self.M.H(f, fu, zu), self.eps) @ R)
                e = float(np.max(_db(Cp) - _db(R)))
                if e > wm:
                    wm, wa = e, float(f)
            per.append((p, wm, wa))
            if wm > worst:
                worst, at = wm, wa
        return worst > self.tol_db, worst, at, per

    # ------------------------------------------------------------------ the loop
    def _start_pass(self, x):
        if self.hist is not None:
            self._fit_pass()
        self.level_cur = x.level
        self.hist = dict(f=[], C=[], D=[])
        self.v = np.zeros(self.setup.n, complex)
        if self.predict and self.dep:
            _, self.fn, self.z = self.predict_amplitudes(x.level)
        else:
            self.fn, self.z = self.M.f0.copy(), self.M.z0.copy()

    def finish(self):
        """Fit the last pass (call after the run)."""
        if self.hist is not None:
            self._fit_pass()
            self.hist = None

    def engage(self, x):
        self._start_pass(x)
        D = self.Z(x.f) @ x.R
        self.D_last = D
        return TickOut(D, values={'err_db': 0.0, 'g': 0.0, 'pass': float(len(self.passes)), 'shift_bw_max': 0.0})

    def tick(self, x):
        if self.level_cur is None or abs(x.level - self.level_cur) > 1e-9 * max(1.0, x.level):
            self._start_pass(x)
        if x.hold == 'free' and not x.settle_open:
            self.hist['f'].append(x.f); self.hist['C'].append(x.C.copy()); self.hist['D'].append(x.D_in_force.copy())
        if self.D_last is not None and np.linalg.norm(x.D_in_force - self.D_last) > 1e-9 * max(1e-12, np.linalg.norm(self.D_last)):
            self.v = self.M.H(x.f, self.fn, self.z) @ x.D_in_force - x.R          # rebase on the drive in force
        g = 0.0 if x.hold == 'engage' else loop_gain(self.setup, x.f, self.g, 'bandwidth', self.beta)
        self.v = self.v + g * (x.R - x.C)
        f1 = float(np.clip(x.f + x.sweep_hz_s * self.setup.dt_c, *self.setup.band))
        R1 = x.R if f1 == x.f else x.level * self.setup.ref_at(f1)
        D = self.Z(f1) @ (R1 + self.v)
        self.D_last = D
        sh = np.max(np.abs((self.fn[self.lad] - self.M.f0[self.lad]) / (2 * self.M.z0[self.lad] * self.M.f0[self.lad])))
        return TickOut(D, values={'err_db': float(np.max(np.abs(_db(x.C) - _db(x.R)))), 'g': float(g),
                                  'pass': float(len(self.passes)), 'shift_bw_max': float(sh)})

    def describe(self):
        return dict(
            name=self.name, version=self.version,
            summary='Level ladder: fit each mode after every pass, predict the next level, run the next pass through '
                    'the predicted modal model in a carried loop (catalog K4 with G1, G8; derivation in the D2 report).',
            parameters={
                'truth': ('study only: true low-level shapes and poles', None, 'fixed'),
                'g': ('gain ceiling before the bandwidth rule', 0.5, '(0, 1]', 'fixed'),
                'beta': ('bandwidth rule factor', 0.5, '(0, 1]', 'fixed'),
                'eps': ('pseudo-inverse regularization', 1e-3, '[1e-6, 1e-1]', 'fixed'),
                'predict': ('use the ladder prediction (False: carried loop through the frozen low-level model)', True, 'bool', 'fixed'),
                'forms': ('exponents tried for the amplitude dependence', self.forms, 'positive', 'fixed'),
                'p_default': ('exponent with one pass point (frequency)', 1.0, '>0', 'fixed'),
                'q_default': ('exponent with one pass point (damping)', 1.0, '>0', 'fixed'),
                'comp_delay': ('undo the swept lag in the fit', True, 'bool', 'fixed'),
                'fit_bw': ('fit window, low-level bandwidths each side', 3.0, '>0', 'fixed'),
                'fit_frac': ('points kept: within this fraction of the peak', 0.5, '(0, 1)', 'fixed'),
                'step_bw': ('step rule: largest predicted shift per level, bandwidths', 0.5, '>0', 'fixed'),
                'tol_db': ('G8 tolerance on the predicted over-test, dB', 3.0, '>0', 'fixed'),
            },
            shapes=('square', 'tall', 'wide'),
            reads=('f', 'sweep_hz_s', 'level', 'R', 'C', 'D_in_force', 'hold', 'settle_open'),
        )
