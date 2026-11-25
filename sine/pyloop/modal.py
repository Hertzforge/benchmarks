"""D1 modal laws: K2 (centred model), K2f (family inverse, closely spaced) and K3 (followed poles)
with the prediction gate G2, all in the carried loop of K1 (catalog K2, K2f, K3, G2; Derivations
LD-2 amendment and LD-3 amendment). NumPy only. Depends on the sibling draft k1_carried_inverse.py
for the carried loop (the only dependency outside the testbed's interface).

Notation (one per file):
  f            tone frequency, Hz;  w = 2 pi f
  Hhat0(f)     the low-level identification (n controls by m drives), interpolated on its grid
  r            a mode of the modal form; theta_r = (f_r, zeta_r); theta0_r its low-level value
  A_r          the mode's residue at the controls (n by m, complex); for real shapes psi_r lam_r^T
  q(f; theta)  -u^2 / (1 - u^2 + 2 j zeta u), u = f / f_r  (the testbed's acceleration-per-force form)
  p_r          the mode's pole, -zeta w_r + j w_r sqrt(1 - zeta^2); k_r = p_r^2 / (2 j Im p_r)
  q = q_s + q_d, q_d = k_r / (j w - p_r) (the resonant part, given an envelope state), q_s the rest
  Hhat(f; theta) = Hhat0(f) + sum_r A_r [q(f; theta_r) - q(f; theta0_r)]   (equals Hhat0 at theta0)
  bw_r         half-power bandwidth f_r / Q = 2 zeta_r f_r (every shift in this file is in bw_r)

The modal form comes either from the law's own fit of Hhat0 (peak picking plus a local least-squares
fit of poles and residues over a window, with a constant-plus-slope background; written for this
work from the textbook description of single- and multi-mode curve fitting), or, FOR A LABELLED
STUDY ONLY, from the testbed's true structure (parameter `truth`). The law never sees the truth
otherwise.
"""
from __future__ import annotations

import numpy as np

from .interface import TickOut, reg_pinv
from .carried import CarriedInverse, _db, smooth_grid

TWO_PI = 2.0 * np.pi


def q_full(f, fr, zr):
    """q(f; theta) for each mode, shape (modes,) for scalar f or (lines, modes)."""
    u = np.asarray(f, float)[..., None] / fr
    return -(u * u) / (1.0 - u * u + 2j * zr * u)


def pole(fr, zr):
    wr = TWO_PI * fr
    wd = wr * np.sqrt(np.maximum(1.0 - zr * zr, 1e-6))
    p = -zr * wr + 1j * wd
    return p, p * p / (2j * wd)


def q_split(f, fr, zr):
    """(q_s, q_d) with q = q_s + q_d; q_d = k / (j w - p). f, fr, zr broadcast together (the caller
    shapes them)."""
    w = TWO_PI * np.asarray(f, float)
    p, k = pole(fr, zr)
    qd = k / (1j * w - p)
    qs = 1.0 + np.conj(k) / (1j * w - np.conj(p))
    return qs, qd


# ---------------------------------------------------------------------------------------------------
class ModalForm:
    """Poles and residues of the modes in band, kept as a difference on the identification."""

    def __init__(self, ident, f0, z0, A, source):
        self.ident = ident
        self.f0 = np.asarray(f0, float)
        self.z0 = np.asarray(z0, float)
        self.A = np.asarray(A, complex)            # (modes, n, m)
        self.source = source
        self.base = None

    def smooth_base(self, span_lines=8):
        """Replace the identification under the modal form by its modal part plus the remainder
        smoothed over +-span_lines (local quadratic). The remainder is smooth in truth; the
        identification's line-to-line scatter is not, and a carried drive follows it (MSN-D1-01
        probe: 1 to 4 percent scatter took K1 from 6 to 13 times better than P1 to worse than P1)."""
        g = self.ident.grid_hz
        Q = q_full(g, self.f0, self.z0) if self.M else np.zeros((g.size, 0))
        modal = np.einsum('lr,rnm->lnm', Q, self.A) if self.M else 0.0
        self.base = smooth_grid(g, self.ident.H0 - modal, span_lines) + modal
        return self

    def H0_at(self, f):
        return self.ident._interp(self.base, f) if self.base is not None else self.ident.H0_at(f)

    @property
    def M(self):
        return self.f0.size

    def bw(self, fr=None, zr=None):
        fr = self.f0 if fr is None else fr
        zr = self.z0 if zr is None else zr
        return 2.0 * zr * fr

    def H(self, f, fr=None, zr=None):
        H0 = self.H0_at(f)
        if self.M == 0 or fr is None:
            return H0
        d = q_full(f, fr, zr) - q_full(f, self.f0, self.z0)
        return H0 + np.einsum('r,rnm->nm', d, self.A)

    # --- from the truth: STUDY ONLY
    @staticmethod
    def from_truth(st, ident, band):
        ratio = ident.level.get('peak_ratio')
        f0, z0 = st.level_state(np.asarray(ratio) if ratio is not None else 0.0)
        c = st.rows('control')
        keep = [r for r in range(st.n_modes) if band[0] * 0.9 <= f0[r] <= band[1] * 1.1]
        A = np.array([np.outer(st.output_shape[r, c], st.drive_shape[r]) for r in keep], complex)
        return ModalForm(ident, f0[keep], z0[keep], A.reshape(len(keep), len(c), st.m), 'truth (study only)')

    # --- the law's own fit
    @staticmethod
    def fit(ident, band, min_prom_db=4.0, max_modes=24):
        g = ident.grid_hz
        H = ident.H0
        L = g.size
        s = np.linalg.svd(H, compute_uv=False)                    # (L, min(n, m))
        cands = []
        for i in range(min(2, s.shape[1])):
            y = 20 * np.log10(np.maximum(s[:, i], 1e-300))
            for k in range(2, L - 2):
                if y[k] >= y[k - 1] and y[k] > y[k + 1] and y[k] >= y[k - 2] and y[k] > y[k + 2]:
                    # prominence against the lower of the two minima within +-8 percent
                    a, b = np.searchsorted(g, g[k] * 0.92), np.searchsorted(g, g[k] * 1.08)
                    lo = min(y[a:k + 1].min(), y[k:b + 1].min())
                    if y[k] - lo >= min_prom_db and band[0] <= g[k] <= band[1]:
                        cands.append((g[k], y[k] - lo, i))
        cands.sort(key=lambda c: -c[1])
        picked = []
        for fk, pr, i in cands:
            if all(abs(fk - p) > 0.004 * fk for p in picked):
                picked.append(fk)
        picked = sorted(picked[:max_modes])
        if not picked:
            return ModalForm(ident, [], [], np.zeros((0, H.shape[1], H.shape[2])), 'fit (none found)')
        # half-power estimate of damping from the largest singular value
        s1 = s[:, 0]
        z_init = []
        for fk in picked:
            k = int(np.argmin(np.abs(g - fk)))
            half = s1[k] / np.sqrt(2.0)
            a = k
            while a > 0 and s1[a] > half and g[k] - g[a] < 0.1 * fk:
                a -= 1
            b = k
            while b < L - 1 and s1[b] > half and g[b] - g[k] < 0.1 * fk:
                b += 1
            z_init.append(float(np.clip((g[b] - g[a]) / (2.0 * fk), 0.003, 0.1)))
        # group modes whose windows overlap, fit each group jointly
        f0, z0, A = [], [], []
        groups, cur = [], [0]
        for j in range(1, len(picked)):
            if picked[j] - picked[cur[-1]] < 5.0 * (z_init[j] * picked[j] + z_init[cur[-1]] * picked[cur[-1]]):
                cur.append(j)
            else:
                groups.append(cur); cur = [j]
        groups.append(cur)
        for grp in groups:
            fr = np.array([picked[j] for j in grp]); zr = np.array([z_init[j] for j in grp])
            lo = fr.min() * (1 - 5 * zr.max()); hi = fr.max() * (1 + 5 * zr.max())
            sel = (g >= lo) & (g <= hi)
            if sel.sum() < 2 * len(grp) + 4:
                c = np.argmin(np.abs(g - fr.mean()))
                sel = np.zeros(L, bool); sel[max(0, c - len(grp) - 3):c + len(grp) + 4] = True
            gg, HH = g[sel], H[sel]
            fr, zr, Ar = _fit_group(gg, HH, fr, zr)
            for r in range(fr.size):
                if band[0] * 0.9 <= fr[r] <= band[1] * 1.1 and 1e-4 < zr[r] < 0.2:
                    f0.append(fr[r]); z0.append(zr[r]); A.append(Ar[r])
        return ModalForm(ident, f0, z0, np.array(A, complex).reshape(len(f0), H.shape[1], H.shape[2]), 'fit')


def _lsq_group(gg, HH, fr, zr):
    """For fixed poles: least squares of residues and a constant-plus-slope background, all elements
    at once. Returns (cost, residues (K, n, m))."""
    Q = q_full(gg, fr, zr)                                      # (l, K)
    fc = gg.mean()
    B = np.column_stack([Q, np.ones_like(gg), (gg - fc) / fc])  # (l, K + 2)
    Y = HH.reshape(gg.size, -1)
    X, *_ = np.linalg.lstsq(B, Y, rcond=None)
    res = Y - B @ X
    return float(np.sum(np.abs(res) ** 2)), X[:fr.size].reshape(fr.size, HH.shape[1], HH.shape[2])


def _fit_group(gg, HH, fr, zr, iters=40):
    """Levenberg-Marquardt over (f_r, ln zeta_r) with the residues by variable projection."""
    th = np.concatenate([fr, np.log(zr)])
    K = fr.size

    def cost(t):
        return _lsq_group(gg, HH, t[:K], np.exp(t[K:]))[0]
    c0 = cost(th)
    lam = 1e-2
    for _ in range(iters):
        h = np.concatenate([1e-5 * th[:K], np.full(K, 1e-4)])
        # Gauss-Newton on the square root of the cost via finite-difference gradient and Hessian proxy
        gr = np.zeros(2 * K); Hs = np.zeros((2 * K, 2 * K))
        for i in range(2 * K):
            e = np.zeros(2 * K); e[i] = h[i]
            cp, cm = cost(th + e), cost(th - e)
            gr[i] = (cp - cm) / (2 * h[i])
            Hs[i, i] = (cp - 2 * c0 + cm) / h[i] ** 2
        for i in range(2 * K):
            for j in range(i + 1, 2 * K):
                ei = np.zeros(2 * K); ei[i] = h[i]
                ej = np.zeros(2 * K); ej[j] = h[j]
                Hs[i, j] = Hs[j, i] = (cost(th + ei + ej) - cost(th + ei - ej) - cost(th - ei + ej) + cost(th - ei - ej)) / (4 * h[i] * h[j])
        ok = False
        for _t in range(8):
            Hd = Hs + lam * np.diag(np.maximum(np.abs(np.diag(Hs)), 1e-30))
            try:
                stp = -np.linalg.solve(Hd, gr)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            stp[:K] = np.clip(stp[:K], -0.01 * th[:K], 0.01 * th[:K])
            stp[K:] = np.clip(stp[K:], -0.5, 0.5)
            c1 = cost(th + stp)
            if c1 < c0:
                th, c0, lam, ok = th + stp, c1, max(lam / 3, 1e-6), True
                break
            lam *= 10
        if not ok or np.max(np.abs(stp[:K] / th[:K])) < 1e-7:
            break
    fr, zr = th[:K], np.exp(th[K:])
    return fr, zr, _lsq_group(gg, HH, fr, zr)[1]


_FIT_CACHE = {}


def modal_form(setup, truth=None, cache_dir=None):
    """The modal form the law uses. The fit is deterministic in the identification, so it is cached
    in memory, and on disk when cache_dir is given (a study convenience; the law is the same)."""
    if truth is not None:
        return ModalForm.from_truth(truth, setup.ident, setup.band)
    import hashlib
    import os
    key = hashlib.sha1(np.ascontiguousarray(setup.ident.H0).tobytes()).hexdigest()[:16]
    if key in _FIT_CACHE:
        f0, z0, A = _FIT_CACHE[key]
        return ModalForm(setup.ident, f0, z0, A, 'fit')
    path = os.path.join(cache_dir, key + '.npz') if cache_dir else None
    if path and os.path.exists(path):
        z = np.load(path)
        mf = ModalForm(setup.ident, z['f0'], z['z0'], z['A'], 'fit')
    else:
        mf = ModalForm.fit(setup.ident, setup.band)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(path, f0=mf.f0, z0=mf.z0, A=mf.A)
    _FIT_CACHE[key] = (mf.f0, mf.z0, mf.A)
    return mf


def fit_error(mf, st, setup):
    """Fit against the truth (for the report only): per true mode in band, the nearest fitted mode's
    frequency error in bandwidths, damping ratio, and relative residue error."""
    tr = ModalForm.from_truth(st, setup.ident, setup.band)
    out = []
    for r in range(tr.M):
        if mf.M == 0:
            out.append(dict(f=float(tr.f0[r]), found=False)); continue
        j = int(np.argmin(np.abs(mf.f0 - tr.f0[r])))
        dbw = (mf.f0[j] - tr.f0[r]) / tr.bw()[r]
        found = abs(dbw) < 1.0
        out.append(dict(f=float(tr.f0[r]), zeta=float(tr.z0[r]), found=bool(found), df_bw=float(dbw),
                        zeta_ratio=float(mf.z0[j] / tr.z0[r]),
                        residue_rel=float(np.linalg.norm(mf.A[j] - tr.A[r]) / np.linalg.norm(tr.A[r]))))
    return out


# ---------------------------------------------------------------------------------------------------
class _ModelLaw(CarriedInverse):
    """K1's carried loop with the model replaced (H(f) and Z(f) of the subclass)."""

    def __init__(self, setup, truth=None, fit_cache=None, smooth_base=8, **kw):
        super().__init__(setup, **kw)
        self.mf = modal_form(setup, truth, fit_cache)
        if smooth_base:
            self.mf.smooth_base(int(smooth_base))


def _expected(mf, expected_drop, expected_damp):
    """Per modal-form mode: the expected fractional drop s_r (negative for a rise) and damping factor
    m_r at full level. Each argument: a scalar, an array per mode of the modal form, or a callable of
    the mode's low-level frequency (how a study hands the law an operator's expectation)."""
    def per(v, default):
        if v is None:
            return np.full(mf.M, default)
        if callable(v):
            return np.array([v(f) for f in mf.f0], float)
        return np.broadcast_to(np.asarray(v, float), (mf.M,)).copy()
    return per(expected_drop, 0.0), per(expected_damp, 1.0)


class SmoothedCarried(_ModelLaw):
    """K1 through the modal-smoothed identification (no shift): the K1 the other modal laws build on."""
    name = 'k1m_carried_smoothed'

    def H(self, f):
        return self.mf.H0_at(f)


class CentredModel(_ModelLaw):
    """K2: mode r at f_r (1 - s_r / 2), damping zeta_r sqrt(m_r)."""
    name = 'k2_centred_model'

    def __init__(self, setup, truth=None, expected_drop=None, expected_damp=None, **kw):
        super().__init__(setup, truth=truth, **kw)
        s, m = _expected(self.mf, expected_drop, expected_damp)
        self.fr = self.mf.f0 * (1.0 - 0.5 * s)
        self.zr = self.mf.z0 * np.sqrt(m)

    def H(self, f):
        return self.mf.H(f, self.fr, self.zr)


class FamilyInverse(_ModelLaw):
    """K2f: Z = (sum_j H_j^H H_j / J + mu^2 I)^-1 sum_j H_j^H / J over members j with every moving
    mode at t_j of its span [0, span_margin * s_r] (damping interpolated in log to m_r), members one
    half-power bandwidth apart at the mode that moves most in bandwidths (spacing_bw), at most
    max_members. The model reported for the record (H) is the middle member."""
    name = 'k2f_family_inverse'

    def __init__(self, setup, truth=None, expected_drop=None, expected_damp=None, span_margin=1.5,
                 spacing_bw=1.0, max_members=9, mu=1e-3, **kw):
        super().__init__(setup, truth=truth, **kw)
        s, m = _expected(self.mf, expected_drop, expected_damp)
        mf = self.mf
        span_bw = np.abs(span_margin * s * mf.f0) / mf.bw() if mf.M else np.zeros(0)
        top = float(span_bw.max()) if mf.M else 0.0
        J = int(np.clip(np.ceil(top / spacing_bw) + 1, 1, max_members))
        self.span_bw = top
        self.members = []
        for t in (np.linspace(0.0, 1.0, J) if J > 1 else [0.5]):
            self.members.append((mf.f0 * (1.0 - t * span_margin * s), mf.z0 * np.exp(t * np.log(np.maximum(m, 1e-9)))))
        self.mu = mu
        self.J = J

    def H(self, f):
        fr, zr = self.members[len(self.members) // 2]
        return self.mf.H(f, fr, zr)

    def Z(self, f):
        Hs = [self.mf.H(f, fr, zr) for fr, zr in self.members]
        G = sum(h.conj().T @ h for h in Hs) / len(Hs)
        smax = np.linalg.norm(G, 2)
        b = sum(h.conj().T for h in Hs) / len(Hs)
        return np.linalg.solve(G + (self.mu ** 2) * smax * np.eye(G.shape[0]), b)


# ---------------------------------------------------------------------------------------------------
class FollowedPoles(_ModelLaw):
    """K3: the poles of the modes near the tone are refitted to the operating responses over a window
    of updates, with an anchored prior, a data gate per mode, and the prediction gate G2.

    Prediction (accounts for the structure's build-up and the tracking filter in a sweep): the law
    runs the model in time at the update rate. Each mode's resonant part q_d is an envelope state
    Y_r' = (p_r - j w) Y_r + k_r A_r D (exact for one pole with the drive held over an interval);
    the rest is quasi-static; the sum passes through the tracking filter's two one-pole stages at the
    present bandwidth. This is the model's own filtered prediction of C (catalog K3 item 3).

    Fit (every fit_every updates, over the last `window` updates, modes within near_bw bandwidths of
    the tone over the window): minimise
        sum_j ||C_j - Chat_j(theta)||^2 / sigma_d^2
          + sum_r ((f_r - f_r,anchor) / (prior_bw bw_r))^2 + ((ln zeta_r - ln zeta_r,anchor) / prior_lnz)^2
    by a few Gauss-Newton steps from the present estimate. sigma_d = data_rel |R| (the reading's scatter
    plus the model's error in the remainder, an engineering choice). Data gate: a mode is updated only
    when one bandwidth of shift would change the window's prediction by more than gate_snr sigma_d.
    Speed limit: at most step_bw bandwidths per fit. Range: within range_bw bandwidths of the anchor.

    G2 (gate): over the window, the follower's prediction residual against the fallback's (the frozen
    identification run in time the same way, or K2f when fallback='family'): the fallback's inverse is
    used from the update the follower is worse by gate_ratio until it is better by gate_ratio.

    refit_scale: also fit one complex scale per near mode on its residue (catalog K3 item 5), prior 1
    with spread prior_scale.
    """
    name = 'k3_followed_poles'
    values = ('err_db', 'g', 'gated', 'n_near', 'shift_bw', 'rho', 'rebased', 'leaking')

    def __init__(self, setup, truth=None, window=40, fit_every=2, near_bw=4.0, prior_bw=4.0,
                 prior_lnz=1.0, range_bw=12.0, step_bw=0.25, data_rel=0.02, gate_snr=1.0, iters=2,
                 gate=True, gate_ratio=2.0, fallback='frozen', expected_drop=None, expected_damp=None,
                 refit_scale=False, prior_scale=0.3, anchor=None, **kw):
        super().__init__(setup, truth=truth, **kw)
        mf = self.mf
        self.fa, self.za = mf.f0.copy(), mf.z0.copy()                  # the anchor
        if anchor is not None:                                         # a predicted start (e.g. from a ladder)
            s, m = _expected(mf, *anchor)
            self.fa, self.za = mf.f0 * (1 - s), mf.z0 * m
        self.fr, self.zr = self.fa.copy(), self.za.copy()
        self.cs = np.ones(mf.M, complex)                               # residue scales (refit_scale)
        self.W, self.every, self.near_bw = int(window), int(fit_every), near_bw
        self.prior_bw, self.prior_lnz, self.range_bw, self.step_bw = prior_bw, prior_lnz, range_bw, step_bw
        self.data_rel, self.gate_snr, self.iters = data_rel, gate_snr, int(iters)
        self.use_gate, self.gate_ratio = gate, gate_ratio
        self.refit_scale, self.prior_scale = refit_scale, prior_scale
        self.fallback = None
        if fallback == 'family':
            self.fallback = FamilyInverse(setup, truth=truth, expected_drop=expected_drop, expected_damp=expected_damp)
        n = setup.n
        self.Y = np.zeros((mf.M, n), complex)           # follower's envelope states
        self.Y0 = np.zeros((mf.M, n), complex)          # fallback (frozen) envelope states
        self.F = np.zeros((2, n), complex)              # follower's filter stages
        self.F0 = np.zeros((2, n), complex)
        self.hist = []                                  # per update: dict(f, D, C, Y, F, P, Pf, P0f, R)
        self.gated = False
        self.D_prev = np.zeros(setup.m, complex)
        self.k = 0
        self.fit_count = 0

    # ---- the model for the inverse
    def H(self, f):
        if self.mf.M == 0:
            return self.mf.H0_at(f)
        return self.H_theta(f, self.fr, self.zr, self.cs)

    def H_theta(self, f, fr, zr, cs):
        mf = self.mf
        d = cs * q_full(f, fr, zr) - q_full(f, mf.f0, mf.z0)
        return self.mf.H0_at(f) + np.einsum('r,rnm->nm', d, mf.A)

    def Z(self, f):
        if self.gated:
            if self.fallback is not None:
                return self.fallback.Z(f)
            return reg_pinv(self.mf.H0_at(f), self.eps)
        return reg_pinv(self.H(f), self.eps)

    # ---- the model run in time
    def _alpha(self, f):
        B = self.setup.tracking_bw(f)
        return 1.0 - np.exp(-TWO_PI * B * self.setup.dt_c)

    def _quasi(self, f, D, fr, zr, cs):
        """Quasi-static part: Hhat0 - sum A q_d(theta0) + sum A (c q_s(theta) - q_s(theta0)), times D."""
        mf = self.mf
        qs0, qd0 = q_split(f, mf.f0, mf.z0)
        qs, _ = q_split(f, fr, zr)
        d = cs * qs - qs0 - qd0
        return (self.mf.H0_at(f) + np.einsum('r,rnm->nm', d, mf.A)) @ D

    def _env_step(self, Y, f, D0, D1, fr, zr, cs):
        """One interval of the envelope states with the drive ramping D0 -> D1 (trapezoid)."""
        mf = self.mf
        p, kk = pole(fr, zr)
        lam = p - 1j * TWO_PI * f
        e = np.exp(lam * self.setup.dt_c)
        AD = np.einsum('rnm,m->rn', mf.A, 0.5 * (D0 + D1))
        return e[:, None] * Y + ((e - 1.0) / lam * kk * cs)[:, None] * AD

    def _filt(self, F, P, f):
        a = self._alpha(f)
        F1 = F[0] + a * (P - F[0])
        F2 = F[1] + a * (F1 - F[1])
        return np.array([F1, F2])

    def _advance_models(self, x):
        """Run the follower's and the frozen fallback's prediction over the interval just played."""
        mf = self.mf
        D0, D1 = self.D_prev, x.D_in_force          # the drive ramps from the previous to the one in force
        if mf.M:
            self.Y = self._env_step(self.Y, x.f, D0, D1, self.fr, self.zr, self.cs)
            self.Y0 = self._env_step(self.Y0, x.f, D0, D1, mf.f0, mf.z0, np.ones(mf.M))
        P = self._quasi(x.f, D1, self.fr, self.zr, self.cs) + self.Y.sum(0)
        P0 = self._quasi(x.f, D1, mf.f0, mf.z0, np.ones(mf.M)) + self.Y0.sum(0)
        Fs = self.F.copy()
        self.F = self._filt(self.F, P, x.f)
        self.F0 = self._filt(self.F0, P0, x.f)
        return Fs

    # ---- the fit
    @staticmethod
    def _scan(c, b, x0):
        """x_j = c_j x_{j-1} + b_j along the last-but-one axis (time), x_{-1} = x0; c (..., W), b (..., W, n)."""
        L = np.cumsum(np.log(c), axis=-1)
        return np.exp(L)[..., None] * (x0[..., None, :] + np.cumsum(b * np.exp(-L)[..., None], axis=-2))

    def _window_prep(self, h, idx):
        mf = self.mf
        f = np.array([e['f'] for e in h])
        D = np.array([e['D'] for e in h]); D0 = np.array([e['D0'] for e in h])
        A = mf.A[idx]
        ADm = np.einsum('knm,wm->kwn', A, 0.5 * (D0 + D))            # (K, W, n)
        ADe = np.einsum('knm,wm->kwn', A, D)
        fr_run = np.array([e['fr'][idx] for e in h]).T                # (K, W)
        zr_run = np.array([e['zr'][idx] for e in h]).T
        cs_run = np.array([e['cs'][idx] for e in h]).T
        qsr, _ = q_split(f[None, :, None], fr_run[:, :, None], zr_run[:, :, None])
        qsr = qsr[..., 0] * cs_run                                    # (K, W)
        Yrun = np.array([e['Y'][idx] for e in h]).transpose(1, 0, 2)  # (K, W, n)
        a = np.array([self._alpha(x) for x in f])                     # (W,)
        Ff = np.array([e['Ff'] for e in h])                           # (W, n)
        C = np.array([e['C'] for e in h])
        return dict(f=f, ADm=ADm, ADe=ADe, qsr=qsr, Yrun=Yrun, a=a, Ff=Ff, C=C, Y0=h[0]['Ystart'][idx])

    def _window_pred(self, w, fr, zr, cs):
        """Predicted C over the window for a batch of candidate poles (b, K) of the near modes; the
        other modes as run. Candidate envelope states start from the stored states at the window's
        start; the change they make passes the tracking filter from zero there."""
        dt = self.setup.dt_c
        f = w['f']
        p, kk = pole(fr, zr)                                          # (b, K)
        lam = p[..., None] - 1j * TWO_PI * f                          # (b, K, W)
        ex = np.exp(lam * dt)
        bt = ((ex - 1.0) / lam * (kk * cs)[..., None])[..., None] * w['ADm'][None]   # (b, K, W, n)
        Y = self._scan(ex, bt, np.broadcast_to(w['Y0'], fr.shape + w['Y0'].shape[-1:]))
        qs, _ = q_split(f[None, None, :, None], fr[..., None, None], zr[..., None, None])
        dq = qs[..., 0] * cs[..., None] - w['qsr'][None]               # (b, K, W)
        dP = np.einsum('bkw,kwn->bwn', dq, w['ADe']) + (Y - w['Yrun'][None]).sum(1)  # (b, W, n)
        c = np.broadcast_to(1.0 - w['a'], dP.shape[:-1])
        z0 = np.zeros(dP.shape[:1] + dP.shape[-1:], complex)
        x1 = self._scan(c, w['a'][None, :, None] * dP, z0)
        x2 = self._scan(c, w['a'][None, :, None] * x1, z0)
        return w['Ff'][None] + x2

    def _fit(self, x):
        h = self.hist
        mf = self.mf
        if mf.M == 0 or len(h) < self.W:
            return
        fs = np.array([e['f'] for e in h])
        bw = mf.bw(self.fr, self.zr)
        dist = np.min(np.abs(fs[:, None] - self.fr[None, :]), 0) / bw
        idx = np.where(dist < self.near_bw)[0]
        if idx.size == 0:
            return
        w = self._window_prep(h, idx)
        sd = self.data_rel * max(np.linalg.norm(x.R), 1e-12)
        K = idx.size
        rs = self.refit_scale
        npar = 4 * K if rs else 2 * K
        bw0 = mf.bw()[idx]

        def resid(T):                                                 # T (b, npar) -> (b, nres)
            fr = self.fa[idx] + T[:, :K] * bw0
            zr = self.za[idx] * np.exp(T[:, K:2 * K])
            cs = (1.0 + T[:, 2 * K:3 * K] + 1j * T[:, 3 * K:]) if rs else np.broadcast_to(self.cs[idx], fr.shape)
            r = (w['C'][None] - self._window_pred(w, fr, zr, cs)).reshape(T.shape[0], -1) / sd
            pri = [T[:, :K] / self.prior_bw, T[:, K:2 * K] / self.prior_lnz] + ([T[:, 2 * K:] / self.prior_scale] if rs else [])
            return np.concatenate([r.real, r.imag] + pri, axis=1)
        t = np.concatenate([(self.fr[idx] - self.fa[idx]) / bw0, np.log(self.zr[idx] / self.za[idx])]
                           + ([self.cs[idx].real - 1.0, self.cs[idx].imag] if rs else []))
        nd = 2 * w['C'].size
        hstep = np.where(np.arange(npar) < K, 0.05, 0.02)
        for _ in range(self.iters):
            Rb = resid(np.vstack([t, t + np.diag(hstep)]))
            r0 = Rb[0]
            Jm = ((Rb[1:] - r0) / hstep[:, None]).T                    # (nres, npar)
            sens = np.linalg.norm(Jm[:nd, :K], axis=0)                 # data gate: one bandwidth's effect
            upd = sens > self.gate_snr
            if not upd.any():
                break
            u = np.where(upd)[0]
            cols = np.concatenate([u, K + u] + ([2 * K + u, 3 * K + u] if rs else []))
            Ju = Jm[:, cols]
            stp = np.linalg.solve(Ju.T @ Ju + 1e-6 * np.eye(cols.size), -Ju.T @ r0)
            full = np.zeros(npar); full[cols] = stp
            full[:K] = np.clip(full[:K], -self.step_bw, self.step_bw)
            full[K:2 * K] = np.clip(full[K:2 * K], -0.2, 0.2)
            tn = t + full
            tn[:K] = np.clip(tn[:K], -self.range_bw, self.range_bw)
            tn[K:2 * K] = np.clip(tn[K:2 * K], np.log(0.3), np.log(5.0))
            if np.sum(resid(tn[None])[0] ** 2) < np.sum(r0 ** 2):
                t = tn
            else:
                break
        self.fr[idx] = self.fa[idx] + t[:K] * bw0
        self.zr[idx] = self.za[idx] * np.exp(t[K:2 * K])
        if rs:
            self.cs[idx] = 1.0 + t[2 * K:3 * K] + 1j * t[3 * K:]
        self.fit_count += 1
        return idx

    def engage(self, x):
        out = super().engage(x)
        self.D_prev = np.zeros(self.setup.m, complex)
        return out

    def tick(self, x):
        self.k += 1
        Fs = self._advance_models(x)
        mf = self.mf
        e = dict(f=x.f, D0=self.D_prev.copy(), D=x.D_in_force.copy(), C=x.C.copy(), Ff=self.F[1].copy(),
                 F0f=self.F0[1].copy(), Y=self.Y.copy(), fr=self.fr.copy(), zr=self.zr.copy(), cs=self.cs.copy())
        # the state at the START of this interval is what a window beginning here re-simulates from
        e['Ystart'] = self._Ylast if hasattr(self, '_Ylast') else self.Y.copy()
        self._Ylast = self.Y.copy()
        self.D_prev = x.D_in_force.copy()
        if x.hold != 'engage':
            self.hist.append(e)
            if len(self.hist) > self.W:
                self.hist.pop(0)
        rho = np.nan
        if len(self.hist) >= self.W:
            if self.k % self.every == 0:
                self._fit(x)
            C = np.array([h['C'] for h in self.hist])
            rf = np.sum(np.abs(C - np.array([h['Ff'] for h in self.hist])) ** 2)
            r0 = np.sum(np.abs(C - np.array([h['F0f'] for h in self.hist])) ** 2)
            rho = float(rf / max(r0, 1e-300))
            if self.use_gate:
                if not self.gated and rho > self.gate_ratio:
                    self.gated = True
                elif self.gated and rho < 1.0 / self.gate_ratio:
                    self.gated = False
        out = super().tick(x)
        if mf.M:
            bw = mf.bw()
            dist = np.abs(x.f - self.fr) / bw
            near = dist < self.near_bw
            sh = float(np.max(np.abs(self.fr - mf.f0)[near] / bw[near])) if near.any() else 0.0
            lz = float(np.max(np.abs(np.log(self.zr / mf.z0))[near])) if near.any() else 0.0
        else:
            near, sh, lz = np.zeros(0, bool), 0.0, 0.0
        v = dict(out.values)
        v.update(gated=float(self.gated), n_near=float(near.sum()), shift_bw=sh, lnz=lz, rho=rho if np.isfinite(rho) else -1.0)
        v.pop('u_norm', None)
        return TickOut(out.D, values=v)

    def describe(self):
        d = super().describe()
        d.update(name=self.name, version='0.1',
                 summary='Followed poles: the frequency and damping of the modes near the tone are refitted to '
                         'the operating responses over a window, with an anchored prior and a data gate, inside '
                         'the carried loop; a prediction gate falls back to the frozen model (catalog K3, G2).')
        d['params'].update({k: dict(default=v, live=False) for k, v in dict(
            window=40, fit_every=2, near_bw=4.0, prior_bw=4.0, prior_lnz=1.0, range_bw=12.0, step_bw=0.25,
            data_rel=0.02, gate_snr=1.0, iters=2, gate=True, gate_ratio=2.0, fallback='frozen',
            refit_scale=False, truth='STUDY ONLY').items()})
        d['reads'] = d['reads'] + ('setup.tracking_bw (to run the tracking filter on its prediction)',)
        return d


# ---------------------------------------------------------------------------------------------------
def _truth_shift(st, factor_s=1.0, factor_m=1.0):
    """STUDY ONLY: an operator's expectation made from the truth: for a mode near frequency f, the
    true full-level fractional drop times factor_s and damping factor to the power factor_m."""
    f0, z0 = st.level_state(0.0)
    f1, z1 = st.level_state(1.0)
    if st.nonlinear:
        for r, nl in st.nonlinear.items():
            f1[r] = f0[r] * (1.04 if nl['kind'] == 'cubic' else 0.96)
    s = 1.0 - f1 / f0
    m = z1 / z0

    def drop(f):
        return float(factor_s * s[int(np.argmin(np.abs(f0 - f)))])

    def damp(f):
        return float(m[int(np.argmin(np.abs(f0 - f)))] ** factor_m)
    return drop, damp


REGISTRY = {
    'K3_true': (FollowedPoles, dict(_truth=True)),
    'K3_fit': (FollowedPoles, {}),
    'K3_fit_nogate': (FollowedPoles, dict(gate=False)),
    'K3_fit_scale': (FollowedPoles, dict(refit_scale=True)),
    'K1_G3': (CarriedInverse, dict(watch=True)),
    'K1m': (SmoothedCarried, {}),
    'K1m_int': (SmoothedCarried, dict(form='integrated')),
    'K1m_G3': (SmoothedCarried, dict(watch=True)),
    'K1s3': (CarriedInverse, dict(smooth_lines=3)),
    'K2_0': (CentredModel, dict(_expect=0.0)),
    'K2_half': (CentredModel, dict(_expect=0.5)),
    'K2_right': (CentredModel, dict(_expect=1.0)),
    'K2_double': (CentredModel, dict(_expect=2.0)),
    'K2f': (FamilyInverse, dict(_expect=1.0)),
    'K2f_2bw': (FamilyInverse, dict(_expect=1.0, spacing_bw=2.0)),
}
