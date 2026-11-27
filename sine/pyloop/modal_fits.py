"""The modal-fit package's fits (the MIMO Sine benchmark's modal-fit study, methods M7d and M9d), ported into the suite's harness
so the candidates K3 and K4 can be built on them (`modal_fit: m7d`, `modal_fit: m9d`).

Source: the modal-fit study's `methods.py` (research code, not published), functions
`plscf_orders`, `_fz`, `_mac`, `stabilize_cluster`, `plscf_poles_cluster`, `octave_sub_bands`, `_in_band`,
`_background`, `als`, `lsfd`, `contribution`, `m5` (with M7's orders), `variance_weights`, `_groups`, `_als_resid`,
`refine_vp`, `rank_one_real`, `_delayed`, `_with_delay`. The arithmetic is unchanged line for line; what is left out is
what M7 and M9 never reach (`als` without starting participations, `free_residues`, the other methods).

One change, as the study recommends: the delay is not estimated. The study's M7d and M9d
estimate a pure delay from a complex fit's residue phases (`estimate_delay`, +-1 ms) and remove it before the real
fit. Here the delay is the one the harness knows: the simulator block's own delay, which the law's identification
holds (the exact small-signal response of the stepped structure, or an engine run's file turned back to it), set on it
by the harness as `block_delay` (samples, fs; harness.py `block_delay_samples`, read from the structure file). It is
taken out of the identification before the fit and put back on the residues after it, exactly as the study's `_delayed` and
`_with_delay` do with an estimate. A fit asked for without a known delay is refused (ValueError): a real-residue fit
of a delayed identification puts no mode within 10 percent (found in the study).

Not the lag: `lag_frames` (less the pipeline, the harness's `gd_turn`) was once proposed was once proposed as the known delay. It is not
the identification's delay. The engine's lag is the peak of a cross-correlation, so it carries the structure's
response time as well: on frame-6x6-realistic s00 the lag less the pipeline is 8 samples and on frame-12x6 s00 14,
where the block's delays are 2 and 4 and the study's own estimate reads 2.0013 and 3.9404.

M7: half-octave pLSCF fit ranges overlapping a quarter octave each side; orders 12 to 24; poles kept by clustering
over the orders (1 % in frequency, 25 % in damping, participation MAC 0.8, in at least half the orders); LSFD with
real rank-one residues (participations held) and a background of 1, (fc/f)^2, (f/fc)^2; modes under a tenth of the
response in their own window dropped and the LSFD repeated, up to three times.
M9: M7 refined, group by group, by Levenberg-Marquardt on the coherence-weighted residual with the shapes and
participations projected out (variable projection), at most 8 steps.
"""
from __future__ import annotations

import copy

import numpy as np

from .modal import MODAL_FITS, ModalForm, q_full

TWO_PI = 2.0 * np.pi
M7_ORDERS = range(12, 25)
MIN_SHARE = 0.1


# ---------------------------------------------------------------------------------------------------- pLSCF
def _in_band(ident, band):
    g = ident.grid_hz
    sel = (g >= band[0]) & (g <= band[1])
    return g[sel], ident.H0[sel]


def plscf_orders(g, H, orders, f0=0.0, W=None):
    """pLSCF over the lines g, H (L, n, m) at each order in `orders`: {p: (lambda (P,), participations (P, m))},
    physical half only (methods.py `plscf_orders`)."""
    L_, n, m = H.shape
    dt = 1.0 / (2.0 * (g[-1] - f0))
    z = np.exp(1j * TWO_PI * (g - f0) * dt)
    pmax = max(orders)
    Om = z[:, None] ** np.arange(pmax + 1)[None, :]
    W = np.ones((L_, n)) if W is None else W
    out = {}
    blocks = {}
    for o in range(n):
        w = W[:, o]
        X = Om * w[:, None]
        Y = -(Om[:, :, None] * H[:, o, None, :]).reshape(L_, -1) * w[:, None]
        R = np.real(X.conj().T @ X)
        S = np.real(X.conj().T @ Y)
        T = np.real(Y.conj().T @ Y)
        blocks[o] = (R, S, T)
    for p in sorted(orders):
        k = m * (p + 1)
        M = np.zeros((k, k))
        for o in range(n):
            R, S, T = blocks[o]
            Rp, Sp, Tp = R[:p + 1, :p + 1], S[:p + 1, :k], T[:k, :k]
            M += Tp - Sp.T @ np.linalg.solve(Rp, Sp)
        a, b = slice(0, m * p), slice(m * p, k)
        try:
            alpha_a = -np.linalg.solve(M[a, a], M[a, b])
        except np.linalg.LinAlgError:
            continue
        alphas = [alpha_a[r * m:(r + 1) * m] for r in range(p)] + [np.eye(m)]
        Cc = np.zeros((m * p, m * p))
        for r in range(p):
            Cc[:m, r * m:(r + 1) * m] = -alphas[p - 1 - r].T
        if p > 1:
            Cc[m:, :-m] = np.eye(m * (p - 1))
        zz, V = np.linalg.eig(Cc)
        lam = np.log(zz.astype(complex)) / dt + 1j * TWO_PI * f0
        Lf = V[-m:, :].T
        f = np.imag(lam) / TWO_PI
        keep = (np.real(lam) < 0) & (f >= g[0]) & (f <= g[-1])
        Lk = Lf[keep]
        Lk = Lk / np.maximum(np.abs(Lk).max(1, keepdims=True), 1e-300)
        out[p] = (lam[keep], Lk)
    return out


def _fz(lam):
    w = np.abs(lam)
    return np.imag(lam) / TWO_PI, -np.real(lam) / w


def _mac(a, b):
    return float(np.abs(np.vdot(a, b)) ** 2 / max(np.real(np.vdot(a, a) * np.vdot(b, b)), 1e-300))


def stabilize_cluster(poles, df=0.01, dz=0.25, mac=0.8, share=0.5, zmax=0.2, df_bw=None):
    """Cluster the poles of every order, highest first; keep clusters present in at least `share` of the orders; the
    representative is the member of median frequency (methods.py `stabilize_cluster`)."""
    orders = sorted(poles, reverse=True)
    clusters = []
    for p in orders:
        lam, Lf = poles[p]
        f, z = _fz(lam)
        for i in np.where((z > 0) & (z < zmax))[0]:
            best, bd = None, np.inf
            for c in clusters:
                if any(o == p for o, _, _ in c['members']):
                    continue
                tol = df_bw * 2 * c['z0'] * c['f0'] if df_bw else df * c['f0']
                if abs(f[i] - c['f0']) <= tol and abs(z[i] - c['z0']) <= dz * c['z0'] and _mac(Lf[i], c['L0']) >= mac:
                    d = abs(f[i] - c['f0'])
                    if d < bd:
                        best, bd = c, d
            if best is None:
                clusters.append(dict(f0=f[i], z0=z[i], L0=Lf[i], members=[(p, lam[i], Lf[i])]))
            else:
                best['members'].append((p, lam[i], Lf[i]))
    lam_out, L_out = [], []
    for c in clusters:
        if len({o for o, _, _ in c['members']}) >= share * len(orders):
            fs = np.array([np.imag(l) for _, l, _ in c['members']])
            k = int(np.argsort(fs)[len(fs) // 2])
            lam_out.append(c['members'][k][1]); L_out.append(c['members'][k][2])
    m = next(iter(poles.values()))[1].shape[1] if poles else 0
    return np.array(lam_out, complex), (np.array(L_out) if L_out else np.zeros((0, m)))


def octave_sub_bands(band, width=0.5, overlap=0.25):
    """Fit ranges `width` octave wide with `overlap` octave each side; each keeps the poles in its central part."""
    out = []
    lo = band[0]
    while lo < band[1]:
        hi = min(lo * 2 ** width, band[1])
        out.append((max(band[0], lo * 2 ** -overlap), min(band[1], hi * 2 ** overlap), lo, hi if hi < band[1] else np.inf))
        lo = hi
    return out


def plscf_poles_cluster(ident, band, orders=M7_ORDERS, sub_bands=None):
    g, H = _in_band(ident, band)
    m = H.shape[2]
    sub_bands = sub_bands or octave_sub_bands(band)
    lams, Ls = [], []
    for lo, hi, klo, khi in sub_bands:
        sel = (g >= lo) & (g <= hi)
        P = plscf_orders(g[sel], H[sel], list(orders), f0=lo)
        lam, Lf = stabilize_cluster(P)
        f, _ = _fz(lam) if lam.size else (np.zeros(0), None)
        k = (f >= klo) & (f < khi)
        lams.append(lam[k]); Ls.append(Lf[k] if lam.size else np.zeros((0, m)))
    lam = np.concatenate(lams); Lf = np.concatenate(Ls)
    o = np.argsort(np.imag(lam))
    return lam[o], Lf[o]


# ---------------------------------------------------------------------------------------------------- LSFD
def _background(g, fc):
    return np.column_stack([np.ones_like(g), (fc / g) ** 2, (g / fc) ** 2])


def rank_one_real(A):
    """Nearest real rank-one matrix to a complex one (methods.py `rank_one_real`)."""
    U, s, Vh = np.linalg.svd(A)
    A1 = s[0] * np.outer(U[:, 0], Vh[0])
    a = -0.5 * np.angle(np.sum(A1 * A1))
    R = np.real(np.exp(1j * a) * A1)
    if np.real(np.sum(np.conj(A1) * R)) < 0:
        R = -R
    u, s2, vh = np.linalg.svd(R)
    return s2[0] * np.outer(u[:, 0], vh[0])


def als(g, H, fr, zr, lam0, W=None, real=True, iters=60, tol=1e-9, lam_fixed=False):
    """Rank-one residues by alternating least squares with the poles fixed, from starting participations lam0
    (methods.py `als`; its path without lam0 is not reached by M7 or M9). Returns psi (K, n), lam (K, m), C (3, n, m),
    cost."""
    L, n, m = H.shape
    K = fr.size
    Q = q_full(g, fr, zr) if K else np.zeros((L, 0))
    Phi = _background(g, np.sqrt(g[0] * g[-1]))
    W = np.ones(H.shape) if W is None else W
    dt = float if real else complex
    lam = np.array(lam0, dt).reshape(K, m)
    psi = np.zeros((K, n), dt)
    C = np.zeros((3, n, m), dt)

    def solve(design, y, w):
        Dw = design * w[:, None]
        yw = y * w
        if real:
            Dw = np.vstack([Dw.real, Dw.imag]); yw = np.concatenate([yw.real, yw.imag])
        x, *_ = np.linalg.lstsq(Dw, yw, rcond=None)
        return x

    prev = np.inf
    cost = np.inf
    for it in range(iters):
        Dpsi = (Q[:, None, :] * lam.T[None, :, :]).reshape(L * m, K)
        Dbg = np.zeros((L * m, 3 * m), dt if not real else complex)
        for b in range(3):
            Dbg[:, b * m:(b + 1) * m] = (Phi[:, b][:, None, None] * np.eye(m)[None]).reshape(L * m, m)
        Dd = np.hstack([Dpsi, Dbg])
        for i in range(n):
            x = solve(Dd, H[:, i, :].reshape(-1), W[:, i, :].reshape(-1))
            psi[:, i] = x[:K]
            C[:, i, :] = x[K:].reshape(3, m)
        if not lam_fixed:
            Dlam = (Q[:, None, :] * psi.T[None, :, :]).reshape(L * n, K)
            Dbg2 = np.zeros((L * n, 3 * n), complex)
            for b in range(3):
                Dbg2[:, b * n:(b + 1) * n] = (Phi[:, b][:, None, None] * np.eye(n)[None]).reshape(L * n, n)
            Dd2 = np.hstack([Dlam, Dbg2])
            for j in range(m):
                x = solve(Dd2, H[:, :, j].reshape(-1), W[:, :, j].reshape(-1))
                lam[:, j] = x[:K]
                C[:, :, j] = x[K:].reshape(3, n)
        s = np.maximum(np.linalg.norm(lam, axis=1), 1e-300)
        lam = lam / s[:, None]; psi = psi * s[:, None]
        Hm = np.einsum('lk,ki,kj->lij', Q, psi, lam) + np.einsum('lb,bij->lij', Phi, C)
        cost = float(np.sum((W * np.abs(H - Hm)) ** 2))
        if lam_fixed or abs(prev - cost) <= tol * max(cost, 1e-300):
            break
        prev = cost
    return psi, lam, C, cost


def lsfd(ident, band, lam, Lf, real=True, W=None, refine_lam=False):
    """LSFD in the q form: shapes by least squares with the poles and participations held."""
    g, H = _in_band(ident, band)
    f, z = _fz(lam)
    if real:
        L0 = np.array([np.real(np.exp(-1j * np.angle(l[np.argmax(np.abs(l))])) * l) for l in Lf]) if len(Lf) else np.zeros((0, H.shape[2]))
    else:
        L0 = Lf
    L0 = L0 / np.maximum(np.linalg.norm(L0, axis=1, keepdims=True), 1e-300)
    Wb = None if W is None else W[(ident.grid_hz >= band[0]) & (ident.grid_hz <= band[1])]
    psi, lm, C, cost = als(g, H, f, z, lam0=L0, real=real, W=Wb, lam_fixed=not refine_lam)
    return f, z, np.einsum('ki,kj->kij', psi, lm)


def contribution(ident, band, fit):
    """Each mode's largest share of the identification over its own window (f_k +- 3 bw_k)."""
    g, H = _in_band(ident, band)
    f, z, A = fit
    out = np.zeros(f.size)
    nH = np.linalg.norm(H.reshape(g.size, -1), axis=1)
    for k in range(f.size):
        w = np.abs(g - f[k]) <= 3 * 2 * z[k] * f[k]
        if not w.any():
            continue
        q = q_full(g[w], np.array([f[k]]), np.array([z[k]]))[:, 0]
        out[k] = np.max(np.abs(q) * np.linalg.norm(A[k]) / np.maximum(nH[w], 1e-300))
    return out


def m7(ident, band, sub_bands=None, min_share=MIN_SHARE, real=True):
    """The study's M7 (methods.py `m5` with orders 12 to 24)."""
    lam, Lf = plscf_poles_cluster(ident, band, orders=M7_ORDERS, sub_bands=sub_bands)
    fit = lsfd(ident, band, lam, Lf, real=real)
    for _ in range(3):
        c = contribution(ident, band, fit)
        keep = c >= min_share
        if keep.all():
            break
        lam, Lf = lam[keep], Lf[keep]
        fit = lsfd(ident, band, lam, Lf, real=real)
    return fit


# ---------------------------------------------------------------------------------------------------- M9's refinement
def variance_weights(ident):
    """1 / sigma of each element of the identification, from its coherence (methods.py `variance_weights`)."""
    H = ident.H0
    s2 = np.asarray(ident.level.get('drive_rms', np.ones(H.shape[2])), float) ** 2
    coh = np.clip(ident.coh, 1e-6, 1 - 1e-9)
    snn = (1 - coh) / coh * np.einsum('loj,j->lo', np.abs(H) ** 2, s2)
    var = snn[:, :, None] / s2[None, None, :]
    w = 1.0 / np.sqrt(np.maximum(var, 1e-300))
    return w / np.median(w)


def _groups(f, z, gap_bw=5.0):
    o = np.argsort(f)
    groups, cur = [], [o[0]] if f.size else []
    for a, b in zip(o[:-1], o[1:]):
        if f[b] - f[a] < gap_bw * (z[a] * f[a] + z[b] * f[b]):
            cur.append(b)
        else:
            groups.append(cur); cur = [b]
    if cur:
        groups.append(cur)
    return groups


def _als_resid(g, Hr, fr, zr, lam0, W, real, iters):
    psi, lm, C, cost = als(g, Hr, fr, zr, lam0=lam0, W=W, real=real, iters=iters)
    Q = q_full(g, fr, zr)
    Phi = _background(g, np.sqrt(g[0] * g[-1]))
    R = W * (Hr - np.einsum('lk,ki,kj->lij', Q, psi, lm) - np.einsum('lb,bij->lij', Phi, C))
    return np.concatenate([R.real.ravel(), R.imag.ravel()]), psi, lm


def refine_vp(ident, band, start, W=None, iters=8, win_bw=5.0, real=True, inner=15):
    f, z, A = (np.array(x, copy=True) for x in start)
    if f.size == 0:
        return f, z, A
    g_all, H_all = _in_band(ident, band)
    Wb = (np.ones(H_all.shape) if W is None else W[(ident.grid_hz >= band[0]) & (ident.grid_hz <= band[1])])
    work = []
    for grp in _groups(f, z):
        if len(grp) <= 8:
            work.append(grp)
        else:
            o = sorted(grp, key=lambda k: f[k])
            chunks = [o[i:i + 6] for i in range(0, max(len(o) - 3, 1), 3)]
            work += chunks + chunks
    for grp in work:
        grp = np.array(grp)
        lo = (f[grp] * (1 - win_bw * z[grp])).min(); hi = (f[grp] * (1 + win_bw * z[grp])).max()
        sel = (g_all >= lo) & (g_all <= hi)
        if sel.sum() < 3 * grp.size + 6:
            continue
        g, H, Wg = g_all[sel], H_all[sel], Wb[sel]
        others = np.setdiff1d(np.arange(f.size), grp)
        Hr = H - (np.einsum('lk,kij->lij', q_full(g, f[others], z[others]), A[others]) if others.size else 0)
        lam0 = np.array([rank_one_real(a) for a in A[grp]])
        lam0 = np.array([a[np.argmax(np.abs(a).sum(1))] for a in lam0])
        lam0 = lam0 / np.maximum(np.linalg.norm(lam0, axis=1, keepdims=True), 1e-300)
        K = grp.size
        t = np.concatenate([f[grp], np.log(z[grp])])
        r0, psi, lm = _als_resid(g, Hr, t[:K], np.exp(t[K:]), lam0, Wg, real, 2 * inner)
        c0 = float(r0 @ r0)
        mu = 1e-3
        h = np.concatenate([1e-4 * z[grp] * f[grp], np.full(K, 1e-3)])
        for _ in range(iters):
            J = np.empty((r0.size, 2 * K))
            for i in range(2 * K):
                e = np.zeros(2 * K); e[i] = h[i]
                J[:, i] = (_als_resid(g, Hr, t[:K] + e[:K], np.exp(t[K:] + e[K:]), lm, Wg, real, inner)[0] - r0) / h[i]
            JtJ = J.T @ J; Jtr = J.T @ r0
            ok = False
            for _t in range(6):
                try:
                    stp = -np.linalg.solve(JtJ + mu * np.diag(np.maximum(np.diag(JtJ), 1e-300)), Jtr)
                except np.linalg.LinAlgError:
                    mu *= 10; continue
                stp[:K] = np.clip(stp[:K], -0.25 * z[grp] * f[grp], 0.25 * z[grp] * f[grp])
                stp[K:] = np.clip(stp[K:], -0.3, 0.3)
                tn = t + stp
                r1, p1, l1 = _als_resid(g, Hr, tn[:K], np.exp(tn[K:]), lm, Wg, real, inner)
                c1 = float(r1 @ r1)
                if c1 < c0:
                    t, r0, c0, psi, lm, ok = tn, r1, c1, p1, l1, True
                    mu = max(mu / 3, 1e-7)
                    break
                mu *= 10
            if not ok or np.max(np.abs(stp[:K]) / (z[grp] * f[grp])) < 1e-4:
                break
        f[grp], z[grp] = t[:K], np.exp(t[K:])
        A[grp] = np.einsum('ki,kj->kij', psi, lm)
    return f, z, A


# ---------------------------------------------------------------------------------------------------- the known delay
def known_delay_s(ident):
    """The identification's own delay in seconds, as the harness knows it (`block_delay` = (samples, fs), the simulator
    block's delay, harness.py `block_delay_samples`). None when it is not known."""
    bd = getattr(ident, 'block_delay', None)
    if bd is None or bd[0] is None:
        return None
    return float(bd[0]) / float(bd[1])


def _delayed(ident, tau):
    """The identification with a delay tau taken out (methods.py `_delayed`)."""
    idn = copy.copy(ident)
    idn.H0 = ident.H0 * np.exp(1j * TWO_PI * ident.grid_hz * tau)[:, None, None]
    idn._gd = None
    return idn


def _with_delay(fit, tau):
    """The fit's residues turned by the delay at each mode's own frequency (methods.py `_with_delay`)."""
    f, z, A = fit
    return f, z, np.asarray(A, complex) * np.exp(-1j * TWO_PI * f * tau)[:, None, None]


def m7d(ident, band, tau, sub_bands=None):
    """M7 on the identification with the known delay tau (s) taken out, the delay put back on the residues."""
    return _with_delay(m7(_delayed(ident, tau), band, sub_bands=sub_bands), tau)


def m9d(ident, band, tau, sub_bands=None):
    """M9 likewise (the weights from the identification as given, as methods.py `m9d`)."""
    idn = _delayed(ident, tau)
    return _with_delay(refine_vp(idn, band, m7(idn, band, sub_bands=sub_bands), W=variance_weights(ident)), tau)


def _form(name, method):
    def fit(ident, band):
        tau = known_delay_s(ident)
        if tau is None:
            raise ValueError(f'the modal fit {name} needs the identification\'s known delay (the simulator block\'s, '
                             'set by the harness as block_delay); none is set')
        f, z, A = method(ident, band, tau, sub_bands=octave_sub_bands(band))
        keep = (f >= band[0] * 0.9) & (f <= band[1] * 1.1) & (z > 1e-4) & (z < 0.2)   # as the package's k3loop keeps
        A = np.asarray(A, complex)[keep]
        mf = ModalForm(ident, f[keep], z[keep], A.reshape(int(keep.sum()), ident.H0.shape[1], ident.H0.shape[2]),
                       f'fit {name} (known delay {tau * float(ident.block_delay[1]):.6g} samples)')
        mf.delay_s = tau
        return mf
    return fit


FITS = {'m7d': _form('m7d', m7d), 'm9d': _form('m9d', m9d)}
MODAL_FITS.update(FITS)
