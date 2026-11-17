"""an earlier study: which metric tracks the stress truth in a multi-shaker random test.

One command reruns everything (simulation, tables, figures):
    python x1_random_study.py --seeds 10 --workers 16
    python x1_random_study.py --analyze      (tables/figures only)

Idealized converged control: no closed-loop dynamics, no identification error (the lab unit's true
FRF is used by every law). Bulk results go to the scratch folder SCR; tables print to stdout and to
SCR/x1_tables.txt; figures to ./figs.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, argparse, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
from multiprocessing import Pool
from scipy.optimize import linprog
from scipy.stats import spearmanr
import frame_model as fm

SCR = os.path.join(os.environ.get('OBS_GEN_SCRATCH', os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'gen_scratch')), 'x1')   # the study: was another session's scratchpad
HERE = os.path.dirname(os.path.abspath(__file__))
F = np.arange(20.0, 2000.0001, 0.5); DF = 0.5; W = 2 * np.pi * F; NF = F.size
NS_LIST, NC_LIST, PERTS, COND = (3, 6, 12), (6, 12, 24), (0, 1, 2), 1e3
B_EXP = (8, 4)


# ------------------------------------------------------------------ bands and SDOF kernels
def band_matrix(N):
    """Partition of lines into 1/N-octave bands (nearest base-2 center); rows average a band."""
    idx = np.round(N * np.log2(F / 1000.0)).astype(int); u = np.unique(idx)
    Bm = (idx[None, :] == u[:, None]).astype(float); return Bm / Bm.sum(1, keepdims=True), u
BANDS = {N: band_matrix(N)[0] for N in (12, 6, 3)}
FN = 1000 * 2.0 ** (np.arange(np.ceil(12 * np.log2(.02)), np.floor(12 * np.log2(2.0)) + 1) / 12)


def sdof_kernels(Q):
    z = 1 / (2 * Q); r = F[None, :] / FN[:, None]; den = (1 - r ** 2) ** 2 + (2 * z * r) ** 2
    return (1 + (2 * z * r) ** 2) / den * DF, 1 / ((2 * np.pi * FN[:, None]) ** 4 * den) * DF
KER = {Q: sdof_kernels(Q) for Q in (10, 25, 50)}


# ------------------------------------------------------------------ helpers
def db10(a, b):
    return 10 * np.log10(np.maximum(a, 1e-300) / np.maximum(b, 1e-300))


def auto(Y):
    return (Y.real ** 2 + Y.imag ** 2).sum(1)


def pinv_trunc(H):
    """Batched pseudo-inverse over the leading axis, singular values below smax/COND dropped."""
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    sinv = np.where(s >= s[:, :1] / COND, 1 / s, 0)
    return np.conj(Vh).transpose(0, 2, 1) @ (sinv[:, :, None] * np.conj(U).transpose(0, 2, 1))


def modal_q(md, in_dofs, Fin):
    """Modal displacement factor (n_m, r, n_f) for force factor Fin (n_in, r, n_f)."""
    return np.einsum('im,irf->mrf', md.Phi[in_dofs], Fin, optimize=True) * md.h(F)[:, None, :]


def acc(md, dofs, Q):
    return -W ** 2 * np.einsum('cm,mrf->crf', md.Phi[dofs], Q, optimize=True)


def moments(Smod, Q, chunk=600):
    """Per stress row: m0 = int G df and m2 = int f^2 G df."""
    m0 = np.zeros(Smod.shape[0]); m2 = np.zeros_like(m0)
    for a in range(0, NF, chunk):
        q = Q[:, :, a:a + chunk]; nm, r, nfc = q.shape
        Y = (Smod @ q.reshape(nm, r * nfc)).reshape(-1, r, nfc); G = auto(Y)
        m0 += G.sum(1) * DF; m2 += (G * F[a:a + chunk] ** 2).sum(1) * DF
    return m0, m2


def element_truth(m0, m2):
    """Governing (worst of 8 fibre points) RMS stress and damage rate (b=8, 4) per element."""
    rms = np.sqrt(m0.reshape(-1, 8)); nu = np.sqrt(m2 / np.maximum(m0, 1e-300)).reshape(-1, 8)
    return dict(rms=rms.max(1), **{f'd{b}': (nu * rms ** b).max(1) for b in B_EXP})


def truth_summary(lab, ref):
    x = 20 * np.log10(lab['rms'] / ref['rms']); out = dict(x=x.astype(np.float32))
    out.update(t_rms=np.sqrt(np.mean(x ** 2)), t_bias=x.mean(), t_p95=np.percentile(x, 95),
               t_p5=np.percentile(x, 5), t_in3=np.mean(np.abs(x) <= 3), t_max=x.max())
    for b in B_EXP:
        d = db10(lab[f'd{b}'], ref[f'd{b}'])
        out.update({f'd{b}_rms': np.sqrt(np.mean(d ** 2)), f'd{b}_bias': d.mean(),
                    f'd{b}_p95': np.percentile(d, 95), f'd{b}_p5': np.percentile(d, 5)})
    out['good'] = bool(out['t_in3'] >= 0.9 and out['t_max'] <= 6)
    return out


def forms(E, pre, out):
    """E (channels, points) dB errors -> rms, bias, spread, worst channel, percent outside 3 dB."""
    out[pre + '_rms'] = np.sqrt(np.mean(E ** 2)); out[pre + '_bias'] = E.mean(); out[pre + '_std'] = E.std()
    out[pre + '_worst'] = np.sqrt(np.mean(E ** 2, axis=1)).max(); out[pre + '_out3'] = np.mean(np.abs(E) > 3)


def chan_metrics(Yl, Yr, Phi_ch, out=None):
    """Candidate metrics for one channel set. Yl, Yr: response factors (n_ch, r, n_f)."""
    out = {} if out is None else out; Gl, Gr = auto(Yl), auto(Yr); E = db10(Gl, Gr)
    forms(E, 'line', out); out['line_maxabs'] = np.abs(E).max()
    for N, Bm in BANDS.items():
        forms(db10(Gl @ Bm.T, Gr @ Bm.T), f'b{N}', out)
    g = db10(Gl.sum(1), Gr.sum(1)); out['grms_rms'] = np.sqrt(np.mean(g ** 2))
    out['grms_bias'] = g.mean(); out['grms_worst'] = np.abs(g).max()
    for Q, (Ka, Kd) in KER.items():
        forms(db10(Gl @ Ka.T, Gr @ Ka.T), f'ers{Q}', out)
        if Q == 10:
            forms(4 * db10(Gl @ Kd.T, Gr @ Kd.T), 'fds10', out)  # damage dB, b = 8
    Gv_l, Gv_r = Gl / W ** 2, Gr / W ** 2
    for N in (6, 3):
        forms(db10(Gv_l @ BANDS[N].T, Gv_r @ BANDS[N].T), f'vel{N}', out)
    w = Gr / Gr.sum(0, keepdims=True); e527 = (w * E).sum(0)
    out['g527_rms'] = np.sqrt(np.mean(e527 ** 2)); out['g527_maxabs'] = np.abs(e527).max()
    out['g527_out3'] = np.mean(np.abs(e527) > 3)
    for N in (6, 3):
        s = db10(Gl.sum(0) @ BANDS[N].T, Gr.sum(0) @ BANDS[N].T)
        out[f'sum{N}_rms'] = np.sqrt(np.mean(s ** 2)); out[f'sum{N}_maxabs'] = np.abs(s).max()
    for k in (6, 12):
        if k <= Yl.shape[0]:
            Psi = np.linalg.pinv(Phi_ch[:, :k])
            El = auto(np.einsum('kc,crf->krf', Psi, Yl)); Er = auto(np.einsum('kc,crf->krf', Psi, Yr))
            forms(db10(El @ BANDS[6].T, Er @ BANDS[6].T), f'mod{k}', out)
    gl = np.einsum('crf,csf->rsf', np.conj(Yl), Yl); gr = np.einsum('crf,csf->rsf', np.conj(Yr), Yr)
    x = np.einsum('crf,csf->rsf', np.conj(Yr), Yl)
    num = (np.abs(gl) ** 2).sum((0, 1)) + (np.abs(gr) ** 2).sum((0, 1)) - 2 * (np.abs(x) ** 2).sum((0, 1))
    den = (np.abs(gr) ** 2).sum((0, 1))
    out['frob_line'] = np.mean(np.sqrt(np.maximum(num, 0) / den))
    out['frob_int'] = np.sqrt(max(num.sum(), 0) / den.sum())
    return out


# ------------------------------------------------------------------ model set-up per seed
def service_forces(seed, art, host, na):
    """8-12 host forces from 6 sources (random mixing with delays) plus 2 weaker article forces."""
    rng = np.random.default_rng(seed + 104729)
    def shapes(n):
        lf = np.log10(F); S = np.zeros((n, NF))
        for k in range(n):
            s = rng.uniform(-1, .5) * (lf - 2)
            for _ in range(3):
                s += rng.uniform(-.5, .5) * np.exp(-.5 * ((lf - rng.uniform(1.3, 3.3)) / rng.uniform(.2, .5)) ** 2)
            S[k] = 10 ** s / np.mean(10 ** s)
        return S
    nh = int(rng.integers(8, 13)); hn = rng.choice(host.meta['nodes'], nh, replace=False)
    hd = rng.standard_normal((nh, 3)); hd /= np.linalg.norm(hd, axis=1, keepdims=True)
    B = rng.standard_normal((nh, 6)) * (rng.random((nh, 6)) < .7); tau = rng.uniform(0, 1.5e-3, (nh, 6))
    Bf = B[:, :, None] * np.exp(-1j * W[None, None, :] * tau[:, :, None]) * np.sqrt(shapes(6))[None]
    an = rng.choice([n for n in art.meta['chassis'] if n not in art.meta['mounts']], 2, replace=False)
    ad = rng.standard_normal((2, 3)); ad /= np.linalg.norm(ad, axis=1, keepdims=True)
    Af = 0.15 * np.sqrt(np.mean(B ** 2) * 6) * np.einsum('ij,jf->ijf', np.eye(2), np.sqrt(shapes(2)))
    r = 8; dofs, Fin = [], []
    for i in range(nh):
        for d in range(3):
            dofs.append(na + 6 * hn[i] + d); Fin.append(np.concatenate([hd[i, d] * Bf[i], np.zeros((2, NF))]))
    for i in range(2):
        for d in range(3):
            dofs.append(6 * an[i] + d); Fin.append(np.concatenate([np.zeros((6, NF)), ad[i, d] * Af[i]]))
    return np.array(dofs), np.array(Fin)  # (n_in,), (n_in, r, n_f)


def efi(Phi, n):
    idx = np.arange(Phi.shape[0])
    while idx.size > n:
        P = Phi[idx]; Ed = np.einsum('ij,jk,ik->i', P, np.linalg.pinv(P.T @ P), P); idx = np.delete(idx, np.argmin(Ed))
    return idx


def choose_rig(seed, art_nom, mlab_nom):
    """Held-out chassis channels (random), controls by Effective Independence on the nominal lab modes,
    shakers greedy on mean log condition number of H(controls, shakers) over 40 log-spaced lines."""
    rng = np.random.default_rng(seed + 31337); m = art_nom.meta
    held_nodes = rng.choice([n for n in m['chassis'] if n not in m['bases']], 4, replace=False)
    held = np.array([6 * n + d for n in held_nodes for d in range(3)])
    cand = np.array([6 * n + d for n in m['chassis'] if n not in held_nodes for d in range(3)])
    ctrls = {nc: np.sort(cand[efi(mlab_nom.Phi[cand][:, :nc], nc)]) for nc in NC_LIST}
    f40 = np.geomspace(20, 2000, 40); hq = mlab_nom.h(f40) * -(2 * np.pi * f40) ** 2
    shk_cand = np.array([6 * n + d for n in m['chassis'] for d in range(3)])
    shakers = {}
    for nc, c in ctrls.items():
        H = np.einsum('cm,mf,sm->fcs', mlab_nom.Phi[c], hq, mlab_nom.Phi[shk_cand])  # (40, nc, ncand)
        sel = [int(np.argmax(np.log(np.linalg.norm(H, axis=1)).mean(0)))]
        for ns in range(2, max(NS_LIST) + 1):
            rest = [j for j in range(len(shk_cand)) if j not in sel]
            Hs = np.stack([H[:, :, sel + [j]] for j in rest])  # (nrest, 40, nc, ns)
            s = np.linalg.svd(Hs, compute_uv=False); cnd = np.log10(s[..., 0] / s[..., -1]).mean(1)
            sel.append(rest[int(np.argmin(cnd))])
            if ns in NS_LIST:
                shakers[(ns, nc)] = shk_cand[sel].copy()
    return held, ctrls, shakers


# ------------------------------------------------------------------ control laws
_PROB = {}
def _band_prob(nc, ns):
    import cvxpy as cp
    if (nc, ns) not in _PROB:
        D = cp.Variable((ns, ns), hermitian=True); Bw = cp.Parameter((nc, ns * ns), complex=True); tw = cp.Parameter(nc)
        obj = cp.sum_squares(cp.real(Bw @ cp.vec(D, order='F')) - tw) + 1e-4 * cp.real(cp.trace(D))
        _PROB[(nc, ns)] = (cp.Problem(cp.Minimize(obj), [D >> 0]), D, Bw, tw)
    return _PROB[(nc, ns)]


def band_law(Hc, Gt, N):
    """One flat correlated PSD drive per 1/N-octave band, chosen so band-averaged control autospectra
    match band-averaged target autospectra. Weighted least squares on linear PSD with 3 reweighting
    passes (weights 1/sqrt(target*response), i.e. a first-order dB residual); cvxpy + Clarabel SDP."""
    import cvxpy as cp
    nc, ns, _ = Hc.shape; prob, D, Bw, tw = _band_prob(nc, ns)
    Fd = np.zeros((ns, ns, NF), complex); resid = []
    for row in BANDS[N]:
        idx = np.nonzero(row)[0]; H = Hc[:, :, idx]
        Bc = np.einsum('cif,cjf->cij', H, np.conj(H)) / idx.size; t = Gt[:, idx].mean(1)
        alpha = t.mean() / np.real(np.einsum('cii->c', Bc)).mean(); Bv = alpha * Bc.reshape(nc, -1, order='F')
        w = 1 / t; Dv = alpha * np.eye(ns, dtype=complex)
        for _ in range(3):
            Bw.value = Bv * w[:, None]; tw.value = t * w; val = None
            for solver in (cp.CLARABEL, cp.SCS):
                try:
                    prob.solve(solver=solver); val = D.value
                except Exception:
                    val = None
                if val is not None:
                    break
            if val is not None:  # otherwise keep the previous pass (or scaled identity)
                Dv = alpha * val; Dv = (Dv + Dv.conj().T) / 2
            r = np.maximum(np.real(np.einsum('cij,ij->c', Bc, Dv)), 1e-6 * t); w = 1 / np.sqrt(t * r)
        lam, V = np.linalg.eigh(Dv); lam = np.clip(lam, 0, None)
        Fd[:, :, idx] = (V * np.sqrt(lam))[:, :, None]; resid.append(db10(r, t))
    return Fd, np.sqrt(np.mean(np.concatenate(resid) ** 2))


def line_law(Hc, Lref):
    return (pinv_trunc(Hc.transpose(2, 0, 1)) @ Lref.transpose(2, 0, 1)).transpose(1, 2, 0)


def modal_law(Hc, Lref, Phi_c, k):
    Psi = np.linalg.pinv(Phi_c[:, :k])
    return line_law(np.einsum('kc,csf->ksf', Psi, Hc), np.einsum('kc,crf->krf', Psi, Lref))


def envelope(Gm):
    """Straight-line (log-log) envelope of 1/3-octave-smoothed autospectra: knots at octave
    frequencies 20..1280 Hz and 2000 Hz, minimum log-area LP subject to enveloping every band."""
    Bm = BANDS[3]; y = np.log10((Gm @ Bm.T).max(0)); fc = np.exp((Bm @ np.log(F)))
    xk = np.log10([20, 40, 80, 160, 320, 640, 1280, 2000]); xc = np.log10(fc)
    A = np.array([np.interp(xc, xk, np.eye(len(xk))[k]) for k in range(len(xk))]).T
    c = np.r_[np.diff(xk), 0] / 2 + np.r_[0, np.diff(xk)] / 2
    v = linprog(c, A_ub=-A, b_ub=-y, bounds=[(None, None)] * len(xk), method='highs').x
    return 10 ** np.interp(np.log10(F), xk, v)


# ------------------------------------------------------------------ one group (seed, perturbation)
def run_group(args):
    seed, pert = args; t0 = time.time()
    art_nom = fm.make_article(seed)
    units = {u: fm.make_article(seed, np.random.default_rng([seed, u, pert]), pert) for u in (1, 2)}
    host = fm.make_host(seed, art_nom.meta['dims']); zbase = fm.DampingField(seed)
    zs = {u: zbase.unit(np.random.default_rng([seed, u, pert, 9]), 0.3 * pert) for u in (1, 2)}
    mlab_nom = fm.Modal(*fm.suspend(art_nom), zbase)
    held, ctrls, shakers = choose_rig(seed, art_nom, mlab_nom)
    m = art_nom.meta
    tips = np.array([6 * n + d for n in m['tips'] for d in range(3)])
    mnt = np.array([6 * n + d for n in m['mounts'] for d in range(3)])
    U = np.unique(np.concatenate([held, tips, mnt] + list(ctrls.values())))
    pos = {d: i for i, d in enumerate(U)}; ix = lambda dofs: np.array([pos[d] for d in dofs])
    # service, units A (1) and B (2)
    serv, YS, truthS, Smods = {}, {}, {}, {}
    for u in (1, 2):
        K, M, na = fm.couple(units[u], host); md = fm.Modal(K, M, zs[u])
        dofs, Fin = service_forces(seed, units[u], host, na); Q = modal_q(md, dofs, Fin)
        S = units[u].stress_matrix(); Smods[u] = S
        YS[u] = acc(md, U, Q); truthS[u] = element_truth(*moments(S @ md.Phi[:na], Q)); serv[u] = (md, Q, na)
    A, Bu = 1, 2; recs = []
    mlab = fm.Modal(*fm.suspend(units[Bu]), zs[Bu]); SB = Smods[Bu] @ mlab.Phi
    PhiU = mlab.Phi[U]
    def metric_sets(Ylab, nc, tip_ref=Bu):
        out = {}
        c, h = ix(ctrls[nc]), ix(held)
        for tag, sel in (('c_', c), ('h_', h)):
            for k, v in chan_metrics(Ylab[sel], YS[A][sel], PhiU[sel]).items():
                out[tag + k] = v
        g = db10(auto(Ylab[ix(tips)]).sum(1), auto(YS[tip_ref][ix(tips)]).sum(1))
        out['tip_rms'] = np.sqrt(np.mean(g ** 2)); out['tip_bias'] = g.mean()
        out['hz_pass'] = bool(out['c_line_maxabs'] <= 1.0)
        return out
    # S1 floor: unit B in service against unit A in service
    for nc in NC_LIST:
        r = dict(seed=seed, pert=pert, test='S1', ns=0, nc=nc, k=0, drive=np.nan, resid=np.nan)
        r.update(truth_summary(truthS[Bu], truthS[A])); r.update(metric_sets(YS[Bu], nc, tip_ref=A)); recs.append(r)
    # S4: service stress carried by the first k elastic free-free modes of unit B
    mdB, QB, naB = serv[Bu]; Ka, Ma = units[Bu].assemble()
    P = mlab.Phi.T @ Ma @ mdB.Phi[:naB]; full = truthS[Bu]
    s4 = {}
    for ke in (1, 2, 3, 5, 10, 20, 40, mlab.Phi.shape[1] - 6):
        k = 6 + ke; tr = element_truth(*moments(Smods[Bu] @ mlab.Phi[:, :k] @ P[:k], QB))
        x = 20 * np.log10(tr['rms'] / full['rms']); s4[ke] = (np.median(x), np.mean(np.abs(x) <= 3), np.mean(x))
    # lab multi-shaker tests on unit B
    for (ns, nc), shk in shakers.items():
        Hu = np.einsum('um,mf,sm->usf', PhiU, mlab.h(F) * -W ** 2, mlab.Phi[shk], optimize=True)
        c = ix(ctrls[nc]); Hc = Hu[c]; Lref = YS[A][c]
        laws = [('LINE', 0, lambda: (line_law(Hc, Lref), np.nan))]
        laws += [(f'BAND{N}', 0, (lambda N=N: band_law(Hc, auto(Lref), N))) for N in (6, 3)]
        laws += [('MODAL', k, (lambda k=k: (modal_law(Hc, Lref, PhiU[c], k), np.nan)))
                 for k in (3, 6, 9, 12) if k <= min(ns, nc)]
        for name, k, fn in laws:
            Fd, resid = fn(); Ylab = np.einsum('usf,srf->urf', Hu, Fd, optimize=True)
            Q = mlab.h(F)[:, None, :] * np.einsum('sm,srf->mrf', mlab.Phi[shk], Fd, optimize=True)
            r = dict(seed=seed, pert=pert, test=name, ns=ns, nc=nc, k=k, resid=resid,
                     drive=10 * np.log10((np.abs(Fd) ** 2).sum() * DF))
            r.update(truth_summary(element_truth(*moments(SB, Q)), truthS[Bu])); r.update(metric_sets(Ylab, nc))
            recs.append(r)
    # single-axis sequence on a rigid base, unit B
    mf = fm.Modal(Ka, Ma, zs[Bu], fixed=fm.fixed_base_dofs(units[Bu])); Sf = Smods[Bu] @ mf.Phi
    hB = mf.h(F); GA = auto(YS[A]); Ysa = np.zeros((U.size, 3, NF), complex); tr_ax = []
    for ax in range(3):
        spec = envelope(GA[ix(mnt[ax::3])]); rr = units[Bu].rigid_translation(ax)
        Gam = mf.Phi.T @ Ma @ rr; qa = Gam[:, None] * hB
        Ysa[:, ax] = (rr[U][:, None] + W ** 2 * (mf.Phi[U] @ qa)) * np.sqrt(spec)
        tr_ax.append(element_truth(*moments(Sf, (-qa * np.sqrt(spec))[:, None, :])))
    for gain_db in (0, 3):
        g = 10 ** (gain_db / 10)
        tr = dict(rms=np.max([t['rms'] for t in tr_ax], 0) * np.sqrt(g),
                  **{f'd{b}': np.sum([t[f'd{b}'] for t in tr_ax], 0) * g ** (b / 2) for b in B_EXP})
        for nc in NC_LIST:
            r = dict(seed=seed, pert=pert, test=f'SA{gain_db}', ns=0, nc=nc, k=0, resid=np.nan, drive=np.nan)
            r.update(truth_summary(tr, truthS[Bu])); r.update(metric_sets(Ysa * np.sqrt(g), nc)); recs.append(r)
    info = dict(seed=seed, pert=pert, s4=s4, n_el=len(units[Bu].el), n_modes_lab=mlab.Phi.shape[1],
                n_modes_serv=serv[Bu][0].Phi.shape[1], lab_f=mlab.fn[6:12], f_target=m['f_target'],
                secs=time.time() - t0)
    if seed == 0 and pert == 1:  # example spectra for the figure
        ex_ns, ex_nc = 6, 12; shk = shakers[(ex_ns, ex_nc)]; c = ix(ctrls[ex_nc])
        Hu = np.einsum('um,mf,sm->usf', PhiU, mlab.h(F) * -W ** 2, mlab.Phi[shk], optimize=True)
        ex = dict(ref=GA, servB=auto(YS[Bu]), line=auto(np.einsum('usf,srf->urf', Hu, line_law(Hu[c], YS[A][c]))),
                  band3=auto(np.einsum('usf,srf->urf', Hu, band_law(Hu[c], GA[c], 3)[0])), sa=auto(Ysa),
                  ctrl=c, tips=ix(tips), held=ix(held))
        info['example'] = ex
    print(f'group seed={seed} pert={pert} cases={len(recs)} {info["secs"]:.0f}s', flush=True)
    return recs, info


# ------------------------------------------------------------------ analysis
def auc(score, good):
    """Area under ROC for 'larger score = worse' predicting not-good."""
    s = np.asarray(score, float); bad = ~np.asarray(good, bool)
    if bad.all() or (~bad).all():
        return np.nan
    from scipy.stats import rankdata
    rk = rankdata(s); n1 = bad.sum(); n0 = (~bad).sum()
    return (rk[bad].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


METRICS = ['line_rms', 'line_out3', 'line_worst', 'b12_rms', 'b6_rms', 'b3_rms', 'b3_out3', 'b3_worst', 'b3_bias',
           'grms_rms', 'grms_worst', 'ers10_rms', 'ers25_rms', 'ers50_rms', 'ers10_worst', 'fds10_rms',
           'vel6_rms', 'vel3_rms', 'g527_rms', 'g527_maxabs', 'sum6_rms', 'sum3_rms', 'mod6_rms', 'frob_line', 'frob_int']


def analyze():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    recs, infos = pickle.load(open(os.path.join(SCR, 'x1_results.pkl'), 'rb'))
    lines = []; P = lambda s='': (print(s), lines.append(s))
    for r in recs:  # derived flags (no re-simulation needed)
        x = r['x']; r['good6'] = bool(np.mean(np.abs(x) <= 6) >= 0.9 and x.max() <= 10)
        r['hz_rms_pass'] = bool(r['c_line_worst'] <= 1.0)  # present Hertzforge criterion: every control's line-rms dB <= 1
    col = lambda key, rs: np.array([r[key] for r in rs], float)
    figs = os.path.join(HERE, 'figs'); os.makedirs(figs, exist_ok=True)
    P(f'cases {len(recs)}  groups {len(infos)}  runtime per group (s) median {np.median([i["secs"] for i in infos]):.0f}')
    P(f'modes lab/serv median {np.median([i["n_modes_lab"] for i in infos])}/{np.median([i["n_modes_serv"] for i in infos])}'
      f'  elements median {np.median([i["n_el"] for i in infos])}')
    T = ['t_rms', 't_bias', 't_p95', 't_p5', 't_in3', 'd8_rms', 'd4_rms', 'tip_rms']
    def table(rs, keys, label):
        P(f'{label:22s} n={len(rs):4d} ' + ' '.join(f'{k}={np.nanmean(col(k, rs)):7.2f}' for k in keys)
          + f' good={np.mean([r["good"] for r in rs]):.2f}')
    P('\n== S1 floor: unit B in service vs unit A in service (truth B/A) ==')
    for p in (1, 2):
        rs = [r for r in recs if r['test'] == 'S1' and r['pert'] == p]
        table(rs, T + ['c_line_rms', 'c_b3_rms', 'c_grms_rms', 'c_ers10_rms', 'c_g527_rms', 'c_frob_line', 'h_line_rms', 'h_b3_rms'], f'S1 pert={p}')
    P('\n== S2 line vs band (all rigs pooled) ==')
    for p in PERTS:
        for t in ('LINE', 'BAND6', 'BAND3'):
            rs = [r for r in recs if r['test'] == t and r['pert'] == p]
            table(rs, T + ['drive', 'resid', 'c_line_rms', 'c_b3_rms', 'h_line_rms', 'h_b3_rms'], f'{t} pert={p}')
        L = {(r['seed'], r['ns'], r['nc']): r for r in recs if r['test'] == 'LINE' and r['pert'] == p}
        for t in ('BAND6', 'BAND3'):
            pr = [(L[(r['seed'], r['ns'], r['nc'])], r) for r in recs if r['test'] == t and r['pert'] == p]
            P(f'  pert={p} {t}-LINE: d t_rms {np.mean([b["t_rms"] - a["t_rms"] for a, b in pr]):+.2f} dB, '
              f'band better in {np.mean([b["t_rms"] < a["t_rms"] for a, b in pr]):.2f}, d d8_rms '
              f'{np.mean([b["d8_rms"] - a["d8_rms"] for a, b in pr]):+.2f}, d drive {np.mean([b["drive"] - a["drive"] for a, b in pr]):+.2f} dB')
        for ns in NS_LIST:
            for nc in NC_LIST:
                g = {t: [r for r in recs if r['test'] == t and r['pert'] == p and r['ns'] == ns and r['nc'] == nc] for t in ('LINE', 'BAND3')}
                if p == 1:
                    P(f'    ns={ns:2d} nc={nc:2d} LINE t_rms {np.mean(col("t_rms", g["LINE"])):5.2f} drive {np.mean(col("drive", g["LINE"])):6.1f} | '
                      f'BAND3 t_rms {np.mean(col("t_rms", g["BAND3"])):5.2f} drive {np.mean(col("drive", g["BAND3"])):6.1f}')
    P('\n== S3 multi-shaker vs single-axis (pert=1) ==')
    for ns in NS_LIST:
        for nc in NC_LIST:
            for t in ('LINE', 'BAND3'):
                rs = [r for r in recs if r['test'] == t and r['pert'] == 1 and r['ns'] == ns and r['nc'] == nc]
                table(rs, T, f'{t} ns={ns} nc={nc}')
    for p in PERTS:
        for t in ('SA0', 'SA3'):
            table([r for r in recs if r['test'] == t and r['pert'] == p and r['nc'] == 12], T, f'{t} pert={p}')
    P('\n== S4 service stress carried by first k elastic free-free modes (median dB, frac within 3 dB, mean dB) ==')
    for ke in (1, 2, 3, 5, 10, 20, 40):
        v = np.array([i['s4'][ke] for i in infos]); P(f'  k_el={ke:3d}  median {v[:, 0].mean():6.2f}  in3 {v[:, 1].mean():.2f}  mean {v[:, 2].mean():6.2f}')
    v = np.array([list(i['s4'].values())[-1] for i in infos]); P(f'  all modes  median {v[:, 0].mean():6.2f}  in3 {v[:, 1].mean():.2f}')
    for k in (3, 6, 9, 12):
        table([r for r in recs if r['test'] == 'MODAL' and r['k'] == k and r['pert'] == 1], T + ['drive', 'c_line_rms', 'h_b3_rms'], f'MODAL k={k} pert=1')
    P('\n== S5 Spearman (metric vs truth) and AUC (truth-good) ==')
    lab = [r for r in recs if r['test'] != 'S1']
    groups = {'pooled': lab, 'LINE': [r for r in lab if r['test'] == 'LINE'], 'BAND': [r for r in lab if r['test'].startswith('BAND')],
              'MODAL': [r for r in lab if r['test'] == 'MODAL'], 'SA': [r for r in lab if r['test'].startswith('SA')],
              'multi': [r for r in lab if not r['test'].startswith('SA')]}
    P(f'truth-good rate pooled {np.mean([r["good"] for r in lab]):.2f}; by test ' +
      ', '.join(f'{t} {np.mean([r["good"] for r in lab if r["test"] == t]):.2f}' for t in ('LINE', 'BAND6', 'BAND3', 'MODAL', 'SA0', 'SA3')))
    res = {}
    for gname, rs in groups.items():
        good = np.array([r['good'] for r in rs])
        P(f'-- {gname} (n={len(rs)}, good={good.mean():.2f}) rho vs t_rms | rho vs d8_rms | rho vs d4_rms | AUC   [ctrl / held]')
        for mname in METRICS:
            row = []
            for tag in ('c_', 'h_'):
                x = col(tag + mname, rs); x = np.abs(x) if 'bias' in mname else x
                vals = []
                for tk in ('t_rms', 'd8_rms', 'd4_rms'):
                    y = col(tk, rs); ok = np.isfinite(x) & np.isfinite(y)
                    vals.append(spearmanr(x[ok], y[ok])[0] if ok.sum() > 5 and np.ptp(x[ok]) > 1e-9 else np.nan)
                vals.append(auc(np.nan_to_num(x, nan=np.nanmax(x) if np.isfinite(x).any() else 0), good))
                row.append(vals); res[(gname, tag, mname)] = vals
            P(f'  {mname:12s} ' + ' | '.join(f'{a:5.2f}/{b:5.2f}' for a, b in zip(row[0], row[1])))
    P('\n== S5 sensitivity: Spearman vs t_rms within perturbation level (multi-shaker), ctrl/held ==')
    for mname in ('line_rms', 'b3_rms', 'grms_rms', 'ers10_rms', 'g527_rms', 'frob_line', 'sum3_rms'):
        s = []
        for p in PERTS:
            rs = [r for r in groups['multi'] if r['pert'] == p]
            s.append('/'.join(f'{spearmanr(col(tag + mname, rs), col("t_rms", rs))[0]:5.2f}' for tag in ('c_', 'h_')))
        P(f'  {mname:10s} pert0 {s[0]}  pert1 {s[1]}  pert2 {s[2]}')
    P('\n== S6 thresholds on band metrics and the Hertzforge 1 dB line criterion ==')
    good = np.array([r['good'] for r in lab])
    for mname in ('c_b3_rms', 'c_b6_rms', 'h_b3_rms', 'c_ers10_rms', 'h_ers10_rms', 'c_grms_rms', 'c_g527_rms', 'h_line_rms'):
        x = col(mname, lab)
        for tau in (0.5, 1, 1.5, 2, 3, 4, 6):
            pas = x <= tau
            if pas.sum():
                P(f'  {mname:12s} <= {tau:3.1f} dB: pass {pas.mean():.2f}  precision {good[pas].mean():.2f}  recall {pas[good].mean():.2f}')
    hz = np.array([r['hz_pass'] for r in lab])
    P(f'  Hertzforge 1 dB line criterion: pass rate {hz.mean():.2f}; among truth-good {hz[good].mean():.2f}; '
      f'among truth-bad {hz[~good].mean():.2f}; precision {good[hz].mean() if hz.any() else np.nan:.2f}')
    for t in ('LINE', 'BAND6', 'BAND3', 'MODAL', 'SA0'):
        sel = np.array([r['test'] == t for r in lab])
        P(f'    {t:6s} pass {hz[sel].mean():.2f}  truth-good {good[sel].mean():.2f}  good&fail {(good & ~hz & sel).sum()}  bad&pass {(~good & hz & sel).sum()}')
    P('\n== S6b present Hertzforge criterion (every control: RMS over lines of dB error <= 1.0 dB) ==')
    hz2 = np.array([r['hz_rms_pass'] for r in lab]); good6 = np.array([r['good6'] for r in lab])
    P(f'  pass {hz2.mean():.2f}; among truth-good {hz2[good].mean():.2f}; among good6 {hz2[good6].mean():.2f}; '
      f'precision(good) {good[hz2].mean():.2f}; precision(good6) {good6[hz2].mean():.2f}')
    for t in ('LINE', 'BAND6', 'BAND3', 'MODAL', 'SA0'):
        sel = np.array([r['test'] == t for r in lab])
        P(f'    {t:6s} pass {hz2[sel].mean():.2f}  good {good[sel].mean():.2f}  good6 {good6[sel].mean():.2f}  '
          f'good6&fail {(good6 & ~hz2 & sel).sum()}  notgood6&pass {(~good6 & hz2 & sel).sum()}')
    for p in (1, 2):
        rs = [r for r in recs if r['test'] == 'S1' and r['pert'] == p]
        P(f'  S1 floor pert={p}: present criterion passes {np.mean([r["hz_rms_pass"] for r in rs]):.2f}, '
          f'every-line form {np.mean([r["hz_pass"] for r in rs]):.2f}, worst control line-rms mean {np.mean(col("c_line_worst", rs)):.2f} dB')
    P('\n== Looser truth criterion good6: within 6 dB on 90 % of elements, none over +10 dB ==')
    P(f'  good6 rate: ' + ', '.join(f'{t} {np.mean([r["good6"] for r in lab if r["test"] == t]):.2f}' for t in ('LINE', 'BAND6', 'BAND3', 'MODAL', 'SA0', 'SA3')))
    for mname in ('line_rms', 'b3_rms', 'b3_bias', 'grms_rms', 'ers10_rms', 'fds10_rms', 'g527_rms', 'sum3_rms', 'mod6_rms', 'frob_line', 'line_worst'):
        P(f'  AUC good6 {mname:10s} ' + '  '.join(f'{g}: ' + '/'.join(f'{auc(np.abs(col(tag + mname, rs)), [r["good6"] for r in rs]):.2f}' for tag in ('c_', 'h_'))
              for g, rs in groups.items() if g in ('pooled', 'multi', 'LINE', 'BAND', 'MODAL')))
    for mname in ('c_sum3_rms', 'c_b3_rms', 'h_sum3_rms', 'h_ers10_rms'):
        x = col(mname, lab)
        for tau in (1, 2, 3, 4, 6):
            pas = x <= tau
            if pas.sum():
                P(f'  {mname:12s} <= {tau} dB: pass {pas.mean():.2f}  precision(good6) {good6[pas].mean():.2f}  recall {pas[good6].mean():.2f}')
    P('\n== S2b paired LINE vs BAND3 excluding square rigs (ns == nc) ==')
    for p in PERTS:
        L = {(r['seed'], r['ns'], r['nc']): r for r in recs if r['test'] == 'LINE' and r['pert'] == p}
        pr = [(L[(r['seed'], r['ns'], r['nc'])], r) for r in recs if r['test'] == 'BAND3' and r['pert'] == p and r['ns'] != r['nc']]
        P(f'  pert={p}: BAND3-LINE d t_rms {np.mean([b["t_rms"] - a["t_rms"] for a, b in pr]):+.2f}, band better in '
          f'{np.mean([b["t_rms"] < a["t_rms"] for a, b in pr]):.2f}, d bias {np.mean([b["t_bias"] - a["t_bias"] for a, b in pr]):+.2f}, '
          f'd drive {np.mean([b["drive"] - a["drive"] for a, b in pr]):+.2f}')
    open(os.path.join(SCR, 'x1_tables.txt'), 'w').write('\n'.join(lines))
    make_figs(recs, infos, res, figs, plt)


def make_figs(recs, infos, res, figs, plt):
    sty = dict(lw=1.4, ms=4)
    mk = ['o', 's', '^', 'D', 'v', 'x', '+']; ls = ['-', '--', ':', '-.', (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1))]
    cl = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#000000']
    # Fig 1: ECDF of element stress dB error, pert=1
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sets = [('S1 floor (two units)', 'S1', {}), ('LINE', 'LINE', {}), ('BAND 1/6', 'BAND6', {}), ('BAND 1/3', 'BAND3', {}),
            ('MODAL k=max', 'MODAL', {}), ('Single-axis, 0 dB', 'SA0', {}), ('Single-axis, +3 dB', 'SA3', {})]
    for i, (lab_, t, _) in enumerate(sets):
        rs = [r for r in recs if r['test'] == t and r['pert'] == 1 and (t != 'MODAL' or r['k'] == min(r['ns'], r['nc'], 12))]
        x = np.sort(np.concatenate([r['x'] for r in rs])); y = np.linspace(0, 1, x.size)
        ax.plot(x, y, color=cl[i], ls=ls[i], marker=mk[i], markevery=max(1, x.size // 8), label=lab_, **sty)
    ax.axvspan(-3, 3, color='0.9'); ax.set_xlim(-25, 25); ax.set_xlabel('element RMS stress, lab / service (dB)')
    ax.set_ylabel('cumulative fraction of elements'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title('an earlier study: stress error distribution, 3 % unit perturbation, all rigs pooled', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figs, 'x1_fig1_stress_ecdf.png'), dpi=110); plt.close(fig)
    # Fig 2: Spearman vs t_rms, ctrl vs held, pooled
    fig, ax = plt.subplots(figsize=(7, 5.5)); y = np.arange(len(METRICS))
    for j, (g, tag, m_, c_) in enumerate([('pooled', 'c_', 'o', cl[0]), ('pooled', 'h_', 's', cl[1]), ('multi', 'c_', '^', cl[2]), ('multi', 'h_', 'D', cl[3])]):
        v = [res[(g, tag, mm)][0] for mm in METRICS]
        ax.plot(v, y + (j - 1.5) * .15, ls='none', marker=m_, color=c_, mfc='none' if j % 2 else c_,
                label=f'{g} {"controls" if tag == "c_" else "held-out"}')
    ax.set_yticks(y); ax.set_yticklabels(METRICS, fontsize=7); ax.axvline(0, color='k', lw=.8)
    ax.set_xlabel('Spearman rank correlation with RMS-over-elements stress dB error'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, 'x1_fig2_spearman.png'), dpi=110); plt.close(fig)
    # Fig 3: S4 and MODAL vs k
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.6)); kes = [1, 2, 3, 5, 10, 20, 40]
    v = np.array([[i['s4'][k] for k in kes] for i in infos])
    ax[0].semilogx(kes, v[:, :, 0].mean(0), 'o-', color=cl[0], label='median element (dB)')
    ax[0].semilogx(kes, 10 * v[:, :, 1].mean(0) - 10, 's--', color=cl[1], label='10 x frac within 3 dB - 10')
    ax[0].set_xlabel('elastic free-free modes kept'); ax[0].set_ylabel('service stress from first k modes / full (dB)')
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    for i, ns in enumerate(NS_LIST):
        ks = [k for k in (3, 6, 9, 12) if k <= ns]
        t = [np.mean([r['t_rms'] for r in recs if r['test'] == 'MODAL' and r['k'] == k and r['ns'] == ns and r['pert'] == 1]) for k in ks]
        ax[1].plot(ks, t, marker=mk[i], ls=ls[i], color=cl[i], label=f'MODAL, {ns} shakers')
        ln = np.mean([r['t_rms'] for r in recs if r['test'] == 'LINE' and r['ns'] == ns and r['pert'] == 1])
        ax[1].axhline(ln, color=cl[i], ls=ls[i], lw=.8)
    ax[1].set_xlabel('modal coordinates controlled k (incl. 6 rigid-body)'); ax[1].set_ylabel('stress dB error, RMS over elements')
    ax[1].legend(fontsize=7, title='thin lines: LINE', title_fontsize=7); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, 'x1_fig3_modes.png'), dpi=110); plt.close(fig)
    # Fig 4: scatter truth vs best band metric (controls), marker by test
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, t in enumerate(('LINE', 'BAND6', 'BAND3', 'MODAL', 'SA0', 'SA3')):
        rs = [r for r in recs if r['test'] == t]
        ax.plot([r['c_b3_rms'] for r in rs], [r['t_rms'] for r in rs], ls='none', marker=mk[i], color=cl[i], ms=3, alpha=.6, label=t)
    ax.set_xscale('symlog', linthresh=0.1); ax.set_xlabel('controls: 1/3-octave band dB error, RMS (dB)')
    ax.set_ylabel('truth: element stress dB error, RMS (dB)'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(figs, 'x1_fig4_metric_vs_truth.png'), dpi=110); plt.close(fig)
    # Fig 5: example spectra
    ex = next((i['example'] for i in infos if 'example' in i), None)
    if ex is not None:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
        for a, ch, ttl in ((ax[0], ex['ctrl'][0], 'a control channel'), (ax[1], ex['tips'][2], 'a component tip (held out)')):
            for i, (k, lab_) in enumerate((('ref', 'service unit A (target)'), ('servB', 'service unit B (truth)'), ('line', 'LINE'), ('band3', 'BAND 1/3'), ('sa', 'single-axis sum'))):
                a.loglog(F, ex[k][ch], color=cl[i], ls=ls[i], lw=1, label=lab_)
            a.set_title(ttl, fontsize=9); a.set_xlabel('Hz'); a.grid(alpha=.3)
        ax[0].set_ylabel('acceleration ASD (arb. units)'); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(figs, 'x1_fig5_example_spectra.png'), dpi=110); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--analyze', action='store_true'); a = ap.parse_args()
    os.makedirs(SCR, exist_ok=True)
    if not a.analyze:
        t0 = time.time(); jobs = [(s, p) for s in range(a.seeds) for p in PERTS]
        out = []
        with Pool(a.workers) as pool:
            for o in pool.imap_unordered(run_group, jobs, chunksize=1):
                out.append(o); pickle.dump(out, open(os.path.join(SCR, 'x1_partial.pkl'), 'wb'))
        recs = [r for rs, _ in out for r in rs]; infos = [i for _, i in out]
        pickle.dump((recs, infos), open(os.path.join(SCR, 'x1_results.pkl'), 'wb'))
        print(f'simulation {time.time() - t0:.0f}s')
    analyze()
