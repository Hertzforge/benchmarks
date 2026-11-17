"""an earlier study: independent audit of an earlier study, an earlier study and an earlier study. Imports the study scripts unchanged.

    PY=python
    $PY a1_audit.py repro            # rerun each study's --analyze on its saved results, into scratch
    $PY a1_audit.py fresh            # unseen seeds: X1 10..19 (all perts), X1b part A 10..19 (p=1), X2 6..11
    $PY a1_audit.py fresh_analyze
    $PY a1_audit.py variants         # artifact hunt (seeds 0..9 random, 0..5 shock)
    $PY a1_audit.py saved_checks     # re-analysis of saved results (orientation, p95 truth, counts)

Everything is written to SCR (scratch); nothing in the study folders is modified.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
from scipy import linalg as sla

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import frame_model as fm
import x1_random_study as x1
import x1b_followup_study as x1b
import x2_shock_study as x2

ROOT = x1.SCR.rsplit('/', 1)[0]
SCR = os.path.join(ROOT, 'a1')
os.makedirs(SCR, exist_ok=True)


def _redirect(mod, sub):
    d = os.path.join(SCR, sub); os.makedirs(os.path.join(d, 'figs'), exist_ok=True)
    mod.SCR = d; mod.HERE = d
    return d


def _link(src, dst):
    if not os.path.exists(dst):
        os.symlink(src, dst)


# ------------------------------------------------------------------ 1. reproduction from saved results
def repro():
    saved = {k: os.path.join(ROOT, k) for k in ('x1', 'x1b', 'x2')}
    d1 = _redirect(x1, 'repro/x1'); _link(os.path.join(saved['x1'], 'x1_results.pkl'), os.path.join(d1, 'x1_results.pkl'))
    x1.analyze()
    d1b = _redirect(x1b, 'repro/x1b'); _link(os.path.join(saved['x1b'], 'x1b_results.pkl'), os.path.join(d1b, 'x1b_results.pkl'))
    x1b.analyze()
    d2 = _redirect(x2, 'repro/x2'); _link(os.path.join(saved['x2'], 'x2_results.pkl'), os.path.join(d2, 'x2_results.pkl'))
    x2.analyze([])
    for k, d, f in (('x1', d1, 'x1_tables.txt'), ('x1b', d1b, 'x1b_tables.txt'), ('x2', d2, 'x2_tables.txt')):
        a = open(os.path.join(saved[k], f)).read().splitlines(); b = open(os.path.join(d, f)).read().splitlines()
        diff = [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
        print(f'REPRO {k}: saved {len(a)} lines, rerun {len(b)} lines, differing lines {len(diff) + abs(len(a) - len(b))}')
        for i, x, y in diff[:8]:
            print(f'   line {i}\n     saved: {x}\n     rerun: {y}')


# ------------------------------------------------------------------ 2. fresh seeds
def _job(j):
    kind, args = j; t0 = time.time()
    if kind == 'x1':
        out = ('x1', x1.run_group(args))
    elif kind == 'x1b':
        out = ('x1b', x1b.run_group(args))
    else:
        x2.NREAL, x2.NREAL_M = 5, 2
        out = ('x2', x2.run_part(args))
    print(f'{kind} {args} {time.time() - t0:.0f}s', flush=True)
    return out


def fresh(workers=16):
    jobs = [('x1b', (s, 1, 'A')) for s in range(10, 20)]
    jobs += [('x2', (s, p, f'syn:{ns}:{nc}')) for (ns, nc) in x2.SYN_RIGS for s in range(6, 12) for p in x2.PERTS]
    jobs += [('x1', (s, p)) for s in range(10, 20) for p in x1.PERTS]
    jobs += [('x2', (s, p, 'base')) for s in range(6, 12) for p in x2.PERTS]
    t0 = time.time(); out = {'x1': [], 'x1b': [], 'x2': []}
    with Pool(workers) as pool:
        for k, o in pool.imap_unordered(_job, jobs, chunksize=1):
            out[k].append(o)
    d = os.path.join(SCR, 'fresh'); os.makedirs(d, exist_ok=True)
    pickle.dump(([r for rs, _ in out['x1'] for r in rs], [i for _, i in out['x1']]), open(os.path.join(d, 'x1_results.pkl'), 'wb'))
    pickle.dump([r for o in out['x1b'] for r in o], open(os.path.join(d, 'x1b_results.pkl'), 'wb'))
    pickle.dump(([r for rs, _ in out['x2'] for r in rs], [i for _, i in out['x2']]), open(os.path.join(d, 'x2_results.pkl'), 'wb'))
    print(f'fresh simulation {time.time() - t0:.0f}s')


# ------------------------------------------------------------------ 3. artifact variants: random
F, W, NF, DF = x1.F, x1.W, x1.NF, x1.DF


def rsetup(seed, pert):
    """Same set-up as x1.run_group, keeping the modal models."""
    art_nom = fm.make_article(seed)
    units = {u: fm.make_article(seed, np.random.default_rng([seed, u, pert]), pert) for u in (1, 2)}
    host = fm.make_host(seed, art_nom.meta['dims']); zbase = fm.DampingField(seed)
    zs = {u: zbase.unit(np.random.default_rng([seed, u, pert, 9]), 0.3 * pert) for u in (1, 2)}
    mlab_nom = fm.Modal(*fm.suspend(art_nom), zbase)
    held, ctrls, shakers = x1.choose_rig(seed, art_nom, mlab_nom); m = art_nom.meta
    tips = np.array([6 * n + d for n in m['tips'] for d in range(3)]); mnt = np.array(fm.fixed_base_dofs(art_nom))
    U = np.unique(np.concatenate([held, tips, mnt] + list(ctrls.values())))
    pos = {d: i for i, d in enumerate(U)}; ix = lambda dofs: np.array([pos[d] for d in dofs])
    md_s, Qs, YS, truthS = {}, {}, {}, {}
    for u in (1, 2):
        K, M, na = fm.couple(units[u], host); md = fm.Modal(K, M, zs[u])
        dofs, Fin = x1.service_forces(seed, units[u], host, na); Q = x1.modal_q(md, dofs, Fin)
        md_s[u], Qs[u] = md, Q; YS[u] = x1.acc(md, U, Q)
        truthS[u] = x1.element_truth(*x1.moments(units[u].stress_matrix() @ md.Phi[:na], Q))
    return dict(units=units, host=host, zs=zs, held=held, ctrls=ctrls, shakers=shakers, U=U, ix=ix, mnt=mnt, m=m,
                md_s=md_s, Qs=Qs, YS=YS, truthS=truthS, na=na)


def line_eval(R, md, shk, nc, cond=1e3, nart=None):
    """LINE law with the plant md (article dofs first); truth vs service unit B; held / EfI-uncontrolled sums."""
    U, ix = R['U'], R['ix']; nart = nart or md.Phi.shape[0]
    Hu = np.einsum('um,mf,sm->usf', md.Phi[U], md.h(F) * -W ** 2, md.Phi[shk], optimize=True)
    c = ix(R['ctrls'][nc]); old = x1.COND; x1.COND = cond
    Fd = x1.line_law(Hu[c], R['YS'][1][c]); x1.COND = old
    Q = md.h(F)[:, None, :] * np.einsum('sm,srf->mrf', md.Phi[shk], Fd, optimize=True)
    SB = R['units'][2].stress_matrix() @ md.Phi[:nart]
    r = x1.truth_summary(x1.element_truth(*x1.moments(SB, Q)), R['truthS'][2]); r.pop('x')
    Gl = x1.auto(np.einsum('usf,srf->urf', Hu, Fd, optimize=True)); GA = x1.auto(R['YS'][1]); B3 = x1.BANDS[3]
    s3 = lambda sel: np.sqrt(np.mean(x1.db10(Gl[sel].sum(0) @ B3.T, GA[sel].sum(0) @ B3.T) ** 2))
    r['h_sum3'] = s3(ix(R['held']))
    unc = np.setdiff1d(R['ctrls'][24], R['ctrls'][nc])
    r['e_sum3'] = s3(ix(unc)) if unc.size else np.nan
    r['c_b3'] = np.sqrt(np.mean(x1.db10(Gl[c] @ B3.T, GA[c] @ B3.T) ** 2))
    r['drive'] = 10 * np.log10((np.abs(Fd) ** 2).sum() * DF)
    return r


def base_ops(unit, zeta, mf=None):
    """Fixed-interface modes at the 12 mount translations plus static constraint modes (exact for
    prescribed, non-uniform mount motion)."""
    Ka, Ma = unit.assemble(); b = np.array(fm.fixed_base_dofs(unit)); f = np.setdiff1d(np.arange(Ka.shape[0]), b)
    Psi = np.zeros((Ka.shape[0], b.size)); Psi[b] = np.eye(b.size)
    Psi[f] = -np.linalg.solve(Ka[np.ix_(f, f)], Ka[np.ix_(f, b)])
    mf = mf if mf is not None else fm.Modal(Ka, Ma, zeta, fixed=b); S = unit.stress_matrix()
    return dict(mf=mf, Lb=mf.Phi.T @ Ma @ Psi, SPsi=S @ Psi, SPhi=S @ mf.Phi, Ma=Ma, Psi=Psi)


def base_truth_rand(O, Ab):
    """Ab (12, r, NF) mount acceleration factors -> element truth (random)."""
    nm = O['mf'].Phi.shape[1]; r = Ab.shape[1]
    q = -(O['Lb'] @ Ab.reshape(Ab.shape[0], -1)).reshape(nm, r, NF) * O['mf'].h(F)[:, None, :]
    Qa = np.concatenate([-Ab / W ** 2, q], 0)
    return x1.element_truth(*x1.moments(np.hstack([O['SPsi'], O['SPhi']]), Qa))


def seq_combine(trs):
    return dict(rms=np.max([t['rms'] for t in trs], 0), **{f'd{b}': np.sum([t[f'd{b}'] for t in trs], 0) for b in x1.B_EXP})


def rv_job(seed, pert=1):
    """Random-study artifact variants for one seed: single-axis split, square-rig causes, EfI held-out, rigid share."""
    t0 = time.time(); R = rsetup(seed, pert); ix = R['ix']; out = dict(seed=seed, pert=pert)
    u2 = R['units'][2]; O = base_ops(u2, R['zs'][2]); GA = x1.auto(R['YS'][1]); ref = R['truthS'][2]
    ts = lambda tr: (lambda s: {k: s[k] for k in ('t_rms', 't_bias', 't_p5', 't_p95', 't_in3')})(x1.truth_summary(tr, ref))
    # --- single-axis decomposition
    mntix = ix(R['mnt'])
    def rigid_axis(ax, spec_or_factor):
        Ab = np.zeros((12,) + spec_or_factor.shape, complex); Ab[ax::3] = spec_or_factor[None]
        return Ab
    sa = {}
    for name, fn in (('ENV (study)', lambda g: x1.envelope(g)), ('MAX unsmoothed', lambda g: g.max(0)), ('MEAN over mounts', lambda g: g.mean(0))):
        sa[name] = seq_combine([base_truth_rand(O, rigid_axis(ax, np.sqrt(fn(GA[mntix[ax::3]]))[None, :])) for ax in range(3)])
    YB = R['YS'][2][mntix]; mean_ax = [YB[ax::3].mean(0) for ax in range(3)]   # (r, NF) per axis, unit B's own service motion
    sa['TRUE B, per axis, sequential'] = seq_combine([base_truth_rand(O, rigid_axis(ax, mean_ax[ax])) for ax in range(3)])
    Ab3 = np.zeros_like(YB)
    for ax in range(3):
        Ab3[ax::3] = mean_ax[ax][None]
    sa['TRUE B, 3 axes at once (rigid 3-DOF table)'] = base_truth_rand(O, Ab3)
    sa['EXACT B, all 12 mount dofs prescribed'] = base_truth_rand(O, YB)
    out['sa'] = {k: ts(v) for k, v in sa.items()}
    # --- square-rig causes (LINE), plus non-square for contrast
    mlab = fm.Modal(*fm.suspend(u2), R['zs'][2]); rigs = [(6, 6), (12, 12), (6, 12), (12, 24), (3, 6), (3, 12), (3, 24), (6, 24), (12, 6)]
    cand = np.array([6 * n + d for n in R['m']['chassis'] for d in range(3)]); rng = np.random.default_rng(seed + 999)
    labs = {'susp 1 Hz': fm.Modal(*fm.suspend(u2, 1.0), R['zs'][2]), 'susp 10 Hz': fm.Modal(*fm.suspend(u2, 10.0), R['zs'][2]),
            'modes to 8 kHz': fm.Modal(*fm.suspend(u2), R['zs'][2], fmax=8000.0)}
    sq = {}
    for (ns, nc) in rigs:
        shk = R['shakers'][(ns, nc)]; base = line_eval(R, mlab, shk, nc); sq[(ns, nc, 'base')] = base
        if ns != nc and (ns, nc) not in ((6, 12), (12, 24)):
            continue
        sq[(ns, nc, 'cond 1e4')] = line_eval(R, mlab, shk, nc, cond=1e4)
        sq[(ns, nc, 'cond 1e2')] = line_eval(R, mlab, shk, nc, cond=1e2)
        for k, md in labs.items():
            sq[(ns, nc, k)] = line_eval(R, md, shk, nc)
        sq[(ns, nc, 'random shakers')] = line_eval(R, mlab, rng.choice(cand, ns, replace=False), nc)
        sq[(ns, nc, 'lab on the host (service BC)')] = line_eval(R, R['md_s'][2], shk, nc, nart=u2.ndof)
    out['sq'] = sq
    # --- rigid-body share of the service reference at the controls (random study: drift defect check)
    md = R['md_s'][1]; c = ix(R['ctrls'][12]); rb = np.arange(6)
    Yr = -W ** 2 * np.einsum('cm,mrf->crf', md.Phi[R['U'][c]][:, rb], R['Qs'][1][rb], optimize=True)
    share = x1.auto(Yr).sum(0) / x1.auto(R['YS'][1][c]).sum(0)
    out['rigid_share'] = {f: float(np.mean(share[(F >= a) & (F < b)])) for f, (a, b) in (('20-40 Hz', (20, 40)), ('40-100 Hz', (40, 100)), ('100-2000 Hz', (100, 2000)))}
    out['rigid_stress'] = float(np.abs(u2.stress_matrix() @ md.Phi[:u2.ndof, :6]).max() / np.abs(u2.stress_matrix() @ md.Phi[:u2.ndof, 6:12]).max())
    print(f'rv seed={seed} {time.time() - t0:.0f}s', flush=True)
    return out


# ------------------------------------------------------------------ band-law objective variants
_BP = {}
def _bprob(nc, ns):
    import cvxpy as cp
    if (nc, ns) not in _BP:
        D = cp.Variable((ns, ns), hermitian=True); Bw = cp.Parameter((nc, ns * ns), complex=True); tw = cp.Parameter(nc)
        tp = cp.Parameter(nonneg=True)
        obj = cp.sum_squares(cp.real(Bw @ cp.vec(D, order='F')) - tw) + tp * cp.real(cp.trace(D))
        _BP[(nc, ns)] = (cp.Problem(cp.Minimize(obj), [D >> 0]), D, Bw, tw, tp)
    return _BP[(nc, ns)]


def band_law_v(Hc, Gt, N=3, trace=1e-4, mode='study', npass=3):
    """x1.band_law with the trace weight exposed and an optional log-residual (dB-symmetric) Gauss-Newton."""
    import cvxpy as cp
    nc, ns, _ = Hc.shape; prob, D, Bw, tw, tp = _bprob(nc, ns); tp.value = trace
    Fd = np.zeros((ns, ns, NF), complex); resid = []
    for row in x1.BANDS[N]:
        idx = np.nonzero(row)[0]; H = Hc[:, :, idx]
        Bc = np.einsum('cif,cjf->cij', H, np.conj(H)) / idx.size; t = Gt[:, idx].mean(1)
        alpha = t.mean() / np.real(np.einsum('cii->c', Bc)).mean(); Bv = alpha * Bc.reshape(nc, -1, order='F')
        w = 1 / t; tt = t.copy(); Dv = alpha * np.eye(ns, dtype=complex)
        for _ in range(npass):
            Bw.value = Bv * w[:, None]; tw.value = tt * w; val = None
            for solver in (cp.CLARABEL, cp.SCS):
                try:
                    prob.solve(solver=solver); val = D.value
                except Exception:
                    val = None
                if val is not None:
                    break
            if val is not None:
                Dv = alpha * val; Dv = (Dv + Dv.conj().T) / 2
            r = np.maximum(np.real(np.einsum('cij,ij->c', Bc, Dv)), 1e-6 * t)
            if mode == 'study':
                w = 1 / np.sqrt(t * r)
            else:  # minimize sum (ln r - ln t)^2 by Gauss-Newton: ln r ~ ln r0 + (r - r0)/r0
                w = 1 / r; tt = r * (1 - np.log(r / t))
        lam, V = np.linalg.eigh(Dv); lam = np.clip(lam, 0, None)
        Fd[:, :, idx] = (V * np.sqrt(lam))[:, :, None]; resid.append(x1.db10(r, t))
    return Fd, np.concatenate(resid)


def bv_job(seed, pert=1, seeds_check=(0,), only_log=False):
    t0 = time.time(); R = rsetup(seed, pert); ix = R['ix']; u2 = R['units'][2]
    mlab = fm.Modal(*fm.suspend(u2), R['zs'][2]); SB = u2.stress_matrix() @ mlab.Phi; out = {}
    vars_ = [('trace 0', dict(trace=0.0)), ('trace 1e-2', dict(trace=1e-2)), ('log-residual GN, 6 passes', dict(mode='log', npass=6))]
    if only_log:
        vars_ = [('study', dict()), ('log-residual GN, 6 passes', dict(mode='log', npass=6))]; seeds_check = ()
    if seed in seeds_check:
        vars_ = [('study (check)', dict())] + vars_
    for (ns, nc), shk in R['shakers'].items():
        Hu = np.einsum('um,mf,sm->usf', mlab.Phi[R['U']], mlab.h(F) * -W ** 2, mlab.Phi[shk], optimize=True)
        c = ix(R['ctrls'][nc]); Gt = x1.auto(R['YS'][1][c])
        for name, kw in vars_:
            Fd, res = band_law_v(Hu[c], Gt, **kw)
            Q = mlab.h(F)[:, None, :] * np.einsum('sm,srf->mrf', mlab.Phi[shk], Fd, optimize=True)
            r = x1.truth_summary(x1.element_truth(*x1.moments(SB, Q)), R['truthS'][2]); r.pop('x')
            r.update(ctrl_res_bias=res.mean(), ctrl_res_rms=np.sqrt(np.mean(res ** 2)), drive=10 * np.log10((np.abs(Fd) ** 2).sum() * DF))
            out[(ns, nc, name)] = r
    print(f'bv seed={seed} {time.time() - t0:.0f}s', flush=True)
    return dict(seed=seed, pert=pert, bv=out)


# ------------------------------------------------------------------ artifact variants: shock
def sv_job(seed, pert=1):
    t0 = time.time(); G = x2.Group(seed, pert); out = dict(seed=seed, pert=pert)
    u2 = fm.make_article(seed, np.random.default_rng([seed, 2, pert]), pert)
    O = base_ops(u2, None, mf=G.fix.md)   # the study's own fixed-interface modes (same K, M, damping)
    H = G.fix.H; ref = G.tr[2]
    def truth_from_Ab(Ab):   # Ab (12, NT) time histories
        A = np.fft.rfft(Ab * x2.EW, axis=1); Qs = -H * (O['Lb'] @ A); Ub = A / x2.S2
        s = np.fft.irfft(O['SPsi'] @ Ub + O['SPhi'] @ Qs, n=x2.NT, axis=1) * x2.EWI * 1e-6
        pk = np.abs(s).max(1); d = np.diff(s, axis=1); ext = d[:, :-1] * d[:, 1:] < 0
        dm = ((s[:, 1:-1] ** 2) ** 4 * ext).sum(1)
        return dict(pk=pk.reshape(-1, 8).max(1), dm=dm.reshape(-1, 8).max(1), tip=ref['tip'])
    def summ(tr):
        s = x2.truth_summary(tr, ref); return {k: s[k] for k in ('t_rms', 't_bias', 't_p5', 't_p95', 't_in3', 'd8_bias')}
    def seq(trs):
        return dict(pk=np.max([t['pk'] for t in trs], 0), dm=np.sum([t['dm'] for t in trs], 0), tip=ref['tip'])
    mix = G.ix(G.mnt); sa = {}
    for name, agg in (('ENV max over mounts (study)', lambda R: R.max(0, keepdims=True)), ('MEAN over mounts', lambda R: R.mean(0, keepdims=True))):
        res = []
        for real in range(2):
            rng = np.random.default_rng([seed, pert, 77, real]); trs = []
            for ax in range(3):
                sel = G.ix(G.mnt[ax::3]); Rt = agg(G.RA[sel]); B = x2.synth_basis(np.array([2 * np.median(G.TEA[sel])]), rng)
                ab = np.einsum('nk,nkt->nt', x2.synth_amp(B, Rt), B)[0]; Ab = np.zeros((12, x2.NT)); Ab[ax::3] = ab
                trs.append(truth_from_Ab(Ab))
            res.append(summ(seq(trs)))
        sa[name] = {k: np.mean([r[k] for r in res]) for k in res[0]}
    YB = G.Y[2][mix]; mean_ax = [YB[ax::3].mean(0) for ax in range(3)]
    trs = []
    for ax in range(3):
        Ab = np.zeros_like(YB); Ab[ax::3] = mean_ax[ax]; trs.append(truth_from_Ab(Ab))
    sa['TRUE B, per axis, sequential'] = summ(seq(trs))
    Ab = np.zeros_like(YB)
    for ax in range(3):
        Ab[ax::3] = mean_ax[ax]
    sa['TRUE B, 3 axes at once'] = summ(truth_from_Ab(Ab))
    sa['EXACT B, 12 mount dofs prescribed'] = summ(truth_from_Ab(YB))
    out['sa'] = sa
    # square WAVEFORM with the lab article mounted on the host (service boundary condition)
    host = fm.make_host(seed, fm.make_article(seed).meta['dims']); zb = fm.DampingField(seed)
    z2 = zb.unit(np.random.default_rng([seed, 2, pert, 9]), 0.3 * pert)
    K, M, na = fm.couple(u2, host); mdh = fm.Modal(K, M, z2); sysh = x2.Sys(mdh, u2.stress_matrix(), na, G.m, 6)
    sq = {}
    for (ns, nc) in ((6, 6), (12, 12), (6, 12), (12, 24)):
        shk = G.shakers[(ns, nc)]; c = G.ix(G.ctrls[nc])
        for name, sysx in (('free-free lab (study)', G.lab), ('lab on the host', sysh)):
            d = x2.drive_from(x2.pinv_cond(x2.hd(sysx.md, G.ctrls[nc], shk), 1e3), G.Y[1][c])
            y, tr = sysx.fire(shk, d, G.U); s = summ(tr); s['c_srs_rms'] = float(np.sqrt(np.mean(x2.dbr(x2.srs(y[c]), G.RA[c]) ** 2)))
            sq[(ns, nc, name)] = s
    out['sq'] = sq
    print(f'sv seed={seed} {time.time() - t0:.0f}s', flush=True)
    return out


# ------------------------------------------------------------------ 4. checks on saved results, and fresh analysis
def _rho(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[ok], y[ok])[0]


def x1_checks(recs, label):
    col = lambda k, rs: np.array([r[k] for r in rs], float)
    lab = [r for r in recs if r['test'] != 'S1']
    from collections import Counter
    print(f'\n[{label}] X1 records: {len(recs)}; lab {len(lab)}; by test {dict(Counter(r["test"] for r in recs))}')
    sa = [r for r in lab if r['test'].startswith('SA')]
    print(f'  single-axis records {len(sa)}, distinct truths {len({(r["seed"], r["pert"], r["test"]) for r in sa})} (each repeated for nc = 6, 12, 24)')
    for r in lab:
        x = r['x']; r['good6'] = bool(np.mean(np.abs(x) <= 6) >= .9 and x.max() <= 10)
    dedup = [r for r in lab if not (r['test'].startswith('SA') and r['nc'] != 12)]
    for gname, rs in (('pooled', lab), ('pooled, SA de-duplicated', dedup), ('LINE', [r for r in lab if r['test'] == 'LINE']),
                      ('multi-shaker', [r for r in lab if not r['test'].startswith('SA')])):
        out = []
        for m in ('line_rms', 'b3_rms', 'ers10_rms', 'sum3_rms', 'g527_rms'):
            v = [f'{_rho(col(t + m, rs), col(tk, rs)):+.2f}' for tk in ('t_rms', 't_p95') for t in ('c_', 'h_')]
            out.append(f'{m} rms c/h {v[0]}/{v[1]} p95 c/h {v[2]}/{v[3]}')
        print(f'  Spearman vs t_rms and vs t_p95 [{gname}, n={len(rs)}]: ' + ' | '.join(out))
        print(f'    AUC sanity: t_rms itself predicting not-good6 = {x1.auc(col("t_rms", rs), [r["good6"] for r in rs]):.2f}; '
              f'h_sum3 AUC good6 = {x1.auc(col("h_sum3_rms", rs), [r["good6"] for r in rs]):.2f}')
    L = [r for r in lab if r['test'] == 'LINE']
    w = [_rho(col('h_sum3_rms', g), col('t_rms', g)) for g in ([r for r in L if (r['ns'], r['nc']) == q] for q in sorted({(r['ns'], r['nc']) for r in L}))]
    w2 = [_rho(col('h_sum3_rms', g), col('t_rms', g)) for g in ([r for r in L if (r['ns'], r['nc'], r['pert']) == q] for q in sorted({(r['ns'], r['nc'], r['pert']) for r in L}))]
    print(f'  LINE within-rig Spearman h_sum3 vs t_rms: mean {np.nanmean(w):+.2f} (by rig {np.round(w, 2).tolist()}); within rig and pert {np.nanmean(w2):+.2f}')
    for p in (0, 1, 2):
        for q in ((6, 6), (12, 12)):
            rs = [r for r in L if r['pert'] == p and (r['ns'], r['nc']) == q]
            print(f'  LINE {q} pert {p}: bias {np.mean(col("t_bias", rs)):+.2f}, c_b3 {np.mean(col("c_b3_rms", rs)):.2f}', end=';')
    print()
    for p in (0, 1, 2):
        pr = {(r['seed'], r['ns'], r['nc']): r for r in L if r['pert'] == p}
        b = [(pr[(r['seed'], r['ns'], r['nc'])], r) for r in lab if r['test'] == 'BAND3' and r['pert'] == p and r['ns'] != r['nc']]
        print(f'  pert {p} non-square BAND3-LINE: d t_rms {np.mean([y["t_rms"] - x["t_rms"] for x, y in b]):+.2f}, LINE better in '
              f'{np.mean([x["t_rms"] < y["t_rms"] for x, y in b]):.2f}, d bias {np.mean([y["t_bias"] - x["t_bias"] for x, y in b]):+.2f}')
    for p in (1, 2):
        rs = [r for r in recs if r['test'] == 'S1' and r['pert'] == p]
        print(f'  S1 floor pert {p}: stress RMS {np.mean(col("t_rms", rs)):.2f}, ctrl line RMS {np.mean(col("c_line_rms", rs)):.2f}, worst ctrl {np.mean(col("c_line_worst", rs)):.2f}')
    for t in ('SA0', 'SA3'):
        rs = [r for r in lab if r['test'] == t and r['pert'] == 1 and r['nc'] == 12]
        print(f'  {t} pert 1: bias {np.mean(col("t_bias", rs)):+.2f}, p5 {np.mean(col("t_p5", rs)):+.2f}; seed 0..: {np.round(col("t_bias", rs), 2).tolist()}')
    for t, m in (('pooled', lab), ('LINE', L)):
        print(f'  [{t}] ctrl metrics vs t_rms: ' + ', '.join(f'{k} {_rho(col("c_" + k, m), col("t_rms", m)):+.2f}' for k in ('line_rms', 'b3_rms', 'sum3_rms')) +
              ' | held: ' + ', '.join(f'{k} {_rho(col("h_" + k, m), col("t_rms", m)):+.2f}' for k in ('line_rms', 'b3_rms', 'sum3_rms', 'ers10_rms')))


def x2_checks(recs, label, nsyn=None):
    col = lambda k, rs: np.array([r.get(k, np.nan) for r in rs], float)
    from collections import Counter
    full = [r for r in recs if 't_rms' in r and r['test'] != 'S1']
    if nsyn is not None:
        full = [r for r in full if not (r['test'].startswith('SYNTH') and r.get('real', 0) >= nsyn)]
    print(f'\n[{label}] X2 records {len(recs)}; with truth (no S1) {len(full)}; by test {dict(Counter(r["test"] for r in recs))}')
    for r in full:
        r['good6'] = bool(np.mean(np.abs(r['x']) <= 6) >= .9 and r['x'].max() <= 10)
    good = np.array([r['good6'] for r in full])
    sets = (('pooled', full), ('pooled, no square rigs', [r for r in full if r['ns'] != r['nc']]),
            ('pooled, SA de-duplicated', [r for r in full if not (r['test'].startswith('SA') and r['nc'] != 12)]))
    for name, rs in sets:
        print(f'  [{name}, n={len(rs)}] rho(c_srs12_rms, t_rms) {_rho(col("c_srs12_rms", rs), col("t_rms", rs)):+.2f}, '
              f'rho(h_srs12_rms, t_rms) {_rho(col("h_srs12_rms", rs), col("t_rms", rs)):+.2f}, '
              f'signed c bias {_rho(col("c_srs12_bias", rs), col("t_bias", rs)):+.2f}, h bias {_rho(col("h_srs12_bias", rs), col("t_bias", rs)):+.2f}')
    ps = np.array([bool(r.get('crit_e')) for r in full])
    print(f'  rule (e): pass {ps.mean():.2f}, precision {good[ps].mean():.2f}, base rate {good.mean():.2f}, recall {ps[good].mean():.2f}')
    for t in ('SA0', 'SA3'):
        rs = [r for r in full if r['test'] == t and r['pert'] == 1 and r['nc'] == 12]
        print(f'  {t} pert 1 bias {np.mean(col("t_bias", rs)):+.2f}; per seed (mean of 2 realizations): ' +
              str({s: round(float(np.mean([r['t_bias'] for r in rs if r['seed'] == s])), 2) for s in sorted({r['seed'] for r in rs})}))
    for q in ((6, 6), (12, 12)):
        rs = [r for r in full if r['test'] == 'WAVEFORM' and (r['ns'], r['nc']) == q]
        print(f'  WAVEFORM {q} (both perts) bias {np.mean(col("t_bias", rs)):+.2f}, ctrl SRS RMS {np.mean(col("c_srs12_rms", rs)):.2f}')


def x1b_s9_s10(recs, x1recs):
    col = lambda k, rs: np.array([r[k] for r in rs], float)
    sel = lambda **kw: [r for r in recs if all(r.get(k) == v for k, v in kw.items())]
    for sq in (False, True):
        o = []
        for law in ('LINE', 'CAP30', 'CVc', 'BAND3'):
            rs = [r for r in sel(square=sq) if r['test'] == f'S9_{law}']
            o.append(f'{law} worst ctrl 1/3-oct over {col("c_over_b3", rs).max():.1f} (mean +1%/+3% {np.mean(col("c_over_b3", [r for r in rs if r["model"] != "zeta/2"])):.1f})')
        print(f'  S9 square={sq}: ' + '; '.join(o))
    for t in ('CAP1000', 'BAND3', 'CVc'):
        rs = sel(test=t, pert=1, square=False); v = []
        for r in rs:
            x = r['x'].astype(float); E = r['h_E3'].ravel(); g_t = -np.percentile(x, 5); g_p = -3 - np.percentile(E, 10)
            v.append((g_t, g_p, np.mean(x + g_p), np.mean(x + g_p < -3)))
        v = np.array(v); print(f'  S10 {t} non-square: truth cover-95 gain {v[:, 0].mean():+.2f}, rule P gain {v[:, 1].mean():+.2f} '
                                f'(|gap| mean {np.abs(v[:, 1] - v[:, 0]).mean():.2f}, case p90 {np.percentile(np.abs(v[:, 1] - v[:, 0]), 90):.2f}), '
                                f'mean over-test {v[:, 2].mean():+.2f}, under>3 dB {v[:, 3].mean():.3f}')
    sa = [r for r in x1recs if r['test'] == 'SA0' and r['pert'] == 1 and r['nc'] == 12]
    print(f'  single-axis as run mean over-test {np.mean([r["x"].mean() for r in sa]):+.2f}')


def saved_checks():
    recs, _ = pickle.load(open(os.path.join(ROOT, 'x1', 'x1_results.pkl'), 'rb')); x1_checks(recs, 'saved seeds 0-9')
    rb = pickle.load(open(os.path.join(ROOT, 'x1b', 'x1b_results.pkl'), 'rb')); print('\n[saved] X1b S9/S10'); x1b_s9_s10(rb, recs)
    r2, _ = pickle.load(open(os.path.join(ROOT, 'x2', 'x2_results.pkl'), 'rb')); x2_checks(r2, 'saved seeds 0-5')
    x2_checks(pickle.load(open(os.path.join(ROOT, 'x2', 'x2_results.pkl'), 'rb'))[0], 'saved seeds 0-5, SYNTH cut to 5 realizations', nsyn=5)


def fresh_analyze():
    d = os.path.join(SCR, 'fresh')
    recs, _ = pickle.load(open(os.path.join(d, 'x1_results.pkl'), 'rb')); x1_checks(recs, 'FRESH seeds 10-19')
    rb = pickle.load(open(os.path.join(d, 'x1b_results.pkl'), 'rb')); print('\n[FRESH] X1b S9/S10 (seeds 10-19, p=1)'); x1b_s9_s10(rb, recs)
    r2, _ = pickle.load(open(os.path.join(d, 'x2_results.pkl'), 'rb')); x2_checks(r2, 'FRESH seeds 6-11 (5 SYNTH realizations)')
    # full study tables on the fresh results
    _redirect(x1, 'fresh'); x1.analyze()
    _redirect(x2, 'fresh'); x2.analyze([])


def vsum(fname='variants_rv_bv_sv.pkl'):
    V = pickle.load(open(os.path.join(SCR, fname), 'rb'))
    keys = ('t_rms', 't_bias', 't_p5', 't_p95', 't_in3')
    if 'rv' in V:
        rv = V['rv']; print(f'\n== random variants, p=1, seeds {sorted(o["seed"] for o in rv)} ==')
        for k in rv[0]['sa']:
            print(f'  SA {k:45s} ' + ' '.join(f'{q} {np.mean([o["sa"][k][q] for o in rv]):+6.2f}' for q in keys))
        for k in sorted({k for o in rv for k in o['sq']}, key=str):
            rs = [o['sq'][k] for o in rv if k in o['sq']]
            print(f'  LINE {str(k):45s} ' + ' '.join(f'{q} {np.mean([r[q] for r in rs]):+6.2f}' for q in ('t_rms', 't_bias', 'c_b3', 'drive')))
        base = [o['sq'][k] for o in rv for k in o['sq'] if k[2] == 'base' and k[1] < 24]
        print(f'  LINE base, nc 6 and 12 (n={len(base)}): rho(held sum3, t_rms) {_rho([r["h_sum3"] for r in base], [r["t_rms"] for r in base]):+.2f}, '
              f'rho(EfI-chosen uncontrolled sum3, t_rms) {_rho([r["e_sum3"] for r in base], [r["t_rms"] for r in base]):+.2f}, '
              f'rho(ctrl b3, t_rms) {_rho([r["c_b3"] for r in base], [r["t_rms"] for r in base]):+.2f}')
        nsq = [r for r in base if True]
        print('  rigid-body share of control acceleration power: ' + ', '.join(f'{b} {np.mean([o["rigid_share"][b] for o in rv]):.2f}' for b in rv[0]['rigid_share'])
              + f'; max stress from rigid modes / elastic {max(o["rigid_stress"] for o in rv):.1e}')
    if 'bv' in V:
        bv = V['bv']; recs, _ = pickle.load(open(os.path.join(ROOT, 'x1', 'x1_results.pkl'), 'rb'))
        B3 = {(r['seed'], r['ns'], r['nc']): r for r in recs if r['test'] == 'BAND3' and r['pert'] == 1}
        print(f'\n== band-law variants, p=1, seeds {sorted(o["seed"] for o in bv)} (study BAND3 from saved results) ==')
        names = sorted({k[2] for o in bv for k in o['bv']})
        for sq in (False, True):
            st = [B3[(o['seed'], k[0], k[1])] for o in bv for k in o['bv'] if k[2] == names[-1] and (k[0] == k[1]) == sq]
            print(f'  square={sq} study BAND3: t_rms {np.mean([r["t_rms"] for r in st]):.2f} bias {np.mean([r["t_bias"] for r in st]):+.2f}')
            for nm in names:
                rs = [(o['seed'], k, v) for o in bv for k, v in o['bv'].items() if k[2] == nm and (k[0] == k[1]) == sq]
                d = [v['t_rms'] - B3[(s, k[0], k[1])]['t_rms'] for s, k, v in rs]; db = [v['t_bias'] - B3[(s, k[0], k[1])]['t_bias'] for s, k, v in rs]
                print(f'    {nm:28s} n={len(rs):3d} t_rms {np.mean([v["t_rms"] for _, _, v in rs]):5.2f} bias {np.mean([v["t_bias"] for _, _, v in rs]):+6.2f} '
                      f'(paired d t_rms {np.mean(d):+.2f}, d bias {np.mean(db):+.2f}) ctrl band residual bias {np.mean([v["ctrl_res_bias"] for _, _, v in rs]):+.2f} '
                      f'rms {np.mean([v["ctrl_res_rms"] for _, _, v in rs]):.2f} drive {np.mean([v["drive"] for _, _, v in rs]):.1f}')
    if 'sv' in V:
        sv = V['sv']; print(f'\n== shock variants, p=1, seeds {sorted(o["seed"] for o in sv)} ==')
        for k in sv[0]['sa']:
            print(f'  SA {k:40s} ' + ' '.join(f'{q} {np.mean([o["sa"][k][q] for o in sv]):+6.2f}' for q in keys + ('d8_bias',)))
        for k in sv[0]['sq']:
            print(f'  WAVEFORM {str(k):40s} ' + ' '.join(f'{q} {np.mean([o["sq"][k][q] for o in sv]):+6.2f}' for q in ('t_rms', 't_bias', 'c_srs_rms')))


def _vjob(j):
    k, s = j
    return k, {'rv': rv_job, 'bv': bv_job, 'sv': sv_job, 'bvlog': lambda s: bv_job(s, only_log=True)}[k](s)


def variants(workers='6', which='rv,bv,sv'):
    which = which.split(','); jobs = []
    jobs += [('bv', s) for s in range(10)] if 'bv' in which else []
    jobs += [('sv', s) for s in range(6)] if 'sv' in which else []
    jobs += [('rv', s) for s in range(10)] if 'rv' in which else []
    jobs += [('bvlog', s) for s in range(10, 16)] if 'bvlog' in which else []
    out = {}
    with Pool(int(workers)) as pool:
        for k, o in pool.imap_unordered(_vjob, jobs, chunksize=1):
            out.setdefault(k, []).append(o)
            pickle.dump(out, open(os.path.join(SCR, f'variants_{"_".join(which)}.pkl'), 'wb'))
    print('variants done')


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'repro':
        repro()
    elif cmd == 'fresh':
        fresh(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
    else:
        globals()[cmd](*sys.argv[2:])
