"""an earlier study: a banded specification run as converted narrowband control or as direct band control,
open loop on a level ladder, on a plant whose modes soften and damp with level.

    PY=python
    $PY x3_banded_spec_study.py verify          # Wishart and H_id estimator checks, fast band fit vs a1.band_law_v
    $PY x3_banded_spec_study.py run 16          # seeds 0..19, all rigs, tiers, conditions (16 workers)
    $PY x3_banded_spec_study.py analyze         # tables to scratch x3/x3_tables.txt, figures figs/x3_*.png

Imports frame_model, x1_random_study, x1b_followup_study and a1_audit unchanged.

Level law (per mode m of a modal model, linear frequency f_m, damping z_m, level fraction s in [0, 1]):
    f_m(s) = f_m (1 - s e_m d k1_m),   z_m(s) = z_m (1 + s e_m (g - 1) k2_m)
    tier (d, g): linear (0, 1), realistic (0.01, 1.5), strong (0.03, 3)
    form A (main): k1, k2 lognormal, median 1, sigma 0.5, correlation 0.7, drawn per seed and unit on a
        1/12-octave frequency field (a mode takes the value of its bin, so lab and service modes of the
        same unit near the same frequency move alike); s = (a - a_id)/(1 - a_id), a = 10^(L/20), L_id = -18 dB
    form B (sensitivity): k1, k2 independent lognormal sigma 0.8 drawn per mode and per configuration
        (lab and service draws unrelated); s = (L - L_id)/(0 - L_id), linear in dB
    e_m = share of the mode's kinetic energy in the article (1 for lab elastic modes, 0 for the 6
        suspension or rigid modes); shapes unchanged. Unit A (spec) and unit B (article) both follow the law.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'RAYON_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, copy, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import frame_model as fm
import x1_random_study as x1
import x1b_followup_study as x1b
import a1_audit as a1
from x1_random_study import F, DF, W, NF, db10, auto

ROOT = x1.SCR.rsplit('/', 1)[0]; SCR = os.path.join(ROOT, 'x3'); os.makedirs(SCR, exist_ok=True)
SEEDS = range(20); PERT = 1
RIGS = {12: ((12, 24), (12, 12)), 6: ((6, 24), (6, 12), (6, 6)), 3: ((3, 12),)}
LEVELS = (-12, -9, -6, -3, 0); RUNLV = LEVELS + (0, 0, 0); LID = -18.0; NAVG = 60; NOISE = 1e-3
TIERS = {'linear': (0.0, 1.0, 'A'), 'realistic': (0.01, 1.5, 'A'), 'strong': (0.03, 3.0, 'A'),
         'realistic-B': (0.01, 1.5, 'B'), 'strong-B': (0.03, 3.0, 'B')}
CASES = [(3, 'exact', t) for t in TIERS] + [(3, 'noise', t) for t in ('linear', 'realistic', 'strong')] + \
        [(6, 'exact', t) for t in ('linear', 'realistic', 'strong')]
LAWS = ('NBF-a1', 'NBF-a05', 'NBF-r', 'NB-U', 'NB-R', 'BAND-G', 'BAND-U', 'LS-F-a1', 'LS-F-a05', 'LS-U')
IDEALS = ('I-NB', 'I-BAND', 'I-LS')
H_ = lambda A: np.conj(A).transpose(0, 2, 1)


def band_of(N):
    return (x1.BANDS[N] > 0).argmax(0)


# ------------------------------------------------------------------ level law
def s_of(L, form):
    if form == 'A':
        a, a0 = 10 ** (L / 20), 10 ** (LID / 20); return (a - a0) / (1 - a0)
    return (L - LID) / (0 - LID)


def mode_factors(md, e, seed, u, form, cfg):
    """(k1, k2) per mode for unit u; e = article energy share per mode."""
    if form == 'A':
        rng = np.random.default_rng([seed, u, 3303]); nb = int(np.log2(1e4 / 0.5) * 12) + 1
        z1 = rng.standard_normal(nb); z2 = 0.7 * z1 + np.sqrt(1 - 0.49) * rng.standard_normal(nb)
        idx = np.clip((np.log2(np.maximum(md.fn, 0.5) / 0.5) * 12).astype(int), 0, nb - 1)
        return e * np.exp(0.5 * z1[idx]), e * np.exp(0.5 * z2[idx])
    rng = np.random.default_rng([seed, u, {'lab': 1, 'serv': 2}[cfg], 4404]); n = md.fn.size
    return e * np.exp(0.8 * rng.standard_normal(n)), e * np.exp(0.8 * rng.standard_normal(n))


def at_level(md, kk, tier, L):
    d, g, form = TIERS[tier]; s = s_of(L, form); m = copy.copy(md)
    m.wn = md.wn * (1 - s * d * kk[0]); m.fn = m.wn / (2 * np.pi); m.zeta = md.zeta * (1 + s * (g - 1) * kk[1]); return m


# ------------------------------------------------------------------ linear algebra helpers (layout (nf, a, b))
def sand(Z, G):
    return Z @ G @ H_(Z)


def factor(D):
    D = (D + H_(D)) / 2; lam, V = np.linalg.eigh(D); return V * np.sqrt(np.clip(lam, 0, None))[:, None, :]


def psd(D):
    Fd = factor(D); return Fd @ H_(Fd)


def cn(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def est_H(H, rng):
    """H1 estimate from NAVG blocks of uncorrelated unit drives, channel noise NOISE x response autospectrum."""
    nf, nc, ns = H.shape; X = cn(rng, (nf, ns, NAVG)); s2 = NOISE * (np.abs(H) ** 2).sum(2)
    Nn = cn(rng, (nf, nc, NAVG)) * np.sqrt(s2)[:, :, None]
    return H + (Nn @ H_(X)) @ np.linalg.inv(X @ H_(X))


def wishart(rng, m):
    Z = cn(rng, (NF, m, NAVG)); return ((Z @ H_(Z)) / NAVG).astype(np.complex64)


def frf(md, rows, shk):
    return np.einsum('um,mf,sm->fus', md.Phi[rows], md.h(F) * -W ** 2, md.Phi[shk], optimize=True)


# ------------------------------------------------------------------ fast log-residual band fit (warm-started polish)
def band_fit(H, t, D0, N=3, lam=1e-4, maxiter=400):
    """Minimize sum_(c,b) (ln r_cb - ln t_cb)^2 + lam tr(D_b/alpha_b) over one flat PSD matrix per 1/N-octave band,
    D_b = L L^H, by L-BFGS from D0 (the objective of a1.band_law_v mode='log'). H (nf, nc, ns), t (nc, nb)."""
    Bm = x1.BANDS[N]; nb = Bm.shape[0]; nf, nc, ns = H.shape; bo = band_of(N)
    B = np.zeros((nb, nc, ns, ns), complex); Ls = np.zeros((nb, ns, ns), complex); al = np.zeros(nb)
    for b in range(nb):
        idx = bo == b; Hb = H[idx]
        B[b] = np.einsum('fci,fcj->cij', Hb, np.conj(Hb)) / idx.sum()
        al[b] = t[:, b].mean() / np.real(np.einsum('cii->c', B[b])).mean(); B[b] *= al[b]
        Db = D0[idx].mean(0) / al[b]; lam_, V = np.linalg.eigh((Db + Db.conj().T) / 2)
        Ls[b] = V * np.sqrt(np.clip(lam_, 0, None))
    BT = B.transpose(0, 1, 3, 2); lt = np.log(t.T)
    def fg(v):
        L = (v[:v.size // 2] + 1j * v[v.size // 2:]).reshape(nb, ns, ns); D = L @ H_(L)
        r = np.maximum(np.real(np.einsum('bcij,bij->bc', B, D)), 1e-12); e = np.log(r) - lt
        G = np.einsum('bc,bcij->bij', 2 * e / r, BT) + lam * np.eye(ns); g = 2 * (G @ L)
        return (e ** 2).sum() + lam * np.real(np.einsum('bii->', D)), np.r_[g.real.ravel(), g.imag.ravel()]
    v0 = np.r_[Ls.real.ravel(), Ls.imag.ravel()]
    v = minimize(fg, v0, jac=True, method='L-BFGS-B', options=dict(maxiter=maxiter, gtol=1e-9, ftol=1e-13)).x
    L = (v[:v.size // 2] + 1j * v[v.size // 2:]).reshape(nb, ns, ns); Db = (L @ H_(L)) * al[:, None, None]
    return Db[bo]


def db_from_v(Fd):  # a1.band_law_v factor (ns, ns, nf) -> D (nf, ns, ns)
    Ft = Fd.transpose(2, 0, 1); return Ft @ H_(Ft)


# ------------------------------------------------------------------ set-up per seed
def setup(seed):
    S = x1b.setup(seed, PERT); art_nom = fm.make_article(seed); host = fm.make_host(seed, art_nom.meta['dims'])
    serv = {}
    for u in (1, 2):
        un = S['units'][u]; K, M, na = fm.couple(un, host); md = fm.Modal(K, M, S['zs'][u])
        dofs, Fin = x1.service_forces(seed, un, host, na); Ma = un.assemble()[1]
        e = np.einsum('im,ij,jm->m', md.Phi[:na], Ma, md.Phi[:na]) * (md.fn > 1.0)
        serv[u] = dict(md=md, dofs=dofs, Fin=Fin, na=na, S=un.stress_matrix(), e=e)
    mlab = S['mlab']; S['elab'] = (np.arange(mlab.fn.size) >= 6).astype(float); S['serv2'] = serv
    return S


def service(S, seed, u, tier):
    d, g, form = TIERS[tier]; sv = S['serv2'][u]; md = sv['md']
    md1 = at_level(md, mode_factors(md, sv['e'], seed, u, form, 'serv'), tier, 0.0)
    Q = x1.modal_q(md1, sv['dofs'], sv['Fin'])
    return x1.acc(md1, S['U'], Q), x1.element_truth(*x1.moments(sv['S'] @ md1.Phi[:sv['na']], Q))


def stress_m0(md, shk, SB, Fd):
    P = md.Phi[shk].T; A = (P[None] @ Fd) * md.h(F).T[:, :, None]; nm = P.shape[0]
    A2 = A.transpose(1, 0, 2).reshape(nm, -1); C = (A2.real @ A2.real.T + A2.imag @ A2.imag.T) * DF
    return np.einsum('rm,mn,rn->r', SB, C, SB, optimize=True)


# ------------------------------------------------------------------ one case
def run_case(S, seed, shk, nc, N, cond, tier, Db_lin, ex=False):
    ix = S['ix']; c = ix(S['ctrls'][nc]); held = ix(S['held']); m = S['mlab'].Phi
    tips = ix(np.array([6 * n + d for n in S['units'][2].meta['tips'] for d in range(3)]))
    mlab, SB = S['mlab'], S['SB']; d_, g_, form = TIERS[tier]; Bm = x1.BANDS[N]; bo = band_of(N)
    kk = mode_factors(mlab, S['elab'], seed, 2, form, 'lab')
    YA, _ = service(S, seed, 1, tier); YB, trB = service(S, seed, 2, tier)
    GAl = auto(YA[c]); t = GAl @ Bm.T; YAc = YA[c].transpose(2, 0, 1); GA = YAc @ H_(YAc)
    GBU = auto(YB); rmsB = trB['rms']
    mds = {L: at_level(mlab, kk, tier, L) for L in (LID,) + LEVELS}
    Ht = {L: frf(mds[L], S['U'][c], shk) for L in mds}; ns = shk.size
    rng = np.random.default_rng([seed, ns, nc, N, list(TIERS).index(tier), 17])
    if cond == 'exact':
        Hid = Ht[LID]; Hp = [Ht[L] for L in (LID,) + LEVELS[:-1]] + [Ht[0]] * 3
    else:
        Hid = est_H(Ht[LID], rng); est = {L: est_H(Ht[L], rng) for L in LEVELS}
        Hp = [Hid] + [est[L] for L in LEVELS[:-1]] + [est[0]] * 3
    Db = fit_cached = band_fit(Hid, t, Db_lin, N)
    Gref = sand(Hid, Db); Zid = x1.pinv_trunc(Hid); Zp = [x1.pinv_trunc(h) for h in Hp]
    Wk = [wishart(rng, ns + nc) for _ in RUNLV] if cond == 'noise' else None
    Dbu, cache = [], {}
    for h in Hp:
        if id(h) not in cache:
            cache[id(h)] = Db if h is Hid else band_fit(h, t, Db, N)
        Dbu.append(cache[id(h)])
    dRef, dA = np.real(np.einsum('fii->fi', Gref)), GAl.T
    HU = frf(mds[0], S['U'], shk)

    def metrics(Dn, k, L, H, md, final):
        a = 10 ** (L / 10); Fd = factor(Dn); Y = H @ Fd; G = (np.abs(Y) ** 2).sum(2)
        Eb = db10(G.T @ Bm.T, t); Er, Ea = db10(G, dRef), db10(G, dA)
        x = 20 * np.log10(np.sqrt(a * stress_m0(md, shk, SB, Fd).reshape(-1, 8)).max(1) / rmsB)
        Dss = np.real(np.einsum('fii->fi', Dn)) * a
        r = dict(k=k, L=L, Eb=Eb.astype(np.float32), x=x.astype(np.float32),
                 in3_ref=np.mean(np.abs(Er) <= 3), in6_ref=np.mean(np.abs(Er) <= 6), over_ref=Er.max(),
                 in3_A=np.mean(np.abs(Ea) <= 3), in6_A=np.mean(np.abs(Ea) <= 6), over_A=Ea.max(),
                 drv=(Dss.sum(0) * DF).astype(np.float32))
        if final:
            Gu = (np.abs(HU @ Fd) ** 2).sum(2)
            for Q in (10, 50):
                Ka = x1.KER[Q][0]
                for tag, sel in (('tip', tips), ('held', held)):
                    r[f'ers{Q}_{tag}'] = db10(Gu[:, sel].T @ Ka.T, GBU[sel] @ Ka.T).astype(np.float32)
            fl = np.array([[Dss[bo == b, s].max() / max(Dss[bo == b, s].mean(), 1e-300) for b in range(Bm.shape[0])] for s in range(ns)])
            r['flat'] = (10 * np.log10(fl)).astype(np.float32)
            if ex:
                r['ex'] = dict(G=G.astype(np.float32), Dss=Dss.astype(np.float32))
        return r, Y, G

    def measure(Y, G, k, full):  # Y, G are at unit level (drive normalized by the run level), so is the estimate
        if Wk is None:
            return Y @ H_(Y) if full else G
        A = np.concatenate([Y, np.sqrt(NOISE * G)[:, :, None] * np.eye(nc)[None]], 2).astype(np.complex64)
        Gm = A @ Wk[k] @ H_(A)
        return Gm.astype(complex) if full else np.real(np.einsum('fii->fi', Gm)).astype(float)

    out = {}
    for law in LAWS:
        tgt = GA if law.startswith('LS') else Gref
        Dn = sand(Zid, GA) if law.startswith('LS') else Db.copy(); Gmod = Gref.copy(); runs = []
        for k, L in enumerate(RUNLV):
            if law in ('NB-U', 'LS-U'):
                Dn = sand(Zp[k], tgt)
            elif law == 'NB-R':
                Dn = sand(Zp[k], sand(Hp[k], Db))
            elif law == 'BAND-U':
                Dn = Dbu[k]
            r, Y, G = metrics(Dn, k, L, Ht[L], mds[L], k == len(RUNLV) - 1); runs.append(r)
            if '-a' in law:
                g = 0.5 if law.endswith('a05') else 1.0
                Dn = psd(Dn + g * sand(Zid, tgt - measure(Y, G, k, True)))
            elif law == 'NBF-r':
                gm = measure(Y, G, k, False); cc = np.sqrt(dRef / np.maximum(gm, 1e-300))
                Gmod = cc[:, :, None] * Gmod * cc[:, None, :]; Dn = sand(Zid, Gmod)
            elif law == 'BAND-G':
                mb = measure(Y, G, k, False).T @ Bm.T
                gb = np.exp(np.mean(np.log(t / np.maximum(mb, 1e-300)), 0)); Dn = Dn * gb[bo][:, None, None]
        out[law] = runs
    if cond == 'exact':
        H0 = Ht[0]; Z0 = x1.pinv_trunc(H0)
        for law, Dn in (('I-NB', sand(Z0, Gref)), ('I-BAND', band_fit(H0, t, Db, N)), ('I-LS', sand(Z0, GA))):
            out[law] = [metrics(Dn, 0, 0, H0, mds[0], True)[0]]
    rec = dict(seed=seed, ns=ns, nc=nc, square=ns == nc, N=N, cond=cond, tier=tier, laws=out,
               polish=None)
    if ex:
        rec['ex'] = dict(dRef=dRef.astype(np.float32), t=t, GAl=GAl.astype(np.float32), fn_id=mds[LID].fn, fn_full=mds[0].fn,
                         c=c)
    return rec


def job(args):
    seed, ns_grp = args; t0 = time.time(); S = setup(seed); recs = []
    for (ns, nc) in RIGS[ns_grp]:
        shk = S['shakers'][(ns, nc)]; c = S['ix'](S['ctrls'][nc])
        Hlin = frf(S['mlab'], S['U'][c], shk)
        Fd3, _ = a1.band_law_v(Hlin.transpose(1, 2, 0), auto(S['YS'][1][c]), N=3, mode='log', npass=6)
        Dv = db_from_v(Fd3); lin = {}
        for N, cond, tier in CASES:
            ex = seed == 0 and (ns, nc) == (6, 12) and tier == 'realistic' and cond == 'exact' and N == 3
            r = run_case(S, seed, shk, nc, N, cond, tier, Dv, ex)
            if N == 3 and cond == 'exact' and tier == 'linear':
                r['bandlog3_v'] = Dv  # verification: a1.band_law_v drive itself, evaluated below
                Fd = factor(Dv); x = 20 * np.log10(np.sqrt(stress_m0(S['mlab'], shk, S['SB'], Fd).reshape(-1, 8)).max(1) / S['truthS'][2]['rms'])
                r['v_trms'] = float(np.sqrt(np.mean(x ** 2))); r.pop('bandlog3_v')
            recs.append(r)
        print(f'seed {seed} rig {ns}x{nc} {time.time() - t0:.0f}s', flush=True)
    return recs


def run(workers='16'):
    jobs = [(s, g) for g in (12, 6, 3) for s in SEEDS]
    t0 = time.time(); recs = []
    with Pool(int(workers)) as pool:
        for o in pool.imap_unordered(job, jobs, chunksize=1):
            recs += o
    pickle.dump(recs, open(os.path.join(SCR, 'x3_results.pkl'), 'wb'))
    print(f'simulation {time.time() - t0:.0f}s, {len(recs)} case records')


# ------------------------------------------------------------------ verification
def verify():
    rng = np.random.default_rng(5); lines = []
    P = lambda s: (print(s), lines.append(s))
    G = np.array([[2.0, 0.6 + 0.3j], [0.6 - 0.3j, 1.0]]); Lc = np.linalg.cholesky(G)
    Wd = wishart(rng, 2)  # NF independent draws of a 2x2 Wishart with identity scale
    Gh = Lc[None] @ Wd @ H_(Lc[None])
    P(f'Wishart: mean G11 {Gh[:, 0, 0].real.mean():.4f} (2.0), rel SD G11 {Gh[:, 0, 0].real.std() / 2:.4f} '
      f'(1/sqrt({NAVG}) = {1 / np.sqrt(NAVG):.4f}), mean G12 {Gh[:, 0, 1].mean():.4f} (0.6+0.3j)')
    H = cn(rng, (NF, 12, 6)); He = est_H(H, rng)
    rel = np.abs(He - H) ** 2 / (np.abs(H) ** 2).sum(2, keepdims=True)
    P(f'H_id estimate: mean |dH|^2 / sum_s |H|^2 per element {rel.mean():.2e} (expected NOISE/(NAVG-ns) = {NOISE / (NAVG - 6):.2e})')
    S = setup(0); shk = S['shakers'][(6, 12)]; c = S['ix'](S['ctrls'][12]); Hl = frf(S['mlab'], S['U'][c], shk)
    for N in (3, 6):
        t0 = time.time(); Fd, res = a1.band_law_v(Hl.transpose(1, 2, 0), auto(S['YS'][1][c]), N=N, mode='log', npass=6); tv = time.time() - t0
        Dv = db_from_v(Fd); t = auto(S['YS'][1][c]) @ x1.BANDS[N].T
        t0 = time.time(); Df = band_fit(Hl, t, Dv, N); tf = time.time() - t0
        t0 = time.time(); Dc = band_fit(Hl, t, np.broadcast_to(np.eye(6) * np.real(np.einsum('fii->f', Dv)).mean() / 6, Dv.shape).copy(), N); tc = time.time() - t0
        for name, D in (('band_law_v', Dv), ('fast, warm', Df), ('fast, cold', Dc)):
            Gb = np.real(np.einsum('fii->fi', sand(Hl, D))).T @ x1.BANDS[N].T; e = db10(Gb, t)
            x = 20 * np.log10(np.sqrt(stress_m0(S['mlab'], shk, S['SB'], factor(D)).reshape(-1, 8)).max(1) / S['truthS'][2]['rms'])
            P(f'1/{N} oct seed 0 6x12 {name:11s}: band resid mean {e.mean():+.3f} rms {np.sqrt(np.mean(e ** 2)):.3f} dB, stress RMS {np.sqrt(np.mean(x ** 2)):.3f} dB, '
              f'drive {10 * np.log10(np.real(np.einsum("fii->", D)) * DF):.2f} dB')
        P(f'   times: band_law_v {tv:.1f}s, fast warm {tf:.1f}s, fast cold {tc:.1f}s')
    open(os.path.join(SCR, 'x3_verify.txt'), 'w').write('\n'.join(lines))


# ------------------------------------------------------------------ analysis
def boot(d, seeds, n=4000, rng=None):
    rng = rng or np.random.default_rng(1); us = np.unique(seeds)
    m = np.array([d[seeds == s].mean() for s in us]); k = np.array([np.sum(seeds == s) for s in us])
    idx = rng.integers(0, us.size, (n, us.size)); bs = (m[idx] * k[idx]).sum(1) / k[idx].sum(1)
    return np.percentile(bs, [2.5, 97.5])


def tsum(x):
    return dict(t_rms=np.sqrt(np.mean(x ** 2)), t_bias=x.mean(), t_p5=np.percentile(x, 5), t_p95=np.percentile(x, 95),
                t_in3=np.mean(np.abs(x) <= 3), t_max=x.max())


def final_row(r):
    """Scalar summary of a law's final run (or ideal)."""
    f = r[-1]; o = tsum(f['x'].astype(float)); Eb = f['Eb'].astype(float)
    o.update(b_rms=np.sqrt(np.mean(Eb ** 2)), b_over=Eb.max(), b_abs=np.abs(Eb).max(), b_bias=Eb.mean(),
             in3_ref=f['in3_ref'], in6_ref=f['in6_ref'], over_ref=f['over_ref'], in3_A=f['in3_A'], in6_A=f['in6_A'],
             over_A=f['over_A'], drv_w=10 * np.log10(f['drv'].max()), flat_max=f['flat'].max(), flat_med=np.median(f['flat']))
    for Q in (10, 50):
        for tag in ('tip', 'held'):
            E = f[f'ers{Q}_{tag}'].astype(float); o[f'e{Q}{tag}_rms'] = np.sqrt(np.mean(E ** 2)); o[f'e{Q}{tag}_over'] = E.max()
    if len(r) > 1:
        fin = Eb; o['conv'] = next(i + 1 for i, q in enumerate(r) if np.sqrt(np.mean((q['Eb'] - fin) ** 2)) <= 1.0)
        o['lad_bover'] = max(q['Eb'].max() for q in r[:5]); o['lad_xmax_low'] = max(q['x'].max() for q in r[:4])
        o['sd_b'] = np.std([np.sqrt(np.mean(q['Eb'].astype(float) ** 2)) for q in r[4:]])
        o['sd_t'] = np.std([np.sqrt(np.mean(q['x'].astype(float) ** 2)) for q in r[4:]])
    return o


def analyze():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    recs = pickle.load(open(os.path.join(SCR, 'x3_results.pkl'), 'rb')); lines = []
    P = lambda s='': (print(s), lines.append(s))
    P(f'records {len(recs)}, seeds {len({r["seed"] for r in recs})}')
    # verification against an earlier study BANDLOG3 (linear, exact)
    xc = pickle.load(open(os.path.join(ROOT, 'x1c', 'x1c_results.pkl'), 'rb'))
    X = {(r['seed'], r['ns'], r['nc']): r['t_rms'] for r in xc if r['law'] == 'BANDLOG3' and r['model'] == 'true'}
    lin = [r for r in recs if r['N'] == 3 and r['cond'] == 'exact' and r['tier'] == 'linear']
    d1 = [abs(r['v_trms'] - X[(r['seed'], r['ns'], r['nc'])]) for r in lin]
    d2 = [final_row(r['laws']['I-BAND'])['t_rms'] - r['v_trms'] for r in lin]
    P(f'check: a1.band_law_v drive evaluated here vs an earlier study BANDLOG3 t_rms: max |d| {max(d1):.2e} dB (n={len(d1)})')
    P(f'check: fast band fit (warm polish) vs band_law_v, linear: d t_rms mean {np.mean(d2):+.3f}, max |d| {np.max(np.abs(d2)):.3f} dB')
    for sq in (False, True):
        P(f'  X1c BANDLOG3 rows recomputed here, sq={int(sq)}: t_rms {np.mean([r["v_trms"] for r in lin if r["square"] == sq]):.2f}')
    T = {}
    for r in recs:
        for law, runs in r['laws'].items():
            T[(r['N'], r['cond'], r['tier'], r['seed'], r['ns'], r['nc'], law)] = final_row(runs)
    allaw = LAWS + IDEALS
    # pick the better NB-F and LS-F rule per (N, cond, tier, class) by narrowband yardstick (share in 3 dB of reference)
    def rows(N, cond, tier, sq, law, rig=None):
        return [v for (n, c_, t_, s, ns, nc, l), v in T.items() if n == N and c_ == cond and t_ == tier and l == law
                and (ns == nc) == sq and (rig is None or (ns, nc) == rig)]
    def keys(N, cond, tier, sq, law):
        return [(s, ns, nc) for (n, c_, t_, s, ns, nc, l) in T if n == N and c_ == cond and t_ == tier and l == law and (ns == nc) == sq]
    best = {}
    P('\n== NB-F / LS-F rule choice: share of control lines within 3 dB of own reference at the final run (higher is better) ==')
    for (N, cond, tier) in CASES:
        for sq in (False, True):
            o = {l: np.mean([v['in3_ref' if l.startswith('NB') else 'in3_A'] for v in rows(N, cond, tier, sq, l)]) for l in ('NBF-a1', 'NBF-a05', 'NBF-r', 'LS-F-a1', 'LS-F-a05')}
            best[(N, cond, tier, sq, 'NB-F')] = max(('NBF-a1', 'NBF-a05', 'NBF-r'), key=lambda l: o[l])
            best[(N, cond, tier, sq, 'LS-F')] = max(('LS-F-a1', 'LS-F-a05'), key=lambda l: o[l])
            P(f'  1/{N} {cond:5s} {tier:12s} sq={int(sq)} ' + ' '.join(f'{l} {v:.3f}' for l, v in o.items()) +
              f'  -> {best[(N, cond, tier, sq, "NB-F")]}, {best[(N, cond, tier, sq, "LS-F")]}')
    SHOW = ('NB-F', 'NB-U', 'NB-R', 'BAND-G', 'BAND-U', 'LS-F', 'LS-U', 'I-NB', 'I-BAND', 'I-LS')
    real = lambda N, cond, tier, sq, l: best.get((N, cond, tier, sq, l), l)
    COLS = [('t_rms', 'stress RMS'), ('t_bias', 'bias'), ('t_p5', 'p5'), ('t_p95', 'p95'), ('t_in3', 'in3'),
            ('e50tip_rms', 'Q50 tip rms'), ('e50tip_over', 'Q50 tip worst over'), ('e50held_over', 'Q50 held worst over'),
            ('e10tip_rms', 'Q10 tip rms'), ('b_rms', 'band rms'), ('b_over', 'band worst over'), ('in3_ref', 'lines in 3 of ref'),
            ('in6_ref', 'in 6 of ref'), ('over_ref', 'line worst over'), ('drv_w', 'worst shaker dB'), ('flat_max', 'flat max'),
            ('flat_med', 'flat med'), ('conv', 'runs to 1 dB'), ('lad_bover', 'ladder band over'), ('lad_xmax_low', 'low-level max stress'),
            ('sd_b', 'SD band'), ('sd_t', 'SD stress')]
    TAB = {}
    for (N, cond, tier) in CASES:
        for sq in (False, True):
            ks = keys(N, cond, tier, sq, 'BAND-G'); nrig = len({k[1:] for k in ks})
            P(f'\n== 1/{N} oct, {cond}, tier {tier}, {"square" if sq else "non-square"}: {len(ks)} cases ({nrig} rigs x {len({k[0] for k in ks})} seeds); final run ==')
            P(f'{"law":9s} ' + ' '.join(f'{h[:9]:>9s}' for _, h in COLS))
            for l in SHOW:
                lr = real(N, cond, tier, sq, l); rs = rows(N, cond, tier, sq, lr)
                if not rs:
                    continue
                v = {k: np.nanmean([q.get(k, np.nan) for q in rs]) for k, _ in COLS}; TAB[(N, cond, tier, sq, l)] = v
                P(f'{lr:9s} ' + ' '.join(f'{v[k]:9.2f}' for k, _ in COLS))
    # paired headline differences, BAND-G minus NB-F (and others), seed-cluster CI
    P('\n== paired differences (law A minus law B), mean [seed-cluster 95 % CI], share of cases A lower ==')
    PAIRS = (('BAND-G', 'NB-F'), ('BAND-U', 'NB-U'), ('BAND-G', 'NB-R'), ('BAND-G', 'BAND-U'), ('I-BAND', 'I-NB'), ('BAND-G', 'LS-F'))
    PD = {}
    for (N, cond, tier) in CASES:
        for sq in (False, True):
            for la, lb in PAIRS:
                A_, B_ = real(N, cond, tier, sq, la), real(N, cond, tier, sq, lb)
                ks = keys(N, cond, tier, sq, A_)
                for m in ('t_rms', 't_bias', 't_p95', 'e50tip_over', 'e50tip_rms', 'drv_w', 'b_rms', 'in3_ref'):
                    d = np.array([T[(N, cond, tier) + k + (A_,)][m] - T[(N, cond, tier) + k + (B_,)][m] for k in ks]); s = np.array([k[0] for k in ks])
                    ci = boot(d, s); PD[(N, cond, tier, sq, la, lb, m)] = (d.mean(), ci, np.mean(d < 0), d.size)
                    if m in ('t_rms', 'e50tip_over', 'drv_w') or (la, lb) == ('BAND-G', 'NB-F'):
                        P(f'  1/{N} {cond:5s} {tier:12s} sq={int(sq)} {A_:8s}-{B_:8s} {m:11s} {d.mean():+6.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] lower {np.mean(d < 0):.2f} n={d.size}')
    # E5 narrow peaks: element and Q = 50 comparisons between BAND-G and best NB-F, final run
    P('\n== E5: does band control let a narrow peak grow that NB-F catches? (final run, exact and noise, 1/3 oct) ==')
    E5 = {}
    for (N, cond, tier) in CASES:
        if N != 3:
            continue
        for sq in (False, True):
            nb_ = real(N, cond, tier, sq, 'NB-F'); stats = []
            for r in recs:
                if (r['N'], r['cond'], r['tier'], r['square']) != (N, cond, tier, sq):
                    continue
                fb, fn = r['laws']['BAND-G'][-1], r['laws'][nb_][-1]
                dx = fb['x'].astype(float) - fn['x'].astype(float)
                q = np.concatenate([fb['ers50_tip'] - fn['ers50_tip'], fb['ers50_held'] - fn['ers50_held']], 0).astype(float)
                qb = np.concatenate([fb['ers50_tip'], fb['ers50_held']], 0); qn = np.concatenate([fn['ers50_tip'], fn['ers50_held']], 0)
                stats.append((np.mean(dx > 3), np.mean((dx > 3) & (fb['x'] > 0)), np.mean(q > 3), np.mean((q > 3) & (qb > 3)),
                              np.mean(dx < -3), np.mean((dx < -3) & (fn['x'] > 0)), np.mean(q < -3), np.mean((q < -3) & (qn > 3)),
                              float(np.mean(dx > 3) > 0), float(np.mean((q > 3) & (qb > 3)) > 0), float(np.mean((q < -3) & (qn > 3)) > 0)))
            s = np.array(stats); E5[(cond, tier, sq)] = s.mean(0)
            P(f'  {cond:5s} {tier:12s} sq={int(sq)} n={len(s)}: elements BAND>NB+3dB {s[:, 0].mean():.3f} (and over service {s[:, 1].mean():.3f}); '
              f'Q50 points BAND>NB+3 {s[:, 2].mean():.3f} (and >3 over service {s[:, 3].mean():.4f}); reverse: elements NB>BAND+3 {s[:, 4].mean():.3f} '
              f'(over service {s[:, 5].mean():.3f}), Q50 NB>BAND+3 {s[:, 6].mean():.3f} (and >3 over {s[:, 7].mean():.4f}); '
              f'cases with any BAND Q50 excess over service {s[:, 9].mean():.2f}, any NB {s[:, 10].mean():.2f}')
    # ladder by level
    P('\n== ladder, exact, 1/3 oct: per level mean worst control band over-test / worst element stress re full service ==')
    LAD = {}
    for tier in ('linear', 'realistic', 'strong'):
        for sq in (False, True):
            for l in ('NB-F', 'NB-U', 'NB-R', 'BAND-G', 'BAND-U', 'LS-F', 'LS-U'):
                lr = real(3, 'exact', tier, sq, l)
                rr = [r['laws'][lr] for r in recs if (r['N'], r['cond'], r['tier'], r['square']) == (3, 'exact', tier, sq)]
                bo_ = np.array([[q['Eb'].max() for q in x] for x in rr]); xo = np.array([[q['x'].max() for q in x] for x in rr])
                lo_ = np.array([[q['over_ref'] for q in x] for x in rr])
                LAD[(tier, sq, l)] = (bo_.mean(0), xo.mean(0), bo_.max(0))
                P(f'  {tier:10s} sq={int(sq)} {lr:8s} n={len(rr)} band over ' + ' '.join(f'{v:5.1f}' for v in bo_.mean(0)) + ' | line over (G_ref) ' +
                  ' '.join(f'{v:5.1f}' for v in lo_.mean(0)) + ' | stress max ' + ' '.join(f'{v:5.1f}' for v in xo.mean(0)))
    # by rig, realistic exact
    P('\n== by rig, exact, realistic tier, 1/3 oct: stress RMS (NB-F / NB-U / NB-R / BAND-G / BAND-U / I-NB / I-BAND), BAND-G minus NB-F ==')
    for rig in sorted({(r['ns'], r['nc']) for r in recs}):
        sq = rig[0] == rig[1]; m = [np.mean([v['t_rms'] for v in rows(3, 'exact', 'realistic', sq, real(3, 'exact', 'realistic', sq, l), rig)]) for l in ('NB-F', 'NB-U', 'NB-R', 'BAND-G', 'BAND-U', 'I-NB', 'I-BAND')]
        P(f'  {rig[0]:2d}x{rig[1]:2d} ' + ' / '.join(f'{x:5.2f}' for x in m) + f'   d {m[3] - m[0]:+.2f}')
    open(os.path.join(SCR, 'x3_tables.txt'), 'w').write('\n'.join(lines))
    pickle.dump(dict(TAB=TAB, PD=PD, LAD=LAD, E5=E5, best=best), open(os.path.join(SCR, 'x3_summary.pkl'), 'wb'))
    figs(recs, TAB, LAD, best, plt)


def figs(recs, TAB, LAD, best, plt):
    fd = os.path.join(HERE, 'figs')
    sty = {'NB-F': ('s', '-', 'C0'), 'NB-U': ('v', '--', 'C9'), 'NB-R': ('<', ':', 'C4'), 'BAND-G': ('o', '-', 'C2'),
           'BAND-U': ('^', '--', 'C1'), 'LS-F': ('x', '-.', 'k'), 'LS-U': ('+', ':', 'C7'), 'I-NB': ('D', '', 'C0'), 'I-BAND': ('P', '', 'C2')}
    # 1 worked example
    ex = next(r for r in recs if 'ex' in r); e = ex['ex']; nbl = best[(3, 'exact', 'realistic', False, 'NB-F')]
    Gn, Gb = ex['laws'][nbl][-1]['ex']['G'], ex['laws']['BAND-G'][-1]['ex']['G']
    lo = (F > 60) & (F < 1000); j, fi = np.unravel_index(np.argmax(np.abs(db10(Gn[lo], Gb[lo]))), Gn[lo].shape); f0 = F[lo][j]
    win = (F > f0 / 1.5) & (F < f0 * 1.5); bo = band_of(3); Bm = x1.BANDS[3]
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True); ax = axs[0]
    ax.semilogy(F[win], e['t'][fi][bo][win], 'k-', lw=2, label='banded specification (1/3 oct)')
    ax.semilogy(F[win], e['dRef'][win, fi], color='C7', ls=':', lw=1.5, label='converted reference G_ref (from H_id)')
    ax.semilogy(F[win], Gn[win, fi], color='C0', ls='-', marker='s', markevery=25, ms=4, label=f'{nbl} at full level, final')
    ax.semilogy(F[win], Gb[win, fi], color='C2', ls='--', marker='o', markevery=25, ms=4, label='BAND-G at full level, final')
    for fn_i, fn_f in zip(e['fn_id'], e['fn_full']):
        if win[0] is not None and f0 / 1.5 < fn_f < f0 * 1.5:
            ax.axvline(fn_i, color='gray', lw=.6, ls=':'); ax.axvline(fn_f, color='gray', lw=.6, ls='-')
    ax.set_ylabel('control autospectrum'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title(f'Seed 0, 6x12, realistic tier: control {fi} near {f0:.0f} Hz (grey: lab modes at ID dotted, full level solid)', fontsize=9)
    ax = axs[1]; s0 = 0
    ax.semilogy(F[win], ex['laws'][nbl][-1]['ex']['Dss'][win, s0], color='C0', ls='-', marker='s', markevery=25, ms=4, label=f'{nbl} drive, shaker {s0}')
    ax.semilogy(F[win], ex['laws']['BAND-G'][-1]['ex']['Dss'][win, s0], color='C2', ls='--', marker='o', markevery=25, ms=4, label=f'BAND-G drive, shaker {s0}')
    ax.set_xscale('log'); ax.set_xlabel('Hz'); ax.set_ylabel('drive autospectrum'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(fd, 'x3_fig1_worked_example.png'), dpi=100); plt.close(fig)
    # 2 stress fidelity by tier and law; 4 yardsticks; 5 drive; 6 noise
    tiers = ('linear', 'realistic', 'strong'); laws = ('NB-F', 'NB-U', 'NB-R', 'BAND-G', 'BAND-U', 'I-NB', 'I-BAND')
    def panel(fname, keys_, ylabs, cond='exact', laws=laws):
        fig, axs = plt.subplots(len(keys_), 2, figsize=(9, 3.2 * len(keys_)), squeeze=False)
        for i, k in enumerate(keys_):
            for jj, sq in enumerate((False, True)):
                ax = axs[i, jj]
                for l in laws:
                    if (3, cond, 'linear', sq, l) not in TAB:
                        continue
                    mk, ls, cl = sty[l]; ax.plot(range(3), [TAB[(3, cond, t, sq, l)][k] for t in tiers], marker=mk, ls=ls or '-', color=cl, label=l, alpha=.85)
                ax.set_xticks(range(3)); ax.set_xticklabels(tiers); ax.grid(alpha=.3); ax.set_ylabel(ylabs[i], fontsize=8)
                ax.set_title(('square rigs' if sq else 'non-square rigs') + f', {cond}', fontsize=9)
        axs[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(fd, fname), dpi=100); plt.close(fig)
    panel('x3_fig2_stress_by_tier.png', ('t_rms', 'e50tip_over'), ('stress RMS error, dB', 'worst Q=50 tip over-test, dB'))
    panel('x3_fig4_yardsticks.png', ('in3_ref', 'b_rms'), ('share of control lines within 3 dB of G_ref', 'control band error RMS, dB'))
    panel('x3_fig5_drive.png', ('drv_w', 'flat_max'), ('worst shaker drive, dB (absolute)', 'max line / band mean of drive, dB'))
    panel('x3_fig6_noise.png', ('sd_b', 'sd_t'), ('SD of control band error RMS over full-level runs, dB', 'SD of stress RMS over full-level runs, dB'),
          cond='noise', laws=('NB-F', 'NB-U', 'NB-R', 'BAND-G', 'BAND-U'))
    # 3 ladder
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5))
    for jj, sq in enumerate((False, True)):
        for l in ('NB-F', 'NB-U', 'BAND-G', 'BAND-U', 'LS-F'):
            mk, ls, cl = sty[l]; b, x, _ = LAD[('strong', sq, l)]
            axs[0, jj].plot(range(8), b, marker=mk, ls=ls, color=cl, label=l); axs[1, jj].plot(range(8), x, marker=mk, ls=ls, color=cl, label=l)
        for i in range(2):
            axs[i, jj].set_xticks(range(8)); axs[i, jj].set_xticklabels(['-12', '-9', '-6', '-3', '0', '0c1', '0c2', '0c3'], fontsize=8); axs[i, jj].grid(alpha=.3)
        axs[0, jj].set_title(('square' if sq else 'non-square') + ' rigs, strong tier, exact', fontsize=9)
        axs[0, jj].set_ylabel('worst control 1/3-oct over-test vs level target, dB', fontsize=8)
        axs[1, jj].set_ylabel('worst element stress re full-level service, dB', fontsize=8)
    axs[0, 0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(fd, 'x3_fig3_ladder.png'), dpi=100); plt.close(fig)


if __name__ == '__main__':
    {'run': run, 'analyze': analyze, 'verify': verify}[sys.argv[1]](*sys.argv[2:])
