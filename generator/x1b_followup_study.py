"""an earlier study: regularization, cross-validation, autospectra-only line control, wrong-model first shot,
coverage margin. Extends an earlier study on the same seeds, rigs and units (imports frame_model and
x1_random_study unchanged).

    python x1b_followup_study.py --seeds 10 --workers 14
    python x1b_followup_study.py --analyze

S7  LINE truncation caps 1000/100/30/10, Tikhonov with lambda per 1/3-octave band chosen by
    leave-group-out cross-validation over the controls (two scores), oracle lambda (true stress).
S8  AUTO-LINE: per-line autospectra-only PSD drive (BAND machinery with one line per band) on every
    8th line, compared on the same lines with LINE and BAND3.
S9  Drive computed from a wrong model unit (stiffness x1.02, x1.06, damping x0.5), applied once.
S10 Coverage margin and level-setting rules from held-out (channel, 1/3-octave) errors.
Idealized: true FRFs except in S9; no noise; no closed loop.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import sys, time, argparse, pickle, copy, warnings
warnings.filterwarnings('ignore')
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
import frame_model as fm
import x1_random_study as x1
from x1_random_study import F, DF, W, NF, db10, auto, modal_q, acc, element_truth, truth_summary

SCR = x1.SCR.replace('/x1', '/x1b')
HERE = os.path.dirname(os.path.abspath(__file__))
B3 = x1.BANDS[3]; IND3 = (B3 > 0).astype(float); NB = B3.shape[0]; BAND_OF = IND3.argmax(0)
LAMS = np.logspace(-7, 0, 8); CAPS = (1000, 100, 30, 10)
SUB = np.arange(0, NF, 8); MASK = np.zeros(NF); MASK[SUB] = 1
x1.BANDS['sub8'] = np.eye(NF)[SUB]  # one line per "band": the BAND law becomes AUTO-LINE


# ------------------------------------------------------------------ linear algebra
def svdH(H):  # H (nc, ns, nf) -> batched SVD over lines
    return np.linalg.svd(H.transpose(2, 0, 1), full_matrices=False)


def drive_from_svd(sv, L, lam=None, cap=None):
    """Drive factor (ns, r, nf) for target factor L (nc, r, nf). Tikhonov (lam) or truncation (cap)."""
    U, s, Vh = sv; smax = s[:, :1]
    g = s / (s ** 2 + lam * smax ** 2) if lam is not None else np.where(s >= smax / cap, 1 / np.maximum(s, 1e-300), 0)
    Z = np.conj(Vh).transpose(0, 2, 1) @ (g[:, :, None] * np.conj(U).transpose(0, 2, 1))
    return (Z @ L.transpose(2, 0, 1)).transpose(1, 2, 0)


def folds_for(ctrl_dofs, seed, nfold=4):
    """Leave-group-out folds over control channels; triaxial (same node) groups kept together."""
    rng = np.random.default_rng(seed + 4242); nodes = ctrl_dofs // 6; un = rng.permutation(np.unique(nodes))
    groups = [np.nonzero(nodes == n)[0] for n in un]
    if len(groups) < nfold:
        groups = [np.array([i]) for i in rng.permutation(len(ctrl_dofs))]
    groups.sort(key=len, reverse=True); fo = [[] for _ in range(nfold)]
    for g in groups:
        min(fo, key=len).extend(g.tolist())
    return [np.array(sorted(f)) for f in fo if f]


def cv_scores(Hc, L, folds, laws):
    """Per law and band: mean over folds of (score1 per-channel 1/3-oct dB RMS, score2 |dB| of sum)."""
    Gt = auto(L) @ B3.T; out = {k: np.zeros((2, NB)) for k in laws}
    for te in folds:
        tr = np.setdiff1d(np.arange(Hc.shape[0]), te); sv = svdH(Hc[tr])
        for k, kw in laws.items():
            Fd = drive_from_svd(sv, L[tr], **kw); Gp = auto(np.einsum('csf,srf->crf', Hc[te], Fd)) @ B3.T
            E = db10(Gp, Gt[te]); out[k][0] += np.sqrt(np.mean(E ** 2, 0)) / len(folds)
            out[k][1] += np.abs(db10(Gp.sum(0), Gt[te].sum(0))) / len(folds)
    return out


# ------------------------------------------------------------------ evaluation of a drive
def evaluate(Fd, Hu, mlab, shk, SB, chunk=500):
    """Band-resolved stress moments, lab autospectra at U, drive power per line."""
    Q = mlab.h(F)[:, None, :] * np.einsum('sm,srf->mrf', mlab.Phi[shk], Fd, optimize=True)
    m0 = np.zeros((SB.shape[0], NB)); m2 = np.zeros_like(m0)
    for a in range(0, NF, chunk):
        q = Q[:, :, a:a + chunk]; nm, r, nfc = q.shape
        G = auto((SB @ q.reshape(nm, r * nfc)).reshape(-1, r, nfc)) * DF
        m0 += G @ IND3[:, a:a + chunk].T; m2 += (G * F[a:a + chunk] ** 2) @ IND3[:, a:a + chunk].T
    return dict(m0b=m0, m2b=m2, Gu=auto(np.einsum('usf,srf->urf', Hu, Fd, optimize=True)),
                dl=(np.abs(Fd) ** 2).sum((0, 1)) * DF)


def compose(evs, choice):
    """Combine evaluated drives by band: choice[b] = index into evs."""
    ch = np.asarray(choice); lines = ch[BAND_OF]
    m0 = sum(evs[i]['m0b'][:, b] for b, i in enumerate(ch)); m2 = sum(evs[i]['m2b'][:, b] for b, i in enumerate(ch))
    Gu = np.zeros_like(evs[0]['Gu']); dl = np.zeros(NF)
    for i in np.unique(ch):
        Gu[:, lines == i] = evs[i]['Gu'][:, lines == i]; dl[lines == i] = evs[i]['dl'][lines == i]
    return m0, m2, Gu, dl


def summarize(m0, m2, Gu, dl, refT, GA, c, h, extra=None):
    r = truth_summary(element_truth(m0, m2), refT); r['drive'] = 10 * np.log10(dl.sum())
    for tag, sel in (('c', c), ('h', h)):
        Gl, Gr = Gu[sel], GA[sel]; E = db10(Gl, Gr); Eb = db10(Gl @ B3.T, Gr @ B3.T)
        r[f'{tag}_line_rms'] = np.sqrt(np.mean(E ** 2)); r[f'{tag}_b3_rms'] = np.sqrt(np.mean(Eb ** 2))
        r[f'{tag}_sum3_rms'] = np.sqrt(np.mean(db10(Gl.sum(0) @ B3.T, Gr.sum(0) @ B3.T) ** 2))
        r[f'{tag}_over_b3'] = Eb.max(); r[f'{tag}_over_line'] = E.max()
        if tag == 'h':
            r['h_E3'] = Eb.astype(np.float32)
    if extra:
        r.update(extra)
    return r


# ------------------------------------------------------------------ one group
def setup(seed, pert):
    """Identical to the set-up part of x1_random_study.run_group."""
    art_nom = fm.make_article(seed)
    units = {u: fm.make_article(seed, np.random.default_rng([seed, u, pert]), pert) for u in (1, 2)}
    host = fm.make_host(seed, art_nom.meta['dims']); zbase = fm.DampingField(seed)
    zs = {u: zbase.unit(np.random.default_rng([seed, u, pert, 9]), 0.3 * pert) for u in (1, 2)}
    mlab_nom = fm.Modal(*fm.suspend(art_nom), zbase)
    held, ctrls, shakers = x1.choose_rig(seed, art_nom, mlab_nom); m = art_nom.meta
    tips = np.array([6 * n + d for n in m['tips'] for d in range(3)])
    mnt = np.array([6 * n + d for n in m['mounts'] for d in range(3)])
    U = np.unique(np.concatenate([held, tips, mnt] + list(ctrls.values())))
    pos = {d: i for i, d in enumerate(U)}; ix = lambda dofs: np.array([pos[d] for d in dofs])
    YS, truthS, serv = {}, {}, {}
    for u in (1, 2):
        K, M, na = fm.couple(units[u], host); md = fm.Modal(K, M, zs[u])
        dofs, Fin = x1.service_forces(seed, units[u], host, na); Q = modal_q(md, dofs, Fin)
        S = units[u].stress_matrix(); YS[u] = acc(md, U, Q)
        truthS[u] = element_truth(*x1.moments(S @ md.Phi[:na], Q)); serv[u] = (S @ md.Phi[:na], Q)
    mlab = fm.Modal(*fm.suspend(units[2]), zs[2])
    return dict(units=units, zs=zs, held=held, ctrls=ctrls, shakers=shakers, U=U, ix=ix, YS=YS,
                truthS=truthS, serv=serv, mlab=mlab, SB=units[2].stress_matrix() @ mlab.Phi)


def frf_u(md, U, shk):
    return np.einsum('um,mf,sm->usf', md.Phi[U], md.h(F) * -W ** 2, md.Phi[shk], optimize=True)


def run_group(args):
    seed, pert, part = args; t0 = time.time(); S = setup(seed, pert)
    ix, YS, mlab, SB, refT = S['ix'], S['YS'], S['mlab'], S['SB'], S['truthS'][2]
    GA = auto(YS[1]); h = ix(S['held']); recs = []
    base = dict(seed=seed, pert=pert)
    if part == 'B':  # S8 reference: service truth of unit B on the subsampled lines
        Ssv, Qs = S['serv'][2]; refT_sub = element_truth(*x1.moments(Ssv, Qs * MASK))
    if pert == 1 and part == 'A':  # S9 model units: stiffness x1.02 / x1.06 (frequencies +1 / +3 %), damping halved
        models = {}
        for name, kf, zf in (('K+1%', 1.02, 1.0), ('K+3%', 1.06, 1.0), ('zeta/2', 1.0, 0.5)):
            un = copy.deepcopy(S['units'][2]); zz = copy.deepcopy(S['zs'][2])
            for e in un.el:
                e['Efac'] *= kf
            zz.logz = zz.logz + np.log(zf); models[name] = fm.Modal(*fm.suspend(un), zz)
    for (ns, nc), shk in S['shakers'].items():
        Hu = frf_u(mlab, S['U'], shk); cdofs = S['ctrls'][nc]; c = ix(cdofs); Hc = Hu[c]; L = YS[1][c]
        fl = folds_for(cdofs, seed); rig = dict(base, ns=ns, nc=nc, square=ns == nc); sv = svdH(Hc)
        if part == 'B':  # ---- S8 on every 8th line, same lines for all three laws
            Fb = x1.band_law(Hc, auto(L), 3)[0]; Fa = x1.band_law(Hc, auto(L), 'sub8')[0]
            for name, Fd in (('S8_LINE', drive_from_svd(sv, L, cap=1000)), ('S8_AUTO', Fa), ('S8_BAND3', Fb)):
                e = evaluate(Fd * MASK, Hu, mlab, shk, SB); r = truth_summary(element_truth(e['m0b'].sum(1), e['m2b'].sum(1)), refT_sub)
                r.pop('x'); r.update(rig, test=name, drive=10 * np.log10(e['dl'].sum()),
                                     c_line_rms=np.sqrt(np.mean(db10(e['Gu'][c][:, SUB], GA[c][:, SUB]) ** 2)))
                recs.append(r)
            continue
        # ---- S7
        laws = {f'cap{k}': dict(cap=k) for k in CAPS}; laws.update({f'lam{i}': dict(lam=l) for i, l in enumerate(LAMS)})
        cvs = cv_scores(Hc, L, fl, laws)
        evs = {k: evaluate(drive_from_svd(sv, L, **kw), Hu, mlab, shk, SB) for k, kw in laws.items()}
        lam_keys = [f'lam{i}' for i in range(len(LAMS))]; ev_l = [evs[k] for k in lam_keys]
        S_l = np.array([cvs[k] for k in lam_keys])  # (nlam, 2, NB)
        def rec(test, m0, m2, Gu, dl, cv1, cv2, **kw):
            return summarize(m0, m2, Gu, dl, refT, GA, c, h, dict(rig, test=test, cv1=np.sqrt(np.mean(cv1 ** 2)),
                                                                   cv2=np.sqrt(np.mean(cv2 ** 2)), **kw))
        for k in CAPS:
            e = evs[f'cap{k}']; recs.append(rec(f'CAP{k}', e['m0b'].sum(1), e['m2b'].sum(1), e['Gu'], e['dl'], *cvs[f'cap{k}']))
        for j, name in ((0, 'CVc'), (1, 'CVs')):
            ch = S_l[:, j].argmin(0); sc = S_l[ch, :, np.arange(NB)].T
            recs.append(rec(name, *compose(ev_l, ch), sc[0], sc[1], lam_idx=ch))
        def t_of(ch):
            m0, m2, _, _ = compose(ev_l, ch); return truth_summary(element_truth(m0, m2), refT)['t_rms']
        ch = np.full(NB, int(np.argmin([t_of(np.full(NB, i)) for i in range(len(LAMS))])))
        best = t_of(ch)
        for _ in range(3):
            for b in range(NB):
                for i in range(len(LAMS)):
                    tr = ch.copy(); tr[b] = i; v = t_of(tr)
                    if v < best - 1e-9:
                        best, ch = v, tr
        sc = S_l[ch, :, np.arange(NB)].T; recs.append(rec('ORACLE', *compose(ev_l, ch), sc[0], sc[1], lam_idx=ch))
        if pert != 1:
            continue
        # ---- BAND3, full grid (S7/S10 reference)
        Fb, res = x1.band_law(Hc, auto(L), 3); e = evaluate(Fb, Hu, mlab, shk, SB)
        recs.append(rec('BAND3', e['m0b'].sum(1), e['m2b'].sum(1), e['Gu'], e['dl'], np.zeros(NB), np.zeros(NB), resid=res))
        # ---- S9 wrong model, one shot
        for mname, mm in models.items():
            Hm = frf_u(mm, S['U'], shk)[c]; svm = svdH(Hm)
            lm = {f'lam{i}': dict(lam=l) for i, l in enumerate(LAMS)}; cvm = cv_scores(Hm, L, fl, lm)
            chm = np.array([cvm[k][0] for k in lm]).argmin(0)
            Fcv = np.zeros((ns, L.shape[1], NF), complex)
            for i in np.unique(chm):
                sel = chm[BAND_OF] == i; Fcv[:, :, sel] = drive_from_svd(svm, L, lam=LAMS[i])[:, :, sel]
            for law, Fd in (('LINE', drive_from_svd(svm, L, cap=1000)), ('CAP30', drive_from_svd(svm, L, cap=30)),
                            ('CVc', Fcv), ('BAND3', x1.band_law(Hm, auto(L), 3)[0])):
                e = evaluate(Fd, Hu, mlab, shk, SB)
                r = summarize(e['m0b'].sum(1), e['m2b'].sum(1), e['Gu'], e['dl'], refT, GA, c, h, dict(rig, test=f'S9_{law}', model=mname))
                r.pop('h_E3'); recs.append(r)
    print(f'group seed={seed} pert={pert} part={part} recs={len(recs)} {time.time() - t0:.0f}s', flush=True)
    return recs


# ------------------------------------------------------------------ analysis
def analyze():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    recs = pickle.load(open(os.path.join(SCR, 'x1b_results.pkl'), 'rb'))
    x1r, _ = pickle.load(open(os.path.join(x1.SCR, 'x1_results.pkl'), 'rb'))
    lines = []; P = lambda s='': (print(s), lines.append(s))
    col = lambda k, rs: np.array([r[k] for r in rs], float)
    sel = lambda **kw: [r for r in recs if all(r.get(k) == v for k, v in kw.items())]
    rho = lambda a, b: spearmanr(a, b)[0]
    # consistency with an earlier study
    X1L = {(r['seed'], r['pert'], r['ns'], r['nc']): r for r in x1r if r['test'] == 'LINE'}
    d = [abs(r['t_rms'] - X1L[(r['seed'], r['pert'], r['ns'], r['nc'])]['t_rms']) for r in sel(test='CAP1000')]
    P(f'check: CAP1000 vs an earlier study LINE t_rms, max abs diff {max(d):.2e} dB over {len(d)} cases')
    T = ['t_rms', 't_bias', 't_p5', 't_p95', 't_in3', 'drive', 'c_line_rms', 'c_b3_rms', 'c_sum3_rms', 'h_sum3_rms', 'cv1', 'cv2']
    tests7 = ['CAP1000', 'CAP100', 'CAP30', 'CAP10', 'CVc', 'CVs', 'ORACLE', 'BAND3']
    P('\n== S7 by variant (all perts; BAND3 pert 1 only), square vs non-square ==')
    P('variant  sq  n  ' + ' '.join(f'{k:>9s}' for k in T))
    for sq in (False, True):
        for t in tests7:
            rs = sel(test=t, square=sq)
            P(f'{t:8s} {int(sq)} {len(rs):3d} ' + ' '.join(f'{np.nanmean(col(k, rs)):9.2f}' for k in T))
    P('\n== S7 t_rms / bias / drive by rig, pert 1 ==')
    for (ns, nc) in sorted({(r['ns'], r['nc']) for r in recs}):
        P(f'{ns:2d}x{nc:2d} ' + ' | '.join(f'{t} {np.mean(col("t_rms", sel(test=t, ns=ns, nc=nc, pert=1))):5.2f}/'
                                          f'{np.mean(col("t_bias", sel(test=t, ns=ns, nc=nc, pert=1))):+5.1f}/'
                                          f'{np.mean(col("drive", sel(test=t, ns=ns, nc=nc, pert=1))):5.1f}' for t in tests7))
    P('\n== S7 paired differences vs CAP1000 (t_rms), non-square / square ==')
    base = {(r['seed'], r['pert'], r['ns'], r['nc']): r for r in sel(test='CAP1000')}
    for t in tests7[1:]:
        o = []
        for sq in (False, True):
            pr = [(base[(r['seed'], r['pert'], r['ns'], r['nc'])], r) for r in sel(test=t, square=sq)]
            o.append(f'{np.mean([b["t_rms"] - a["t_rms"] for a, b in pr]):+6.2f} dB, better in {np.mean([b["t_rms"] < a["t_rms"] for a, b in pr]):.2f}')
        P(f'  {t:8s} non-sq {o[0]} | sq {o[1]}')
    P('  CV gap to oracle (t_rms): ' + ', '.join(f'{t} {np.mean(col("t_rms", sel(test=t)) - col("t_rms", sel(test="ORACLE"))):+.2f}' for t in ('CVc', 'CVs', 'CAP30', 'CAP1000')))
    for t in ('CVc', 'CVs', 'ORACLE'):
        li = np.concatenate([r['lam_idx'] for r in sel(test=t)]); P(f'  {t} lambda index histogram (0=1e-7 .. 7=1): {np.bincount(li, minlength=8).tolist()}')
    P('\n== S7 ranking: Spearman vs t_rms ==')
    for label, rs in (('CVc cases', sel(test='CVc')), ('CVs cases', sel(test='CVs')),
                      ('all S7 variants (LINE family)', [r for r in recs if r['test'] in tests7[:7]])):
        rigs = sorted({(r['ns'], r['nc']) for r in rs})
        def within(k):
            return np.nanmean([rho(col(k, g), col('t_rms', g)) for g in ([r for r in rs if (r['ns'], r['nc']) == q] for q in rigs)])
        P(f'  {label} (n={len(rs)}): ' + ', '.join(f'{k} {rho(col(k, rs), col("t_rms", rs)):.2f} (within rig {within(k):.2f})'
                                                     for k in ('cv1', 'cv2', 'c_b3_rms', 'c_sum3_rms', 'h_b3_rms', 'h_sum3_rms')))
    # ---- S8
    P('\n== S8 every 8th line, pert 1: LINE (full SDM) / AUTO-LINE / BAND3 ==')
    for sq in (False, True):
        for t in ('S8_LINE', 'S8_AUTO', 'S8_BAND3'):
            rs = sel(test=t, square=sq)
            P(f'  sq={int(sq)} {t:9s} n={len(rs)} ' + ' '.join(f'{k} {np.mean(col(k, rs)):6.2f}' for k in ('t_rms', 't_bias', 't_p5', 't_p95', 't_in3', 'drive', 'c_line_rms')))
    pk = lambda t: {(r['seed'], r['ns'], r['nc']): r for r in sel(test=t)}
    Lr, Ar, Br = pk('S8_LINE'), pk('S8_AUTO'), pk('S8_BAND3')
    for sq in (False, True):
        ks = [k for k in Lr if (k[1] == k[2]) == sq]
        dd = lambda a, b, f: np.mean([b[k][f] - a[k][f] for k in ks])
        P(f'  sq={int(sq)} LINE->AUTO: d t_rms {dd(Lr, Ar, "t_rms"):+.2f}, d bias {dd(Lr, Ar, "t_bias"):+.2f}, d drive {dd(Lr, Ar, "drive"):+.2f} | '
          f'AUTO->BAND3: d t_rms {dd(Ar, Br, "t_rms"):+.2f}, d bias {dd(Ar, Br, "t_bias"):+.2f}, d drive {dd(Ar, Br, "drive"):+.2f}')
    # ---- S9
    P('\n== S9 wrong model, one shot (pert 1). converged = S7 true-FRF pert 1 ==')
    K9 = ['t_rms', 't_bias', 't_p5', 't_p95', 't_in3', 'drive', 'c_over_b3', 'c_over_line', 'h_over_b3', 'h_over_line', 'c_b3_rms']
    P('  law/model         sq ' + ' '.join(f'{k:>11s}' for k in K9))
    conv = {'LINE': 'CAP1000', 'CAP30': 'CAP30', 'CVc': 'CVc', 'BAND3': 'BAND3'}
    for sq in (False, True):
        for law in ('LINE', 'CAP30', 'CVc', 'BAND3'):
            for mname in ('converged', 'K+1%', 'K+3%', 'zeta/2'):
                rs = sel(test=conv[law], square=sq, pert=1) if mname == 'converged' else sel(test=f'S9_{law}', model=mname, square=sq)
                P(f'  {law:6s}{mname:10s} {int(sq)} ' + ' '.join(f'{np.mean(col(k, rs)):11.2f}' for k in K9))
        for law in ('LINE', 'CAP30', 'CVc', 'BAND3'):
            rs = [r for r in sel(square=sq) if r['test'] == f'S9_{law}']
            P(f'  {law:6s} sq={int(sq)} worst case over all models: c_over_b3 max {col("c_over_b3", rs).max():.1f}, '
              f'h_over_b3 max {col("h_over_b3", rs).max():.1f}, t_p95 max {col("t_p95", rs).max():.1f}')
    # ---- S10
    P('\n== S10 coverage margin (pert 1) ==')
    cov = []
    for t in ('CAP1000', 'CVc', 'BAND3'):
        for sq in (False, True):
            rs = sel(test=t, pert=1, square=sq); o = {}
            for r in rs:
                x = r['x'].astype(float); E = r['h_E3'].ravel()
                for rule, g in (('as run', 0.0), ('cover 95% (truth)', -np.percentile(x, 5)),
                                ('held median >= 0', -np.median(E)), ('held p10 >= -3', -3 - np.percentile(E, 10))):
                    o.setdefault(rule, []).append((g, np.mean(x + g < 0), np.mean(x + g < -3), np.mean(x + g)))
            for rule, v in o.items():
                v = np.array(v); cov.append((t, sq, rule, *v.mean(0)))
                P(f'  {t:8s} sq={int(sq)} {rule:18s} gain {v[:, 0].mean():+6.2f} dB  under {v[:, 1].mean():.2f}  under>3dB {v[:, 2].mean():.2f}  mean over-test {v[:, 3].mean():+6.2f}')
    sa = [r for r in x1r if r['test'] == 'SA0' and r['pert'] == 1 and r['nc'] == 12]
    xs = [r['x'].astype(float) for r in sa]
    P(f'  single-axis as run: mean over-test {np.mean([x.mean() for x in xs]):+.2f}, under {np.mean([np.mean(x < 0) for x in xs]):.3f}; '
      f'lowered to cover 95%: gain {np.mean([-np.percentile(x, 5) for x in xs]):+.2f}, mean over-test {np.mean([x.mean() - np.percentile(x, 5) for x in xs]):+.2f}')
    # an earlier study saved results, all perts, LINE and BAND3: truth-based cover-95 margin
    for t in ('LINE', 'BAND3'):
        for p in (0, 1, 2):
            xs2 = [r['x'].astype(float) for r in x1r if r['test'] == t and r['pert'] == p and r['ns'] != r['nc']]
            P(f'  an earlier study {t} pert {p} non-square: cover-95 gain {np.mean([-np.percentile(x, 5) for x in xs2]):+.2f}, '
              f'mean over-test {np.mean([x.mean() - np.percentile(x, 5) for x in xs2]):+.2f}')
    open(os.path.join(SCR, 'x1b_tables.txt'), 'w').write('\n'.join(lines))
    figs(recs, cov, plt, sel, col)


def figs(recs, cov, plt, sel, col):
    out = os.path.join(HERE, 'figs'); mk = ['o', 's', '^', 'D', 'v', 'x', '+', 'P']
    cl = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#000000', '#999999']
    ls = ['-', '--', ':', '-.', (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2))]
    rigs = sorted({(r['ns'], r['nc']) for r in recs})
    # Fig 1: S7 stress RMS by rig and variant
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, t in enumerate(['CAP1000', 'CAP100', 'CAP30', 'CAP10', 'CVc', 'CVs', 'ORACLE', 'BAND3']):
        for a, k in zip(ax, ('t_rms', 't_bias')):
            v = [np.mean(col(k, sel(test=t, ns=ns, nc=nc, pert=1))) for ns, nc in rigs]
            a.plot(range(len(rigs)), v, marker=mk[i], ls=ls[i], color=cl[i], label=t, ms=5)
    for a, yl in zip(ax, ('element stress dB error, RMS over elements', 'mean element stress dB error (bias)')):
        a.set_xticks(range(len(rigs))); a.set_xticklabels([f'{a_}x{b_}' for a_, b_ in rigs], fontsize=7)
        a.set_xlabel('rig (shakers x controls)'); a.set_ylabel(yl); a.grid(alpha=.3)
    ax[0].legend(fontsize=7); fig.suptitle('an earlier study S7: regularized LINE laws, 3 % unit perturbation, 10 seeds', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'x1b_fig1_s7_by_rig.png'), dpi=110); plt.close(fig)
    # Fig 2: CV score vs truth, held-out sum vs truth
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, t in enumerate(['CAP1000', 'CAP30', 'CVc', 'CVs', 'ORACLE']):
        rs = sel(test=t)
        ax[0].plot(col('cv1', rs), col('t_rms', rs), ls='none', marker=mk[i], color=cl[i], ms=3, alpha=.6, label=t)
        ax[1].plot(col('h_sum3_rms', rs), col('t_rms', rs), ls='none', marker=mk[i], color=cl[i], ms=3, alpha=.6, label=t)
    ax[0].set_xlabel('CV score 1: left-out per-channel 1/3-oct dB error, RMS over bands'); ax[1].set_xlabel('held-out gauges: sum of autospectra 1/3-oct dB RMS')
    ax[0].set_ylabel('element stress dB error, RMS'); [a.grid(alpha=.3) for a in ax]; ax[0].set_xscale('log'); ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'x1b_fig2_cv_vs_truth.png'), dpi=110); plt.close(fig)
    # Fig 3: S9
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    models = ['converged', 'K+1%', 'K+3%', 'zeta/2']; conv = {'LINE': 'CAP1000', 'CAP30': 'CAP30', 'CVc': 'CVc', 'BAND3': 'BAND3'}
    for i, law in enumerate(('LINE', 'CAP30', 'CVc', 'BAND3')):
        for a, sq in zip(ax, (False, True)):
            xs, ys = [], []
            for m in models:
                rs = sel(test=conv[law], square=sq, pert=1) if m == 'converged' else sel(test=f'S9_{law}', model=m, square=sq)
                xs.append(np.mean(col('c_over_b3', rs))); ys.append(np.mean(col('t_rms', rs)))
            a.plot(xs, ys, marker=mk[i], ls=ls[i], color=cl[i], label=law)
            a.annotate('conv.', (xs[0], ys[0]), fontsize=6)
    for a, s in zip(ax, ('non-square rigs', 'square rigs')):
        a.set_title(s, fontsize=9); a.set_xlabel('worst control 1/3-oct over-test (dB), mean over cases'); a.grid(alpha=.3)
    ax[0].set_ylabel('element stress dB error, RMS'); ax[0].legend(fontsize=7, title='points: conv., K+1%, K+3%, zeta/2', title_fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'x1b_fig3_s9_first_shot.png'), dpi=110); plt.close(fig)
    # Fig 4: S10
    fig, ax = plt.subplots(figsize=(7, 4.2)); rules = ['as run', 'cover 95% (truth)', 'held median >= 0', 'held p10 >= -3']
    i = 0
    for t in ('CAP1000', 'CVc', 'BAND3'):
        for sq in (False,):
            v = [c_ for c_ in cov if c_[0] == t and c_[1] == sq]
            ax.plot([c_[6] for c_ in v], [c_[5] for c_ in v], marker=mk[i], ls=ls[i], color=cl[i], label=f'{t} (non-square)')
            for c_ in v:
                ax.annotate(rules.index(c_[2]) + 1, (c_[6], c_[5]), fontsize=7, xytext=(3, 3), textcoords='offset points')
            i += 1
    ax.set_xlabel('mean element over-test (dB)'); ax.set_ylabel('share of elements under-tested by more than 3 dB')
    ax.set_title('an earlier study S10: 1 as run, 2 truth cover-95, 3 held median >= 0, 4 held p10 >= -3', fontsize=8)
    ax.legend(fontsize=7); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(out, 'x1b_fig4_s10_coverage.png'), dpi=110); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=10); ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--analyze', action='store_true'); ap.add_argument('--one', type=str, default=None); a = ap.parse_args()
    os.makedirs(SCR, exist_ok=True)
    if a.one:
        s, p, q = a.one.split(','); r = run_group((int(s), int(p), q)); pickle.dump(r, open(os.path.join(SCR, 'one.pkl'), 'wb')); sys.exit()
    if not a.analyze:
        t0 = time.time(); jobs = [(s, 1, q) for s in range(a.seeds) for q in 'AB'] + [(s, p, 'A') for s in range(a.seeds) for p in (0, 2)]
        out = []
        with Pool(a.workers) as pool:
            for o in pool.imap_unordered(run_group, jobs, chunksize=1):
                out.append(o)
        pickle.dump([r for o in out for r in o], open(os.path.join(SCR, 'x1b_results.pkl'), 'wb'))
        print(f'simulation {time.time() - t0:.0f}s')
    analyze()
