"""an earlier study offline truth and report card, from the drive spectral matrix a closed-loop run played.

    PY=python
    $PY x4_truth.py verify                               # converged line law, seed 0, 6 x 12, against x1/x1b
    $PY x4_truth.py score DRIVE.npz --seed 0 --ns 6 --nc 12 [--tier linear] [--out card.json]

DRIVE.npz holds `D` (lines x drives x drives, or drives x drives x lines), optionally `f` (Hz, the
lines of D; must contain 20 to 2000 Hz in 0.5 Hz steps) and `lab_fn`, `lab_zeta` (the lab plant's
modal frequencies and damping as played, when the plant changed with level).

Plant, rigs, gauges and service truth are those of an earlier study and an earlier study (imported unchanged): unit A
(1) in service gives the specification, unit B (2) is the article in the lab, rig (ns, nc) is
ns shakers and nc controls. Reference at controls and held-out gauges for the report card is unit
A in service; reference for element stress and response spectra is unit B in service.
"""
import os, sys, json, copy
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = HERE; sys.path.insert(0, EXP)   # the study: the study modules sit beside this file
import x1_random_study as x1
import x1b_followup_study as x1b
from x1_random_study import F, DF, W, NF, db10, auto

PERT = 1
B3 = x1.BANDS[3]; BO3 = (B3 > 0).argmax(0); NB3 = B3.shape[0]
H_ = lambda A: np.conj(A).transpose(0, 2, 1)
_SETUP = {}


def context(seed, ns, nc, tier='linear'):
    """Everything fixed for a seed, rig and service tier. Indices c, held, tips are into U."""
    key = (seed, tier != 'linear')
    if key not in _SETUP:
        if tier == 'linear':
            _SETUP[key] = x1b.setup(seed, PERT)
        else:
            import x3_banded_spec_study as x3
            _SETUP[key] = x3.setup(seed)
    S = _SETUP[key]; ix = S['ix']
    if tier == 'linear':
        YA, YB, trB = S['YS'][1], S['YS'][2], S['truthS'][2]
    else:
        import x3_banded_spec_study as x3
        YA, _ = x3.service(S, seed, 1, tier); YB, trB = x3.service(S, seed, 2, tier)
    c = ix(S['ctrls'][nc]); YAc = YA[c].transpose(2, 0, 1)
    tips = ix(np.array([6 * n + d for n in S['units'][2].meta['tips'] for d in range(3)]))
    return dict(S=S, seed=seed, ns=ns, nc=nc, tier=tier, shk=S['shakers'][(ns, nc)], c=c, held=ix(S['held']),
                tips=tips, mlab=S['mlab'], SB=S['SB'], U=S['U'], YA=YA, GA=YAc @ H_(YAc), GAU=auto(YA),
                GBU=auto(YB), truthB=trB)


def lab_modal(ctx, lab_fn=None, lab_zeta=None):
    md = ctx['mlab']
    if lab_fn is None and lab_zeta is None:
        return md
    m = copy.copy(md)
    if lab_fn is not None:
        m.fn = np.asarray(lab_fn, float); m.wn = 2 * np.pi * m.fn
    if lab_zeta is not None:
        m.zeta = np.asarray(lab_zeta, float)
    return m


def frf(md, rows, shk):
    """Acceleration FRF (nf, rows, shakers) of a modal model on F."""
    return np.einsum('um,mf,sm->fus', md.Phi[rows], md.h(F) * -W ** 2, md.Phi[shk], optimize=True)


def factor(D):
    D = (D + H_(D)) / 2; lam, V = np.linalg.eigh(D); return V * np.sqrt(np.clip(lam, 0, None))[:, None, :]


def on_grid(D, f=None):
    """D as (NF, ns, ns) on the study grid F."""
    D = np.asarray(D)
    if D.shape[0] == D.shape[1] and D.shape[2] != D.shape[0]:
        D = D.transpose(2, 0, 1)
    if f is None:
        assert D.shape[0] == NF, f'D has {D.shape[0]} lines, the study grid has {NF}; pass f'
        return D
    f = np.asarray(f, float); idx = np.round((F - f[0]) / (f[1] - f[0])).astype(int)
    assert idx.min() >= 0 and idx.max() < f.size and np.allclose(f[idx], F, atol=1e-6), \
        'f must contain 20 to 2000 Hz in 0.5 Hz steps'
    return D[idx]


def stress_moments(md, shk, SB, Fd):
    """m0 and m2 per stress row from a drive factor Fd (nf, ns, r) through the lab modal model."""
    P = md.Phi[shk].T; A = (P[None] @ Fd) * md.h(F).T[:, :, None]; nm = P.shape[0]
    def mom(Aw):
        A2 = Aw.transpose(1, 0, 2).reshape(nm, -1); C = (A2.real @ A2.real.T + A2.imag @ A2.imag.T) * DF
        return np.einsum('rm,mn,rn->r', SB, C, SB, optimize=True)
    return mom(A), mom(A * F[:, None, None])


def _st(x):
    x = np.asarray(x, float)
    return dict(mean=float(x.mean()), rms=float(np.sqrt(np.mean(x ** 2))), max=float(x.max()), min=float(x.min()))


def score(D, seed, ns, nc, f=None, tier='linear', lab_fn=None, lab_zeta=None, ctx=None, arrays=False):
    """Stress truth, response spectra and report card for one played drive spectral matrix."""
    ctx = ctx or context(seed, ns, nc, tier); D = on_grid(D, f); assert D.shape[1] == ns
    md = lab_modal(ctx, lab_fn, lab_zeta); shk, c, held, tips = ctx['shk'], ctx['c'], ctx['held'], ctx['tips']
    Fd = factor(D); out = dict(seed=seed, ns=ns, nc=nc, tier=tier)
    # element stress against unit B in service
    ts = x1.truth_summary(x1.element_truth(*stress_moments(md, shk, ctx['SB'], Fd)), ctx['truthB'])
    x = ts.pop('x')
    out['stress'] = dict(rms=float(ts['t_rms']), bias=float(ts['t_bias']), p5=float(ts['t_p5']),
                         p95=float(ts['t_p95']), in3=float(ts['t_in3']), max=float(ts['t_max']),
                         d8_rms=float(ts['d8_rms']), d8_bias=float(ts['d8_bias']),
                         d4_rms=float(ts['d4_rms']), d4_bias=float(ts['d4_bias']))
    # responses at every gauge of U, as played
    HU = frf(md, ctx['U'], shk); Gu = (np.abs(HU @ Fd) ** 2).sum(2).T  # (nU, nf)
    out['ers'] = {}
    for Q in (10, 50):
        Ka = x1.KER[Q][0]
        for tag, sel in (('tip', tips), ('held', held)):
            out['ers'][f'q{Q}_{tag}'] = _st(db10(Gu[sel] @ Ka.T, ctx['GBU'][sel] @ Ka.T))
    # report card against unit A in service (the specification's source)
    GA = ctx['GAU']; card = {}; Eb = {}
    for tag, sel in (('c', c), ('h', held)):
        Gl, Gr = Gu[sel], GA[sel]; Eb[tag] = db10(Gl @ B3.T, Gr @ B3.T)
        card[f'{tag}_b3_rms'] = float(np.sqrt(np.mean(Eb[tag] ** 2)))
        card[f'{tag}_sum3_rms'] = float(np.sqrt(np.mean(db10(Gl.sum(0) @ B3.T, Gr.sum(0) @ B3.T) ** 2)))
        card[f'{tag}_line_rms'] = float(np.sqrt(np.mean(db10(Gl, Gr) ** 2)))
    allb = np.concatenate([Eb['c'], Eb['h']]); alll = db10(Gu[np.r_[c, held]], GA[np.r_[c, held]])
    card.update(level_mean=float(allb.mean()), worst_over_b3=float(allb.max()), worst_over_line=float(alll.max()),
                share_over6_b3=float(np.mean(allb > 6)), share_over6_line=float(np.mean(alll > 6)),
                held_under3_b3=float(np.mean(Eb['h'] < -3)), ctrl_under6_b3=float(np.mean(Eb['c'] < -6)))
    # declared deviation: the best the rig reaches at the controls, P G P with P = H pinv(H), linear lab plant
    Hc = frf(ctx['mlab'], ctx['U'][c], shk); P = Hc @ x1.pinv_trunc(Hc); Gp = P @ ctx['GA'] @ H_(P)
    gp = np.real(np.einsum('fii->if', Gp)); dev = db10(gp @ B3.T, GA[c] @ B3.T); vs = db10(Gu[c] @ B3.T, gp @ B3.T)
    card.update(declared_b3_rms=float(np.sqrt(np.mean(dev ** 2))), declared_b3_max=float(dev.max()),
                declared_b3_min=float(dev.min()), vs_declared_b3_rms=float(np.sqrt(np.mean(vs ** 2))))
    # drives
    Dss = np.real(np.einsum('fii->fi', D))
    card['drive_rms'] = [float(v) for v in np.sqrt(Dss.sum(0) * DF)]
    card['drive_line_to_band_db'] = [float(10 * np.log10(max(
        Dss[BO3 == b, s].max() / max(Dss[BO3 == b, s].mean(), 1e-300) for b in range(NB3)))) for s in range(ns)]
    out['card'] = card
    if arrays:
        out['arrays'] = dict(x=x, Eb_c=Eb['c'].astype(np.float32), Eb_h=Eb['h'].astype(np.float32),
                             Gu=Gu.astype(np.float32), Dss=Dss.astype(np.float32))
    return out


def line_drive(ctx):
    """The converged line law on the linear plant: pinv(H) G pinv(H)^H, (nf, ns, ns)."""
    Hc = frf(ctx['mlab'], ctx['U'][ctx['c']], ctx['shk']); Z = x1.pinv_trunc(Hc); return Z @ ctx['GA'] @ H_(Z)


def verify(seed=0, ns=6, nc=12):
    """Feed the converged line-law drive through score() and compare with the x1/x1b path for the same case."""
    ctx = context(seed, ns, nc); S = ctx['S']; c, held, shk = ctx['c'], ctx['held'], ctx['shk']
    new = score(line_drive(ctx), seed, ns, nc, ctx=ctx)
    Hu = x1b.frf_u(S['mlab'], S['U'], shk); Fd = x1.line_law(Hu[c], S['YS'][1][c])
    ev = x1b.evaluate(Fd, Hu, S['mlab'], shk, S['SB'])
    old = x1b.summarize(ev['m0b'].sum(1), ev['m2b'].sum(1), ev['Gu'], ev['dl'], S['truthS'][2], auto(S['YS'][1]), c, held)
    pairs = [('stress rms dB', new['stress']['rms'], old['t_rms']), ('stress bias dB', new['stress']['bias'], old['t_bias']),
             ('stress p5 dB', new['stress']['p5'], old['t_p5']), ('stress p95 dB', new['stress']['p95'], old['t_p95']),
             ('share within 3 dB', new['stress']['in3'], old['t_in3']), ('d8 rms dB', new['stress']['d8_rms'], old['d8_rms']),
             ('ctrl b3 rms dB', new['card']['c_b3_rms'], old['c_b3_rms']), ('held b3 rms dB', new['card']['h_b3_rms'], old['h_b3_rms']),
             ('held sum3 rms dB', new['card']['h_sum3_rms'], old['h_sum3_rms']),
             ('drive power dB', 10 * np.log10(sum(v ** 2 for v in new['card']['drive_rms'])), old['drive'])]
    worst = 0.0
    print(f'verify seed {seed} rig {ns} x {nc}: x4_truth.score against x1.line_law + x1b.evaluate + x1b.summarize')
    for name, a, b in pairs:
        worst = max(worst, abs(a - b)); print(f'  {name:20s} x4 {a:10.5f}   x1b {b:10.5f}   diff {a - b:+.2e}')
    print(f'  largest difference {worst:.2e}  ->  {"PASS" if worst < 1e-6 else "FAIL"}')
    print('  card:', json.dumps({k: (round(v, 3) if isinstance(v, float) else [round(u, 3) for u in v])
                                 for k, v in new['card'].items()}))
    print('  ers:', json.dumps({k: {a: round(b, 2) for a, b in v.items()} for k, v in new['ers'].items()}))
    return worst < 1e-6


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('verify', 'score')); ap.add_argument('drive', nargs='?')
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ns', type=int, default=6)
    ap.add_argument('--nc', type=int, default=12); ap.add_argument('--tier', default='linear'); ap.add_argument('--out')
    a = ap.parse_args()
    if a.cmd == 'verify':
        sys.exit(0 if verify(a.seed, a.ns, a.nc) else 1)
    z = np.load(a.drive); g = lambda k: z[k] if k in z.files else None
    r = score(z['D'], a.seed, a.ns, a.nc, f=g('f'), tier=a.tier, lab_fn=g('lab_fn'), lab_zeta=g('lab_zeta'))
    txt = json.dumps(r, indent=1); print(txt)
    if a.out:
        open(a.out, 'w').write(txt)


# ======================================================================================
# Agent R, 2026-11-14: scoring from what the plant played (additions; nothing above is changed).
#
# score() rebuilds stress and gauge responses from the played DRIVE through the linear lab model, which
# is wrong once the plant changes with level. score_played() takes them from the played OUTPUT
# autospectra instead: fibre stress from the stress monitor outputs of x4_export.build_b
# (G_sigma = G_a / scale^2 / (2 pi f)^4), gauge responses from the physical outputs themselves.
def _on_grid_psd(psd, f):
    psd = np.asarray(psd, float)
    if f is None:
        assert psd.shape[0] == NF; return psd
    f = np.asarray(f, float); idx = np.round((F - f[0]) / (f[1] - f[0])).astype(int)
    assert idx.min() >= 0 and idx.max() < f.size and np.allclose(f[idx], F, atol=1e-6); return psd[idx]


def stress_moments_played(psd_stress, scale):
    """m0 and m2 per fibre stress row from the stress monitors' played autospectra (nf, rows)."""
    G = psd_stress.T / (np.asarray(scale, float)[:, None] ** 2) / W ** 4
    return G.sum(1) * DF, (G * F ** 2).sum(1) * DF


def stress_moments_modal(M, side, SB):
    """m0 and m2 per fibre stress row from the modal monitors' displacement band moments M (2, nm, nm).

    The runner's M0_ij = sum over band lines of Re(G_ij) df / (2 pi f)^4 for the scaled modal accelerations
    a_i = scale_i q_i'', so M0 / (scale_i scale_j) is the covariance of the modal displacements as played and
    fibre stress row r has m0 = SB_r C SB_r^T. The same quantity as the fibre monitors give, to rounding.
    """
    sc = np.asarray(side['modal_scale'], float); C0, C2 = M[0] / np.outer(sc, sc), M[1] / np.outer(sc, sc)
    return np.einsum('rm,mn,rn->r', SB, C0, SB, optimize=True), np.einsum('rm,mn,rn->r', SB, C2, SB, optimize=True)


def score_played(D, psd, side, seed, ns, nc, f=None, tier='linear', ctx=None, arrays=False, moments=None):
    """As score(), from the played output autospectra psd (lines, outputs) and the side arrays of build_b.

    Same keys as score(). Stress, response spectra and the gauge figures of the card are as played; the
    declared deviation (what the system can reach) is still the linear lab model's, as in score().
    Fibre stress comes from the fibre stress monitors when the plant has them, else from `moments`
    (the modal monitors' displacement band moments, x4_export.read_moments).
    """
    ctx = ctx or context(seed, ns, nc, tier); D = on_grid(D, f); psd = _on_grid_psd(psd, f); assert D.shape[1] == ns
    c, held, tips = ctx['c'], ctx['held'], ctx['tips']; n_c, n_h, n_t = c.size, held.size, tips.size
    n_phys, n_s = int(side['n_phys']), int(side['n_stress']); assert n_phys == n_c + n_h + n_t
    out = dict(seed=seed, ns=ns, nc=nc, tier=tier)
    if n_s:
        assert n_s == ctx['SB'].shape[0]; mm = stress_moments_played(psd[:, n_phys:n_phys + n_s], side['stress_scale'])
    else:
        assert moments is not None, 'this plant has no fibre stress monitors: pass the modal band moments'
        mm = stress_moments_modal(moments, side, ctx['SB'])
    ts = x1.truth_summary(x1.element_truth(*mm), ctx['truthB'])
    x = ts.pop('x')
    out['stress'] = dict(rms=float(ts['t_rms']), bias=float(ts['t_bias']), p5=float(ts['t_p5']),
                         p95=float(ts['t_p95']), in3=float(ts['t_in3']), max=float(ts['t_max']),
                         d8_rms=float(ts['d8_rms']), d8_bias=float(ts['d8_bias']),
                         d4_rms=float(ts['d4_rms']), d4_bias=float(ts['d4_bias']))
    G = {'c': psd[:, :n_c].T, 'h': psd[:, n_c:n_c + n_h].T, 't': psd[:, n_c + n_h:n_phys].T}     # (gauges, nf), as played
    out['ers'] = {}
    for Q in (10, 50):
        Ka = x1.KER[Q][0]
        for tag, k, sel in (('tip', 't', tips), ('held', 'h', held)):
            out['ers'][f'q{Q}_{tag}'] = _st(db10(G[k] @ Ka.T, ctx['GBU'][sel] @ Ka.T))
    GA = ctx['GAU']; card = {}; Eb = {}
    for tag, sel in (('c', c), ('h', held)):
        Gl, Gr = G[tag], GA[sel]; Eb[tag] = db10(Gl @ B3.T, Gr @ B3.T)
        card[f'{tag}_b3_rms'] = float(np.sqrt(np.mean(Eb[tag] ** 2)))
        card[f'{tag}_sum3_rms'] = float(np.sqrt(np.mean(db10(Gl.sum(0) @ B3.T, Gr.sum(0) @ B3.T) ** 2)))
        card[f'{tag}_line_rms'] = float(np.sqrt(np.mean(db10(Gl, Gr) ** 2)))
    allb = np.concatenate([Eb['c'], Eb['h']]); alll = db10(np.concatenate([G['c'], G['h']]), GA[np.r_[c, held]])
    card.update(level_mean=float(allb.mean()), worst_over_b3=float(allb.max()), worst_over_line=float(alll.max()),
                share_over6_b3=float(np.mean(allb > 6)), share_over6_line=float(np.mean(alll > 6)),
                held_under3_b3=float(np.mean(Eb['h'] < -3)), ctrl_under6_b3=float(np.mean(Eb['c'] < -6)))
    shk = ctx['shk']; Hc = frf(ctx['mlab'], ctx['U'][c], shk); P = Hc @ x1.pinv_trunc(Hc); Gp = P @ ctx['GA'] @ H_(P)
    gp = np.real(np.einsum('fii->if', Gp)); dev = db10(gp @ B3.T, GA[c] @ B3.T); vs = db10(G['c'] @ B3.T, gp @ B3.T)
    card.update(declared_b3_rms=float(np.sqrt(np.mean(dev ** 2))), declared_b3_max=float(dev.max()),
                declared_b3_min=float(dev.min()), vs_declared_b3_rms=float(np.sqrt(np.mean(vs ** 2))))
    Dss = np.real(np.einsum('fii->fi', D))
    card['drive_rms'] = [float(v) for v in np.sqrt(Dss.sum(0) * DF)]
    card['drive_line_to_band_db'] = [float(10 * np.log10(max(
        Dss[BO3 == b, s].max() / max(Dss[BO3 == b, s].mean(), 1e-300) for b in range(NB3)))) for s in range(ns)]
    out['card'] = card
    if arrays:
        out['arrays'] = dict(x=x, Eb_c=Eb['c'].astype(np.float32), Eb_h=Eb['h'].astype(np.float32), Dss=Dss.astype(np.float32))
    return out


def stress_routes(D, psd, side, seed, ns, nc, f=None, ctx=None):
    """Fibre and element stress RMS by both routes, for the check on a plant that does not change.

    'drive': the played drive through the linear lab model (score()). 'played': the stress monitors.
    Returns the dB differences (played over drive) per fibre row and per element, and each route's card figures.
    """
    ctx = ctx or context(seed, ns, nc); Dg = on_grid(D, f); pg = _on_grid_psd(psd, f)
    n_phys, n_s = int(side['n_phys']), int(side['n_stress'])
    m0d, m2d = stress_moments(ctx['mlab'], ctx['shk'], ctx['SB'], factor(Dg))
    m0p, m2p = stress_moments_played(pg[:, n_phys:n_phys + n_s], side['stress_scale'])
    ed, ep = x1.element_truth(m0d, m2d), x1.element_truth(m0p, m2p)
    return dict(fibre_db=10 * np.log10(m0p / m0d), element_db=20 * np.log10(ep['rms'] / ed['rms']),
                d8_db=db10(ep['d8'], ed['d8']), m0_drive=m0d, m0_played=m0p,
                drive=score(Dg, seed, ns, nc, ctx=ctx), played=score_played(Dg, pg, side, seed, ns, nc, ctx=ctx))


def modal_state(D, psd, side, f=None, span=4.0):
    """Each lab mode's frequency and apparent damping as played, from the modal monitors.

    |H_m|^2 = G_am / (phi_m^T G_dd phi_m) with phi_m the mode's drive shape (per newton here: D is in N^2/Hz),
    and for one mode  w^4 / |H_m|^2 = w^4 + (4 zeta^2 - 2) wn^2 w^2 + wn^4,  linear in (b, c) = ((4 zeta^2 - 2) wn^2, wn^4).
    Fitted by least squares on the lines within span half-power bandwidths of the small-signal frequency
    (at least 6 lines), weighted by |H|^2 so the peak governs. On a plant that wanders in level the
    result is the best single mode through the time-averaged response: an apparent frequency and damping.
    Returns dict(fn, zeta, f0, zeta0, ok) per mode (nan where the mode is outside the band or unresolved).
    """
    Dg = on_grid(D, f); pg = _on_grid_psd(psd, f); n_phys, n_s, n_m = int(side['n_phys']), int(side['n_stress']), int(side['n_modal'])
    assert n_m > 0, 'this plant has no modal monitors'
    Ga = pg[:, n_phys + n_s:n_phys + n_s + n_m].T / (np.asarray(side['modal_scale'])[:, None] ** 2)        # (nm, nf)
    phi = np.asarray(side['drive_shape']) / float(side['force_per_volt'])                                   # (nm, ns) per newton
    Gu = np.real(np.einsum('ms,fst,mt->mf', phi, Dg, phi, optimize=True))
    f0, z0 = np.asarray(side['fn'], float), np.asarray(side['zeta'], float); fn = np.full(n_m, np.nan); ze = np.full(n_m, np.nan)
    for m in range(n_m):
        if not (F[0] + 5 < f0[m] < F[-1] - 5):
            continue
        for _ in range(3):      # recentre on the fitted frequency
            fc = f0[m] if np.isnan(fn[m]) else fn[m]; zc = z0[m] if np.isnan(ze[m]) else max(ze[m], z0[m])
            half = max(span * zc * fc, 6 * DF); k = np.flatnonzero((np.abs(F - fc) <= half) & (Gu[m] > 0) & (Ga[m] > 0))
            if k.size < 6:
                break
            w2 = W[k] ** 2; H2 = Ga[m, k] / Gu[m, k]; y = w2 ** 2 / H2 - w2 ** 2; wt = H2 / H2.max()
            A = np.stack([w2, np.ones_like(w2)], 1) * wt[:, None]; sol = np.linalg.lstsq(A / np.abs(y).max(), y * wt / np.abs(y).max(), rcond=None)[0]
            b, cc = sol
            if cc <= 0:
                break
            wn2 = np.sqrt(cc); z2 = (b / wn2 + 2) / 4
            if z2 <= 0:
                break
            fn[m] = np.sqrt(wn2) / (2 * np.pi); ze[m] = np.sqrt(z2)
    return dict(fn=fn, zeta=ze, f0=f0, zeta0=z0, ok=~np.isnan(fn))
