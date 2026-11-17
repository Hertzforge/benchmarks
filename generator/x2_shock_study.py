"""an earlier study: which metric tracks the stress truth in a multi-shaker shock test.

One command reruns everything (verification, simulation, tables, figures):
    python x2_shock_study.py --seeds 6 --workers 16
    python x2_shock_study.py --analyze     (tables/figures only)

Reuses frame_model.py and the rig selection of x1_random_study.py unchanged (imported, not edited).
Time domain: fs = 20480 Hz, 2.0 s records (40960 samples), event at 0.2 s. Responses by modal
superposition in the frequency domain with an exponential window (force windowed by exp(-sig t),
FRFs evaluated at s = sig + j w, result multiplied by exp(sig t), sig T = 6), so wrap-around is
suppressed by e^-6 on top of the natural decay and rigid-body modes need no special case.
SRS: Smallwood ramp-invariant filter, maximax absolute acceleration, Q = 10, 1/12-octave 10-2000 Hz
(92 points); points below 200 Hz are computed on the record decimated by 8 (2560 Hz), points
from 200 Hz up on the first 1.0 s at full rate. Idealized converged plant knowledge as in an earlier study
(the lab unit's own FRF designs the drives) except the '_m' variants, designed on the nominal unit.
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
import time, argparse, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
from multiprocessing import Pool
from scipy.signal import lfilter, resample_poly
from scipy.optimize import nnls
from scipy.stats import spearmanr
import frame_model as fm
import x1_random_study as x1

SCR = os.path.join(os.environ.get('OBS_GEN_SCRATCH', os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'gen_scratch')), 'x2')   # the study
HERE = os.path.dirname(os.path.abspath(__file__))
FS, NT, T0 = 20480, 40960, 0.2
TT = np.arange(NT) / FS; FK = np.fft.rfftfreq(NT, 1 / FS); NF = FK.size
SIG = 6.0 / (NT / FS); S = SIG + 2j * np.pi * FK; S2 = S ** 2; EW = np.exp(-SIG * TT); EWI = 1 / EW
DL = np.nonzero((FK > 0) & (FK <= 3000))[0]; WD = 2 * np.pi * FK[DL]
KS = np.arange(-79, 13); FN = 1000 * 2.0 ** (KS / 12)                   # 92 pts, 10.1..2000 Hz
I6, I3 = np.nonzero(KS % 2 == 0)[0], np.nonzero(KS % 4 == 0)[0]
I3A = np.nonzero((KS % 4 == 0) & (FN >= 49))[0]                         # present law check grid
BK = np.arange(-40, 7); FB = 1000 * 2.0 ** (BK / 6); NB = FB.size       # 1/6-octave bands
BL = [np.nonzero((FK >= f * 2 ** (-1 / 12)) & (FK < f * 2 ** (1 / 12)))[0] for f in FB]
PB = np.floor(KS / 2).astype(int) - BK[0]                               # band of each SRS point
NS_LIST, NC_LIST = x1.NS_LIST, x1.NC_LIST
SYN_RIGS = ((6, 6), (12, 12), (3, 12), (6, 12), (3, 24), (6, 24)); S6_RIGS = ((3, 12), (3, 24), (6, 12), (6, 24))
NREAL, NREAL_M, PERTS = 20, 5, (0, 1)


# ------------------------------------------------------------------ SRS and signal helpers
def srs(x, fn=FN, Q=10.0):
    x = np.atleast_2d(x); out = np.zeros(x.shape[:-1] + (len(fn),)); z = 1 / (2 * Q)
    lo = fn < 200; xd = resample_poly(x, 1, 8, axis=-1) if lo.any() else None; xh = x[..., :FS]
    for k, f in enumerate(fn):
        fs, y = (FS / 8, xd) if lo[k] else (FS, xh)
        wn = 2 * np.pi * f; E = np.exp(-z * wn / fs); K = wn * np.sqrt(1 - z * z) / fs
        C = E * np.cos(K); Sp = E * np.sin(K) / K
        out[..., k] = np.abs(lfilter([1 - Sp, 2 * (Sp - C), E * E - Sp], [1, -2 * C, E * E], y, axis=-1)).max(-1)
    return out


def dbr(a, b):
    return 20 * np.log10(np.maximum(a, 1e-30) / np.maximum(b, 1e-30))


def teff(y):
    a = np.abs(y); m = a >= a.max(-1, keepdims=True) / 3
    return (NT - m[..., ::-1].argmax(-1) - m.argmax(-1)) / FS


def rmsdur(y):
    w = y * y; w = w / w.sum(-1, keepdims=True); tm = (w * TT).sum(-1)
    return np.sqrt((w * (TT - tm[..., None]) ** 2).sum(-1))


def pol(y):
    return np.sign(y[np.arange(y.shape[0]), np.abs(y).argmax(-1)])


def wavelet(f, N, t0):
    tw = TT - t0; d = N / (2 * f); m = (tw >= 0) & (tw <= d); w = np.zeros(NT)
    w[m] = np.sin(2 * np.pi * f * tw[m] / N) * np.sin(2 * np.pi * f * tw[m]); return w


def bandE(y):
    P = np.abs(np.fft.rfft(y, axis=-1)) ** 2; return np.stack([P[..., l].sum(-1) for l in BL], -1)


# ------------------------------------------------------------------ modal machinery
def hs(md):
    wn, z = md.wn[:, None], md.zeta[:, None]; return 1 / (wn ** 2 + S2[None] + 2 * z * wn * S[None])


def hd(md, out, inn):
    """Design FRF (acceleration/force) at the real design lines, (n_f, n_out, n_in)."""
    wn, z = md.wn[:, None], md.zeta[:, None]; h = -WD ** 2 / (wn ** 2 - WD ** 2 + 2j * z * wn * WD)
    return np.einsum('om,mf,im->foi', md.Phi[out], h, md.Phi[inn], optimize=True)


def tf(md, H, out, inn):
    """Exp-window transfer (n_out, n_in, NF), acceleration per force."""
    return np.stack([(md.Phi[out] * md.Phi[i]) @ (H * S2) for i in inn], 1)


class Sys:
    """One configuration: modal model, exp-window modal FRF, stress and tip-deflection operators."""
    def __init__(self, md, Sfull, nart, m, el0):
        self.md, self.H, self.el = md, hs(md), np.arange(el0, md.Phi.shape[1])
        P = md.Phi[:nart, self.el]; self.Smod = Sfull @ P
        tip = [6 * n + d for n in m['tips'] for d in range(3)]; base = [6 * n + d for n in m['bases'] for d in range(3)]
        self.Tip = P[tip] - P[base]

    def truth(self, Qs):
        qt = np.fft.irfft(Qs[self.el], n=NT, axis=1) * EWI; s = (self.Smod @ qt) * 1e-6
        pk = np.abs(s).max(1); d = np.diff(s, axis=1); ext = d[:, :-1] * d[:, 1:] < 0
        dm = ((s[:, 1:-1] ** 2) ** 4 * ext).sum(1)
        tp = np.sqrt(((self.Tip @ qt).reshape(-1, 3, NT) ** 2).sum(1)).max(1)
        return dict(pk=pk.reshape(-1, 8).max(1), dm=dm.reshape(-1, 8).max(1), tip=tp)

    def fire(self, inn, ft, out, truth=True):
        Qs = self.H * (self.md.Phi[inn].T @ np.fft.rfft(ft * EW, axis=1))
        y = np.fft.irfft(self.md.Phi[out] @ (Qs * S2), n=NT, axis=1) * EWI
        return y, (self.truth(Qs) if truth else None)


def truth_summary(lab, ref):
    x = dbr(lab['pk'], ref['pk']); d = 10 * np.log10(lab['dm'] / ref['dm']); tp = dbr(lab['tip'], ref['tip'])
    out = dict(x=x.astype(np.float32), xd=d.astype(np.float32), t_rms=np.sqrt(np.mean(x ** 2)), t_bias=x.mean(),
               t_p95=np.percentile(x, 95), t_p5=np.percentile(x, 5), t_in3=np.mean(np.abs(x) <= 3), t_max=x.max(),
               d8_rms=np.sqrt(np.mean(d ** 2)), d8_bias=d.mean(), tip_rms=np.sqrt(np.mean(tp ** 2)), tip_bias=tp.mean())
    out['good'] = bool(out['t_in3'] >= 0.9 and out['t_max'] <= 6); return out


# ------------------------------------------------------------------ metrics and criteria
def criteria(E, yl, yr, Rr):
    """Acceptance rules on control dB errors E (nc, 92)."""
    out = {}
    ter = teff(yl) / teff(yr); polok = pol(yl) == pol(yr)
    out['crit_a'] = bool(((E[:, I3A] >= -1.5) & (E[:, I3A] <= 3)).all() and ((ter >= .8) & (ter <= 1.2)).all() and polok.all())
    out['crit_a_srs'] = bool(((E[:, I3A] >= -1.5) & (E[:, I3A] <= 3)).all())
    out['crit_a2'] = bool(((E >= -1.5) & (E <= 3)).all())
    ch = (((E >= -1.5) & (E <= 3)).mean(1) >= .9) & ((E >= -3) & (E <= 6)).all(1); out['crit_b'] = bool(ch.all())
    z6 = E[:, I6].mean(0)
    out['crit_c'] = bool((np.abs(z6) <= 1.5).all() and (np.abs(E) <= 6).all() and (np.abs(E) <= 3).mean() >= .9)
    w = Rr / Rr.sum(0, keepdims=True); g = (w * E).sum(0); out['crit_d'] = bool((np.abs(g) <= 3).all())
    z = E.mean(0); out['crit_e'] = bool((((z >= -1.5) & (z <= 3)).mean() >= .9) and ((z >= -3) & (z <= 6)).all())
    out['Emax'], out['Emin'] = E.max(), E.min(); out['ch_pass_b'] = ch.mean()
    out['te_ok'] = ((ter >= .8) & (ter <= 1.2)).mean(); out['pol_ok'] = polok.mean()
    out['reach3'] = (np.mean(np.abs(E) <= 3, 1) >= .9).mean(); out['reach6'] = (np.mean(np.abs(E) <= 6, 1) >= .9).mean()
    return out


def metrics(Rl, Rr, yl, yr, Phi_ch, pre, out):
    E = dbr(Rl, Rr); o = {}
    o['srs12_rms'] = np.sqrt(np.mean(E ** 2)); o['srs12_bias'] = E.mean(); o['srs12_std'] = E.std()
    o['srs12_max'], o['srs12_min'] = E.max(), E.min()
    o['srs6_rms'] = np.sqrt(np.mean(E[:, I6] ** 2)); o['srs3_rms'] = np.sqrt(np.mean(E[:, I3] ** 2))
    o['srs3_max'] = np.abs(E[:, I3]).max()
    o['in153'] = np.mean((E >= -1.5) & (E <= 3)); o['in3'] = np.mean(np.abs(E) <= 3); o['in6'] = np.mean(np.abs(E) <= 6)
    o['p5168'] = np.mean((((E >= -1.5) & (E <= 3)).mean(1) >= .9) & ((E >= -3) & (E <= 6)).all(1))
    z = E.mean(0); o['zone_max'] = np.abs(z).max(); o['zone_rms'] = np.sqrt(np.mean(z ** 2))
    w = Rr / Rr.sum(0, keepdims=True); g = (w * E).sum(0); o['g527_rms'] = np.sqrt(np.mean(g ** 2)); o['g527_max'] = np.abs(g).max()
    be = 10 * np.log10(np.maximum(bandE(yl), 1e-30) / np.maximum(bandE(yr), 1e-30))[:, 1:]
    o['be6_rms'] = np.sqrt(np.mean(be ** 2))
    te = dbr(teff(yl), teff(yr)); o['te_db'] = np.sqrt(np.mean(te ** 2)); o['te_bias'] = te.mean()
    o['te_ratio_med'] = np.median(teff(yl) / teff(yr))
    o['rd_db'] = np.sqrt(np.mean(dbr(rmsdur(yl), rmsdur(yr)) ** 2))
    pk = dbr(np.abs(yl).max(1), np.abs(yr).max(1)); o['pk_db'] = np.sqrt(np.mean(pk ** 2)); o['pk_bias'] = pk.mean()
    o['corr'] = np.mean((yl * yr).sum(1) / np.sqrt((yl * yl).sum(1) * (yr * yr).sum(1)))
    for k in (6, 12):
        if k <= Phi_ch.shape[0]:
            Psi = np.linalg.pinv(Phi_ch[:, :k]); Em = dbr(srs(Psi @ yl), srs(Psi @ yr))
            o[f'mod{k}_rms'] = np.sqrt(np.mean(Em ** 2))
        else:
            o[f'mod{k}_rms'] = np.nan
    pv = Rr / (2 * np.pi * FN); pv = pv / pv.sum()
    o['pv_rms'] = np.sqrt((pv * E ** 2).sum()); o['pv_bias'] = (pv * E).sum()
    for k, v in o.items():
        out[pre + k] = v
    return E


# ------------------------------------------------------------------ waveform synthesis
def synth_basis(te, rng):
    """Smallwood wavelets, one per 1/12-octave point, random odd half-cycle count, sign and delay
    inside the effective duration te. Returns (n_ch, 92, NT) float32 basis."""
    n = te.size; B = np.zeros((n, FN.size, NT), np.float32)
    for i in range(n):
        for k, f in enumerate(FN):
            nmax = int(np.clip(2 * f * te[i] * .7, 3, 41)); N = 2 * int(rng.integers(1, (nmax - 1) // 2 + 1)) + 1
            d = N / (2 * f); B[i, k] = rng.choice((-1, 1)) * wavelet(f, N, T0 + rng.uniform(0, max(te[i] - d, 0)))
    return B


def synth_amp(B, Rt, n_it=20):
    """Amplitude iteration on the isolated waveform's SRS; keeps the best iterate (RMS dB error)."""
    A = Rt / 4; best = (np.inf, A)
    for _ in range(n_it):
        M = srs(np.einsum('nk,nkt->nt', A, B)); e = np.sqrt(np.mean(dbr(M, Rt) ** 2))
        if e < best[0]:
            best = (e, A)
        A = A * np.clip(Rt / M, .5, 2)
    return best[1]


# ------------------------------------------------------------------ set-up per group
def service_shock(seed, host, na):
    rng = np.random.default_rng(seed + 271828); n = int(rng.integers(2, 4))
    nodes = rng.choice(host.meta['nodes'], n, replace=False); dofs, ft = [], []
    for i in range(n):
        d = rng.standard_normal(3); d /= np.linalg.norm(d); amp = 5e3 * np.exp(rng.uniform(np.log(.4), np.log(2)))
        t1 = TT - T0 - rng.uniform(0, 3e-3)
        if rng.random() < .5:
            tau = rng.uniform(.2e-3, 1e-3); p = np.where((t1 >= 0) & (t1 <= tau), np.sin(np.pi * np.clip(t1, 0, tau) / tau), 0)
        else:
            fb = np.exp(rng.uniform(np.log(200), np.log(1500))); tb = rng.uniform(1e-3, 4e-3)
            p = np.where(t1 >= 0, np.sin(2 * np.pi * fb * np.clip(t1, 0, None)) * np.exp(-np.clip(t1, 0, None) / tb), 0)
        p = p - np.roll(p, int(rng.uniform(2e-3, 8e-3) * FS))   # equal and opposite pulse: zero net impulse
        for k in range(3):
            dofs.append(na + 6 * nodes[i] + k); ft.append(amp * d[k] * p)
    return np.array(dofs), np.array(ft)


class Group:
    def __init__(self, seed, pert):
        self.seed, self.pert = seed, pert
        art_nom = fm.make_article(seed)
        units = {u: fm.make_article(seed, np.random.default_rng([seed, u, pert]), pert) for u in (1, 2)}
        host = fm.make_host(seed, art_nom.meta['dims']); zbase = fm.DampingField(seed)
        zs = {u: zbase.unit(np.random.default_rng([seed, u, pert, 9]), 0.3 * pert) for u in (1, 2)}
        self.mnom = fm.Modal(*fm.suspend(art_nom), zbase)
        self.held, self.ctrls, self.shakers = x1.choose_rig(seed, art_nom, self.mnom)
        m = art_nom.meta; self.m = m
        self.mnt = np.array([6 * n + d for n in m['mounts'] for d in range(3)])
        self.U = np.unique(np.concatenate([self.held, self.mnt] + list(self.ctrls.values())))
        pos = {d: i for i, d in enumerate(self.U)}; self.ix = lambda dofs: np.array([pos[d] for d in dofs])
        self.Y, self.tr = {}, {}
        for u in (1, 2):
            K, M, na = fm.couple(units[u], host); md = fm.Modal(K, M, zs[u])
            dofs, ft = service_shock(seed, host, na)
            sy = Sys(md, units[u].stress_matrix(), na, m, 6); self.Y[u], self.tr[u] = sy.fire(dofs, ft, self.U)
        self.RA = srs(self.Y[1]); self.TEA = teff(self.Y[1])
        self.lab = Sys(fm.Modal(*fm.suspend(units[2]), zs[2]), units[2].stress_matrix(), units[2].ndof, m, 6)
        Ka, Ma = units[2].assemble(); self.Ma = Ma; self.rr = [units[2].rigid_translation(a) for a in range(3)]
        mf = fm.Modal(Ka, Ma, zs[2], fixed=fm.fixed_base_dofs(units[2]))
        self.fix = Sys(mf, units[2].stress_matrix(), units[2].ndof, m, 0)
        self.PhiU = self.lab.md.Phi[self.U]

    def record(self, test, ns, nc, yU, truth, Rl=None, **kw):
        r = dict(seed=self.seed, pert=self.pert, test=test, ns=ns, nc=nc, **kw)
        for tag, dofs in (('c_', self.ctrls[nc]), ('h_', self.held)):
            sel = self.ix(dofs); Rl_ = Rl[sel] if Rl is not None else srs(yU[sel])
            E = metrics(Rl_, self.RA[sel], yU[sel], self.Y[1][sel], self.PhiU[sel], tag, r)
            if tag == 'c_':
                r.update(criteria(E, yU[sel], self.Y[1][sel], self.RA[sel]))
        if truth is not None:
            r.update(truth_summary(truth, self.tr[2]))
        return r


# ------------------------------------------------------------------ laws
def pinv_cond(H, cond):
    U, s, Vh = np.linalg.svd(H, full_matrices=False); sinv = np.where(s >= s[:, :1] / cond, 1 / s, 0)
    return np.conj(Vh).transpose(0, 2, 1) @ (sinv[:, :, None] * np.conj(U).transpose(0, 2, 1))


def drive_from(Pinv, yt):
    D = np.zeros((Pinv.shape[1], NF), complex); D[:, DL] = np.einsum('fsc,cf->sf', Pinv, np.fft.rfft(yt, axis=1)[:, DL])
    return np.fft.irfft(D, n=NT, axis=1)


def ctrl_resp(TFc, d):
    return np.fft.irfft(np.einsum('csf,sf->cf', TFc, np.fft.rfft(d * EW, axis=1)), n=NT, axis=1) * EWI


def stop_crit(E):
    return ((E >= -1.5) & (E <= 3)).all()


UNITW = np.stack([wavelet(f, 9, T0) for f in FB]); EU = np.array([(np.abs(np.fft.rfft(UNITW[b])[BL[b]]) ** 2).sum() for b in range(NB)])


def energy_drive(Bm, e, ns, off):
    d = np.zeros((ns, NT))
    for b, f in enumerate(FB):
        if BL[b].size == 0:
            continue
        p = nnls(Bm[b] / e[b][:, None], np.ones(Bm.shape[1]))[0]; dur = 9 / (2 * f); st = min(dur, .5 / ns)
        for s in np.nonzero(p > 0)[0]:
            d[s] += np.sqrt(p[s] / EU[b]) * np.roll(UNITW[b], int(round(((s + off[b]) % ns) * st * FS)))
    return d


def run_part(job):
    seed, pert, part = job; t0 = time.time(); G = Group(seed, pert); recs = []; lab = G.lab; info = {}
    def eval_drive(test, ns, nc, shk, d, **kw):
        y, tr = lab.fire(shk, d, G.U); return G.record(test, ns, nc, y, tr, drive_pk=20 * np.log10(np.abs(d).max()), **kw)
    def iterate(test, ns, nc, shk, TFc, make_d, update, real=0, **extra):
        c = G.ix(G.ctrls[nc]); Rt = G.RA[c]; best = (np.inf, None, 0)
        for it in range(9):
            d = make_d(); M = srs(ctrl_resp(TFc, d)); E = dbr(M, Rt); e = np.sqrt(np.mean(E ** 2))
            if it == 0:
                first, reach0 = E.max(), (np.mean(np.abs(E) <= 3, 1) >= .9).mean()
            if e < best[0]:
                best = (e, d, it)
            if stop_crit(E) or it == 8:
                break
            update(np.clip(Rt / M, .5, 2) ** 0.7)
        return eval_drive(test, ns, nc, shk, best[1], real=real, iters=best[2], first_over=first, reach3_0=reach0, **extra)
    if part == 'base':
        for nc in NC_LIST:
            recs.append(G.record('S1', 0, nc, G.Y[2], None) | truth_summary(G.tr[2], G.tr[1]))
        for (ns, nc), shk in G.shakers.items():
            c = G.ix(G.ctrls[nc]); designs = [('WAVEFORM', lab.md)] + ([('WAVEFORM_m', G.mnom)] if pert else [])
            TFc = tf(lab.md, lab.H, G.ctrls[nc], shk)
            for name, mdd in designs + [('WAVEFORM_c100', lab.md)]:
                d = drive_from(pinv_cond(hd(mdd, G.ctrls[nc], shk), 100 if name.endswith('c100') else 1e3), G.Y[1][c])
                r = eval_drive(name, ns, nc, shk, d); r['first_over'] = r['Emax']; recs.append(r)
            if (ns, nc) in SYN_RIGS:
                yc = G.Y[1][c]; e0 = bandE(yc).T
                for name, mdd in [('ENERGY', lab.md)] + ([('ENERGY_m', G.mnom)] if pert and (ns, nc) in S6_RIGS else []):
                    Hdd = hd(mdd, G.ctrls[nc], shk); Bm = np.stack([(np.abs(Hdd[BL[b] - 1]) ** 2).mean(0) if BL[b].size else np.ones((nc, ns)) for b in range(NB)])
                    st = dict(kb=np.ones(NB)); off = np.random.default_rng([seed, pert, ns, nc]).integers(0, ns, NB)
                    def upd(g, st=st):  # magnitude loop: one gain per band, median over controls of the SRS dB error
                        r_db = 20 * np.log10(g); rb = np.array([r_db[:, PB == b].mean(1) if (PB == b).any() else np.zeros(nc) for b in range(NB)])
                        st['kb'] = st['kb'] * 10 ** (np.median(rb, 1) / 10)
                    recs.append(iterate(name, ns, nc, shk, TFc, lambda st=st, Bm=Bm: energy_drive(Bm, e0 * st['kb'][:, None], ns, off), upd))
                if (ns, nc) in S6_RIGS and pert == 1:  # capped, non-overlapping first pulse (SRS by max over components)
                    Rt = G.RA[c]; cap = np.zeros_like(Rt)
                    for b, f in enumerate(FB):
                        idx = np.nonzero(np.abs(KS - 2 * BK[b]) <= 2)[0]
                        if idx.size == 0:
                            continue
                        yu = np.fft.irfft(TFc * np.fft.rfft(UNITW[b] * EW)[None, None], n=NT, axis=2) * EWI  # (nc, ns, NT)
                        Su = srs(yu, FN[idx]); a = (Rt[:, None, idx] / Su).min((0, 2))
                        cap[:, idx] = np.maximum(cap[:, idx], (a[None, :, None] * Su).max(1))
                    E = dbr(cap, Rt)
                    recs.append(dict(seed=seed, pert=pert, test='ENERGY_CAP', ns=ns, nc=nc, first_over=E.max(),
                                     reach3=(np.mean(np.abs(E) <= 3, 1) >= .9).mean(), reach6=(np.mean(np.abs(E) <= 6, 1) >= .9).mean(),
                                     c_in3=np.mean(np.abs(E) <= 3), c_in6=np.mean(np.abs(E) <= 6), c_srs12_bias=E.mean()))
        # single-axis sequence on a rigid base
        for real in range(2):
            rng = np.random.default_rng([seed, pert, 77, real]); shots = []
            for ax in range(3):
                sel = G.ix(G.mnt[ax::3]); Rt = G.RA[sel].max(0, keepdims=True); B = synth_basis(np.array([2 * np.median(G.TEA[sel])]), rng)
                ab = np.einsum('nk,nkt->nt', synth_amp(B, Rt), B)[0]
                Qs = -(G.fix.md.Phi.T @ G.Ma @ G.rr[ax])[:, None] * G.fix.H * np.fft.rfft(ab * EW)[None]
                y = G.rr[ax][G.U][:, None] * ab + np.fft.irfft(G.fix.md.Phi[G.U] @ (Qs * S2), n=NT, axis=1) * EWI
                shots.append((y, G.fix.truth(Qs)))
            for gdb in (0, 3):
                g = 10 ** (gdb / 20)
                tr = dict(pk=np.max([t['pk'] for _, t in shots], 0) * g, dm=np.sum([t['dm'] for _, t in shots], 0) * g ** 8,
                          tip=np.max([t['tip'] for _, t in shots], 0) * g)
                ysum = sum(y for y, _ in shots) * g; Rmax = np.max([srs(y) for y, _ in shots], 0) * g
                for nc in NC_LIST:
                    r = G.record(f'SA{gdb}', 0, nc, ysum, tr, real=real, Rl=Rmax); recs.append(r)
        if seed == 0 and pert == 1:
            info['ex'] = dict(RA=G.RA, RB=srs(G.Y[2]), c=G.ix(G.ctrls[12]), held=G.ix(G.held), yA=G.Y[1][G.ix(G.ctrls[12])[:1]])
    else:
        ns, nc = map(int, part.split(':')[1:]); shk = G.shakers[(ns, nc)]; c = G.ix(G.ctrls[nc])
        TFc = tf(lab.md, lab.H, G.ctrls[nc], shk)
        variants = [('SYNTH', lab.md, NREAL)] + ([('SYNTH_m', G.mnom, NREAL_M)] if pert and (ns, nc) in S6_RIGS else [])
        for name, mdd, nreal in variants:
            Pinv = x1.pinv_trunc(hd(mdd, G.ctrls[nc], shk))
            for real in range(nreal):
                rng = np.random.default_rng([seed, pert, ns, nc, real, 5]); kd = np.exp(rng.uniform(0, np.log(4)))
                B = synth_basis(G.TEA[c] * kd, rng)
                st = dict(A=synth_amp(B, G.RA[c]))
                def upd(g, st=st):
                    st['A'] = st['A'] * g
                r = iterate(name, ns, nc, shk, TFc, lambda st=st, B=B: drive_from(Pinv, np.einsum('nk,nkt->nt', st['A'], B)), upd, real, kd=kd)
                recs.append(r)
    print(f'part seed={seed} pert={pert} {part} recs={len(recs)} {time.time() - t0:.0f}s', flush=True)
    return recs, info


# ------------------------------------------------------------------ verification
def verify():
    lines = []
    # (1) SRS of a half-sine against the undamped closed form (Q = 5000)
    tau = 1e-3; t1 = TT - T0; x = np.where((t1 >= 0) & (t1 <= tau), np.sin(np.pi * np.clip(t1, 0, tau) / tau), 0)
    num = srs(x[None], Q=5000)[0]; ref = []
    for f in FN:
        r = f * tau; b = 1 / (2 * r); wn = 2 * np.pi * f; tp = np.linspace(0, tau, 4001)
        prim = np.abs((np.sin(np.pi * tp / tau) - b * np.sin(wn * tp)) / (1 - b * b)).max() if abs(b - 1) > 1e-6 else np.pi / 2
        res = abs(4 * r * np.cos(np.pi * r) / (4 * r * r - 1)) if abs(r - .5) > 1e-6 else np.pi / 2
        ref.append(max(prim, res))
    e = dbr(num, np.array(ref)); lines.append(f'V1 half-sine 1 ms SRS vs closed form (Q=5000): max |err| {np.abs(e).max():.3f} dB, '
                                              f'below 200 Hz {np.abs(e[FN < 200]).max():.3f} dB, rms {np.sqrt(np.mean(e ** 2)):.3f} dB')
    # (2) waveform reproduced on a square plant (6 shakers, 6 controls), seed 0, same unit
    G = Group(0, 0); shk = G.shakers[(6, 6)]; c = G.ix(G.ctrls[6]); yt = G.Y[1][c]
    Hdd = hd(G.lab.md, G.ctrls[6], shk); win = slice(int(.1 * FS), int(1.2 * FS))
    Yl = np.fft.rfft(yt, axis=1); Yl[:, (FK > 3000) | (FK == 0)] = 0; ylp = np.fft.irfft(Yl, n=NT, axis=1)
    kappa = np.linalg.svd(Hdd, compute_uv=False); kappa = (kappa[:, 0] / kappa[:, -1]).max()
    for nm, Pv in (('exact inverse', np.linalg.inv(Hdd)), ('pinv truncated at cond 1000', x1.pinv_trunc(Hdd))):
        y, _ = G.lab.fire(shk, drive_from(Pv, yt), G.ctrls[6], truth=False)
        rel = np.linalg.norm(y[:, win] - ylp[:, win]) / np.linalg.norm(ylp[:, win]); E = dbr(srs(y), G.RA[c])
        lines.append(f'V2 square plant 6x6 WAVEFORM, {nm}: time error vs 3 kHz-band-limited target {rel:.2e}, '
                     f'SRS error rms {np.sqrt(np.mean(E ** 2)):.2f} dB, max |err| {np.abs(E).max():.2f} dB (max cond {kappa:.0f})')
    lines.append(f'model: service modes {G.lab.md.Phi.shape[1]} lab, elements {len(G.tr[1]["pk"])}, '
                 f'target T_E median {np.median(G.TEA) * 1e3:.1f} ms, range {G.TEA.min() * 1e3:.1f}-{G.TEA.max() * 1e3:.1f} ms')
    for l in lines:
        print(l)
    return lines


# ------------------------------------------------------------------ analysis
MET = ['srs12_rms', 'srs12_bias', 'srs12_std', 'srs12_max', 'srs12_min', 'srs6_rms', 'srs3_rms', 'in153', 'in3', 'in6',
       'p5168', 'zone_max', 'zone_rms', 'g527_rms', 'g527_max', 'be6_rms', 'te_db', 'rd_db', 'pk_db', 'corr',
       'mod6_rms', 'mod12_rms', 'pv_rms', 'pv_bias']
ORIENT = {'in153': -1, 'in3': -1, 'in6': -1, 'p5168': -1, 'corr': -1, 'srs12_min': -1}


def oriented(rs, key):
    x = np.array([r.get(key, np.nan) for r in rs], float); m = key.split('_', 1)[1]
    return np.abs(x) if 'bias' in m else ORIENT.get(m, 1) * x


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[ok], y[ok])[0] if ok.sum() > 8 and np.ptp(x[ok]) > 1e-9 else np.nan


def analyze(vlines):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    recs, infos = pickle.load(open(os.path.join(SCR, 'x2_results.pkl'), 'rb'))
    L = list(vlines); P = lambda s='': (print(s), L.append(s))
    col = lambda k, rs: np.array([r.get(k, np.nan) for r in rs], float)
    tests = ['WAVEFORM', 'WAVEFORM_m', 'WAVEFORM_c100', 'SYNTH', 'SYNTH_m', 'ENERGY', 'ENERGY_m', 'SA0', 'SA3']
    for r in recs:
        if 'x' in r:
            r['good6'] = bool(np.mean(np.abs(r['x']) <= 6) >= .9 and r['x'].max() <= 10); r['t_in6'] = np.mean(np.abs(r['x']) <= 6)
    full = [r for r in recs if 't_rms' in r and r['test'] != 'S1']
    T = ['t_rms', 't_bias', 't_p95', 't_p5', 't_in3', 'd8_rms', 'd8_bias', 'tip_rms']
    def row(rs, lab_, keys=T):
        P(f'{lab_:26s} n={len(rs):4d} ' + ' '.join(f'{k}={np.nanmean(col(k, rs)):6.2f}' for k in keys) +
          f' good={np.mean([r["good"] for r in rs]):.2f} good6={np.mean([r.get("good6", np.nan) for r in rs]):.2f}')
    P(f'\nrecords {len(recs)} (with truth {len(full)})')
    P('\n== an earlier study-S1 floor: unit B vs unit A in service ==')
    for p in PERTS:
        row([r for r in recs if r['test'] == 'S1' and r['pert'] == p], f'S1 pert={p}', T + ['c_srs12_rms', 'c_srs3_rms', 'h_srs12_rms', 'c_pk_db', 'c_te_db'])
    P('\n== an earlier study-S2 SRS ambiguity: SYNTH realizations per case (20 each) ==')
    cases = sorted({(r['seed'], r['pert'], r['ns'], r['nc']) for r in full if r['test'] == 'SYNTH'})
    agg = []
    for cs in cases:
        rs = [r for r in full if r['test'] == 'SYNTH' and (r['seed'], r['pert'], r['ns'], r['nc']) == cs]
        X = np.array([r['x'] for r in rs]); XD = np.array([r['xd'] for r in rs])
        agg.append(dict(cs=cs, sd_el=np.median(X.std(0)), sd_el90=np.percentile(X.std(0), 90), rng_el=np.median(np.ptp(X, 0)),
                        sdD=np.median(XD.std(0)), rngbias=np.ptp(col('t_bias', rs)), rngd8=np.ptp(col('d8_bias', rs)),
                        srs=np.mean(col('c_srs12_rms', rs)), srs_sd=np.std(col('c_srs12_rms', rs)),
                        rte=rho(col('c_te_bias', rs), col('d8_bias', rs)), rtp=rho(col('c_te_bias', rs), col('t_bias', rs)),
                        rsrs=rho(col('c_srs12_bias', rs), col('t_bias', rs)), te=np.mean(col('c_te_ratio_med', rs)),
                        te_rng=np.ptp(np.log2(col('c_te_ratio_med', rs)))))
    for ns, nc in SYN_RIGS:
        a = [g for g in agg if g['cs'][2:] == (ns, nc)]
        P(f'  rig {ns:2d}x{nc:2d}: ctrl SRS rms {np.mean([g["srs"] for g in a]):4.2f} dB | per-element peak-stress sd across '
          f'realizations median {np.mean([g["sd_el"] for g in a]):4.2f} dB (p90 {np.mean([g["sd_el90"] for g in a]):4.2f}), range {np.mean([g["rng_el"] for g in a]):4.2f} dB | '
          f'damage(b=8) sd {np.mean([g["sdD"] for g in a]):4.2f} dB | range of bias: stress {np.mean([g["rngbias"] for g in a]):4.2f}, damage {np.mean([g["rngd8"] for g in a]):4.2f} dB | '
          f'T_E ratio median {np.mean([g["te"] for g in a]):4.2f}, spread {np.mean([g["te_rng"] for g in a]):4.2f} oct')
    P(f'  within-case Spearman: T_E bias vs damage bias {np.nanmean([g["rte"] for g in agg]):+.2f}, vs stress bias {np.nanmean([g["rtp"] for g in agg]):+.2f}; '
      f'SRS bias vs stress bias {np.nanmean([g["rsrs"] for g in agg]):+.2f}')
    sq = [r for r in full if r['test'] == 'SYNTH' and r['ns'] == r['nc']]
    if sq:
        tight = [r for r in sq if r['c_srs12_rms'] <= 1.0]
        P(f'  square rigs, SRS rms <= 1 dB: n={len(tight)}, stress bias sd {np.std(col("t_bias", tight)):.2f} dB, damage bias sd {np.std(col("d8_bias", tight)):.2f} dB')
    P('\n== an earlier study-S3 truth by test and rig ==')
    for p in PERTS:
        for t in tests:
            rs = [r for r in full if r['test'] == t and r['pert'] == p]
            if rs:
                row(rs, f'{t} pert={p}')
    for t in ('WAVEFORM', 'WAVEFORM_c100', 'SYNTH', 'ENERGY'):
        for ns, nc in sorted({(r['ns'], r['nc']) for r in full if r['test'] == t}):
            rs = [r for r in full if r['test'] == t and r['ns'] == ns and r['nc'] == nc and r['pert'] == 1]
            row(rs, f'  {t} {ns}x{nc} pert=1', ['t_rms', 't_bias', 't_in3', 'd8_rms', 'c_srs12_rms', 'h_srs12_rms'])
    P('\n== an earlier study-S4 Spearman of metric vs truth (worse-is-larger orientation), controls / held-out ==')
    lab_ = full; good = np.array([r['good6'] for r in lab_])
    P(f'strict truth-good (90% of elements within 3 dB, none over 6) share {np.mean([r["good"] for r in lab_]):.3f}, n={sum(r["good"] for r in lab_)}; '
      'AUC and S5 below use the looser truth-good6 (90% within 6 dB, none over 10 dB)')
    P(f'truth-good6 share pooled {good.mean():.2f}; ' + ', '.join(f'{t} {np.mean([r["good6"] for r in lab_ if r["test"] == t]):.2f}' for t in tests))
    res = {}
    groups = {'pooled': lab_, 'WAVE': [r for r in lab_ if r['test'].startswith('WAVE')], 'SYNTH': [r for r in lab_ if r['test'].startswith('SYNTH')],
              'ENERGY': [r for r in lab_ if r['test'].startswith('ENERGY')], 'SA': [r for r in lab_ if r['test'].startswith('SA')]}
    P(f'{"metric":12s} {"t_rms c/h":>12s} {"d8_rms c/h":>12s} {"-t_in3 c/h":>12s} {"tip c/h":>12s} {"AUC c/h":>12s} | within type rho(t_rms) ctrl: WAVE SYNTH ENERGY SA')
    for m in MET:
        cells = []
        for tk in ('t_rms', 'd8_rms', 't_in3', 'tip_rms'):
            v = [rho(oriented(lab_, tag + m), (-1 if tk == 't_in3' else 1) * col(tk, lab_)) for tag in ('c_', 'h_')]
            cells.append(v)
        au = [x1.auc(np.nan_to_num(oriented(lab_, tag + m), nan=np.nanmax(oriented(lab_, tag + m))), good) for tag in ('c_', 'h_')]
        wt = [rho(oriented(g, 'c_' + m), col('t_rms', g)) for k, g in groups.items() if k != 'pooled']
        res[m] = dict(c=cells[0][0], h=cells[0][1], aucc=au[0], auch=au[1], d8c=cells[1][0], d8h=cells[1][1])
        P(f'{m:12s} ' + ' '.join(f'{a:5.2f}/{b:5.2f}' for a, b in cells + [au]) + ' | ' + ' '.join(f'{v:5.2f}' for v in wt))
    P('signed: rho(c_srs12_bias, t_bias) %.2f  rho(h_srs12_bias, t_bias) %.2f  rho(c_pv_bias, t_bias) %.2f  rho(c_srs12_bias, d8_bias) %.2f  rho(c_te_bias, d8_bias) %.2f  rho(c_srs12_max, t_p95) %.2f  rho(c_srs12_min, t_p5) %.2f' % (
        rho(col('c_srs12_bias', lab_), col('t_bias', lab_)), rho(col('h_srs12_bias', lab_), col('t_bias', lab_)), rho(col('c_pv_bias', lab_), col('t_bias', lab_)),
        rho(col('c_srs12_bias', lab_), col('d8_bias', lab_)), rho(col('c_te_bias', lab_), col('d8_bias', lab_)),
        rho(col('c_srs12_max', lab_), col('t_p95', lab_)), rho(col('c_srs12_min', lab_), col('t_p5', lab_))))
    P('\n== an earlier study-S5 acceptance rules (controls): pass rate, precision (truth-good among passes), recall ==')
    for k, lab_k in (('crit_a', '(a) present check: 1/3-oct 50-2k, -1.5/+3 every ctrl, T_E 0.8-1.2, polarity'),
                     ('crit_a_srs', '(a, SRS part only)'), ('crit_a2', '(a2) every ctrl, every 1/12 pt, -1.5/+3'),
                     ('crit_b', '(b) 516.8 90% rule, every ctrl'), ('crit_c', '(c) zone mean 1.5, ctrl 6, 90% pairs 3'),
                     ('crit_d', '(d) 527.2 global SRS +/-3'), ('crit_e', '(e) 517.3 zone log-mean -1.5/+3 90%, -3/+6')):
        ps = np.array([bool(r.get(k)) for r in lab_])
        P(f'  {lab_k:62s} pass {ps.mean():.2f} precision {good[ps].mean() if ps.any() else np.nan:.2f} recall {ps[good].mean():.2f} | '
          + ' '.join(f'{t}:{np.mean([bool(r.get(k)) for r in lab_ if r["test"] == t]):.2f}' for t in tests))
    pe = [r for r in lab_ if r.get('crit_e')]
    if pe:
        P(f'  runs passing (e): n={len(pe)}; worst single control over-test median {np.median(col("Emax", pe)):.1f} dB (p90 {np.percentile(col("Emax", pe), 90):.1f}), '
          f'worst under-test median {np.median(col("Emin", pe)):.1f} dB (p10 {np.percentile(col("Emin", pe), 10):.1f}); share of controls passing 516.8 rule {np.mean(col("ch_pass_b", pe)):.2f}')
    P('  per-control parts: T_E within 0.8-1.2 share %.2f, polarity right share %.2f (all multi-shaker runs)' % (
        np.nanmean(col('te_ok', [r for r in lab_ if not r['test'].startswith('SA')])), np.nanmean(col('pol_ok', [r for r in lab_ if not r['test'].startswith('SA')]))))
    for m in ('c_srs12_rms', 'c_zone_rms', 'c_srs6_rms', 'h_srs12_rms', 'c_pv_rms', 'c_g527_rms', 'c_be6_rms', 'c_in3'):
        x = oriented(lab_, m)
        for tau in ((-.95, -.9, -.8, -.7) if m.endswith('in3') else (1, 1.5, 2, 3, 4, 6)):
            ps = x <= tau
            if ps.any():
                P(f'  {m:12s} <= {tau:5.2f}: pass {ps.mean():.2f} precision {good[ps].mean():.2f} recall {ps[good].mean():.2f}')
    P('\n== an earlier study-S6 more controls than drives: reach, first pulse, truth, drive, model mismatch (pert=1) ==')
    P(f'{"law":12s} {"rig":>6s} {"reach3_0":>8s} {"reach3":>7s} {"in3":>5s} {"in6":>5s} {"first_over":>10s} {"Emax":>6s} {"Emin":>6s} {"t_rms":>6s} {"t_bias":>6s} {"good6":>5s} {"drive":>6s} {"iters":>5s}')
    for ns, nc in S6_RIGS:
        for t in ('WAVEFORM', 'WAVEFORM_m', 'SYNTH', 'SYNTH_m', 'ENERGY', 'ENERGY_m', 'ENERGY_CAP'):
            rs = [r for r in recs if r['test'] == t and r['ns'] == ns and r['nc'] == nc and r['pert'] == 1]
            if not rs:
                continue
            v = lambda k: np.nanmean(col(k, rs))
            P(f'{t:12s} {ns:2d}x{nc:2d} {v("reach3_0"):8.2f} {v("reach3"):7.2f} {v("c_in3"):5.2f} {v("c_in6"):5.2f} {v("first_over"):10.1f} {v("Emax"):6.1f} {v("Emin"):6.1f} '
              f'{v("t_rms"):6.2f} {v("t_bias"):6.2f} {np.nanmean([r.get("good6", np.nan) for r in rs]):5.2f} {v("drive_pk"):6.1f} {v("iters"):5.1f}')
    open(os.path.join(SCR, 'x2_tables.txt'), 'w').write('\n'.join(L))
    make_figs(recs, infos, res, plt)


def make_figs(recs, infos, res, plt):
    figs = os.path.join(HERE, 'figs'); os.makedirs(figs, exist_ok=True)
    mk = ['o', 's', '^', 'D', 'v', 'x', '+', '*']; ls = ['-', '--', ':', '-.', (0, (5, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (8, 2))]
    cl = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#000000', '#999999']
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, t in enumerate(('S1', 'WAVEFORM', 'WAVEFORM_m', 'SYNTH', 'ENERGY', 'SA0', 'SA3')):
        rs = [r for r in recs if r['test'] == t and r['pert'] == 1 and 'x' in r]
        x = np.sort(np.concatenate([r['x'] for r in rs])); y = np.linspace(0, 1, x.size)
        ax.plot(x, y, color=cl[i], ls=ls[i], marker=mk[i], markevery=max(1, x.size // 8), lw=1.4, ms=4,
                label={'S1': 'S1 floor (two units)', 'SA0': 'single-axis, 0 dB', 'SA3': 'single-axis, +3 dB'}.get(t, t))
    ax.axvspan(-3, 3, color='0.9'); ax.set_xlim(-20, 20); ax.set_xlabel('element peak stress, lab / service (dB)')
    ax.set_ylabel('cumulative fraction of elements'); ax.legend(fontsize=7); ax.grid(alpha=.3)
    ax.set_title('an earlier study: peak-stress error distribution, 3 % unit perturbation, all rigs pooled', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figs, 'x2_fig1_stress_ecdf.png'), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5.5)); y = np.arange(len(MET))
    for j, (k, m_, c_, lab_) in enumerate((('c', 'o', cl[0], 'controls vs stress'), ('h', 's', cl[1], 'held-out vs stress'),
                                            ('d8c', '^', cl[2], 'controls vs damage b=8'), ('d8h', 'D', cl[3], 'held-out vs damage b=8'))):
        ax.plot([res[m][k] for m in MET], y + (j - 1.5) * .15, ls='none', marker=m_, color=c_, mfc='none' if j % 2 else c_, label=lab_)
    ax.set_yticks(y); ax.set_yticklabels(MET, fontsize=7); ax.axvline(0, color='k', lw=.8); ax.grid(alpha=.3); ax.legend(fontsize=7)
    ax.set_xlabel('Spearman rank correlation with RMS-over-elements truth error (pooled)')
    fig.tight_layout(); fig.savefig(os.path.join(figs, 'x2_fig2_spearman.png'), dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for i, (ns, nc) in enumerate(SYN_RIGS):
        rs = [r for r in recs if r['test'] == 'SYNTH' and r['ns'] == ns and r['nc'] == nc and 't_bias' in r]
        ax[0].plot([r['c_srs12_rms'] for r in rs], [r['t_bias'] for r in rs], ls='none', marker=mk[i], color=cl[i], ms=3, alpha=.6, label=f'{ns}x{nc}')
        ax[1].plot([r['c_te_ratio_med'] for r in rs], [r['d8_bias'] for r in rs], ls='none', marker=mk[i], color=cl[i], ms=3, alpha=.6)
    ax[0].set_xlabel('controls: SRS dB error, RMS'); ax[0].set_ylabel('truth: peak-stress bias (dB)'); ax[0].legend(fontsize=7, title='shakers x controls', title_fontsize=7)
    ax[1].set_xscale('log'); ax[1].set_xlabel('controls: effective duration ratio, median'); ax[1].set_ylabel('truth: damage (b=8) bias (dB)')
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle('an earlier study-S2: SRS-SYNTH realizations (each point one test)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(figs, 'x2_fig3_srs_ambiguity.png'), dpi=110); plt.close(fig)
    ex = next((i['ex'] for i in infos if 'ex' in i), None)
    if ex is not None:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for i, (ch, lab_) in enumerate(((ex['c'][0], 'control 1'), (ex['c'][5], 'control 6'), (ex['held'][0], 'held-out 1'))):
            ax.loglog(FN, ex['RA'][ch], color=cl[i], ls='-', marker=mk[i], markevery=8, label=f'{lab_}, unit A (target)')
            ax.loglog(FN, ex['RB'][ch], color=cl[i], ls='--', label=f'{lab_}, unit B in service')
        ax.set_xlabel('natural frequency (Hz)'); ax.set_ylabel('SRS, maximax acc. (arb.)'); ax.grid(alpha=.3, which='both'); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(figs, 'x2_fig4_example_srs.png'), dpi=110); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', type=int, default=6); ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--analyze', action='store_true'); ap.add_argument('--base-only', action='store_true')
    a = ap.parse_args(); os.makedirs(SCR, exist_ok=True)
    v = verify()
    if a.base_only:  # rerun the 'base' parts only and merge with the stored SYNTH records
        old, oinf = pickle.load(open(os.path.join(SCR, 'x2_results.pkl'), 'rb'))
        with Pool(a.workers) as pool:
            out = pool.map(run_part, [(s, p, 'base') for s in range(a.seeds) for p in PERTS], chunksize=1)
        recs = [r for r in old if r['test'].startswith('SYNTH')] + [r for rs, _ in out for r in rs]
        pickle.dump((recs, [i for _, i in out]), open(os.path.join(SCR, 'x2_results.pkl'), 'wb'))
    elif not a.analyze:
        t0 = time.time(); parts = ['base'] + [f'syn:{ns}:{nc}' for ns, nc in SYN_RIGS]
        jobs = [(s, p, q) for q in parts[::-1] for s in range(a.seeds) for p in PERTS]
        with Pool(a.workers) as pool:
            out = pool.map(run_part, jobs, chunksize=1)
        recs = [r for rs, _ in out for r in rs]; infos = [i for _, i in out]
        pickle.dump((recs, infos), open(os.path.join(SCR, 'x2_results.pkl'), 'wb'))
        print(f'simulation {time.time() - t0:.0f}s  frame_model md5 check at end recommended')
    analyze(v)
