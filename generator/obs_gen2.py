"""the study part 2: the frame structure with a hardening member, the level rule from physics, service by fixed point,
one absolute level per seed, and the exports (case schema mm-x4-case/2).

    PY=python; export PYTHONDONTWRITEBYTECODE=1
    $PY obs_gen2.py member                                  # L's strip, realistic and the strong resize, checked
    $PY obs_gen2.py kappa                                   # envelope over RMS of the simulator's follower, by damping
    $PY obs_gen2.py export --seed 0 --ns 6 --nc 12 --variant realistic      # one export (fresh process per case)
    $PY obs_gen2.py all --seeds 0,1 [--jobs 4]              # every rig and variant of the seeds, one process each
    $PY obs_gen2.py regress --seed 0                        # member removed: stopped study's structure and rule
    $PY obs_gen2.py table --seed 0 --ns 6 --nc 12           # the study plan's table (realistic and strong)

THE MEMBER (the study, the study's notes, C2): a 301 full-hard strip clamped at both ends on a stiff steel
bracket standing on one chassis node, a compact mass at mid-span. It replaces one of the seed's cantilevered
components (chosen by the seed); the bracket stands on that component's base node, along that component's own
outward direction; the strip's axis and its thin direction are chosen by the seed among the global axes.
Realistic: 50 x 15 x 0.6 mm, 0.15 kg. Strong: resized (member_strong) so the same service motion gives about 4 percent.
Cubic stiffness k3 = 0.514 x the immovable-end value (L's knock-down). Equivalent linearization (L, Section 2): the
member alone rises by sqrt(1 + 3 (k3/k1) sigma^2) - 1 at RMS mid-span motion sigma.

THE ABSOLUTE LEVEL: the seed's service forces are scaled so that in the LINEAR service model of unit A with the
realistic member, the member's mid-span RMS motion relative to its clamps is SIGMA_D (the RMS at which the realistic
member rises 1 percent). Every variant of the seed uses that environment.

THE RULE PER MODE (ruling 4). For mode m with strain-energy shares e_J (article outside the strip) and e_H (the
strip), small-signal circular frequency w_m, RMS modal displacement s_m, envelope A1 = k1env s_m and A2 = k2env s_m
(A1: the study's mean envelope over RMS by damping and frequency, decay_cycles 5; A2: the sine amplitude 2 s_m, on
which the rise is footed so the rule's spectral peak moves as an equivalent-linear cubic does; see kappa()):
    softening  drop  = D k1(f) e_J min(1, u/u_ref),       u = sqrt(e_J / 2) w_m A1   (strain energy in the jointed part)
               zeta factor - 1 = G k2(f) e_J (u/u_ref)  +  G_H e_H (x/x_ref)
    hardening  rise  = e_H R min(c, x/x_ref)^2,            x/x_ref = |phi_x,m| s_m / SIGMA_D
k1, k2: the stopped study's seeded fields over frequency (x3.mode_factors form A, unit u). D, u_ref, G belong to the
unit and the tier: sized in that unit's own service fixed point so the median article mode in 20 to 2000 Hz drops by
the tier's size with damping times the tier's factor and sits at two thirds of saturation (u = u_ref / 1.5).
R, c, G_H: L's table (realistic 0.01, 2.0, 0.25; strong: the resized member's own rise at SIGMA_D, 1.5, 0.5).
In the file, r = A / reference_amplitude, one reference per mode:
    A_J = u_ref / (sqrt(e_J/2) w)     A_H2 = 2 SIGMA_D / |phi_x| (sine)     A_H1 = k1env SIGMA_D / |phi_x|
  The reference is the smaller of A_J and A_H2, so no field is larger than its physical size (a mode that lives almost
  wholly in the member, e_J near 0, takes the member's).
  A_J <= A_H2: reference A_J. drop = D k1 e_J; rise = e_H R (A_J/A_H2)^2, cap c A_H2/A_J; gain = G k2 e_J + G_H e_H A_J/A_H1.
             Exact (both terms are powers of the same amplitude).
  A_H2 < A_J: reference A_H2. rise = e_H R, cap c; drop = D k1 e_J A_H2/A_J (exact below saturation; the drop then
             saturates at r = 1 instead of r = A_J/A_H2 > 1, so it stops early: stated); gain likewise rescaled.
A mode follows its own amplitude (ruling 6): a real member follows its total motion. Stated as a limit.
"""
import os, sys, json, copy, argparse, subprocess, time, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '2')
os.environ.setdefault('OBS_GEN_BULK', os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'exports/obs_gen2'))
import numpy as np
import frame_model as fm
import x1_random_study as x1
import x4_truth as xt
import x4_export as xe
from x1_random_study import F, DF, W, NF, auto

BULK = os.environ['OBS_GEN_BULK']
SCHEMA = 'mm-x4-case/2'
PERT = 1
RIGS = ((3, 12), (6, 12), (12, 24), (6, 6), (12, 12))           # (shakers, controls), the stopped study's design
DIAG_RIGS = ((6, 12), (6, 6))
VARIANTS = ('none', 'realistic', 'strong')
DIAG = ('realistic_soft', 'realistic_hard', 'strong_soft', 'strong_hard')
DECAY_CYCLES = 5.0                                              # the stopped study's jointed plants; structure-wide
CLIP_DROP, CLIP_DAMP = 3.0, 2.0   # the study's ruling 3 (part 4): drop at saturation <= 3 x the tier's, damping factor <= 2 x
TIER = {'realistic': dict(d=0.01, g=1.5, R=0.01, c=2.0, GH=0.25), 'strong': dict(d=0.03, g=3.0, R=None, c=1.5, GH=0.5)}
# ---- the study's material and design numbers (L1, Sections 4 and 6; hardening_sizes.py section F)
E_SS, RHO_SS, SY_SS, SF_SS = 193e9, 7850.0, 965e6, 552e6       # 301 full hard; RHO as L's script (rho_st)
KNOCK = 8.5e9 / (0.72 * 210e9 * 0.014 * 0.0005 / 0.040 ** 3)    # L's E3: identified over ideal immovable-end cubic = 0.514
JC = 4.8                                                        # int psi'^2 of the centre-load shape (L, F1)
PC = None                                                       # int psi^2, computed below as L does
E_BR, RHO_BR = 200e9, 7850.0                                    # bracket steel
G_SS, G_BR = E_SS / (2 * 1.28), E_BR / (2 * 1.3)                # shear moduli (Poisson 0.28 for 301, 0.3 bracket steel)
BRACKET_SEC = dict(fm.sq_tube(0.016, 0.004), G=G_BR)
H_STEM, H_POST = 0.015, 0.015                                   # bracket stem off the chassis node, posts up to the clamps
LAB_RELAX = 0.3
STEP = dict(kind='equalized', band_hz=2000.0, half_taps=2)       # the study, commit bccb6754 (the study's notes)
N_STRIP = 8                                                     # strip elements (4 a half)


def _pc():
    xi = np.linspace(0, 0.5, 4001); psi = 4 * xi ** 2 * (3 - 4 * xi)
    from scipy import integrate
    return 2 * integrate.simpson(psi ** 2, x=xi)


PC = _pc()


# ------------------------------------------------------------------ the member (L's formulas, F1)
def strip_design(L, b, t, M):
    """L's F1 clamped-clamped strip with a centre mass: k1, k3 (knocked down), f0, EL rise and stresses."""
    A, I = b * t, b * t ** 3 / 12; k1 = 192 * E_SS * I / L ** 3; k3 = KNOCK * E_SS * A * JC ** 2 / (2 * L ** 3)
    meff = M + RHO_SS * A * L * PC; f0 = np.sqrt(k1 / meff) / (2 * np.pi); gam = k3 / k1
    rise_el = lambda s: np.sqrt(1 + 3 * gam * s ** 2) - 1
    stress = lambda a: E_SS * (t / 2) * 24 * a / L ** 2 + E_SS * JC * a ** 2 / (2 * L ** 2)     # peak at deflection a (L's F1)
    sine_amp = lambda rise: np.sqrt(((1 + rise) ** 2 - 1) / (0.75 * gam))
    return dict(L=L, b=b, t=t, M=M, k1=k1, k3=k3, gamma=gam, f0=f0, meff=meff, rise_el=rise_el, stress=stress, sine_amp=sine_amp)


REAL = strip_design(0.050, 0.015, 0.0006, 0.15)
SIGMA_D = REAL['sine_amp'](0.01) / 2                             # RMS at which the realistic member rises 1 percent (EL: sine at 2 sigma)


def member_strong(target=0.04, f_keep=None, M=0.15, width_over_length=0.8):
    """The strong tier's member: same mass, thickness set so the EL rise at SIGMA_D is `target`, frequency kept at the
    realistic design's, width b <= width_over_length x L. Longest strip that fits (lowest stress). Returns the design."""
    f_keep = f_keep or REAL['f0']; best = None
    for L in np.arange(0.020, 0.0601, 0.0005):
        lo, hi = 1e-5, 2e-3                                        # t from the rise (independent of L and b)
        for _ in range(80):
            t = np.sqrt(lo * hi); r = strip_design(L, 0.01, t, M)['rise_el'](SIGMA_D)
            lo, hi = (lo, t) if r < target else (t, hi)
        t = np.sqrt(lo * hi); blo, bhi = 1e-4, 1.0                 # b from the frequency
        for _ in range(80):
            b = np.sqrt(blo * bhi); f = strip_design(L, b, t, M)['f0']
            blo, bhi = (b, bhi) if f < f_keep else (blo, b)
        b = np.sqrt(blo * bhi); d = strip_design(L, b, t, M)
        if b <= width_over_length * L:
            best = d
    return best


STRONG = member_strong()
DESIGN = {'realistic': REAL, 'strong': STRONG}


def _plain(d):
    return {k: (_plain(v) if isinstance(v, dict) else v) for k, v in d.items() if not callable(v)}


def member_report():
    out = {}
    for k, d in DESIGN.items():
        s = SIGMA_D; out[k] = dict(L_mm=1e3 * d['L'], b_mm=1e3 * d['b'], t_mm=1e3 * d['t'], M_kg=d['M'], f0_hz=float(d['f0']),
                                   rise_at_sigma_d=float(d['rise_el'](s)), sigma_d_mm=1e3 * s, sigma_over_t=float(s / d['t']),
                                   stress_3sigma_mpa=float(d['stress'](3 * s) / 1e6), stress_2sigma_mpa=float(d['stress'](2 * s) / 1e6),
                                   stress_at_cap_3sigma_mpa=float(d['stress'](3 * s * TIER[k]['c']) / 1e6),
                                   allowable_mpa=SF_SS / 2e6, endurance_mpa=SF_SS / 1e6, third_of_yield_mpa=SY_SS / 3e6,
                                   member=_plain(member_design(d, d['f0'], k)))
    out['knock'] = KNOCK; out['sigma_d_mm'] = 1e3 * SIGMA_D
    return out


N_STRIPS = 2                 # the study's ruling 2 (part 3): the mass rides on two parallel strips, each half the width
# the mass: a block with fixed length along the strips and height in the thin direction, its width from the mass.
# Realistic: steel 20 x 20.7 mm (46 mm wide at 0.15 kg). Strong: the strip is 28.5 mm long, so a steel block pitches
# (rotation about the across axis) at the translation frequency; a tungsten heavy alloy bar 10 x 12 mm (69 mm wide at
# 0.15 kg, density 18,000 kg/m^3, a typical W-Ni-Fe grade) keeps pitch at 1.8 times translation.
BLOCK = {'realistic': dict(l=0.020, h=0.0207, rho=7850.0, material='steel'), 'strong': dict(l=0.010, h=0.012, rho=18000.0, material='tungsten heavy alloy')}
ROCK_MIN = 1.5               # rocking about the strips' axis at least this times the translation frequency (ruling 2)
LINK_SEC = dict(fm.sq_tube(0.016, 0.004), G=G_BR)                # stiff links from the mass to the strips (massless)


def strip_sec(d):
    return dict(fm.strip(d['b'] / N_STRIPS, d['t']), G=G_SS)


def mass_block(M, tier):
    """The block of mass M for a tier: l along the strips and h in the thin direction fixed, w across from the mass;
    moments of inertia about its own axes (along the strips, across, thin)."""
    p = BLOCK[tier]; l, h = p['l'], p['h']; w = M / (p['rho'] * l * h)
    return dict(l=l, w=w, h=h, rho=p['rho'], material=p['material'], I_along=M * (w * w + h * h) / 12, I_across=M * (l * l + h * h) / 12,
                I_thin=M * (l * l + w * w) / 12)


def tuned_mass(sd, f):
    """The mass that puts the member's translation mode (L's formula) at f, the strip kept."""
    A = sd['b'] * sd['t']; return float(sd['k1'] / (2 * np.pi * f) ** 2 - RHO_SS * A * sd['L'] * PC)


def add_member(fr, center, a, tdir, d, ends, group_ends=True):
    """Two strips of half the width spaced s apart (across = a x t), clamped at ends[0] and ends[1] (the clamp-line
    centres), the block at `center` joined to both strips' mid-spans by stiff massless links. Returns the node ids."""
    wv = np.cross(a, tdir); s_ = d['s']; ids = dict(clamps=[], strips=[], mids=[]); e0 = len(fr.el)
    for sg in (1, -1):
        k1, k2 = ends[0] + sg * s_ / 2 * wv, ends[1] + sg * s_ / 2 * wv
        if group_ends:
            fr.line(ends[0], k1, BRACKET_SEC, n=1, group='bracket'); fr.line(ends[1], k2, BRACKET_SEC, n=1, group='bracket')
        e1 = len(fr.el); sid = fr.line(k1, k2, strip_sec(d), n=N_STRIP, group='strip')
        for e in fr.el[e1:]:
            e.update(E=E_SS, rho=RHO_SS, vref=np.cross(a, tdir).tolist())
        ids['clamps'] += [int(sid[0]), int(sid[-1])]; ids['strips'].append(list(range(e1, len(fr.el)))); ids['mids'].append(int(sid[N_STRIP // 2]))
    mc = fr.node(center)
    for m in ids['mids']:
        e1 = len(fr.el); fr.line(center, np.array(fr.xyz[m]), LINK_SEC, n=1, group='link')
        for e in fr.el[e1:]:
            e.update(E=E_BR, rho=1e-9)
    bl = d['block']; I = np.abs(a) * bl['I_along'] + np.abs(wv) * bl['I_across'] + np.abs(tdir) * bl['I_thin']
    fr.add_mass(mc, d['M'], I)
    for e in fr.el[e0:]:
        if e['group'] == 'bracket':
            e.update(E=E_BR, rho=RHO_BR)
    ids['center'] = int(mc); return ids


def fe_member(d):
    """The member alone, clamps fixed: its first modes, classified, and the translation, rocking and next modes."""
    fr = fm.Frame(); a, t = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    ids = add_member(fr, np.array([d['L'] / 2, 0, 0]), a, t, d, [np.zeros(3), np.array([d['L'], 0, 0])], group_ends=False)
    fixed = [6 * n + k for n in ids['clamps'] for k in range(6)]
    K, M = fr.assemble(); md = fm.Modal(K, M, lambda f: 0 * f + 0.01, fmax=1e5, fixed=fixed)
    c = ids['center']; P = md.Phi[6 * c:6 * c + 6, :10] * np.array([1, 1, 1, 0.01, 0.01, 0.01])[:, None]     # rotation x 1 cm
    kind = ['translation, thin' if np.argmax(np.abs(P[:, i])) == 2 else
            {0: 'translation, along', 1: 'translation, across', 3: 'rocking about the strips', 4: 'pitch about across', 5: 'yaw about thin'}.get(int(np.argmax(np.abs(P[:, i]))), '?')
            for i in range(P.shape[1])]
    tr = int(np.argmax(np.abs(P[2]))); rk = int(np.argmax(np.abs(P[3]))); pt = int(np.argmax(np.abs(P[4])))
    nxt = [i for i in range(len(kind)) if i not in (tr, rk)][0]
    return dict(translation_hz=float(md.fn[tr]), rocking_hz=float(md.fn[rk]), rocking_over_translation=float(md.fn[rk] / md.fn[tr]),
                pitch_hz=float(md.fn[pt]), pitch_over_translation=float(md.fn[pt] / md.fn[tr]), lowest_is_translation=bool(tr == 0),
                next_hz=float(md.fn[nxt]), next_kind=kind[nxt], modes=[(round(float(md.fn[i]), 1), kind[i]) for i in range(6)])


def member_design(sd, f, tier):
    """Strip design sd (L's formulas, full width) carrying the block tuned to f, on two strips spaced at the block's
    width, widened if needed until rocking is at least ROCK_MIN times the translation frequency. FE-checked."""
    M = tuned_mass(sd, f); d = strip_design(sd['L'], sd['b'], sd['t'], M); d['block'] = mass_block(M, tier); d['s'] = d['block']['w']
    d['f_tuned'] = float(f); d['tier'] = tier
    for _ in range(60):
        fe = fe_member(d)
        if fe['rocking_over_translation'] >= ROCK_MIN + 0.05:
            break
        d['s'] *= 1.05
    d['fe'] = fe; return d


def fe_clamped(d):
    return fe_member(d if 'block' in d else member_design(d, d['f0'], 'realistic'))


# ------------------------------------------------------------------ the article with the member
def make_article_m(seed, unit_rng=None, pert=0.0, design=None):
    """fm.make_article with one component replaced by the member (design: a strip_design dict, or None for none).
    Every random draw of fm.make_article is made in the same order, so geometry, the other components, the lumped
    masses and their unit perturbations are those of the stopped study; the member's own element perturbations are
    drawn after everything else."""
    rng = np.random.default_rng(seed); fr = fm.Frame()
    Lx, Ly, Lz = 0.6 * rng.uniform(.9, 1.1), 0.4 * rng.uniform(.9, 1.1), 0.3 * rng.uniform(.9, 1.1)
    b = rng.uniform(.020, .030); sec = fm.sq_tube(b, rng.uniform(.0015, .0025))
    secv = fm.sq_tube(b * rng.uniform(.8, 1.0), .002)
    x, y = Lx / 2, Ly / 2
    for z in (0.0, Lz):
        for yy in (-y, y):
            fr.line([-x, yy, z], [x, yy, z], sec, n=6)
        for xx in (-x, 0.0, x):
            fr.line([xx, -y, z], [xx, y, z], sec, n=4)
    for xx in (-x, x):
        for yy in (-y, y):
            fr.line([xx, yy, 0], [xx, yy, Lz], secv, n=3)
    X = np.array(fr.xyz); chassis = list(range(fr.nn))
    mounts = [fr.node([sx * x, sy * y, 0]) for sx in (-1, 1) for sy in (-1, 1)]
    corners = [fr.node([sx * x, sy * y, z]) for sx in (-1, 1) for sy in (-1, 1) for z in (0, Lz)]
    cand = [n for n in chassis if n not in corners]
    nst = int(rng.integers(3, 6)); bases = list(rng.choice(cand, nst, replace=False))
    ftar = np.sort(np.exp(rng.uniform(np.log(80), np.log(800), nst)))
    kmem = (int(design['slot']) if 'slot' in design else int(np.random.default_rng([seed, 5501]).integers(nst))) if design is not None else None
    tips, tipm, member = [], [], None
    lo, hi = X.min(0), X.max(0)
    for k, (nb, ft) in enumerate(zip(bases, ftar)):
        p = X[nb]; dirs = []
        for kk in range(3):
            if abs(p[kk] - hi[kk]) < 1e-6: dirs.append(np.eye(3)[kk])
            if abs(p[kk] - lo[kk]) < 1e-6 and kk < 2: dirs.append(-np.eye(3)[kk])
        if not dirs:
            dirs = [-np.eye(3)[2]]
        d = dirs[rng.integers(len(dirs))]; L = rng.uniform(.06, .14); m = np.exp(rng.uniform(np.log(.05), np.log(.3)))
        I = m * (fm.TWO_PI * ft) ** 2 * L ** 3 / (3 * fm.E_AL); bs = (I / 0.0492) ** 0.25
        if k != kmem:
            ids = fr.line(p, p + L * d, fm.sq_tube(bs, 0.1 * bs), n=3, group='stalk')
            tips.append(ids[-1]); tipm.append(m); fr.add_mass(ids[-1], m, m * 0.015 ** 2 / 6)
            continue
        # the member: stem along d, crossbar along the strip axis a, posts along d, strip between the clamps
        ax = int(np.flatnonzero(np.abs(d) > 0.5)[0]); perp = [i for i in range(3) if i != ax]
        mrng = np.random.default_rng([seed, 5502, k])          # the orientation belongs to the place
        a = np.eye(3)[perp[mrng.integers(2)]]
        tdir = d.copy() if mrng.integers(2) == 0 else np.eye(3)[[i for i in perp if np.abs(a[i]) < 0.5][0]]
        Ls = design['L']; e0 = len(fr.el)
        c0 = p + H_STEM * d; ids = fr.line(p, c0, BRACKET_SEC, n=1, group='bracket')
        e1, e2 = c0 - Ls / 2 * a, c0 + Ls / 2 * a
        fr.line(c0, e1, BRACKET_SEC, n=1, group='bracket'); fr.line(c0, e2, BRACKET_SEC, n=1, group='bracket')
        k1n, k2n = e1 + H_POST * d, e2 + H_POST * d
        fr.line(e1, k1n, BRACKET_SEC, n=1, group='bracket'); fr.line(e2, k2n, BRACKET_SEC, n=1, group='bracket')
        for e in fr.el[e0:]:
            e.update(E=E_BR, rho=RHO_BR)
        ids_m = add_member(fr, c0 + H_POST * d, a, tdir, design, [k1n, k2n])
        mid = ids_m['center']; Mm = design['M']
        tips.append(mid); tipm.append(Mm)
        member = dict(component=k, base=int(nb), mid=int(mid), clamps=ids_m['clamps'], thin=tdir.tolist(), axis=a.tolist(), across=np.cross(a, tdir).tolist(),
                      out=d.tolist(), elements=list(range(e0, len(fr.el))), strip_elements=[i for i in range(e0, len(fr.el)) if fr.el[i]['group'] == 'strip'],
                      design=_plain({k_: v for k_, v in design.items() if k_ != 'fe'}), f_target_replaced=float(ft))
    lumped = list(rng.choice([n for n in cand if n not in bases], int(rng.integers(2, 5)), replace=False))
    for n in lumped:
        fr.add_mass(n, rng.uniform(.3, 1.5), 1e-4)
    if unit_rng is not None and pert > 0:
        mem_el = set(member['elements']) if member else set(); dummy = False
        for i, e in enumerate(fr.el):
            if i in mem_el:
                if not dummy:          # the replaced stalk's three draws, at its place in the sequence
                    unit_rng.standard_normal(3); dummy = True
                continue
            e['Efac'] *= 1 + 0.03 * pert * unit_rng.standard_normal()
        for n, m in zip(tips, tipm):
            dm = m * 0.03 * pert * unit_rng.standard_normal()
            fr.add_mass(n, dm, (dm / Mm) * np.asarray(fr.lump[n][1]) if (member and n == member['mid']) else dm * 0.015 ** 2 / 6)
        for i in sorted(mem_el):       # the member's own, after everything the stopped study drew
            fr.el[i]['Efac'] *= 1 + 0.03 * pert * unit_rng.standard_normal()
    fr.meta = dict(chassis=chassis, mounts=mounts, corners=corners, bases=bases, tips=tips, lumped=lumped, f_target=ftar,
                   dims=(Lx, Ly, Lz), member=member)
    return fr


def selftest(seed=0):
    out = {}
    for u in (None, 1, 2):
        ra = None if u is None else np.random.default_rng([seed, u, 1]); rb = None if u is None else np.random.default_rng([seed, u, 1])
        a = fm.make_article(seed, ra, 0 if u is None else 1); b = make_article_m(seed, rb, 0 if u is None else 1, None)
        Ka, Ma = a.assemble(); Kb, Mb = b.assemble()
        out[f'no_member_unit_{u}'] = bool(np.array_equal(Ka, Kb) and np.array_equal(Ma, Mb) and np.array_equal(a.stress_matrix(), b.stress_matrix()))
    return out


# ------------------------------------------------------------------ energy, member motion
def element_energy(fr, Phi, off=0):
    """(n_el, nm): phi_e^T K_e phi_e per element of frame fr, whose DOFs start at row `off` of Phi."""
    X = np.array(fr.xyz); E = np.zeros((len(fr.el), Phi.shape[1]))
    for ie, e in enumerate(fr.el):
        L = np.linalg.norm(X[e['j']] - X[e['i']]); T = fm._rot(X[e['i']], X[e['j']], e.get('vref'))
        kl, _ = fm._local_KM(e['sec'], L, E=e.get('E', fm.E_AL) * e['Efac'], rho=e.get('rho', fm.RHO_AL))
        d = fr._edofs(e) + off; P = Phi[d]; E[ie] = np.einsum('im,ij,jm->m', P, T.T @ kl @ T, P)
    return E


def shares(art, md, K, host=None):
    """Per mode: e_J (article elements outside the strip), e_H (the strip), e_art (all article elements), closure.
    K is the full stiffness of the configuration (suspension springs or host coupling included)."""
    Ea = element_energy(art, md.Phi); total = np.einsum('im,ij,jm->m', md.Phi, K, md.Phi)
    strip = np.zeros(len(art.el), bool)
    if art.meta.get('member'):
        strip[art.meta['member']['strip_elements']] = True
    eJ, eH = Ea[~strip].sum(0), Ea[strip].sum(0)
    out = dict(e_J=eJ / total, e_H=eH / total, e_art=Ea.sum(0) / total, total=total)
    if host is not None:
        na = 6 * art.nn; Eh = element_energy(host, md.Phi, na).sum(0)
        out['e_host'] = Eh / total; out['e_springs'] = 1 - (Ea.sum(0) + Eh) / total
    return out


def member_motion(art, Phi):
    """Per unit modal displacement: mid-span motion relative to the mean of the two clamps, in the thin direction."""
    m = art.meta['member']
    if m is None:
        return np.zeros(Phi.shape[1])
    t = np.asarray(m['thin']); tr = lambda n: Phi[6 * n:6 * n + 3]
    return t @ (tr(m['mid']) - np.mean([tr(c) for c in m['clamps']], axis=0))


# ------------------------------------------------------------------ the simulator's envelope over RMS
_KAPPA = None


def kappa_table(decay_cycles=DECAY_CYCLES, n=1 << 21, spp=32, seed=20261114):
    """Mean envelope over RMS (k1) and RMS envelope over RMS (k2) of the simulator's follower (jointed.rs: a peak
    follower of sqrt(q^2 + (q'/w)^2), decaying by exp(-1) in decay_cycles periods), for a mode of damping zeta under
    white noise. Stationary Gaussian q from the SDOF filter by FFT; the follower by a cumulative maximum in log form."""
    rng = np.random.default_rng(seed); zs = np.geomspace(0.002, 0.4, 14); out = []
    f = np.fft.rfftfreq(n, 1 / spp); Wn = rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size)
    c = 1.0 / (decay_cycles * spp)
    for z in zs:
        H = 1 / (1 - f ** 2 + 2j * z * f); q = np.fft.irfft(H * Wn, n); qd = np.fft.irfft(2j * np.pi * f * H * Wn, n) / (2 * np.pi)
        a = np.sqrt(q ** 2 + qd ** 2); k = np.arange(n) * c
        env = np.exp(np.maximum.accumulate(np.log(a) + k) - k)[n // 8:]; s = q[n // 8:].std()
        out.append((env.mean() / s, np.sqrt((env ** 2).mean()) / s))
    return zs, np.array(out)


POWER_WEIGHTED = os.environ.get('OBS_POWER_WEIGHTED', '1') == '1'   # V2's sizing; 0 gives part 4's first exports
KAPPA_MODE = os.environ.get('OBS_KAPPA', 'v1')   # the study's ruling after V1: the study's table (V1_Stepper_And_Rule, section 6)
# V1 section 6, decay_cycles 5, 8192 samples a second: mean envelope over RMS modal displacement, by damping and frequency
V1_ZETA = np.array([0.005, 0.015, 0.05]); V1_F = np.array([100.0, 1000.0])
V1_K1 = np.array([[1.348, 1.317], [1.469, 1.409], [1.707, 1.578]])
SINE_OVER_RMS = 2.0     # the rise's reference is a sine amplitude, 2 x RMS (V1 section 5; the study's ruling 2)


# ---- the power-weighted rule (the study's ruling after V2, section 3.3): the spectrum sees the rule averaged over the envelope
# with weight envelope^2. W(r) = E[e^2 min(1, r e)] / E[e^2], e the envelope over its mean, r the mean-envelope ratio; for
# r well under 1 it is rho1 r with rho1 = E[env^3] / (E[env^2] E[env]). Tabulated from this file's simulation of the
# simulator's follower (kappa_table's method) at 100 and 1000 Hz (8192 samples a second), decay_cycles 5; it reproduces
# V2's rho1 within 1.3 percent (1.386, 1.310, 1.205 against V's 1.402, 1.318, 1.203 at 100 Hz) and V1's mean envelope
# within 0.5 percent.
_W = None
W_ZETA = np.geomspace(0.003, 0.4, 12); W_F = np.array([100.0, 1000.0]); W_R = np.geomspace(1e-3, 1e2, 81)


def w_table(n=1 << 21, seed=20261114):
    path = os.path.join(BULK, 'checks', 'w_table.npz')
    if os.path.exists(path):
        z = np.load(path); return z['W'], z['rho1']
    W = np.zeros((W_ZETA.size, W_F.size, W_R.size)); rho1 = np.zeros((W_ZETA.size, W_F.size))
    for j, f0 in enumerate(W_F):
        spp = 8192.0 / f0; rng = np.random.default_rng(seed); fr = np.fft.rfftfreq(n, 1 / spp)
        Wn = rng.standard_normal(fr.size) + 1j * rng.standard_normal(fr.size); c = 1.0 / (DECAY_CYCLES * spp)
        for i, z in enumerate(W_ZETA):
            H = 1 / (1 - fr ** 2 + 2j * z * fr); q = np.fft.irfft(H * Wn, n); qd = np.fft.irfft(2j * np.pi * fr * H * Wn, n) / (2 * np.pi)
            a = np.sqrt(q ** 2 + qd ** 2); k = np.arange(n) * c; env = np.exp(np.maximum.accumulate(np.log(a) + k) - k)[n // 8:]
            e = env / env.mean(); e2 = e * e; m2 = e2.mean(); rho1[i, j] = float((e2 * e).mean() / m2)
            W[i, j] = [float((e2 * np.minimum(1.0, r * e)).mean() / m2) for r in W_R]
    os.makedirs(os.path.dirname(path), exist_ok=True); np.savez(path, W=W, rho1=rho1, zeta=W_ZETA, f=W_F, r=W_R); return W, rho1


def wpow(r, zeta, f):
    """The power-weighted saturating factor W(r) per mode (bilinear in log damping and log frequency, log r)."""
    global _W
    if _W is None:
        _W = w_table()
    W, rho1 = _W; r = np.asarray(r, float); z = np.clip(np.asarray(zeta, float), W_ZETA[0], W_ZETA[-1]); f = np.clip(np.asarray(f, float), W_F[0], W_F[-1])
    lz = np.interp(np.log(z), np.log(W_ZETA), np.arange(W_ZETA.size)); i0 = np.minimum(np.floor(lz).astype(int), W_ZETA.size - 2); tz = lz - i0
    tf = (np.log(f) - np.log(W_F[0])) / (np.log(W_F[1]) - np.log(W_F[0]))
    lr = np.log(np.clip(r, W_R[0], W_R[-1])); out = np.zeros(r.shape)
    for di, wz in ((0, 1 - tz), (1, tz)):
        for dj, wf in ((0, 1 - tf), (1, tf)):
            tab = W[i0 + di, dj]                                      # (modes, nr)
            out += wz * wf * np.array([np.interp(lr[m], np.log(W_R), tab[m]) for m in range(r.size)])
    small = r < W_R[0]                                                # linear below the table: rho1 r
    if small.any():
        out[small] = (out[small] / W_R[0]) * r[small]
    return np.where(r > 0, out, 0.0)


def kappa_v1(zeta, f):
    """V's mean envelope over RMS, bilinear in log damping and log frequency, held flat outside the table (damping
    0.5 to 5 percent, 100 to 1000 Hz)."""
    lz = np.clip(np.log(np.asarray(zeta, float)), np.log(V1_ZETA[0]), np.log(V1_ZETA[-1]))
    lf = np.clip(np.log(np.maximum(np.asarray(f, float), 1e-9)), np.log(V1_F[0]), np.log(V1_F[-1]))
    tf = (lf - np.log(V1_F[0])) / (np.log(V1_F[1]) - np.log(V1_F[0]))
    lo = np.interp(lz, np.log(V1_ZETA), V1_K1[:, 0]); hi = np.interp(lz, np.log(V1_ZETA), V1_K1[:, 1])
    return lo + tf * (hi - lo)


def kappa(zeta, f=None):
    """(mean envelope over RMS for the drop and damping terms, sine amplitude over RMS for the rise term) per mode.
    Default 'v1': V's table and 2 (the study's rulings 1 and 2 after V1). 'stopped': sqrt 2 for both (the first exports of
    part 2). 'follower': kappa_table() (this file's own simulation of the follower, by damping) and 2."""
    global _KAPPA
    z = np.asarray(zeta, float)
    if KAPPA_MODE == 'stopped':
        k = np.full(z.shape, xe.ENV_OVER_RMS); return k, k.copy()
    if KAPPA_MODE == 'v1':
        return kappa_v1(z, f if f is not None else np.full(z.shape, 316.0)), np.full(z.shape, SINE_OVER_RMS)
    if _KAPPA is None:
        _KAPPA = kappa_table()
    zs, K = _KAPPA; lz = np.log(np.clip(zeta, zs[0], zs[-1]))
    return np.interp(lz, np.log(zs), K[:, 0]), np.full(z.shape, SINE_OVER_RMS)


# ------------------------------------------------------------------ setup of a seed and a structure
def k_fields(seed, u, fn):
    """The stopped study's seeded fields over frequency (x3.mode_factors form A with e = 1): k1, k2 per mode."""
    import x3_banded_spec_study as x3
    md = type('M', (), {})(); md.fn = np.asarray(fn)
    return x3.mode_factors(md, np.ones(md.fn.size), seed, u, 'A', 'lab')


def setup(seed, design, rig_design='same', scale=1.0):
    """x1b.setup with the member: units 1 and 2 (A and B), host, damping fields, rig (from the nominal article with
    rig_design; 'same' = design), service models (scaled forces) and the lab model of unit B."""
    art_nom = make_article_m(seed, None, 0, design)
    units = {u: make_article_m(seed, np.random.default_rng([seed, u, PERT]), PERT, design) for u in (1, 2)}
    host = fm.make_host(seed, art_nom.meta['dims']); zbase = fm.DampingField(seed)
    zs = {u: zbase.unit(np.random.default_rng([seed, u, PERT, 9]), 0.3 * PERT) for u in (1, 2)}
    art_rig = art_nom if rig_design == 'same' else make_article_m(seed, None, 0, rig_design)
    held, ctrls, shakers = x1.choose_rig(seed, art_rig, fm.Modal(*fm.suspend(art_rig), zbase)); m = art_rig.meta
    tips = np.array([6 * n + d for n in m['tips'] for d in range(3)])
    mnt = np.array([6 * n + d for n in m['mounts'] for d in range(3)])
    U = np.unique(np.concatenate([held, tips, mnt] + list(ctrls.values())))
    pos = {d: i for i, d in enumerate(U)}; ix = lambda dofs: np.array([pos[d] for d in dofs])
    serv = {}
    for u in (1, 2):
        K, M, na = fm.couple(units[u], host); md = fm.Modal(K, M, zs[u])
        dofs, Fin = x1.service_forces(seed, units[u], host, na)
        sh = shares(units[u], md, K, host); member_floor(md, sh); serv[u] = dict(md=md, K=K, na=na, dofs=dofs, Fin=Fin * scale, S=units[u].stress_matrix(), sh=sh,
                                                         phx=member_motion(units[u], md.Phi), k=k_fields(seed, u, md.fn))
    Kl, Ml = fm.suspend(units[2]); mlab = fm.Modal(Kl, Ml, zs[2]); shl = shares(units[2], mlab, Kl); member_floor(mlab, shl)
    lab = dict(md=mlab, K=Kl, sh=shl, phx=member_motion(units[2], mlab.Phi), k=k_fields(seed, 2, mlab.fn))
    return dict(seed=seed, units=units, host=host, zs=zs, held=held, ctrls=ctrls, shakers=shakers, U=U, ix=ix, serv=serv, lab=lab,
                SB=units[2].stress_matrix() @ mlab.Phi, design=design, scale=scale)


MEMBER_ZETA_MIN = 0.01      # the study's ruling 1 (part 4): the member's own modes have at least 1 percent of critical damping


def member_floor(md, sh):
    """The member's own modes (at least half their strain energy in the strips: translation, rocking, pitch) get damping
    of at least MEMBER_ZETA_MIN, so the translation mode's half-power width is at least 2 Hz at 100 Hz."""
    own = sh['e_H'] >= 0.5; md.zeta = np.where(own, np.maximum(md.zeta, MEMBER_ZETA_MIN), md.zeta); return int(own.sum())


def service_sigma_x(S, u=1, md=None):
    sv = S['serv'][u]; md = md or sv['md']; Q = x1.modal_q(md, sv['dofs'], sv['Fin'])
    X = np.einsum('m,mrf->rf', sv['phx'], Q); return float(np.sqrt((np.abs(X) ** 2).sum() * DF))


LEVEL_UNIT = 2              # the study's ruling 1 (part 5): unit B, the article on the shakers, sets the level (was unit A)
LEVEL_MAX_G = 30.0          # the study's ruling 2 (part 5): prefer a candidate whose control level is at most this (g RMS)


def level_scale(seed):
    """The seed's force scale: unit B, linear service, realistic member, mid-span RMS = SIGMA_D."""
    S = setup(seed, designs_of(seed)['realistic']); return float(SIGMA_D / service_sigma_x(S, LEVEL_UNIT))


def level_of(seed, design_real):
    """(force scale, control level): the scale that gives unit B's member SIGMA_D in its linear service, and unit A's
    linear service response at the 6 x 12 rig's controls at that scale, median over controls, g RMS."""
    S = setup(seed, design_real, design_real, 1.0); sc = SIGMA_D / service_sigma_x(S, LEVEL_UNIT)
    sv = S['serv'][1]; Y = x1.acc(sv['md'], S['U'], x1.modal_q(sv['md'], sv['dofs'], sv['Fin'])); c = S['ix'](S['ctrls'][12])
    g = np.sqrt((np.abs(Y[c]) ** 2).sum((1, 2)) * DF) * sc / 9.80665
    return float(sc), float(np.median(g))


# ------------------------------------------------------------------ placing and tuning the member (the study's ruling 1, part 3)
SHARE_MIN = 0.8                            # the member's translation mode keeps at least this share of its strain energy in the strips
F_TUNE = np.arange(90.0, 160.01, 5.0)      # the stated range of tuned frequencies
F_WIDE = np.arange(80.0, 250.01, 5.0)      # the study's ruling (part 4): tried only when F_TUNE gives no candidate


def member_shares(seed, design):
    """The member's translation mode (the mode with the largest mid-span motion relative to the clamps) in unit A and
    unit B, in service and on the lab suspension: its frequency and its strain-energy share in the strips."""
    host = fm.make_host(seed, fm.make_article(seed).meta['dims']); zbase = fm.DampingField(seed); out = {}
    for u in (1, 2):
        art = make_article_m(seed, np.random.default_rng([seed, u, PERT]), PERT, design)
        zs = zbase.unit(np.random.default_rng([seed, u, PERT, 9]), 0.3 * PERT)
        K, M, na = fm.couple(art, host); md = fm.Modal(K, M, zs); sh = shares(art, md, K); ph = np.abs(member_motion(art, md.Phi)); m = int(ph.argmax())
        out[f'service_{"AB"[u - 1]}'] = dict(f_hz=float(md.fn[m]), e_H=float(sh['e_H'][m]))
        Kl, Ml = fm.suspend(art); ml = fm.Modal(Kl, Ml, zs); shl = shares(art, ml, Kl); ph = np.abs(member_motion(art, ml.Phi)); m = int(ph.argmax())
        out[f'lab_{"AB"[u - 1]}'] = dict(f_hz=float(ml.fn[m]), e_H=float(shl['e_H'][m]))
    out['min_e_H'] = min(v['e_H'] for v in out.values() if isinstance(v, dict)); return out


def choose_member(seed, path=None):
    """The first (place, tuned frequency) whose member keeps SHARE_MIN in its translation mode in both units, in service
    and on the lab suspension, at both tiers. Places in the seed's order, frequencies of F_TUNE in the seed's order.
    Written to choices/s<seed>.json; read from there when it exists. None if no candidate passes (the seed stops)."""
    path = path or os.path.join(BULK, CHOICES, f's{seed}.json')
    if os.path.exists(path):
        return json.load(open(path))
    if seed in AUDIT_SEEDS and not ALLOW_AUDIT:
        raise SystemExit('seeds 20 to 29 belong to the audit (pass --audit as the auditor)')
    old = {}                                   # share rows already computed by an earlier search (the shares do not depend on the level)
    for p in glob.glob(os.path.join(BULK, 'choices*', f's{seed}.json')):
        for r in json.load(open(p)).get('tried', []):
            old[(int(r['slot']), float(r['f_tuned']))] = r
    nst = len(fm.make_article(seed).meta['bases'])
    slots = [int(k) for k in np.random.default_rng([seed, 5501]).permutation(nst)]
    freqs = [float(f) for f in np.random.default_rng([seed, 5503]).permutation(F_TUNE)]
    tried = []; chosen = None; passing = []
    wide = [float(f) for f in np.random.default_rng([seed, 5504]).permutation(F_WIDE) if f not in F_TUNE]
    for k, f in [(k, f) for k in slots for f in freqs] + [(k, f) for k in slots for f in wide]:
        if (k, f) in old:
            row = {kk: v for kk, v in old[(k, f)].items() if kk not in ('scale', 'level_g')}
        else:
            row = dict(slot=k, f_tuned=f)
            for tier, sd in (('realistic', REAL), ('strong', STRONG)):
                d = dict(member_design(sd, f, tier), slot=k); row[tier] = member_shares(seed, d)
            row['min_e_H'] = min(row['realistic']['min_e_H'], row['strong']['min_e_H'])
        tried.append(row)
        if row['min_e_H'] >= SHARE_MIN:
            row['scale'], row['level_g'] = level_of(seed, dict(member_design(REAL, f, 'realistic'), slot=k)); passing.append(row)
        print(f'seed {seed}: place {k}, {f:.0f} Hz: least share {row["min_e_H"]:.3f}' + (f', level {row["level_g"]:.1f} g' if 'level_g' in row else ''),
              file=sys.stderr, flush=True)
        if 'level_g' in row and row['level_g'] <= LEVEL_MAX_G:
            chosen = row; break
    level_flag = False
    if chosen is None and passing:             # no passing candidate at or under LEVEL_MAX_G: the lowest level, flagged
        chosen = min(passing, key=lambda r: r['level_g']); level_flag = True
    res = dict(level_flag=level_flag, level_max_g=LEVEL_MAX_G, level_unit='B', widened=bool(chosen and chosen['f_tuned'] not in F_TUNE), seed=seed, rule=f'translation-mode share in the strips >= {SHARE_MIN} in units A and B, service and lab suspension, both tiers',
               slots_order=slots, freqs_order=freqs, chosen=chosen, tried=tried)
    os.makedirs(os.path.dirname(path), exist_ok=True); json.dump(res, open(path, 'w'), indent=1, default=float); return res


def designs_of(seed):
    """The seed's members, both tiers, at the chosen place and frequency. Raises if the seed has no candidate."""
    ch = choose_member(seed)['chosen']
    if ch is None:
        raise RuntimeError(f'seed {seed}: no place and tuned frequency in {F_WIDE[0]:.0f} to {F_WIDE[-1]:.0f} Hz keeps the member mode at {SHARE_MIN}; seed stopped')
    return {t: dict(member_design(sd, ch['f_tuned'], t), slot=ch['slot']) for t, sd in (('realistic', REAL), ('strong', STRONG))}


# ------------------------------------------------------------------ the rule and the fixed point
def rule_state(md0, sh, phx, kk, sigma, zeta_now, law, sizes, mask, effects=('soft', 'hard')):
    """Frequency factor and damping factor per mode for RMS modal displacements sigma, and the per-mode quantities."""
    k1e, k2e = kappa(zeta_now, md0.fn); A1 = k1e * sigma
    eJ, eH = np.clip(sh['e_J'], 0, 1), np.clip(sh['e_H'], 0, 1); w0 = md0.wn
    u = np.sqrt(eJ / 2) * w0 * A1; xr = np.abs(phx) * sigma / SIGMA_D
    rJ, rr, sat = ratios(u, xr, k1e, k2e, sizes)
    drop = np.zeros_like(w0); rise = np.zeros_like(w0); dz = np.zeros_like(w0); dcoef = np.zeros_like(w0)
    pos = rr > 0; rrs = np.where(pos, rr, 1.0)
    if 'soft' in effects:
        dcoef = np.where(pos, sizes['D'] * kk[0] * eJ * rJ / rrs, 0.0)     # the file's drop field (its value at saturation)
        dz = dz + sizes['G'] * kk[1] * eJ * rJ
    if 'hard' in effects:
        rise = eH * law['R'] * np.minimum(law['c'], xr) ** 2
        dz = dz + law['GH'] * eH * xr
    gain = np.where(pos, dz / rrs, 0.0)                                 # the file's damping gain (cap 1)
    if law is not None and 'drop_max' in law:                           # the tails of the seeded fields clipped (part 4)
        dcoef = np.minimum(dcoef, law['drop_max']); gain = np.minimum(gain, law['gain_max'])
    Wr = wpow(rr, zeta_now, md0.fn) if POWER_WEIGHTED else np.minimum(1.0, rr)
    drop = dcoef * Wr                                                   # what the spectrum shows (V2 section 3.3)
    zf = 1 + gain * Wr                                                  # damping_cap = 1 (the study's ruling 3, part 3)
    drop, rise, zf = np.where(mask, drop, 0), np.where(mask, rise, 0), np.where(mask, zf, 1)
    return 1 - drop + rise, zf, dict(u=u, xr=xr, drop=drop, rise=rise, k1env=k1e, k2env=k2e, drop_field=dcoef, gain_field=gain)


def ratios(u, xr, k1e, k2e, sizes):
    """(rJ, r, sat): the softening amplitude ratio u/u_ref, the file's amplitude ratio r at the mean envelope (its
    reference is the smaller of A_J and A_H2, so r = max(rJ, rH) with rH = k1env x_r / sine-over-RMS), and
    min(1, r) / r: the factor by which the drop and the capped damping stop growing past r = 1."""
    rJ = u / sizes['u_ref'] if sizes else np.zeros_like(u); rH = k1e * xr / k2e; r = np.maximum(rJ, rH)
    sat = np.where(r > 0, np.minimum(1.0, r) / np.where(r > 0, r, 1.0), 1.0)
    return rJ, r, sat


def size_softening(md0, sh, kk, sigma, zeta_now, tier, art_mask, phx=None):
    """D, u_ref, G so the median article mode drops d with damping times g and sits at 2/3 of saturation (damping
    capped at r = 1 as the drop)."""
    k1e, k2e = kappa(zeta_now, md0.fn); eJ = np.clip(sh['e_J'], 0, 1); u = np.sqrt(eJ / 2) * md0.wn * k1e * sigma
    u_ref = 1.5 * np.median(u[art_mask]); xr = np.abs(phx) * sigma / SIGMA_D if phx is not None else np.zeros_like(u)
    rJ, rr, sat = ratios(u, xr, k1e, k2e, dict(u_ref=u_ref))
    if POWER_WEIGHTED:
        sat = np.where(rr > 0, wpow(rr, zeta_now, md0.fn) / np.where(rr > 0, rr, 1.0), 1.0)
    D = tier['d'] / np.median((kk[0] * eJ * rJ * sat)[art_mask]); G = (tier['g'] - 1) / np.median((kk[1] * eJ * rJ * sat)[art_mask])
    return dict(D=float(D), u_ref=float(u_ref), G=float(G))


def modal_sigma(Q):
    return np.sqrt((np.abs(Q) ** 2).sum((1, 2)) * DF)


def fixed_point(md0, sh, phx, kk, qfun, law, sizes, mask, art_mask=None, tier=None, effects=('soft', 'hard'), tol=1e-9, itmax=200, relax=1.0):
    """Iterate: modal RMS under the forcing at the current state, the rule, the new state. With tier and art_mask the
    softening is re-sized at every step (service of the unit that owns the sizes). Returns (md, sizes, info)."""
    md = copy.copy(md0); hist = []
    for it in range(itmax):
        sig = modal_sigma(qfun(md))
        if tier is not None:
            sizes = size_softening(md0, sh, kk, sig, md.zeta, tier, art_mask, phx)
        ff, zf, q = rule_state(md0, sh, phx, kk, sig, md.zeta, law, sizes, mask, effects)
        fn_new = md0.fn * ff; z_new = np.minimum(0.9, md0.zeta * zf)
        if relax != 1.0:     # under-relaxed step (the lab on a square rig oscillates at full step)
            fn_new = md.fn + relax * (fn_new - md.fn); z_new = md.zeta + relax * (z_new - md.zeta)
        dmax = float(np.max((np.abs(fn_new - md.fn) / md0.fn)[mask])); hist.append(dmax)
        md.fn = fn_new; md.wn = 2 * np.pi * fn_new; md.zeta = z_new
        if dmax < tol:
            break
    sig = modal_sigma(qfun(md))
    ffac = np.where(mask, md.fn / np.where(mask, md0.fn, 1.0), 1.0)
    return md, sizes, dict(iterations=it + 1, last_step=hist[-1], converged=bool(hist[-1] < tol), relax=relax, sigma=sig, q=q, ffac=ffac, zfac=md.zeta / md0.zeta)


def law_of(tier_name):
    t = TIER[tier_name]; d = DESIGN[tier_name]
    return dict(R=float(d['rise_el'](SIGMA_D)) if t['R'] is None else t['R'], c=t['c'], GH=t['GH'], kappa=KAPPA_MODE,
                drop_max=CLIP_DROP * t['d'], gain_max=CLIP_DAMP * t['g'] - 1.0)


def effects_of(variant):
    if variant == 'none':
        return ()
    return {'soft': ('soft',), 'hard': ('hard',)}.get(variant.split('_')[1], ('soft', 'hard')) if '_' in variant else ('soft', 'hard')


def tier_of(variant):
    return None if variant == 'none' else variant.split('_')[0]


def service(S, u, law, tier_name, effects, sizes=None):
    """The service fixed point of unit u: (md, sizes, info, Q). Sizes of unit u from its own full-effect fixed point."""
    sv = S['serv'][u]; md0 = sv['md']; qf = lambda md: x1.modal_q(md, sv['dofs'], sv['Fin'])
    mask = md0.fn > 1.0
    if not effects:
        return md0, None, dict(iterations=0, sigma=modal_sigma(qf(md0)), ffac=np.ones(md0.fn.size), zfac=np.ones(md0.fn.size)), qf(md0)
    art = (sv['sh']['e_art'] >= 0.5) & (md0.fn >= 20) & (md0.fn <= 2000) & mask
    if sizes is None:
        _, sizes, _ = fixed_point(md0, sv['sh'], sv['phx'], sv['k'], qf, law, None, mask, art, TIER[tier_name], ('soft', 'hard'))
    md, _, info = fixed_point(md0, sv['sh'], sv['phx'], sv['k'], qf, law, sizes, mask, effects=effects)
    info['article_modes'] = int(art.sum()); return md, sizes, info, qf(md)


# ------------------------------------------------------------------ one export
def build(seed, ns, nc, variant, scale=None, write=True):
    t0 = time.time(); tname = tier_of(variant); eff = effects_of(variant)
    DS = designs_of(seed); design = DS['realistic'] if tname in (None, 'realistic') else DS['strong']
    scale = level_scale(seed) if scale is None else scale
    S = setup(seed, design, rig_design=DS['realistic'], scale=scale); law = law_of(tname) if tname else None
    ix = S['ix']; U = S['U']; c = ix(S['ctrls'][nc]); held = ix(S['held']); shk = S['shakers'][(ns, nc)]
    tips = ix(np.array([6 * n + d for n in S['units'][2].meta['tips'] for d in range(3)]))
    # service, unit A (targets) and unit B (truth); sizes from each unit's own full-effect fixed point
    out = {}
    for u in (1, 2):
        md, sizes, info, Q = service(S, u, law, tname, eff)
        if tname and eff != ('soft', 'hard'):      # a diagnostic keeps the full variant's sizes
            _, sizes_full, _, _ = service(S, u, law, tname, ('soft', 'hard'))
            md, sizes, info, Q = service(S, u, law, tname, eff, sizes_full)
        sv = S['serv'][u]; Y = x1.acc(md, U, Q)
        out[u] = dict(md=md, sizes=sizes, info=info, Y=Y, truth=x1.element_truth(*x1.moments(sv['S'] @ md.Phi[:sv['na']], Q)),
                      sigma_x=float(np.sqrt((np.abs(np.einsum('m,mrf->rf', sv['phx'], Q)) ** 2).sum() * DF)))
    lin = {u: x1.acc(S['serv'][u]['md'], U, x1.modal_q(S['serv'][u]['md'], S['serv'][u]['dofs'], S['serv'][u]['Fin'])) for u in (1, 2)}
    YA = out[1]['Y']; YAc = YA[c].transpose(2, 0, 1)
    mlab = S['lab']['md']
    ctx = dict(S=dict(units=S['units']), seed=seed, ns=ns, nc=nc, tier=variant, shk=shk, c=c, held=held, tips=tips, mlab=mlab, SB=S['SB'], U=U,
               YA=YA, GA=YAc @ xt.H_(YAc), GAU=auto(YA), GBU=auto(out[2]['Y']), truthB=out[2]['truth'])
    # the drive unit and the monitors from the LINEAR structure of the realistic member (one per seed and rig), so every
    # variant has the same volts; the monitors' scale from this structure's linear reference drive
    ylin = lin[1][c].transpose(2, 0, 1); ctx_lin = dict(ctx, GA=ylin @ xt.H_(ylin))
    if design is DS['realistic']:
        Dlin = xt.line_drive(ctx_lin)
    else:
        Sr = setup(seed, DS['realistic'], rig_design=DS['realistic'], scale=scale); mdr = Sr['serv'][1]['md']
        yr = x1.acc(mdr, Sr['U'], x1.modal_q(mdr, Sr['serv'][1]['dofs'], Sr['serv'][1]['Fin']))[c].transpose(2, 0, 1)
        Dlin = xt.line_drive(dict(ctx, mlab=Sr['lab']['md'], GA=yr @ xt.H_(yr)))
    g = float(np.sqrt(np.real(np.einsum('fii->i', Dlin)) * DF).max() / xe.LINE_DRIVE_VRMS)
    Cl = xe.accel_cov(ctx, xt.line_drive(ctx_lin)); rows = U[np.r_[c, held, tips]]; osh = mlab.Phi[rows].T
    ctrl_rms = float(np.median(np.sqrt(np.einsum('mc,mn,nc->c', osh[:, :nc], Cl, osh[:, :nc]))))
    nm = mlab.fn.size; m_scale = xe.MON_LEVEL * ctrl_rms / np.sqrt(np.diag(Cl))
    full_out = np.hstack([osh, np.diag(m_scale)]); dshape = mlab.Phi[shk].T * g
    outputs = ([dict(name=f'c{i + 1:02d}', role='control') for i in range(c.size)] + [dict(name=f'h{i + 1:02d}', role='held_out') for i in range(held.size)]
               + [dict(name=f't{i + 1:02d}', role='held_out') for i in range(tips.size)] + [dict(name=f'm{m:03d}', role='monitor') for m in range(nm)])
    base = dict(n_outputs=int(full_out.shape[1]), n_drives=int(ns), feedthrough=0.0, residual=[], transport_delay_samples=0)
    elastic = np.arange(nm) >= 6
    # the lab: full level on the shakers under the reference drive (line drive on the lab state, to unit A's service)
    lab = S['lab']; kk = lab['k']; info_lab = None
    if eff:
        def qlab(md):
            D = xt.line_drive(dict(ctx, mlab=md)); return (md.Phi[shk].T[None] @ xt.factor(D)).transpose(1, 2, 0) * md.h(F)[:, None, :]
        sizesB = out[2]['sizes']
        md_full, _, info_lab = fixed_point(mlab, lab['sh'], lab['phx'], kk, qlab, law, sizesB, elastic, effects=eff, relax=LAB_RELAX, itmax=600)
        if not info_lab['converged']:      # part 5: one square-rig case oscillated at 0.3; a smaller step, longer
            md_full, _, info_lab = fixed_point(mlab, lab['sh'], lab['phx'], kk, qlab, law, sizesB, elastic, effects=eff, relax=0.1, itmax=3000)
        if not info_lab['converged']:
            print(f'WARNING: lab fixed point not converged, last step {info_lab["last_step"]:.2e}', file=sys.stderr)
        # the targets are the rule at the final state's amplitudes (consistent with the reference amplitudes even if not converged)
        ff_t, zf_t, q_t = rule_state(mlab, lab['sh'], lab['phx'], kk, info_lab['sigma'], md_full.zeta, law, sizesB, elastic, eff)
        info_lab.update(ffac=np.where(elastic, ff_t, 1.0), zfac=np.where(elastic, np.minimum(0.9, mlab.zeta * zf_t) / mlab.zeta, 1.0), q=q_t)
        k1e, k2e = kappa(md_full.zeta, mlab.fn)
        eJ, eH = np.clip(lab['sh']['e_J'], 0, 1), np.clip(lab['sh']['e_H'], 0, 1); w0 = mlab.wn; ph = np.abs(lab['phx'])
        AJ = sizesB['u_ref'] / np.maximum(np.sqrt(eJ / 2) * w0, 1e-300)
        AH2 = np.where(ph > 0, k2e * SIGMA_D / np.maximum(ph, 1e-300), np.inf); AH1 = np.where(ph > 0, k1e * SIGMA_D / np.maximum(ph, 1e-300), np.inf)
        soft, hard = 'soft' in eff, 'hard' in eff
        modes = []; ref = np.zeros(nm); clip_d = np.zeros(nm, bool); clip_g = np.zeros(nm, bool)
        for m in range(nm):
            lv = dict(drop=0.0, drop_exponent=1.0, rise=0.0, rise_exponent=2.0, rise_cap=1.0, damping_gain=0.0, damping_exponent=1.0, damping_cap=1.0)
            if not elastic[m]:
                ref[m] = 1e30
            else:
                use_h = bool(np.isfinite(AH2[m]) and AH2[m] < AJ[m]); ref[m] = AH2[m] if use_h else AJ[m]   # the smaller reference: no field above its physical size
                Dk = sizesB['D'] * kk[0][m] * eJ[m] if soft else 0.0; Gk = sizesB['G'] * kk[1][m] * eJ[m] if soft else 0.0
                lv['drop'] = float(Dk * ref[m] / AJ[m]) if use_h else float(Dk)
                if hard and np.isfinite(AH2[m]):
                    lv['rise'] = float(eH[m] * law['R'] * (ref[m] / AH2[m]) ** 2); lv['rise_cap'] = float(law['c'] * AH2[m] / ref[m])
                lv['damping_gain'] = float(Gk * ref[m] / AJ[m] + ((law['GH'] * eH[m] * ref[m] / AH1[m]) if hard and np.isfinite(AH1[m]) else 0.0))
                clip_d[m] = lv['drop'] > law['drop_max']; clip_g[m] = lv['damping_gain'] > law['gain_max']
                lv['drop'] = min(lv['drop'], law['drop_max']); lv['damping_gain'] = min(lv['damping_gain'], law['gain_max'])
                if lv['rise'] == 0.0:
                    lv['rise_cap'] = 1.0
            modes.append(dict(f_hz=float(mlab.fn[m]), zeta=float(mlab.zeta[m]), drive_shape=dshape[m].tolist(), output_shape=full_out[m].tolist(),
                              reference_amplitude=float(ref[m]), level=lv))
        plant = dict(kind='jointed', **base, modes=modes, output_delays_s=[],
                     joint=dict(delta=0.0, damping_gain=0.0, freq_exponent=1.0, damping_exponent=1.0, decay_cycles=DECAY_CYCLES),
                     step=STEP)        # the study's ruling: the equalized step on every level-dependent structure
        tchange = info_lab['ffac'] - 1; tz = info_lab['zfac']
        env_over_rms = float(np.median(k1e[elastic]))
    else:
        plant = dict(kind='modal', **base, softening=None, noisy_outputs=int(rows.size), discretization=dict(kind='fitted', band_hz=2200.0, taps=9, delay=4),
                     modes=[dict(f_hz=float(mlab.fn[m]), q=float(1 / (2 * mlab.zeta[m])), drive_shape=dshape[m].tolist(), output_shape=full_out[m].tolist(),
                                 response='acceleration') for m in range(nm)])
        tchange = np.zeros(nm); tz = np.ones(nm); env_over_rms = float(np.median(kappa(mlab.zeta, mlab.fn)[0][elastic])); ref = None
    # the study plan's numbers in words
    sA, sB = out[1]['sizes'], out[2]['sizes']
    at_full = None if not eff else dict(
        words=(f'{tname} tier, effects {"+".join(eff)}: softening sized in each unit\'s service fixed point so the median article mode '
               f'drops {TIER[tname]["d"]:.0%} with damping times {TIER[tname]["g"]}, at two thirds of saturation; hardening from the hardening '
               f'member ({1e3 * design["L"]:.1f} x {1e3 * design["b"]:.1f} x {1e3 * design["t"]:.2f} mm 301 FH strip, {design["M"]} kg), '
               f'rise {law["R"]:.4f} at mid-span RMS {1e3 * SIGMA_D:.4f} mm, cap {law["c"]}, member damping gain {law["GH"]}'),
        frequency_change=-TIER[tname]['d'] if 'soft' in eff else 0.0, damping_factor=TIER[tname]['g'] if 'soft' in eff else 1.0,
        hardening_rise=law['R'] if 'hard' in eff else 0.0, member_rms_m=SIGMA_D, placeholder=False)
    level_targets = dict(frequency_change=[float(v) for v in tchange], damping_factor=[float(v) for v in tz], at_full_level=at_full,
                                  envelope_over_rms=env_over_rms,
                                  what='each mode at full level on the shakers under the reference line drive (lab fixed point): signed frequency change '
                                       '(negative a drop) and damping factor')
    gA = np.sqrt(np.real(np.einsum('fii->i', ctx['GA'])) * DF) / 9.80665
    gAlin = np.sqrt((np.abs(lin[1][c]) ** 2).sum((1, 2)) * DF) / 9.80665
    side = dict(fn=mlab.fn, zeta=mlab.zeta, drive_shape=dshape, output_shape=full_out, SB=S['SB'], rows=rows, shk=shk, U=U, c=c, held=held, tips=tips,
                n_control=c.size, n_held=held.size, n_tip=tips.size, n_phys=rows.size, n_stress=0, n_modal=nm, modal_scale=m_scale, mon_level=xe.MON_LEVEL,
                ctrl_rms=ctrl_rms, force_per_volt=g, seed=seed, ns=ns, nc=nc, change=variant, kind=plant['kind'],
                target_change=tchange, target_zfac=tz, law_keys=np.array(['env_over_rms']), law_vals=np.array([env_over_rms]),
                Phi_rows=mlab.Phi[rows], Phi_shk=mlab.Phi[shk], ndof=mlab.Phi.shape[0],
                GAU_c=ctx['GAU'][c], GAU_h=ctx['GAU'][held], GBU_h=ctx['GBU'][held], GBU_t=ctx['GBU'][tips],
                truthB_rms=out[2]['truth']['rms'], truthB_d8=out[2]['truth']['d8'], truthB_d4=out[2]['truth']['d4'],
                lab_e_J=lab['sh']['e_J'], lab_e_H=lab['sh']['e_H'], lab_e_art=lab['sh']['e_art'], lab_phx=lab['phx'], lab_k1=kk[0], lab_k2=kk[1],
                scale=scale, sigma_d=SIGMA_D, sigma_x_A=out[1]['sigma_x'], sigma_x_B=out[2]['sigma_x'],
                ctrl_g_rms=gA, ctrl_g_rms_linear=gAlin, ref_amp=ref if ref is not None else np.zeros(0))
    for u, tag in ((1, 'A'), (2, 'B')):
        sv = S['serv'][u]; side[f'serv{tag}_f0'] = sv['md'].fn; side[f'serv{tag}_ffac'] = out[u]['info']['ffac']; side[f'serv{tag}_zfac'] = out[u]['info']['zfac']
        side[f'serv{tag}_zeta0'] = sv['md'].zeta; side[f'serv{tag}_e_J'] = sv['sh']['e_J']; side[f'serv{tag}_e_H'] = sv['sh']['e_H']; side[f'serv{tag}_e_art'] = sv['sh']['e_art']
        side[f'serv{tag}_sigma'] = out[u]['info']['sigma']; side[f'serv{tag}_phx'] = sv['phx']
        if out[u]['sizes']:
            for k, v in out[u]['sizes'].items():
                side[f'sizes{tag}_{k}'] = v
    if info_lab:
        side.update(lab_ffac=info_lab['ffac'], lab_zfac=info_lab['zfac'], lab_sigma=info_lab['sigma'], lab_xr=info_lab['q']['xr'], lab_u=info_lab['q']['u'],
                    lab_iterations=info_lab['iterations'], lab_k1env=info_lab['q']['k1env'], lab_sine_over_rms=info_lab['q']['k2env'])
    meta = dict(lab_converged=None if not info_lab else info_lab['converged'], lab_last_step=None if not info_lab else info_lab['last_step'],
                seed=seed, ns=ns, nc=nc, variant=variant, design=_plain(design), choice={k: v for k, v in choose_member(seed)['chosen'].items()}, scale=scale,
                member=S['units'][2].meta['member'], law=law, sizes_A=sA, sizes_B=sB,
                service_iterations=[out[u]['info']['iterations'] for u in (1, 2)], lab_iterations=None if not info_lab else info_lab['iterations'],
                seconds=round(time.time() - t0, 1))
    service_state = None
    if eff:   # the study's ruling 2 (part 4): the state in service of unit B, the unit whose stress is the truth
        svB = S['serv'][2]; fB = svB['md'].fn; ffB, zfB = out[2]['info']['ffac'], out[2]['info']['zfac']
        mm = int(np.argmax(np.abs(svB['phx']))); art = (svB['sh']['e_art'] >= 0.5) & (fB >= 20) & (fB <= 2000) & (fB > 1.0); art[mm] = False
        q = lambda v: [float(np.percentile(v, p)) for p in (10, 50, 90)]
        service_state = dict(
            what='unit B in service at the fixed point of its level rule (the state the shakers are meant to reproduce): the member\'s '
                 'translation mode, and the article\'s other modes (at least half their strain energy in the article, 20 to 2000 Hz)',
            unit='B', variant=variant,
            member=dict(f0_hz=float(fB[mm]), frequency_change=float(ffB[mm] - 1), damping_factor=float(zfB[mm]), e_H=float(svB['sh']['e_H'][mm]),
                        mid_span_rms_m=float(out[2]['sigma_x'])),
            article=dict(modes=int(art.sum()), frequency_change_p10_p50_p90=q(ffB[art] - 1), damping_factor_p10_p50_p90=q(zfB[art])))
        meta['clipping'] = dict(lab_modes_drop_clipped=int(clip_d.sum()), lab_modes_gain_clipped=int(clip_g.sum()),
                                largest_drop_at_saturation=float(max(m['level']['drop'] for m in plant['modes'])),
                                largest_damping_factor_at_saturation=float(1 + max(m['level']['damping_gain'] for m in plant['modes'])),
                                drop_max=law['drop_max'], damping_factor_max=law['gain_max'] + 1)
        meta['service_state'] = service_state
    import frequency_bound      # set version 0.2: refuse a structure the simulator would refuse (A1 audit, item 9)
    frequency_bound.check_modes(plant['modes'], f's{seed}_{ns}x{nc}_{variant}')
    if not write:
        return plant, outputs, side, ctx, meta, S, level_targets
    folder = os.path.join(BULK, 'plants', f's{seed}_{ns}x{nc}_{variant}'); os.makedirs(folder, exist_ok=True)
    xe.write_plant_b(folder, plant, outputs, side, tag='v2')
    json.dump(level_targets, open(os.path.join(folder, 'level_targets_v2.json'), 'w'), indent=1)   # the runner refuses unknown plant keys
    if service_state is not None:
        json.dump(service_state, open(os.path.join(folder, 'service_state_v2.json'), 'w'), indent=1)
    import x4_reference as xr
    t, Gb = xr.banded(ctx)
    xr.write_reference(folder, 'ref_banded', Gb); xr.write_reference(folder, 'ref_line', ctx['GA'])
    np.savez(os.path.join(folder, 'ref_side.npz'), t=t, band_of_line=xt.BO3, tier=variant, hid='none (no converted reference, ruling 8)')
    fr = S['units'][2]
    geo = dict(nodes=[[float(v) for v in p] for p in fr.xyz], elements=[dict(i=int(e['i']), j=int(e['j']), group=e['group']) for e in fr.el],
               lumped_masses=[dict(node=int(n), kg=float(mm)) for n, (mm, _) in sorted(fr.lump.items())], mounts=[int(n) for n in fr.meta['mounts']],
               tips=[int(n) for n in fr.meta['tips']], dims_m=[float(v) for v in fr.meta['dims']], member=fr.meta['member'])
    json.dump(geo, open(os.path.join(folder, 'geometry_v2.json'), 'w'))
    json.dump(meta, open(os.path.join(folder, 'meta_v2.json'), 'w'), indent=1, default=float)
    write_case(folder, f's{seed}_{ns}x{nc}_{variant}_pinv_closed_default_line_hold_divopen')
    return folder, meta


def write_case(folder, case_id, ref='line', law='pinv_closed', setting='default', frf='hold'):
    import zlib, x4_matrix as xm
    outputs = json.load(open(os.path.join(folder, 'outputs_v2.json'))); plant = json.load(open(os.path.join(folder, 'plant_v2.json')))
    n_phys = sum(o['role'] != 'monitor' for o in outputs)
    run = dict(xm.RUN, law=dict(id=law, params=xm.law_params(law, setting, frf, plant['n_drives'])), frf_mode=xm.FRF[frf],
               sim_seed=zlib.crc32(case_id.encode()) & 0x7fffffff, response_record='autospectra', protections='default', frames_per_loop=xm.FRAMES_DEFAULT)
    if plant['kind'] == 'jointed':
        run['moments'] = dict(first=n_phys, count=len(outputs) - n_phys)
    run.update(xm.OPENED)
    case = dict(schema=SCHEMA, case_id=case_id, grid=xe.GRID, plant_file=os.path.join(folder, 'plant_v2.json'), outputs=outputs,
                reference_file=os.path.join(folder, f'ref_{ref}.json'), run=run)
    d = os.path.join(BULK, 'cases_v2'); os.makedirs(d, exist_ok=True); p = os.path.join(d, case_id + '.json'); json.dump(case, open(p, 'w')); return p


TUNING_SEEDS = (30, 31)
AUDIT_SEEDS = tuple(range(20, 30))
ALLOW_AUDIT = False          # set by --audit (the auditor only)
CHOICES = 'choices_v5'       # part 5's choices (unit B's level, the level preference); earlier searches' share rows are reused


def jobs_of(seeds):
    """the study's case set (the study's notes): every rig and stage variant on study seeds 0 to 19, the
    diagnostics on 6 x 12 and 6 x 6 for seeds 0 to 9, and on the tuning seeds 30 and 31 the linear and strong variants
    on 6 x 12 and 6 x 6 only."""
    js = []
    for s in seeds:
        if s in AUDIT_SEEDS and not ALLOW_AUDIT:
            raise SystemExit('seeds 20 to 29 belong to the audit (pass --audit as the auditor)')
        if s in TUNING_SEEDS:
            js += [(s, ns, nc, v) for ns, nc in DIAG_RIGS for v in ('none', 'strong')]; continue
        for ns, nc in RIGS:
            js += [(s, ns, nc, v) for v in VARIANTS]
            if (ns, nc) in DIAG_RIGS and s < 10:
                js += [(s, ns, nc, v) for v in DIAG]
    return js


def _one(job):
    s, ns, nc, v, scale = job
    p = subprocess.run([sys.executable, os.path.abspath(__file__), 'export', '--seed', str(s), '--ns', str(ns), '--nc', str(nc), '--variant', v,
                        '--scale', repr(scale)] + (['--audit'] if ALLOW_AUDIT else []), capture_output=True, text=True)
    return job, p.returncode, (p.stdout + p.stderr)[-600:]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('member', 'kappa', 'export', 'all', 'selftest', 'scale', 'choose', 'seeds'))
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--seeds', default='0,1'); ap.add_argument('--ns', type=int, default=6)
    ap.add_argument('--nc', type=int, default=12); ap.add_argument('--variant', default='realistic'); ap.add_argument('--scale', type=float, default=None)
    ap.add_argument('--jobs', type=int, default=4); ap.add_argument('--audit', action='store_true', help='allow seeds 20 to 29 (the auditor only)')
    ap.add_argument('--first', type=int, default=0); ap.add_argument('--count', type=int, default=20); a = ap.parse_args(); ALLOW_AUDIT = a.audit
    if a.cmd == 'member':
        print(json.dumps(member_report(), indent=1, default=float))
    elif a.cmd == 'kappa':
        zs, K = kappa_table(); print(json.dumps(dict(zeta=zs.tolist(), mean_env_over_rms=K[:, 0].tolist(), rms_env_over_rms=K[:, 1].tolist()), indent=1))
    elif a.cmd == 'selftest':
        print(json.dumps(selftest(a.seed)))
    elif a.cmd == 'seeds':
        # the study's ruling 3 (part 5): a seed whose member cannot be placed is not a problem of the set. The study's twenty
        # are seeds first.. first+19 with each failing one replaced, in order, by the next from 32 up (audit: 20 to 29,
        # replacements from 50 up). Writes study_seeds.json.
        first, rep = a.first, (50 if a.first == 20 else 32); out = []; failed = []
        for sd in list(range(first, first + a.count)):
            while choose_member(sd)['chosen'] is None:
                failed.append(sd); sd = rep; rep += 1
            out.append(sd)
        res = dict(first=first, seeds=out, failed=failed, tuning=list(TUNING_SEEDS))
        json.dump(res, open(os.path.join(BULK, 'study_seeds.json' if first == 0 else f'study_seeds_from{first}.json'), 'w'), indent=1); print(json.dumps(res))
    elif a.cmd == 'choose':
        r = choose_member(a.seed); print(json.dumps(dict(seed=a.seed, chosen=r['chosen'], tried=len(r['tried'])), indent=1, default=float))
    elif a.cmd == 'scale':
        print(level_scale(a.seed))
    elif a.cmd == 'export':
        folder, meta = build(a.seed, a.ns, a.nc, a.variant, a.scale); print(folder, json.dumps(meta, default=float)[:1500])
    elif a.cmd == 'all':
        from concurrent.futures import ThreadPoolExecutor
        seeds = [int(s) for s in a.seeds.split(',')]
        if any(s in AUDIT_SEEDS for s in seeds) and not ALLOW_AUDIT:
            sys.exit('seeds 20 to 29 belong to the audit; pass --audit as the auditor')
        scales = {}
        for sd in seeds:
            try:
                scales[sd] = level_scale(sd)
            except RuntimeError as e:      # no candidate: the seed stops (the study's ruling 1, part 3)
                print(f'STOPPED: {e}', flush=True)
        js = [j + (scales[j[0]],) for j in jobs_of(list(scales))]
        with ThreadPoolExecutor(a.jobs) as ex:
            for job, rc, tail in ex.map(_one, js):
                print(job[:4], 'ok' if rc == 0 else f'FAILED {rc}: {tail}', flush=True)
