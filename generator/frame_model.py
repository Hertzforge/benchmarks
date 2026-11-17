"""frame_model: small 3D frame finite-element model for MIMO metric studies (an earlier study, later shock).

Euler-Bernoulli beam elements, 6 DOF per node (ux, uy, uz, rx, ry, rz), consistent mass,
square aluminum tubes. Everything downstream works from modal quantities (natural frequencies,
modal damping, mass-normalized shapes, modal stress matrix) so the same objects serve a
frequency-domain random study and a time-domain transient study.

Main pieces
  Frame            nodes, elements, lumped masses; assemble K, M; stress recovery matrix S
  make_article     seeded chassis (box/ladder) with cantilevered component stalks; unit perturbation
  make_host        seeded flexible next-level assembly (grid frame) with lumped masses
  couple           host + article joined at 4 mounts by stiff 3-translation springs
  suspend          article on a soft suspension (lab multi-shaker, free-free)
  Modal            eigen-solution with optional fixed DOFs, modal damping from a DampingField
  DampingField     per-unit damping as a function of frequency (lognormal per 1/12-octave bin)
  verify()         closed-form checks (cantilever frequencies, tip deflection and root stress,
                   rigid-body modes, total mass)

Stress: for each element end, outer-fiber stress N/A +/- Mz c/I +/- My c/I at the four corners of
the square tube (8 linear outputs per element). The governing value per element is taken as the
worst of the 8 (largest RMS) in post-processing.
"""
import numpy as np
from scipy import linalg as sla

E_AL, RHO_AL, NU_AL = 69e9, 2700.0, 0.33
G_AL = E_AL / (2 * (1 + NU_AL))
TWO_PI = 2 * np.pi


def sq_tube(b, t):
    bi = b - 2 * t
    return dict(A=b * b - bi * bi, I=(b ** 4 - bi ** 4) / 12, J=(b - t) ** 3 * t, c=b / 2)


def round_shaft(d):
    """the study: solid round shaft of diameter d. Same keys as sq_tube; the stress rows are then conservative (corner sum)."""
    return dict(A=np.pi * d * d / 4, I=np.pi * d ** 4 / 64, J=np.pi * d ** 4 / 32, c=d / 2, kind='shaft', dims=dict(d=d))


def round_tube(d, t):
    """the study: round tube, outer diameter d, wall t."""
    di = d - 2 * t
    return dict(A=np.pi * (d * d - di * di) / 4, I=np.pi * (d ** 4 - di ** 4) / 64, J=np.pi * (d ** 4 - di ** 4) / 32, c=d / 2,
                kind='tube', dims=dict(d=d, t=t))


def strip(w, t):
    """the study: rectangular strip, thickness t along the element's local y axis, width w along local z.
    Bending in the local x-y plane (about z) is the soft one. Carries Iy, Iz, cy, cz, so _local_KM and
    stress_matrix use the two bending planes separately; sections without those keys take the old path unchanged."""
    a, b = max(w, t), min(w, t)
    J = a * b ** 3 * (1 / 3 - 0.21 * b / a * (1 - b ** 4 / (12 * a ** 4)))   # Roark, rectangle torsion constant
    return dict(A=w * t, I=w * t ** 3 / 12, Iz=w * t ** 3 / 12, Iy=t * w ** 3 / 12, J=J, c=t / 2, cy=t / 2, cz=w / 2,
                kind='strip', dims=dict(w=w, t=t))


def _local_KM(sec, L, E=E_AL, rho=RHO_AL):
    if 'Iy' in sec:
        return _local_KM_2(sec, L, E, rho)
    A, I, J = sec['A'], sec['I'], sec['J']
    K = np.zeros((12, 12)); M = np.zeros((12, 12))
    def put(Mat, idx, blk):
        Mat[np.ix_(idx, idx)] += blk
    put(K, [0, 6], E * A / L * np.array([[1, -1], [-1, 1]]))
    put(K, [3, 9], sec.get('G', G_AL) * J / L * np.array([[1, -1], [-1, 1]]))   # the study: a section may carry its own shear modulus
    kb = E * I / L ** 3
    put(K, [1, 5, 7, 11], kb * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                                         [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]]))
    put(K, [2, 4, 8, 10], kb * np.array([[12, -6 * L, -12, -6 * L], [-6 * L, 4 * L * L, 6 * L, 2 * L * L],
                                         [-12, 6 * L, 12, 6 * L], [-6 * L, 2 * L * L, 6 * L, 4 * L * L]]))
    m = rho * A * L
    put(M, [0, 6], m / 6 * np.array([[2, 1], [1, 2]]))
    put(M, [3, 9], rho * 2 * I * L / 6 * np.array([[2, 1], [1, 2]]))
    mb = m / 420
    put(M, [1, 5, 7, 11], mb * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                                         [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L * L, -22 * L, 4 * L * L]]))
    put(M, [2, 4, 8, 10], mb * np.array([[156, -22 * L, 54, 13 * L], [-22 * L, 4 * L * L, -13 * L, -3 * L * L],
                                         [54, -13 * L, 156, 22 * L], [13 * L, -3 * L * L, 22 * L, 4 * L * L]]))
    return K, M


def _local_KM_2(sec, L, E=E_AL, rho=RHO_AL):
    """the study: as _local_KM with Iz in the x-y bending plane and Iy in the x-z plane (unequal for a strip)."""
    A, Iy, Iz, J = sec['A'], sec['Iy'], sec['Iz'], sec['J']
    K = np.zeros((12, 12)); M = np.zeros((12, 12))
    def put(Mat, idx, blk):
        Mat[np.ix_(idx, idx)] += blk
    put(K, [0, 6], E * A / L * np.array([[1, -1], [-1, 1]]))
    put(K, [3, 9], sec.get('G', G_AL) * J / L * np.array([[1, -1], [-1, 1]]))   # the study: a section may carry its own shear modulus
    kz, ky = E * Iz / L ** 3, E * Iy / L ** 3
    put(K, [1, 5, 7, 11], kz * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L * L, -6 * L, 2 * L * L],
                                         [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L * L, -6 * L, 4 * L * L]]))
    put(K, [2, 4, 8, 10], ky * np.array([[12, -6 * L, -12, -6 * L], [-6 * L, 4 * L * L, 6 * L, 2 * L * L],
                                         [-12, 6 * L, 12, 6 * L], [-6 * L, 2 * L * L, 6 * L, 4 * L * L]]))
    m = rho * A * L
    put(M, [0, 6], m / 6 * np.array([[2, 1], [1, 2]]))
    put(M, [3, 9], rho * (Iy + Iz) * L / 6 * np.array([[2, 1], [1, 2]]))
    mb = m / 420
    put(M, [1, 5, 7, 11], mb * np.array([[156, 22 * L, 54, -13 * L], [22 * L, 4 * L * L, 13 * L, -3 * L * L],
                                         [54, 13 * L, 156, -22 * L], [-13 * L, -3 * L * L, -22 * L, 4 * L * L]]))
    put(M, [2, 4, 8, 10], mb * np.array([[156, -22 * L, 54, 13 * L], [-22 * L, 4 * L * L, -13 * L, -3 * L * L],
                                         [54, -13 * L, 156, 22 * L], [13 * L, -3 * L * L, 22 * L, 4 * L * L]]))
    return K, M


def _rot(p0, p1, vref=None):
    """the study part 2: an element may carry its own reference vector (key 'vref'); without it, as before."""
    ex = (p1 - p0) / np.linalg.norm(p1 - p0)
    if vref is None:
        vref = np.array([0, 0, 1.0]) if abs(ex[2]) < 0.9 else np.array([0, 1.0, 0])
    else:
        vref = np.asarray(vref, float)
    ey = np.cross(vref, ex); ey /= np.linalg.norm(ey)
    ez = np.cross(ex, ey)
    R = np.vstack([ex, ey, ez])
    T = np.zeros((12, 12))
    for k in range(4):
        T[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
    return T


class Frame:
    """Nodes, beam elements and lumped masses. Elements: dict(i, j, sec, Efac, group)."""

    def __init__(self):
        self.xyz, self._key, self.el, self.lump = [], {}, [], {}

    def node(self, p):
        p = np.asarray(p, float); k = tuple(np.round(p, 6))
        if k not in self._key:
            self._key[k] = len(self.xyz); self.xyz.append(p)
        return self._key[k]

    def line(self, p0, p1, sec, h=0.1, group='chassis', n=None):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = n or max(1, int(np.ceil(np.linalg.norm(p1 - p0) / h - 1e-9)))
        ids = [self.node(p0 + (p1 - p0) * k / n) for k in range(n + 1)]
        for a, b in zip(ids[:-1], ids[1:]):
            self.el.append(dict(i=a, j=b, sec=sec, Efac=1.0, group=group))
        return ids

    def add_mass(self, n, m, irot=0.0):
        m0, i0 = self.lump.get(n, (0.0, 0.0)); self.lump[n] = (m0 + m, i0 + irot)

    @property
    def nn(self):
        return len(self.xyz)

    @property
    def ndof(self):
        return 6 * self.nn

    def _edofs(self, e):
        return np.r_[6 * e['i']:6 * e['i'] + 6, 6 * e['j']:6 * e['j'] + 6]

    def assemble(self):
        K = np.zeros((self.ndof, self.ndof)); M = np.zeros_like(K); X = np.array(self.xyz)
        for e in self.el:
            L = np.linalg.norm(X[e['j']] - X[e['i']]); T = _rot(X[e['i']], X[e['j']], e.get('vref'))
            kl, ml = _local_KM(e['sec'], L, E=e.get('E', E_AL) * e['Efac'], rho=e.get('rho', RHO_AL))   # the study: per-element material
            d = self._edofs(e)
            K[np.ix_(d, d)] += T.T @ kl @ T; M[np.ix_(d, d)] += T.T @ ml @ T
        for n, (m, ir) in self.lump.items():
            for k in range(3):
                M[6 * n + k, 6 * n + k] += m; M[6 * n + 3 + k, 6 * n + 3 + k] += (ir[k] if np.ndim(ir) else ir)   # the study: ir may be (Ix, Iy, Iz)
        return K, M

    def stress_matrix(self):
        """S (8*n_el x ndof): rows [el0 end1 corners(++,+-,-+,--), el0 end2 ..., el1 ...]. Pa per m."""
        X = np.array(self.xyz); S = np.zeros((8 * len(self.el), self.ndof))
        for ie, e in enumerate(self.el):
            L = np.linalg.norm(X[e['j']] - X[e['i']]); T = _rot(X[e['i']], X[e['j']], e.get('vref'))
            kl, _ = _local_KM(e['sec'], L, E=e.get('E', E_AL) * e['Efac'], rho=e.get('rho', RHO_AL))
            R = kl @ T; d = self._edofs(e); s = e['sec']
            for end, (sg, iN, iMy, iMz) in enumerate([(-1, 0, 4, 5), (1, 6, 10, 11)]):
                N, My, Mz = sg * R[iN], sg * R[iMy], sg * R[iMz]
                for c, (s1, s2) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
                    if 'Iy' in s:   # the study: two bending planes (strip); old sections keep the old expression exactly
                        S[8 * ie + 4 * end + c, d] = N / s['A'] + s1 * Mz * s['cy'] / s['Iz'] + s2 * My * s['cz'] / s['Iy']
                    else:
                        S[8 * ie + 4 * end + c, d] = N / s['A'] + (s1 * Mz + s2 * My) * s['c'] / s['I']
        return S

    def rigid_translation(self, axis):
        r = np.zeros(self.ndof); r[axis::6] = 1.0; return r


class DampingField:
    """Modal damping as a function of frequency: lognormal value per 1/12-octave bin.

    A mode at frequency f takes the bin value at f. A unit perturbation multiplies every bin by an
    independent lognormal factor, so the same unit carries a consistent damping signature across
    configurations (service, lab free-free, fixed base) while two units differ."""

    def __init__(self, seed, median=0.015, sig=0.4, bpo=12, f0=0.5, f1=1e4):
        rng = np.random.default_rng(seed); self.bpo, self.f0 = bpo, f0
        self.logz = np.log(median) + sig * rng.standard_normal(int(np.log2(f1 / f0) * bpo) + 1)

    def unit(self, rng, sig_unit):
        c = DampingField.__new__(DampingField); c.bpo, c.f0 = self.bpo, self.f0
        c.logz = self.logz + sig_unit * rng.standard_normal(self.logz.size); return c

    def __call__(self, f):
        idx = np.clip((np.log2(np.maximum(f, self.f0) / self.f0) * self.bpo).astype(int), 0, self.logz.size - 1)
        return np.clip(np.exp(self.logz[idx]), 0.003, 0.08)


class Modal:
    """Modes of (K, M) with optional fixed DOFs; Phi is full-size (zeros at fixed DOFs)."""

    def __init__(self, K, M, zeta_fn, fmax=4000.0, fixed=()):
        n = K.shape[0]; free = np.setdiff1d(np.arange(n), np.asarray(fixed, int))
        w2, V = sla.eigh(K[np.ix_(free, free)], M[np.ix_(free, free)],
                         subset_by_value=(-np.inf, (TWO_PI * fmax) ** 2))
        self.wn = np.sqrt(np.clip(w2, 0, None)); self.fn = self.wn / TWO_PI
        self.Phi = np.zeros((n, len(w2))); self.Phi[free] = V
        self.free, self.zeta = free, zeta_fn(self.fn)

    def h(self, f):
        """Modal displacement receptance per unit modal force, (n_modes, n_f)."""
        w = TWO_PI * np.asarray(f)[None, :]; wn = self.wn[:, None]
        return 1.0 / (wn ** 2 - w ** 2 + 2j * self.zeta[:, None] * wn * w)

    def frf_acc(self, out_dofs, in_dofs, f):
        """Acceleration per force, (n_out, n_in, n_f)."""
        hq = self.h(f) * -(TWO_PI * np.asarray(f)) ** 2
        return np.einsum('om,mf,im->oif', self.Phi[out_dofs], hq, self.Phi[in_dofs], optimize=True)


# ---------------------------------------------------------------- article, host, configurations
def make_article(seed, unit_rng=None, pert=0.0):
    """Seeded box/ladder chassis about 0.6 x 0.4 x 0.3 m with 3 to 5 component stalks.

    Geometry is fixed by `seed`. With `unit_rng`, member stiffness (E per element) and stalk tip
    masses are scaled by (1 + 0.03*pert*N(0,1)): a different serial number of the same design."""
    rng = np.random.default_rng(seed); fr = Frame()
    Lx, Ly, Lz = 0.6 * rng.uniform(.9, 1.1), 0.4 * rng.uniform(.9, 1.1), 0.3 * rng.uniform(.9, 1.1)
    b = rng.uniform(.020, .030); sec = sq_tube(b, rng.uniform(.0015, .0025))
    secv = sq_tube(b * rng.uniform(.8, 1.0), .002)
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
    tips, tipm = [], []
    lo, hi = X.min(0), X.max(0)
    for nb, ft in zip(bases, ftar):
        p = X[nb]; dirs = []
        for k in range(3):
            if abs(p[k] - hi[k]) < 1e-6: dirs.append(np.eye(3)[k])
            if abs(p[k] - lo[k]) < 1e-6 and k < 2: dirs.append(-np.eye(3)[k])
        if not dirs:  # interior node of the bottom cross member: hang the stalk below
            dirs = [-np.eye(3)[2]]
        d = dirs[rng.integers(len(dirs))]; L = rng.uniform(.06, .14); m = np.exp(rng.uniform(np.log(.05), np.log(.3)))
        I = m * (TWO_PI * ft) ** 2 * L ** 3 / (3 * E_AL); bs = (I / 0.0492) ** 0.25
        ids = fr.line(p, p + L * d, sq_tube(bs, 0.1 * bs), n=3, group='stalk')
        tips.append(ids[-1]); tipm.append(m); fr.add_mass(ids[-1], m, m * 0.015 ** 2 / 6)
    lumped = list(rng.choice([n for n in cand if n not in bases], int(rng.integers(2, 5)), replace=False))
    for n in lumped:
        fr.add_mass(n, rng.uniform(.3, 1.5), 1e-4)
    if unit_rng is not None and pert > 0:
        for e in fr.el:
            e['Efac'] *= 1 + 0.03 * pert * unit_rng.standard_normal()
        for n, m in zip(tips, tipm):
            dm = m * 0.03 * pert * unit_rng.standard_normal(); fr.add_mass(n, dm, dm * 0.015 ** 2 / 6)
    fr.meta = dict(chassis=chassis, mounts=mounts, corners=corners, bases=bases, tips=tips,
                   lumped=lumped, f_target=ftar, dims=(Lx, Ly, Lz))
    return fr


def make_host(seed, art_dims):
    """Flexible grid frame 1.6 x 1.0 m (50 x 3 mm tube) with lumped masses; nodes at the 4 mounts."""
    rng = np.random.default_rng(seed + 7919); fr = Frame(); sec = sq_tube(.05, .003)
    x, y = art_dims[0] / 2, art_dims[1] / 2
    xs, ys = [-0.8, -x, x, 0.8], [-0.5, -y, y, 0.5]
    for yy in ys:
        for a, b_ in zip(xs[:-1], xs[1:]): fr.line([a, yy, 0], [b_, yy, 0], sec)
    for xx in xs:
        for a, b_ in zip(ys[:-1], ys[1:]): fr.line([xx, a, 0], [xx, b_, 0], sec)
    mounts = [fr.node([sx * x, sy * y, 0]) for sx in (-1, 1) for sy in (-1, 1)]
    others = [n for n in range(fr.nn) if n not in mounts]
    for n in rng.choice(others, int(rng.integers(4, 7)), replace=False):
        fr.add_mass(n, rng.uniform(2, 10), 1e-3)
    fr.meta = dict(mounts=mounts, nodes=others)
    return fr


def couple(art, host, k=1e8):
    """Block system [article; host] with 3-translation springs at the 4 mounts."""
    Ka, Ma = art.assemble(); Kh, Mh = host.assemble(); na = Ka.shape[0]
    K = sla.block_diag(Ka, Kh); M = sla.block_diag(Ma, Mh)
    for ma, mh in zip(art.meta['mounts'], host.meta['mounts']):
        for d in range(3):
            i, j = 6 * ma + d, na + 6 * mh + d
            K[i, i] += k; K[j, j] += k; K[i, j] -= k; K[j, i] -= k
    return K, M, na


def suspend(art, f_susp=3.0):
    """Article on a soft suspension at its 4 mounts (rigid-body modes near f_susp)."""
    K, M = art.assemble(); mtot = art.rigid_translation(0) @ M @ art.rigid_translation(0)
    ks = mtot * (TWO_PI * f_susp) ** 2 / 4
    for n in art.meta['mounts']:
        for d in range(3):
            K[6 * n + d, 6 * n + d] += ks
    return K, M


def fixed_base_dofs(art):
    return [6 * n + d for n in art.meta['mounts'] for d in range(3)]


# ---------------------------------------------------------------- verification
def verify(verbose=True):
    """Closed-form checks. Returns dict of (computed, expected) pairs."""
    out = {}
    sec = sq_tube(.025, .002); L = 1.0; fr = Frame(); ids = fr.line([0, 0, 0], [L, 0, 0], sec, n=20)
    K, M = fr.assemble(); fixed = list(range(6))
    md = Modal(K, M, lambda f: 0 * f + .01, fmax=2000, fixed=fixed)
    c = np.sqrt(E_AL * sec['I'] / (RHO_AL * sec['A'] * L ** 4)) / TWO_PI
    fb = np.unique(np.round(md.fn, 3))  # y and z bending are degenerate (square)
    out['cantilever_f1_Hz'] = (fb[0], 1.8751 ** 2 * c)
    out['cantilever_f2_Hz'] = (fb[1], 4.6941 ** 2 * c)
    free = np.setdiff1d(np.arange(fr.ndof), fixed); P = 100.0
    F = np.zeros(fr.ndof); F[6 * ids[-1] + 1] = P
    u = np.zeros(fr.ndof); u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    out['tip_deflection_m'] = (u[6 * ids[-1] + 1], P * L ** 3 / (3 * E_AL * sec['I']))
    S = fr.stress_matrix(); s = S @ u
    out['root_stress_Pa'] = (np.abs(s[0:4]).max(), P * L * sec['c'] / sec['I'])
    art = make_article(1); K, M = art.assemble()
    md = Modal(K, M, lambda f: 0 * f + .01, fmax=3000)
    out['free_free_modes_below_1Hz'] = (int((md.fn < 1).sum()), 6)
    out['first_elastic_Hz'] = (md.fn[6], float('nan'))
    X = np.array(art.xyz)
    mexp = sum(RHO_AL * e['sec']['A'] * np.linalg.norm(X[e['j']] - X[e['i']]) for e in art.el) + \
        sum(m for m, _ in art.lump.values())
    out['total_mass_kg'] = (art.rigid_translation(0) @ M @ art.rigid_translation(0), mexp)
    if verbose:
        for k, (a, b) in out.items():
            print(f'{k:28s} computed {a:12.5g}  expected {b:12.5g}')
    return out


if __name__ == '__main__':
    verify()
