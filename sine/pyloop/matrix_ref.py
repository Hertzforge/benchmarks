"""Reference composition of the `matrix` law for package M's parity vectors (outside the repository).

K1m (SmoothedCarried, transport form) with the carry weight, the reset of the drive the controls
cannot see on a wide setup (projection onto the row space of the model at the next frequency),
the rebase switch (False: integrate on the drive the law last asked for), and an amplitude-only
reference whose phases are taken from the measurement."""
import numpy as np
from .interface import TickOut, reg_pinv
from .modal import SmoothedCarried

def _db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))

class RefMatrix(SmoothedCarried):
    name = 'ref_matrix'

    def __init__(self, setup, carry=0.67, reset=True, rebase=True, smooth_base=8, free_phase='in_phase', **kw):
        super().__init__(setup, w=carry, smooth_base=smooth_base, **kw)
        self.carry, self.reset, self.rebase = carry, reset, rebase
        self.asked = None
        self.free_phase = free_phase

    def _ref(self, x, R):
        if self.setup.phase_specified:
            return R
        if self.free_phase == 'in_phase':
            return np.abs(R) + 0j
        a = np.abs(x.C)
        ph = np.where(a > 1e-300, x.C / np.maximum(a, 1e-300), 1.0 + 0j)
        return np.abs(R) * ph

    def _unseen(self, D, f):
        if self.setup.m <= self.setup.n or not self.reset:
            return D, 0.0
        _, _, Vh = np.linalg.svd(self.H(f), full_matrices=False)
        Dn = Vh.conj().T @ (Vh @ D)
        return Dn, float(np.linalg.norm(D - Dn) / max(np.linalg.norm(D), 1e-300))

    def engage(self, x):
        D = self.Z(x.f) @ self._ref(x, x.R)
        D, _ = self._unseen(D, x.f)
        self.asked = D.copy()
        return TickOut(D)

    def tick(self, x):
        g = 0.0 if x.hold == 'engage' else self.gain(x.f)
        Rk = self._ref(x, x.R)
        E = Rk - x.C
        fn = self._f_next(x)
        Rn = self._ref(x, self.setup.ref_at(fn) * x.level)
        Hk = self.H(x.f)
        Zk, Zn = self.Z(x.f), self.Z(fn)
        D0 = x.D_in_force if self.rebase else self.asked
        carried = self.carry * ((Zn - Zk) @ (Hk @ D0)) + Zn @ (Rn - Rk)
        corr = g * (Zk @ E)
        D = D0 + carried + corr
        D, unseen = self._unseen(D, fn)
        self.asked = D.copy()
        return TickOut(D, values=dict(gain=float(g), error_db=float(np.max(np.abs(_db(x.C) - _db(x.R))))))
