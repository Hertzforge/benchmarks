"""The `matrix` law's update as the shipped Rust law has it (hertzforge-mimo-sine-law-math `matrix.rs` at 9a4c7b80),
restated on the research testbed's model (`RefMatrix`'s modal-smoothed identification and its inverse).

What it adds to `RefMatrix` (each a no-op on the parity vectors' flat reference with the guards off, which is why the
vectors do not tell the two apart):
  - the reference's change is carried backward, Zn (Rk - Rp), Rp the reference told at the last update (per unit of
    level, at this update's level and notch), not forward from the grid at the next frequency;
  - the correction is aimed at the reference as the trackers read it (`seen_reference`): the reference per unit of
    level through a one-pole lag of the loop's delay besides the structure (tracker delay + transport + half a tick);
  - `recarry`: the part of the last carry a change cap cut is carried again while nothing holds the loop;
  - the reference aimed at is the reference times the notch scale.
Not restated: the shipped fit's group limits and the windows it leaves out (there the shipped law carries nothing);
the Python fit fits every group, so this law carries everywhere.
"""
import numpy as np

from .interface import TickOut
from .matrix_ref import RefMatrix, _db

TRACKER_DELAY_FACTOR_DEFAULT = 1.0 / np.pi


class ShippedMatrix(RefMatrix):
    name = 'matrix_shipped'

    def __init__(self, setup, seen_reference=True, recarry=True, **kw):
        super().__init__(setup, **kw)
        self.seen_reference, self.recarry = seen_reference, recarry
        n, m = setup.n, setup.m
        self.told_prev = np.zeros(n, complex)
        self.told_valid = False
        self.seen = np.zeros(n, complex)
        self.seen_valid = False
        self.carried_prev = np.zeros(m, complex)

    @staticmethod
    def _notch(x):
        s = x.notch_scale
        return float(np.clip(s, 0.0, 1.0)) if np.isfinite(s) else 1.0

    def _remember_told(self, x):
        if x.level > 0 and np.isfinite(x.level):
            self.told_prev = np.asarray(x.R, complex) / x.level
            self.told_valid = True
        else:
            self.told_valid = False

    def engage(self, x):
        aimed = np.asarray(x.R, complex) * self._notch(x)
        self.seen = np.asarray(x.R, complex) / x.level if (x.level > 0 and np.isfinite(x.level)) else np.asarray(x.R, complex)
        self.seen_valid = True
        self.carried_prev[:] = 0.0
        self.told_valid = False
        self._remember_told(x)
        D = self.Z(x.f) @ self._ref(x, aimed)
        D, _ = self._unseen(D, x.f)
        self.asked = D.copy()
        return TickOut(D)

    def tick(self, x):
        s = self.setup
        g = 0.0 if x.hold == 'engage' else self.gain(x.f)
        notch = self._notch(x)
        R = np.asarray(x.R, complex)
        aimed = R * notch
        rk = self._ref(x, aimed)
        lo, hi = s.ident.grid_hz[0], s.ident.grid_hz[-1]
        fn = float(np.clip(x.f + x.sweep_hz_s * s.dt_c, lo, hi))
        rp = self._ref(x, self.told_prev * (x.level * notch) if self.told_valid else aimed)
        self._remember_told(x)
        B = s.tracking_bw(x.f)
        tau = (getattr(s, 'tracker_delay_factor', TRACKER_DELAY_FACTOR_DEFAULT) / B + getattr(s, 'transport_delay_s', 0.0)
               + 0.5 * s.dt_c)
        a_seen = 1.0 - np.exp(-s.dt_c / max(tau, 1e-12)) if self.seen_valid else 1.0
        self.seen_valid = True
        if x.level > 0 and np.isfinite(x.level):
            self.seen = self.seen + (R / x.level - self.seen) * a_seen
        compared = self._ref(x, self.seen * (x.level * notch)) if self.seen_reference else rk
        E = compared - x.C
        Hk = self.H(x.f)
        Zk, Zn = self.Z(x.f), self.Z(fn)
        D0 = x.D_in_force if self.rebase else self.asked
        if self.recarry and self.rebase and x.hold == 'free':
            cut = self.asked - x.D_in_force
            nc, nk = np.abs(self.carried_prev), np.abs(cut)
            ok = (nk > 0) & ((cut * np.conj(self.carried_prev)).real > 0)
            pend = np.where(ok, cut * np.minimum(1.0, nc / np.maximum(nk, 1e-300)), 0.0)
        else:
            pend = np.zeros(s.m, complex)
        carried = self.carry * ((Zn - Zk) @ (Hk @ D0)) + Zn @ (rk - rp) + pend
        self.carried_prev = carried.copy()
        corr = g * (Zk @ E)
        D = D0 + carried + corr
        D, unseen = self._unseen(D, fn)
        self.asked = D.copy()
        return TickOut(D, values=dict(gain=float(g), error_db=float(np.max(np.abs(_db(x.C) - _db(x.R))))))
