"""Low-level random identification with uncorrelated drives.

Each drive plays its own band-limited Gaussian noise (independent across drives). The structure is
stepped in time from rest (through the actuator), measurement noise is added, the first `skip_s`
seconds are dropped, and the frequency response is estimated per line by H1 = S_yx S_xx^-1 from
Hann-windowed segments with 50 percent overlap, under the negative-exponent transform, so that
C = H D for a drive d = Re{D exp(j w t)} (Notation.md). Gives the controls' Hhat0, the limit
channels' That0, the multiple coherence of each, and a trust mask, on the grid of lines in band.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np



@dataclass
class Identification:
    grid_hz: np.ndarray                 # (L,)
    H0: np.ndarray                      # (L, n, m)
    T0: np.ndarray                      # (L, p, m)
    coh: np.ndarray                     # (L, n) multiple coherence of each control
    coh_limit: np.ndarray               # (L, p)
    trust: np.ndarray                   # (L,) bool
    level: dict = field(default_factory=dict)   # what level it was taken at (drive rms, peak modal ratios)

    def _interp(self, A, f):
        g = self.grid_hz
        f = float(np.clip(f, g[0], g[-1]))
        k = int(np.searchsorted(g, f))
        if k <= 0:
            return A[0]
        if k >= g.size:
            return A[-1]
        t = (f - g[k - 1]) / (g[k] - g[k - 1])
        return (1.0 - t) * A[k - 1] + t * A[k]

    def H0_at(self, f):
        """Hhat0 at frequency f, complex linear interpolation between grid lines (n, m)."""
        return self._interp(self.H0, f)

    def T0_at(self, f):
        return self._interp(self.T0, f)

    def group_delay_at(self, f, span_hz=0.0):
        """The model's group delay at f, s: the largest over the elements of Hhat0 above a tenth of
        the line's largest element of |d arg H / d w|, smoothed over five lines. A stand-in for the
        structure's own response time (about 1 / (zeta w) at a lightly damped resonance). With
        span_hz, the largest over f +- span_hz."""
        if getattr(self, '_gd', None) is None:
            g = self.grid_hz
            H = self.H0.reshape(g.size, -1)
            ph = np.unwrap(np.angle(H), axis=0)
            d = np.abs(np.gradient(ph, 2 * np.pi * g, axis=0))
            big = np.abs(H) >= 0.1 * np.abs(H).max(1, keepdims=True)
            gd = np.where(big, d, 0.0).max(1)
            k = np.ones(5) / 5.0
            self._gd = np.convolve(gd, k, mode='same')
        if span_hz > 0:                     # the largest within +-span_hz: a nearby light mode's free response
            g = self.grid_hz
            a, b = np.searchsorted(g, f - span_hz), np.searchsorted(g, f + span_hz, side='right')
            if b > a:
                return float(max(self._gd[a:b].max(), np.interp(f, g, self._gd)))
        return float(np.interp(f, self.grid_hz, self._gd))

    def trust_at(self, f):
        g = self.grid_hz
        k = int(np.clip(np.searchsorted(g, f), 1, g.size - 1))
        return bool(self.trust[k - 1] and self.trust[k])

    def save(self, path):
        np.savez(path, grid_hz=self.grid_hz, H0=self.H0, T0=self.T0, coh=self.coh, coh_limit=self.coh_limit,
                 trust=self.trust, level=np.array([repr(self.level)]))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        import ast
        return Identification(z['grid_hz'], z['H0'], z['T0'], z['coh'], z['coh_limit'], z['trust'],
                              ast.literal_eval(str(z['level'][0])))


