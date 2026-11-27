"""The law interface: what a multi-shaker sine law is built with, what it sees each update, and
what it returns. It mirrors a proposed `sine-mimo` seam,
following the conventions of the software's `sine-single` and `random-mimo` seams:

- Complex amplitudes: a drive d_i(t) = Re{ D_i exp(j phi(t)) }, volts peak; a control's tracked
  amplitude C_c with y_c(t) = Re{ C_c exp(j phi(t)) }, g peak, so C = H D for a linear structure in
  steady state (phase of response minus phase of the tone, the negative-exponent transform).
- The FRF is response over drive, controls by drives (n, m), as in `random-mimo`.
- Unlike `random-mimo` (and like `sine-single`), the law sees the level and the reference AT this
  level, and returns the drive as it should be played (not a full-level drive).
- The law never decides the frequency, the tone's phase, the level, or an abort. It may ask for a
  hold or a slower sweep; the controller may ignore the request.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ident import Identification


@dataclass
class Setup:
    """Fixed for the run."""
    n: int                          # controls
    m: int                          # drives
    p: int                          # limit channels
    dt_c: float                     # time between updates, s
    fs: float                       # sample rate, Hz
    band: tuple                     # (lo, hi), Hz
    drive_ceiling: np.ndarray       # (m,) volts peak
    ident: Identification           # Hhat0, That0, coherence, trust, the level it was taken at
    ref_grid: np.ndarray            # (L, n) full-level reference on ident.grid_hz (complex; real positive when amplitude-only)
    phase_specified: bool           # False: amplitude-only reference, phases free
    limit_grid: np.ndarray          # (L, p) limit bound on the grid, channel units peak (inf = none)
    alarm_db: float
    abort_db: float
    shape: str                      # 'square', 'tall' (n > m, least squares), 'wide' (n < m, minimum norm)
    tracking: dict                  # {'mode': 'proportional', 'pct':, 'max_hz':} or {'mode': 'fixed', 'hz':}
    identification_level_db: float = float('nan')

    def ref_at(self, f):
        return self.ident._interp(self.ref_grid, f)

    def tracking_bw(self, f):
        t = self.tracking
        if t['mode'] == 'fixed':
            b = t['hz']
        else:
            b = min(f * t['pct'] / 100.0, t['max_hz'])
        return min(b, 0.5 * f)


@dataclass
class TickIn:
    """What the law sees at each update (index 0 at engage)."""
    k: int
    dt: float                       # time since the last call, s
    f: float                        # frequency of the tone now, Hz
    sweep_hz_s: float               # signed sweep rate, Hz/s (0 at a dwell or while held)
    sweep_oct_min: float            # the same in oct/min
    level: float                    # scheduled level, fraction of full level
    R: np.ndarray                   # (n,) reference at this frequency and level
    C: np.ndarray                   # (n,) tracked control amplitudes (zeros at engage)
    L: np.ndarray                   # (p,) tracked limit amplitudes
    D_in_force: np.ndarray          # (m,) drive in force as played (after every guard)
    hold: str                       # 'free', 'notch' (the backstop notch scaled the last drive), 'held' (a granted hold), 'engage' (the engage settle window: filters still rising)
    notch_scale: float              # the backstop notch's present scale (1 = not binding)
    limit_bounds: np.ndarray        # (p,) the limit bounds here
    settle_open: bool


@dataclass
class TickOut:
    D: np.ndarray                   # (m,) complex drive, volts peak
    fault: bool = False             # the law's own judgement, recorded, never relied on
    values: dict = field(default_factory=dict)   # named values for the record (at most 8)
    request: object = None          # None, 'hold', or ('slow', factor in (0, 1])


class Law:
    """Base class. A law is built from the Setup and its own parameters; engage() gives the first
    drive, tick() every drive after it."""
    name = 'law'
    values = ()                     # names of the values it reports

    def __init__(self, setup: Setup, **params):
        self.setup = setup
        self.params = params

    def engage(self, x: TickIn) -> TickOut:
        raise NotImplementedError

    def tick(self, x: TickIn) -> TickOut:
        raise NotImplementedError


def reg_pinv(H, eps=1e-3):
    """Regularized pseudo-inverse Z = V diag(s / (s^2 + (eps s_max)^2)) U^H: least squares for a tall
    H, minimum norm for a wide one, Tikhonov with a weight relative to the largest singular value."""
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    lam = (eps * s[0]) ** 2
    return (Vh.conj().T * (s / (s * s + lam))) @ U.conj().T
