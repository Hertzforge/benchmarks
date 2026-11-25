"""A small synthetic identification for the modal-fit tests (Package G): real rank-one modes in the testbed's q form,
a smooth background, a pure delay and seeded noise, on 0.25 Hz lines. Shared by the tests and by the script that wrote
the parity vector from Package C's own code (`07_Research/MIMO_Sine_Benchmark/experiments/G/make_m7d_vector.py`), so
both fit the same numbers."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pyloop.ident import Identification  # noqa: E402

FS = 8192.0
BAND = (40.0, 160.0)
# (frequency Hz, damping) of each mode: a close pair half a bandwidth apart at 90 Hz, the rest spread
MODES = [(52.0, 0.012), (71.0, 0.010), (90.0, 0.008), (90.7, 0.008), (118.0, 0.015), (141.0, 0.010)]


def truth(n=4, m=3, seed=7):
    rng = np.random.default_rng(seed)
    f = np.array([x[0] for x in MODES]); z = np.array([x[1] for x in MODES])
    psi = rng.standard_normal((f.size, n)); lam = rng.standard_normal((f.size, m))
    return f, z, np.einsum('ki,kj->kij', psi, lam)


def identification(delay_samples=3.0, noise=1e-3, n=4, m=3, seed=7):
    """(Identification, (f, zeta, A) true, delay s). The identification carries the delay, as the harness's does."""
    f, z, A = truth(n, m, seed)
    g = np.arange(BAND[0] / 1.25, BAND[1] * 1.25 + 1e-9, 0.25)
    u = g[:, None] / f[None, :]
    q = -(u * u) / (1.0 - u * u + 2j * z[None, :] * u)
    rng = np.random.default_rng(seed + 1)
    bg = 0.05 * rng.standard_normal((n, m))
    H = np.einsum('lk,kij->lij', q, A) + bg[None]
    H = H * np.exp(-2j * np.pi * g * delay_samples / FS)[:, None, None]
    scale = np.abs(H).mean()
    H = H + noise * scale * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    idn = Identification(g, np.ascontiguousarray(H), np.zeros((g.size, 0, m), complex), np.ones((g.size, n)),
                         np.ones((g.size, 0)), np.ones(g.size, bool), {})
    return idn, (f, z, A), delay_samples / FS
