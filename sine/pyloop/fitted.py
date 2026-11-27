"""The linear tier as the software's simulator steps it: each mode of a modal structure a filter fitted to its
continuous response, restated from the simulator's fitted modes (hertzforge-hal-sim, `fitted.rs`, Discretization::
Fitted), so the harness plays the same discrete structure the runner does.

A modal structure's `simulator.discretization` names it: {"kind": "fitted", "band_hz", "taps", "delay"} (the benchmark
set's linear tier: 2200 Hz, 9 taps, 4 samples). Each mode keeps the continuous mode's exact poles (z = exp(s T)) and its
numerator, `taps` coefficients, is the least-squares fit on 16 lines a tap from 0 to `band_hz` of the continuous
acceleration response s^2 / (s^2 + s w / Q + w^2) (Q = 1 / (2 zeta), at least 0.5) delayed `delay` samples. Stepped in
direct form II: w[n] = u[n] - a1 w[n-1] - a2 w[n-2], the mode's output sum_k b_k w[n-k]; the outputs are the modes'
outputs through the output shapes, the mode's input the drives through the drive shapes. Feedthrough, residual and a
transport delay are not modelled (the benchmark set's linear tier has none; a structure with any is refused).

The reference model's own step for a modal structure is the triangle hold, which reads 20 % off the continuous response
at 2000 Hz on frame-12x6 s00; the simulator's fitted filters are within 0.0001 dB of it in band (fitted.rs).
"""
import numpy as np

TWO_PI = 2.0 * np.pi
LINES_PER_TAP = 16


def is_fitted(st):
    d = (st.raw.get('simulator') or {}).get('discretization') or {}
    return st.kind == 'modal' and d.get('kind') == 'fitted'


def _continuous(fn, q, f):
    w = TWO_PI * fn
    s = 1j * TWO_PI * np.asarray(f, float)
    return s * s / (s * s + s * (w / q) + w * w)


def _poles(fn, q, fs):
    w = TWO_PI * fn
    half = 1.0 / (2.0 * q)
    root = np.sqrt(complex(half * half - 1.0, 0.0))
    z1 = np.exp((-half + root) * w / fs)
    z2 = np.exp((-half - root) * w / fs)
    return -(z1 + z2).real, (z1 * z2).real


def _fit(a1, a2, taps, delay, band, fs, target):
    lines = LINES_PER_TAP * taps
    f = band * np.arange(lines) / (lines - 1)
    zi = np.exp(-1j * TWO_PI * f / fs)
    den = 1.0 + zi * a1 + zi * zi * a2
    want = target(f) * zi ** delay
    basis = (zi[:, None] ** np.arange(taps)[None, :]) / den[:, None]
    A = np.concatenate([basis.real, basis.imag], 0)
    b = np.concatenate([want.real, want.imag])
    return np.linalg.lstsq(A, b, rcond=None)[0]


class FittedModal:
    """The fitted filters of a modal structure's modes, read at `rows` (output indices), at the rate fs."""

    def __init__(self, st, rows, fs):
        sim = st.raw.get('simulator') or {}
        d = sim['discretization']
        if float(sim.get('feedthrough', 0.0)) != 0.0 or sim.get('residual') or int(sim.get('transport_delay_samples', 0)):
            raise ValueError('a modal structure with a feedthrough, a residual or a transport delay is not modelled')
        self.taps, self.delay, self.band = int(d['taps']), int(d['delay']), float(d['band_hz'])
        if self.taps < 3 or self.delay >= self.taps or not (0 < self.band < 0.5 * fs):
            raise ValueError('the fitted discretization is out of the simulator\'s range')
        self.fs = fs
        q = np.maximum(1.0 / (2.0 * st.zeta), 0.5)
        nm = st.n_modes
        self.a1, self.a2 = np.empty(nm), np.empty(nm)
        self.b = np.empty((nm, self.taps))
        for k in range(nm):
            self.a1[k], self.a2[k] = _poles(st.f_hz[k], q[k], fs)
            self.b[k] = _fit(self.a1[k], self.a2[k], self.taps, self.delay, self.band, fs,
                             lambda f, fn=st.f_hz[k], qq=q[k]: _continuous(fn, qq, f))
        self.drive_shape = st.drive_shape                       # (modes, drives)
        self.out = st.output_shape[:, rows]                     # (modes, rows)
        self.w = np.zeros((self.taps, nm))                      # w[n-1] ... w[n-taps], newest first

    def frf(self, f_hz):
        """The exact response of the stepped filters, (lines, rows, drives), z^-1 = exp(-j 2 pi f / fs)."""
        zi = np.exp(-1j * TWO_PI * np.asarray(f_hz, float) / self.fs)                 # (lines,)
        num = (zi[:, None] ** np.arange(self.taps)[None, :]) @ self.b.T                  # (lines, modes)
        den = 1.0 + zi[:, None] * self.a1[None, :] + (zi * zi)[:, None] * self.a2[None, :]
        r = num / den
        return np.einsum('mo,fm,md->fod', self.out, r, self.drive_shape, optimize=True)

    def step(self, d):
        """Play `d` (samples, drives), continuing from the last call; returns (samples, rows)."""
        u = np.asarray(d, float) @ self.drive_shape.T                                    # (samples, modes)
        N = u.shape[0]
        T = self.taps
        W = np.empty((N + T, u.shape[1]))
        W[:T] = self.w[::-1]                                     # oldest first: w[-T] ... w[-1]
        a1, a2 = self.a1, self.a2
        for n in range(N):
            W[T + n] = u[n] - a1 * W[T + n - 1] - a2 * W[T + n - 2]
        y = np.zeros((N, u.shape[1]))
        for k in range(T):
            y += self.b[:, k] * W[T - k:T - k + N]
        self.w = W[N:][::-1].copy()
        return y @ self.out
