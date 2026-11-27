"""The tone and the level schedule: a phase-continuous oscillator that sweeps (logarithmic or
linear, up or down), dwells, steps, or ramps the level at a dwell, segment after segment.

The phase of sample n is phi_n = phi_{n-1} + 2 pi f_n / fs, so the tone never breaks phase at a
change of rate, a step, a hold or a segment boundary. The controller owns the tone; a law never
sets the frequency, the phase or the level.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass
class Segment:
    kind: str                       # 'sweep', 'dwell', 'steps', 'ramp'
    level_db: float = 0.0           # level relative to full level (ramp: level at its start)
    f_start: float = 100.0
    f_end: float | None = None      # sweep and steps: the end frequency (below f_start for a down sweep)
    rate: float = 1.0               # sweep: oct/min (log) or Hz/s (lin)
    spacing: str = 'log'
    duration_s: float = 1.0         # dwell, ramp, and each step's dwell
    steps_per_octave: float = 12.0
    level_end_db: float | None = None
    settle_s: float = 0.0           # settle window at the start of the segment, and after each step


def sweep(f0, f1, rate=1.0, level_db=0.0, spacing='log', settle_s=0.0):
    return Segment('sweep', level_db=level_db, f_start=f0, f_end=f1, rate=rate, spacing=spacing, settle_s=settle_s)


def dwell(f, duration_s, level_db=0.0, settle_s=0.0):
    return Segment('dwell', level_db=level_db, f_start=f, duration_s=duration_s, settle_s=settle_s)


def ramp(f, duration_s, db_from, db_to):
    return Segment('ramp', level_db=db_from, level_end_db=db_to, f_start=f, duration_s=duration_s)


def steps(f0, f1, per_octave, dwell_s, level_db=0.0, settle_s=0.0):
    return Segment('steps', level_db=level_db, f_start=f0, f_end=f1, steps_per_octave=per_octave, duration_s=dwell_s,
                   settle_s=settle_s)


class Tone:
    def __init__(self, segments, fs):
        self.segs = list(segments)
        self.fs = float(fs)
        self.phi = 0.0
        self.t = 0.0                  # run time, s
        self.i = 0
        self.settle_until = -1.0
        self.done = False
        self._start_segment()

    # ------------------------------------------------------------------ segment state
    def _start_segment(self):
        if self.i >= len(self.segs):
            self.done = True
            return
        s = self.segs[self.i]
        self.pos = 0.0               # sweep: octaves (log) or Hz (lin) travelled; others: time in segment
        self.k_step = 0
        if s.kind == 'steps':
            n_oct = abs(np.log2(s.f_end / s.f_start))
            nst = int(np.floor(n_oct * s.steps_per_octave + 1e-9)) + 1
            sign = 1.0 if s.f_end >= s.f_start else -1.0
            self.step_f = s.f_start * 2.0 ** (sign * np.arange(nst) / s.steps_per_octave)
        if s.settle_s > 0:
            self.settle_until = self.t + s.settle_s
        self.f = self._freq(s)

    def _freq(self, s):
        if s.kind == 'sweep':
            sign = 1.0 if s.f_end >= s.f_start else -1.0
            if s.spacing == 'log':
                return s.f_start * 2.0 ** (sign * self.pos)
            return s.f_start + sign * self.pos
        if s.kind == 'steps':
            return self.step_f[min(self.k_step, self.step_f.size - 1)]
        return s.f_start

    def level_db(self):
        if self.done:
            s = self.segs[-1]
            return s.level_end_db if s.kind == 'ramp' else s.level_db
        s = self.segs[self.i]
        if s.kind == 'ramp':
            return s.level_db + (s.level_end_db - s.level_db) * min(1.0, self.pos / s.duration_s)
        return s.level_db

    def sweep_rate(self):
        """Signed sweep rate (Hz/s, oct/min) at the present frequency; zero at a dwell or a hold."""
        if self.done or self.held:
            return 0.0, 0.0
        s = self.segs[self.i]
        if s.kind != 'sweep':
            return 0.0, 0.0
        sign = 1.0 if s.f_end >= s.f_start else -1.0
        if s.spacing == 'log':
            octm = sign * s.rate * self.factor
            return self.f * np.log(2.0) * octm / 60.0, octm
        hz = sign * s.rate * self.factor
        return hz, hz * 60.0 / (self.f * np.log(2.0))

    held = False
    factor = 1.0

    # ------------------------------------------------------------------ advance
    def advance(self, N, hold=False, factor=1.0):
        """Produce N samples: returns (f (N,), phi (N,)), the frequency and phase of each sample.
        hold freezes the schedule's progress (the tone stays at its frequency); factor scales the
        sweep rate (a slowed sweep). Both are the controller's to apply."""
        self.held = bool(hold)
        self.factor = float(factor)
        fs = self.fs
        f = np.empty(N)
        n = 0
        while n < N:
            if self.done or hold:
                f[n:] = self.f
                n = N
                break
            s = self.segs[self.i]
            rem = N - n
            if s.kind == 'sweep':
                sign = 1.0 if s.f_end >= s.f_start else -1.0
                if s.spacing == 'log':
                    total = abs(np.log2(s.f_end / s.f_start))
                    dpos = s.rate * factor / 60.0 / fs
                else:
                    total = abs(s.f_end - s.f_start)
                    dpos = s.rate * factor / fs
                left = int(np.ceil((total - self.pos) / dpos - 1e-9))
                k = min(rem, max(left, 0))
                pos = self.pos + dpos * np.arange(1, k + 1)
                pos = np.minimum(pos, total)
                if s.spacing == 'log':
                    f[n:n + k] = s.f_start * 2.0 ** (sign * pos)
                else:
                    f[n:n + k] = s.f_start + sign * pos
                if k:
                    self.pos = pos[-1]
                    self.f = f[n + k - 1]
                n += k
                if self.pos >= total - 1e-12:
                    self.i += 1
                    self._start_segment()
            elif s.kind == 'steps':
                left = int(round((s.duration_s - self.pos) * fs))
                k = min(rem, max(left, 0))
                f[n:n + k] = self.step_f[self.k_step]
                self.pos += k / fs
                n += k
                if k == left:
                    self.k_step += 1
                    self.pos = 0.0
                    if self.k_step >= self.step_f.size:
                        self.i += 1
                        self._start_segment()
                    else:
                        self.f = self.step_f[self.k_step]
                        if s.settle_s > 0:
                            self.settle_until = self.t + n / fs + s.settle_s
            else:                                   # dwell, ramp
                left = int(round((s.duration_s - self.pos) * fs))
                k = min(rem, max(left, 0))
                f[n:n + k] = s.f_start
                self.pos += k / fs
                n += k
                if k == left:
                    prev = self.level_db()
                    self.i += 1
                    self._start_segment()
        phi = self.phi + TWO_PI * np.cumsum(f) / fs
        self.phi = float(np.mod(phi[-1], TWO_PI))
        self.t += N / fs
        return f, phi

    @property
    def settle_open(self):
        return self.t < self.settle_until

    @property
    def segment(self):
        return min(self.i, len(self.segs) - 1)
