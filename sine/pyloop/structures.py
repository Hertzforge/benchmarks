"""Structures: modal models with acceleration outputs, seeded generators, and the stepping object.

A structure has `n_modes` modes. Mode r has small-signal natural frequency f_hz[r], damping ratio
zeta[r], a drive shape (one entry per drive, modal force per volt) and an output shape (one entry
per output). Outputs are accelerations in g. With w = 2 pi f and w_r = 2 pi f_r the small-signal
frequency response from drive j to output i is (as in the benchmark reference model)

    H_ij(f) = sum_r output_shape[r, i] drive_shape[r, j] (-w^2) / (w_r^2 - w^2 + 2 j zeta_r w_r w)

Outputs have roles: `control` (n of them), `limit` (p), anything else is a monitor.

A level-dependent mode follows the Open Benchmark Set's level rule (format 0.2-draft)
on r_m = envelope / reference_amplitude:

    f    = f0 (1 - drop min(1, r)^drop_exponent + rise min(rise_cap, r)^rise_exponent)
    zeta = min(0.9, zeta0 (1 + damping_gain min(damping_cap, r)^damping_exponent))

A mode may instead carry a directly integrated nonlinearity: a cubic spring (`cubic`, k3 in modal
units, force k3 q^3) or an Iwan joint (`iwan`, Jenkins elements on the modal coordinate).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

MAX_DAMPING = 0.9

TWO_PI = 2.0 * np.pi
LEVEL_FIELDS = ('drop', 'drop_exponent', 'rise', 'rise_exponent', 'rise_cap', 'damping_gain',
                'damping_exponent', 'damping_cap')


@dataclass
class ModalStructure:
    name: str
    f_hz: np.ndarray
    zeta: np.ndarray
    drive_shape: np.ndarray            # (modes, m)
    output_shape: np.ndarray           # (modes, outputs)
    roles: list                        # one role per output: 'control', 'limit', 'monitor'
    rules: dict | None = None          # LEVEL_FIELDS -> (modes,) arrays; None = no rule anywhere
    has_rule: np.ndarray | None = None  # (modes,) bool
    reference_amplitude: np.ndarray | None = None
    decay_cycles: float = 3.0
    nonlinear: dict = field(default_factory=dict)  # mode -> {'kind': 'cubic', 'k3'} | {'kind': 'iwan', 'k', 'f'}
    feedthrough: float = 0.0
    residual: np.ndarray | None = None  # (outputs, m)
    step: tuple | None = None           # None (triangle hold) or (band_hz, half_taps) equalized
    transport_delay_samples: int = 0
    seed: int | None = None
    info: dict = field(default_factory=dict)

    @property
    def n_modes(self):
        return self.f_hz.size

    @property
    def m(self):
        return self.drive_shape.shape[1]

    @property
    def n_outputs(self):
        return self.output_shape.shape[1]

    def rows(self, role):
        return np.array([i for i, r in enumerate(self.roles) if r == role], int)

    @property
    def n(self):
        return self.rows('control').size

    @property
    def p(self):
        return self.rows('limit').size

    def level_state(self, r):
        """(f_hz, zeta) of every mode at amplitude ratio r (scalar or per mode), by the rule. Modes
        without a rule (and nonlinear modes) keep their small-signal values."""
        f, z = self.f_hz.copy(), self.zeta.copy()
        if self.rules is None:
            return f, z
        R = self.rules
        r = np.broadcast_to(np.asarray(r, float), f.shape)
        ff = 1.0 - R['drop'] * np.minimum(1.0, r) ** R['drop_exponent'] + R['rise'] * np.minimum(R['rise_cap'], r) ** R['rise_exponent']
        zz = np.minimum(MAX_DAMPING, self.zeta * (1.0 + R['damping_gain'] * np.minimum(R['damping_cap'], r) ** R['damping_exponent']))
        h = self.has_rule
        f[h] = (self.f_hz * ff)[h]
        z[h] = zz[h]
        return f, z

    def frf(self, f_hz, rows=None, fn=None, zeta=None):
        """Continuous small-signal acceleration FRF (lines, outputs, drives), g per volt (no actuator).
        fn, zeta: the modes' state (default small signal). Feedthrough and residual included."""
        f_hz = np.atleast_1d(np.asarray(f_hz, float))
        fn = self.f_hz if fn is None else np.asarray(fn, float)
        zeta = self.zeta if zeta is None else np.asarray(zeta, float)
        w = TWO_PI * f_hz[None, :]
        wn = TWO_PI * fn[:, None]
        h = -(w ** 2) / (wn ** 2 - w ** 2 + 2j * zeta[:, None] * wn * w)          # (modes, lines)
        osh = self.output_shape if rows is None else self.output_shape[:, rows]
        H = np.einsum('mo,mf,md->fod', osh, h, self.drive_shape, optimize=True)
        H = H + self.feedthrough
        if self.residual is not None:
            H = H + (self.residual if rows is None else self.residual[rows])[None]
        return H

    def to_bm_raw(self):
        """This structure as a benchmark structure.json (0.2-draft) dict, for the reference model's
        stepper (parity tests). Nonlinear modes are not representable and are refused."""
        assert not self.nonlinear, 'a directly integrated nonlinearity has no benchmark form'
        modes = []
        for r in range(self.n_modes):
            md = dict(f_hz=float(self.f_hz[r]), zeta=float(self.zeta[r]), drive_shape=self.drive_shape[r].tolist(),
                      output_shape=self.output_shape[r].tolist(),
                      reference_amplitude=float(self.reference_amplitude[r]) if self.reference_amplitude is not None else 1.0)
            if self.rules is not None and self.has_rule[r]:
                md['level'] = {k: float(self.rules[k][r]) for k in LEVEL_FIELDS}
            modes.append(md)
        rule_any = self.rules is not None and bool(np.any(self.has_rule))
        law = dict(kind='jointed', delta=0.0, damping_gain=0.0, freq_exponent=1.0, damping_exponent=1.0,
                   decay_cycles=self.decay_cycles) if rule_any else None
        if rule_any and not np.all(self.has_rule):
            raise ValueError('mixed rule and no-rule modes: give the no-rule modes a zero rule first')
        sim = dict(feedthrough=self.feedthrough, residual=[] if self.residual is None else self.residual.ravel().tolist(),
                   transport_delay_samples=int(self.transport_delay_samples), output_delays_s=[])
        if self.step is not None:
            sim['step'] = dict(kind='equalized', band_hz=self.step[0], half_taps=self.step[1])
        return dict(schema='hertzforge-benchmark-structure/0.2-draft', kind='jointed' if rule_any else 'modal',
                    n_outputs=self.n_outputs, n_drives=self.m,
                    outputs=[dict(name=f'o{i}', role=r) for i, r in enumerate(self.roles)],
                    drives=[dict(name=f'd{j}') for j in range(self.m)], level_law=law, modes=modes, simulator=sim)


