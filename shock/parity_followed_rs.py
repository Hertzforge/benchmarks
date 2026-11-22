"""Parity of followed through the closed-loop library (followed_rs/, laws_rust.py) against the research bridge's
followed (mshock-research bridge, ctypes), closed loop, pulse by pulse.

    MSHOCK_GYM=<mshock-research>/gym python shock/parity_followed_rs.py --out parity.json [--problems DIR --problem P]

(a) On the shock record's structures (`moderate`, where drops are fitted, and `linear`), through the record's
harness (mshock_gym.harness.run), the record's identification (noise 1e-4, seed 11), no measurement noise, as
parity_rust.py. (b) Optionally on one suite problem through this suite's harness (noise 1e-3, seed 11). Compared
per pulse: the drives (largest absolute difference over the largest drive) and the over-test (dB). (c) followed-hold
through the library against the Python port (laws.FollowedHold) on the record's structures: on `linear` no drop is
fitted, so the warp, where the port departs from the Rust law, is unused and the two should agree to rounding.
"""
import argparse
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get('MSHOCK_GYM', os.path.expanduser('~/dev/mshock-research/gym')))
import laws_rust as LR  # noqa: E402
from mshock_gym import plants, harness as gh, _bridge  # noqa: E402
from mshock_gym import laws as glaws  # noqa: E402


class Rec:
    """Records every drive a law returns; `inner` has the shock harness's Law interface or the gym's."""

    def __init__(self, inner, gym_setup_fn=None):
        self.inner, self.name, self.drives, self.fn = inner, inner.name, [], gym_setup_fn

    def ladder(self, t):
        return self.inner.ladder(list(t))

    def start(self, setup, shoot=None):
        if self.fn is None:
            return self.inner.start(setup, shoot) if shoot is not None else self.inner.start(setup)
        return self.inner.start(self.fn(setup))

    def seed(self, level):
        d = np.array(self.inner.seed(level)); self.drives.append(d); return d

    def iterate(self, p):
        d = np.array(self.inner.iterate(p)); self.drives.append(d); return d

    def next_level(self, history, planned):
        return planned

    def report(self):
        return {}


def gym_view(s):
    """The gym's setup with the drive mask the bridge would form (band and trusted)."""
    v = types.SimpleNamespace(**{k: getattr(s, k) for k in ('fs', 'n', 'nc', 'nd', 'band_lo', 'band_hi', 'reference',
                                                             'natural_hz', 'q', 'points', 'frf')})
    v.drive_mask = lambda: np.asarray(s.trusted, bool)
    v.lines = s.n // 2 + 1
    return v


class BridgeOnSuite:
    """The bridge's followed on this suite's Setup (trusted = the drive band)."""
    name = 'followed (bridge)'

    def ladder(self, t):
        return [-40.0, -34.0] + list(t)

    def start(self, s):
        v = types.SimpleNamespace(**s.__dict__); v.trusted = s.drive_mask()
        self.h = _bridge.setup_to_bridge(v); self.law = _bridge.RustLaw(self.h, {'kind': 'followed'})

    def seed(self, level):
        return self.law.seed(level)

    def iterate(self, p):
        return self.law.iterate(p.index, p.level_db, p.next_level_db, p.drives, p.responses)


def diff(a, b, oa, ob):
    n = min(len(a.drives), len(b.drives))
    dd = [float(np.max(np.abs(a.drives[k] - b.drives[k])) / max(np.max(np.abs(b.drives[k])), 1e-30)) for k in range(n)]
    m = min(len(oa), len(ob))
    return dict(pulses=[len(oa), len(ob)], drive_rel_diff_max=max(dd) if dd else None,
                over_db_diff_max=float(np.max(np.abs(np.array(oa[:m]) - np.array(ob[:m])))) if m else None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    ap.add_argument('--problems'); ap.add_argument('--problem'); ap.add_argument('--kind', default='srs')
    a = ap.parse_args()
    out = dict(what=__doc__.strip().splitlines()[0], library=LR.lib_info(),
               bridge=dict(path=str(_bridge._DEFAULT) if not os.environ.get('MSHOCK_BRIDGE_LIB') else os.environ['MSHOCK_BRIDGE_LIB']),
               cases=[])
    out['bridge']['sha256'] = LR.hashlib.sha256(open(out['bridge']['path'], 'rb').read()).hexdigest()
    for name in ('moderate', 'linear'):
        plant = plants.get(name)
        setup = plant.identify(noise_abs=1e-4, seed=11)
        x, y = Rec(LR.RustFollowed('followed'), gym_view), Rec(glaws.get('followed'))
        rx = gh.run(plant, x, noise=0.0, seed=11, setup=setup)
        ry = gh.run(plant, y, noise=0.0, seed=11, setup=setup)
        r = dict(case=f'record structure {name}', end=[rx.end, ry.end],
                 **diff(x, y, [q.over_db for q in rx.records], [q.over_db for q in ry.records]))
        out['cases'].append(r); print(json.dumps(r), flush=True)
    import laws as L
    for name in ('linear', 'moderate'):
        plant = plants.get(name)
        setup = plant.identify(noise_abs=1e-4, seed=11)
        x, y = Rec(LR.RustFollowed('followed-hold'), gym_view), Rec(L.FollowedHold(), gym_view)
        rx = gh.run(plant, x, noise=0.0, seed=11, setup=setup)
        ry = gh.run(plant, y, noise=0.0, seed=11, setup=setup)
        r = dict(case=f'followed-hold, library against the Python port, record structure {name}', end=[rx.end, ry.end],
                 rust_report=x.inner.report(), port_report=dict(probe_checks=y.inner.checks, holds=y.inner.law.holds),
                 **diff(x, y, [q.over_db for q in rx.records], [q.over_db for q in ry.records]))
        out['cases'].append(r); print(json.dumps({k: v for k, v in r.items() if not k.endswith('_report')}), flush=True)
    if a.problem:
        import harness as hz
        s, st, tg, meta = hz.load(os.path.join(a.problems, a.problem), a.kind)
        x, y = Rec(LR.RustFollowed('followed')), Rec(BridgeOnSuite())
        rx = hz.run(s, st, x, tg, noise=1e-3, seed=11)
        ry = hz.run(s, st, y, tg, noise=1e-3, seed=11)
        r = dict(case=f'suite problem {a.problem} {a.kind}', end=[rx['end'], ry['end']],
                 **diff(x, y, [q['over_db'] for q in rx['records']], [q['over_db'] for q in ry['records']]))
        out['cases'].append(r); print(json.dumps(r), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
