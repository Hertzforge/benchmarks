"""Parity of the Python ports against the shipped Rust laws, on the shock record's own structure.

    MSHOCK_GYM=<mshock-research>/gym python shock/parity_rust.py --out parity.json

Each pair runs through the record's harness (mshock_gym.harness.run) on the record's structure `linear`
(shock-linear-4x4) with the record's engine identification (noise 1e-4, station seed 11), the record's engine ladder
and no measurement noise: the Rust law through the record's bridge and the Python port through an adapter. Compared
per pulse: the drives (largest absolute difference over the largest drive) and the largest SRS error over the
target (dB). Pairs: inverse / Rust `inverse` (InverseParams::default); shape / Rust `frozen` (ShapeParams::default,
safety off, the identified model frozen) and Rust `shape_default` (safety at its defaults); followed / Rust
`followed`. paired has no path through the bridge and is not checked here.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.environ.get('MSHOCK_GYM', os.path.expanduser('~/dev/mshock-research/gym')))
import laws as L  # noqa: E402
from mshock_gym import plants, harness  # noqa: E402
from mshock_gym import laws as glaws  # noqa: E402


class Port(glaws.Law):
    def __init__(self, inner):
        self.inner, self.name = inner, inner.name + ' (Python port)'
        self.drives = []

    def ladder(self, plant_ladder):
        return self.inner.ladder(list(plant_ladder))

    def start(self, setup, shoot):
        self.inner.start(setup)

    def seed(self, level):
        d = self.inner.seed(level); self.drives.append(d); return d

    def iterate(self, p):
        d = self.inner.iterate(p); self.drives.append(d); return d


class Rec(glaws.Law):
    def __init__(self, inner):
        self.inner, self.name, self.drives = inner, inner.name, []

    def ladder(self, plant_ladder):
        return self.inner.ladder(plant_ladder)

    def start(self, setup, shoot):
        self.inner.start(setup, shoot)

    def seed(self, level):
        d = self.inner.seed(level); self.drives.append(np.array(d)); return d

    def iterate(self, p):
        d = self.inner.iterate(p); self.drives.append(np.array(d)); return d


def compare(py, rs, plant, setup):
    a, b = Port(py), Rec(rs)
    ra = harness.run(plant, a, noise=0.0, seed=11, setup=setup)
    rb = harness.run(plant, b, noise=0.0, seed=11, setup=setup)
    n = min(len(a.drives), len(b.drives))
    dd = [float(np.max(np.abs(a.drives[k] - b.drives[k])) / max(np.max(np.abs(b.drives[k])), 1e-30)) for k in range(n)]
    oa = [r.over_db for r in ra.records]; ob = [r.over_db for r in rb.records]
    m = min(len(oa), len(ob))
    return dict(python=a.name, rust=b.name, pulses=[len(oa), len(ob)], end=[ra.end, rb.end],
                drive_rel_diff_first=dd[0] if dd else None, drive_rel_diff_max=max(dd) if dd else None,
                over_db_diff_max=float(np.max(np.abs(np.array(oa[:m]) - np.array(ob[:m])))) if m else None,
                over_db_python=[round(x, 3) for x in oa], over_db_rust=[round(x, 3) for x in ob])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); ap.add_argument('--structure', default='linear')
    a = ap.parse_args()
    plant = plants.get(a.structure)
    setup = plant.identify(noise_abs=1e-4, seed=11)
    out = dict(structure=a.structure, what=__doc__.strip().splitlines()[0], pairs=[])
    for py, rs in ((L.Inverse(), glaws.get('inverse')), (L.Shape(), glaws.get('frozen')),
                   (L.Shape(), glaws.get('shape_default')), (L.Followed(), glaws.get('followed'))):
        try:
            r = compare(py, rs, plant, setup)
        except Exception as ex:
            r = dict(python=py.name, rust=getattr(rs, 'name', '?'), error=f'{type(ex).__name__}: {ex}')
        out['pairs'].append(r)
        print(json.dumps({k: v for k, v in r.items() if not k.startswith('over_db_')}), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
