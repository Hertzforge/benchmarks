"""paired against the shipped Rust law (hertzforge-mimo-shock-law-math, paired.rs, PairedParams::diagonal), and why it
fails on these frames.

    cargo build --release --manifest-path shock/parity_paired_rs/Cargo.toml     # needs a checkout of the repository at
                                                                                 # ../sds beside this one
    python shock/parity_paired.py --bin <target>/release/paired_replay --problems <problems> --out paired_parity.json \
        frame-6x6_s00:half_sine frame-12x6-strong_s00:srs ...

Replay: the Python port runs the problem through the harness; every pulse it was given (level, next level, the drives
as played, the responses as measured) is handed to the Rust law in the same order, and the drives each returns are
compared with the port's (largest absolute difference over the largest drive, per pulse). Equal inputs and equal
outputs at every step make the closed loop equal.
Coupling: per problem, each paired path's in-band RMS magnitude over the largest in its control's row, the median over
lines and controls of |H_cc| over the sum of |H_cd|, d != c, and the seed's predicted error in the problem's own
small-signal model (the RMS of each control's predicted response over its reference's, dB): the seed misses by that
much before any pulse is played, whatever the structure does at level.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as hz  # noqa: E402
import laws  # noqa: E402


class Recorded(laws.Paired):
    def start(self, s):
        super().start(s)
        self.outs, self.ins = [], []

    def seed(self, level):
        self.seed_level = level
        self.outs.append(super().seed(level))
        return self.outs[-1]

    def iterate(self, p):
        self.ins.append((p.level_db, p.next_level_db, np.array(p.drives), np.array(p.responses)))
        self.outs.append(super().iterate(p))
        return self.outs[-1]


def coupling(s):
    m = s.drive_mask()
    H = s.frf[m]
    k = min(s.nc, s.nd)
    A = np.sqrt(np.mean(np.abs(H) ** 2, axis=0))
    dd = np.abs(np.stack([H[:, i, i] for i in range(k)], 1))
    off = np.abs(H[:, :k, :]).sum(2) - dd
    law = laws.Paired()
    law.start(s)
    Y = np.einsum('lcd,dl->cl', s.frf, law.D)
    err = 10 * np.log10(np.sum(np.abs(Y[:, m]) ** 2, 1) / np.sum(np.abs(law.R[:, m]) ** 2, 1))
    return dict(pair_over_row_largest=np.round(np.diag(A[:k]) / A[:k].max(1), 3).tolist(),
                median_pair_over_others=round(float(np.median(dd / off)), 4),
                seed_predicted_error_db=np.round(err, 2).tolist())


def replay(binary, s, st, tg):
    law = Recorded()
    r = hz.run(s, st, law, tg)
    with tempfile.TemporaryDirectory() as d:
        s.frf.real.astype('<f8').tofile(f'{d}/frf_re.bin')
        s.frf.imag.astype('<f8').tofile(f'{d}/frf_im.bin')
        s.reference.astype('<f8').tofile(f'{d}/reference.bin')
        for i, (_, _, dr, rs) in enumerate(law.ins):
            dr.astype('<f8').tofile(f'{d}/drv_{i}.bin')
            rs.astype('<f8').tofile(f'{d}/rsp_{i}.bin')
        json.dump(dict(n=s.n, nc=s.nc, nd=s.nd, fs=s.fs, band_lo=s.band_lo, band_hi=s.band_hi, seed_level=law.seed_level,
                       pulses=[dict(level=a, next=b) for a, b, _, _ in law.ins]), open(f'{d}/meta.json', 'w'))
        subprocess.run([binary, d], check=True)
        ro = np.fromfile(f'{d}/rust_out.bin', '<f8').reshape(len(law.outs), s.nd, s.n)
    rel = [float(np.max(np.abs(a - b)) / np.max(np.abs(b))) for a, b in zip(ro, law.outs)]
    return dict(end=r['end'], pulses=len(r['records']), max_rel_drive_diff=max(rel),
                over_db=[round(x['over_db'], 3) for x in r['records']])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bin', required=True)
    ap.add_argument('--problems', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('cases', nargs='+', help='problem:kind')
    a = ap.parse_args()
    res = {}
    for case in a.cases:
        prob, kind = case.split(':')
        s, st, tg, _ = hz.load(os.path.join(a.problems, prob), kind)
        res[case] = dict(replay(a.bin, s, st, tg), **coupling(s))
        print(case, json.dumps(res[case]), flush=True)
    json.dump(dict(law='paired', rust='hertzforge-mimo-shock-law-math src/paired.rs, PairedParams::diagonal', cases=res),
              open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
