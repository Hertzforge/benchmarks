"""Hold the steady-tone prediction of the level rule to reference_model.simulate_level on one problem.

    python sine/check_tone_state.py <sine problem folder> [--freqs 73.6,250.1] [--n-res 4] [--out result.json]

For each tone frequency (the problem's T4 dwell frequency and the small-signal resonances of `--n-res` more article
modes spread over the band, unless --freqs is given) the full-level ideal drive is played from rest, ramped in over
0.5 s, for max(40 time constants of the slowest mode near the tone, 4 s), and each mode's final envelope ratio
(envelope / reference_amplitude) is set beside the ratio sine_model predicts for that mode under that tone (the fixed
point reached by a slow ramp, sine_model.tone_ratio_curve at the one frequency). The simulated ratio is the largest
envelope over the last tone period: the envelope is a peak follower that decays by exp(-1) in five of the mode's own
periods, so a mode well above the tone sags between the tone's peaks and its level state swings within each tone
cycle; the swing (smallest over largest over the last period) is reported. Modes whose predicted ratio is under 0.05
(where the rule barely acts) are left out; the modes within a quarter octave of the tone are reported apart. The modes
are independent in modal coordinates, so one tone checks every mode at once.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sine_model as sm  # noqa: E402
rm = sm.rm


def predicted(st, f, d, s_unit=1.0):
    """Every mode's predicted envelope ratio under the steady tone d (complex volts peak) at f."""
    F = st.drive_shape @ d
    out = np.zeros(st.n_modes)
    for m in range(st.n_modes):
        if st.reference_amplitude is None or st.reference_amplitude[m] > 1e29:
            continue
        M = sm.tone_ratio_curve(st, m, np.array([f]), np.array([F[m]]))
        out[m] = sm.ratio_at(M, s_unit)[0]
    return out


def check(folder, freqs=None, n_res=4):
    meta = json.load(open(os.path.join(folder, 'problem.json')))
    st = rm.load_structure(folder)
    s = meta['full_level']['spec_amplitude_mps2_pk']
    if freqs is None:
        t4 = json.load(open(os.path.join(folder, 'test_T4.json')))
        freqs = [t4['dwell_hz']]
        am = np.flatnonzero(sm.article_modes(st))
        pick = am[np.linspace(0, am.size - 1, n_res + 2).astype(int)[1:-1]]
        freqs += [float(st.f_hz[m]) for m in pick]
    rows = []
    for f in freqs:
        d = sm.ideal(st, np.array([f]), s)[0][0]
        near = np.abs(np.log2(st.f_hz / f)) < 0.25
        tau = float(np.max(1.0 / (st.zeta[near] * 2 * np.pi * st.f_hz[near]))) if near.any() else 0.1
        T = max(40 * tau, 4.0)
        run = sm.play(st, sm.tone_drive(d, f, T), keep_modes=True)
        last = run.env[-int(np.ceil(sm.FS / f)) - 1:]
        with np.errstate(all='ignore'):
            got = last.max(0) / st.reference_amplitude
            swing = last.min(0) / np.maximum(last.max(0), 1e-300)
        want = predicted(st, f, d)
        act = want >= 0.05
        rel = np.abs(got[act] / want[act] - 1.0)
        nr = near[act]
        m0 = int(np.argmax(np.where(act, want, 0)))
        rows.append(dict(tone_hz=f, seconds=T, modes_compared=int(act.sum()),
                         ratio_error_p50=float(np.median(rel)) if rel.size else None,
                         ratio_error_p95=float(np.percentile(rel, 95)) if rel.size else None,
                         ratio_error_max=float(rel.max()) if rel.size else None,
                         near_modes=int(nr.sum()), near_ratio_error_max=float(rel[nr].max()) if nr.any() else None,
                         swing_min_all=float(swing[act].min()) if act.any() else None,
                         swing_min_near=float(swing[act][nr].min()) if nr.any() else None,
                         worst_mode_over_tone=float(st.f_hz[np.flatnonzero(act)[int(np.argmax(rel))]] / f) if rel.size else None,
                         largest_mode=dict(mode=m0, f0_hz=float(st.f_hz[m0]), predicted=float(want[m0]), simulated=float(got[m0]),
                                           predicted_change=float(rm.frequency_change(st, want)[m0]),
                                           simulated_change=float(rm.frequency_change(st, got)[m0])),
                         finite=bool(np.all(np.isfinite(run.outputs)))))
    return dict(problem=os.path.basename(os.path.normpath(folder)), spec_amplitude_mps2_pk=s, tones=rows,
                what='largest envelope ratio over the last tone period from simulate_level against the steady-tone fixed point, modes with a '
                     'predicted ratio of 0.05 or more; ratio_error = |simulated / predicted - 1|')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folder')
    ap.add_argument('--freqs', default=None)
    ap.add_argument('--n-res', type=int, default=4)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    res = check(a.folder, [float(x) for x in a.freqs.split(',')] if a.freqs else None, a.n_res)
    for r in res['tones']:
        lm = r['largest_mode']
        print(f"tone {r['tone_hz']:.2f} Hz, {r['seconds']:.1f} s: {r['modes_compared']} modes, ratio error p50 "
              f"{r['ratio_error_p50']:.4f} p95 {r['ratio_error_p95']:.4f} max {r['ratio_error_max']:.4f} (worst at f0/tone "
              f"{r['worst_mode_over_tone']:.2f}); {r['near_modes']} near, max {r['near_ratio_error_max']}; swing min "
              f"{r['swing_min_all']:.3f} all, {r['swing_min_near']} near; mode {lm['mode']} "
              f"({lm['f0_hz']:.2f} Hz) predicted {lm['predicted']:.4f} simulated {lm['simulated']:.4f}, change "
              f"{lm['predicted_change']:+.5f} against {lm['simulated_change']:+.5f}")
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
