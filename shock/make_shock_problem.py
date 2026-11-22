"""Make MIMO Shock problems (draft 0.2) on the MIMO Random suite's structures.

    python shock/make_shock_problem.py --random <folder of the Random set's problems> --out <problems folder> \
        --structures frame-6x6-realistic_s00,frame-12x6-strong_s01 [--targets srs,half_sine,sawtooth]

Each shock problem folder is named as the Random folder it is built on and holds that folder's `structure.json`
byte for byte (its SHA-256 is checked against the Random problem's own record), so a structure id means the same in
both suites. What is new is the target and its full level:

- targets: `srs` (the shock record's SRS shape, one synthesized waveform in phase at every control),
  `half_sine` and `sawtooth` (11 ms, MIL-STD-810H 516 shapes, with pre- and post-pulse compensation), each
  projected onto what the frame can reach in its small-signal model (the response at the controls to the
  least-squares drive); the specification's departure from the projection is recorded as the declared deviation;
- the full level: the scale at which the pulse, played by the least-squares drive of the small-signal model, moves
  the structure at the pulse's peak as far as the Random suite's full level does, median over the article's modes
  in 20 to 2000 Hz (`size_full_level`). A linear structure takes the scale of its realistic member at the same seed
  and rig, so the three tiers of a structure share one level.

Checks at generation (a refusal stops the generation): every mode's highest reachable frequency is under FS / 2 (the
simulator's bound; the Random study's item 5), the full-level ideal drive and the same drive 6 dB higher play to
finite responses, and each target's SRS points are positive and finite.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import shock_model as sm  # noqa: E402
rm = sm.rm

FORMAT = '0.2-draft'
SCHEMA = 'hertzforge-benchmark-shock-problem/0.2-draft'
LADDER = dict(table_db=[-12.0, -9.0, -6.0, -3.0, -1.5, 0.0], pulses_a_level=6, passes_at_full=3,
              tolerance_db=dict(inner=list(sm.SRS_INNER_DB), inner_share=sm.SRS_INNER_SHARE, outer=list(sm.SRS_OUTER_DB)),
              predicted_over_limit_db=sm.SRS_INNER_DB[1], min_step_db=0.75, max_pulses=60,
              what="the shock record's engine ladder: a level is left on a passing pulse (every control's SRS inside "
                   "Method 516.8's tolerance: -1.5/+3 dB on at least 90 % of its points, every point within -3/+6 dB); "
                   "a step is halved while the predicted worst point is over +3 dB; six pulses a level without a pass "
                   "ends the run; three passes at full level")
TOL = dict(srs_db=dict(inner=list(sm.SRS_INNER_DB), inner_share=sm.SRS_INNER_SHARE, outer=list(sm.SRS_OUTER_DB)),
           srs_db_secondary=[-3.0, 3.0], effective_duration=[0.8, 1.2], level_mean_db=1.5, limit_line_db=6.0,
           band_516_pct=20.0, duration_pct=10.0, velocity_pct=20.0, compensation_pct=20.0, post_slope_pct_td=10.0,
           what='MIL-STD-810H Method 516.8 (checked 2026-11-22, Method_516_Tolerances_Check_2026-11-22.md): the SRS at '
                'the twelfth-octave centres 10 to 1995 Hz, Q 10, within -1.5/+3 dB on at least 90 % of the points of '
                'each control and every point within -3/+6 dB; the SRS target also its effective duration TE (the '
                'shortest time holding every sample above 10 % of the peak) within 0.8 to 1.2 of the target\'s. '
                'Secondary, labelled: every point within -3/+3 dB (draft 0.1\'s reading, the shock record\'s). The '
                'Level reading: mean SRS error within 1.5 dB and no point over +6 dB (the acceptance proposal). '
                'Classical pulses: the waveform inside +/-20 % of the peak around the target over the pulse window, '
                'the duration (width above 10 % of the peak) within +/-10 % of the target\'s, the velocity change '
                'over the pulse within +/-20 % of the target\'s, the compensation (before and after the pulse) inside '
                '+/-20 % of the peak, and for the sawtooth the post-pulse slope (terminal peak to inside 20 % of the '
                'peak) under 10 % of TD')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def unit_target(kind, st, seed):
    """(reference (controls, N) at unit scale, points (controls, 93), fine points (the same points), band, extra)."""
    nc = len(sm.controls(st))
    if kind == 'srs':
        w, syn = sm.synth_srs_pulse(seed)
        pts = np.tile(sm.srs(w, sm.FS, sm.NAT516)[0], (nc, 1))      # the waveform's own SRS: the reference meets its target
        fine = np.tile(sm.srs(w, sm.FS, sm.NAT516)[0], (nc, 1))
        dev = sm.error_db(pts[0], sm.srs_shape(sm.NAT516))[sm.NAT516 >= 50.0]   # the shape is specified from 50 Hz
        extra = dict(synthesis_rms_db=syn, synthesis_vs_shape_db=[float(dev.min()), float(dev.max())],
                     shape='+6 dB an octave 50 to 100 Hz, flat above', polarity='in phase',
                     points_are="the synthesized waveform's own SRS (so the reference itself is inside the tolerance)")
        band = sm.BAND_SRS
    else:
        w, ideal, win = sm.classical(kind)
        pts = np.tile(sm.srs(w, sm.FS, sm.NAT516)[0], (nc, 1))
        fine = np.tile(sm.srs(w, sm.FS, sm.NAT516)[0], (nc, 1))
        extra = dict(duration_s=sm.T_CLASSICAL, window_s=list(win), compensation=f'pre and post half-sines of {sm.COMP_FACTOR:g} '
                     'times the duration, amplitudes solved for zero end velocity and displacement',
                     ideal_peak=1.0, velocity_change=float(np.sum(ideal) / sm.FS))
        band = (sm.BAND_CLASSICAL_LO, sm.BAND_SRS[1])
    return np.tile(w, (nc, 1)), pts, fine, band, extra


def ideal_drive(H, mask, ref):
    G = sm.regularized_inverse(H, mask)
    R = np.fft.rfft(ref, axis=1)
    return np.fft.irfft(np.einsum('ldc,cl->dl', G, R), n=sm.N, axis=1)


def linear_peaks(st, d):
    """Peak modal envelope per unit drive scale, from a play small enough that the rule does not act (r < 1e-4)."""
    s = 1.0
    for _ in range(3):
        _, run = sm.play(st, d * s)
        r = run.peak_envelope / st.reference_amplitude
        top = float(np.max(r[sm.article_modes(st, (0.0, sm.FS))])) if st.rules is not None else 0.0
        if top <= 1e-4:
            break
        s *= 1e-5 / top
    return run.peak_envelope / s


def size_full_level(st, d_unit):
    """The scale s with median over the article's in-band modes of effect_size(s A / a_ref) equal to the median
    |frequency change| of the Random suite's full level (level_targets). Returns (s, record)."""
    A = linear_peaks(st, d_unit)
    am = sm.article_modes(st)
    want = float(np.median(np.abs(np.asarray(st.raw['level_targets']['frequency_change'])[am])))
    def med(s):
        return float(np.median(sm.effect_size(st, s * A / st.reference_amplitude)[am]))
    lo, hi = 1e-6, 1e6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if med(mid) < want else (lo, mid)
        if hi / lo < 1 + 1e-9:
            break
    s = float(np.sqrt(lo * hi))
    r = s * A / st.reference_amplitude
    return s, dict(random_median_change=want, shock_median_change_linear=med(s), modes_counted=int(am.sum()),
                   peak_ratio_p10_p50_p90_linear=[float(x) for x in np.percentile(r[am], [10, 50, 90])])


def state_at_peak(st, run):
    """Signed frequency change of each article mode at its peak envelope (as tested), summarized by effect."""
    r = run.peak_envelope / st.reference_amplitude
    ch = rm.frequency_change(st, r)
    am = sm.article_modes(st)
    R = st.rules
    soft = am & (R['drop'] * np.minimum(1, r) >= R['rise'] * np.minimum(R['rise_cap'], r) ** R['rise_exponent'])
    hard = am & ~soft
    pc = lambda m: [round(float(x), 6) for x in np.percentile(ch[m], [10, 50, 90])] if m.any() else None
    fpk = st.f_hz * (1 + ch)
    return dict(peak_ratio_p50=float(np.median(r[am])), change_softening_p10_p50_p90=pc(soft), n_softening=int(soft.sum()),
                change_hardening_p10_p50_p90=pc(hard), n_hardening=int(hard.sum()),
                largest_rise=float(ch[am].max()), largest_drop=float(ch[am].min()), highest_hz_reached=float(fpk.max()))


def make(random_root, out_root, sid, kinds, linear_scale_from=None):
    src = os.path.join(random_root, sid)
    rp = json.load(open(os.path.join(src, 'problem.json')))
    s_path = os.path.join(src, 'structure.json')
    assert sha(s_path) == rp['structure']['sha256'], f'{sid}: structure.json does not match its problem record'
    st = rm.load_structure(s_path)
    # the simulator's limit, at generation (the Random study's item 5): refuse, do not record
    fmax = rm.max_frequency(st)
    if np.any(fmax >= sm.FS / 2) or np.any(st.f_hz <= 0):
        raise SystemExit(f'REFUSED {sid}: a mode can reach {fmax.max():.1f} Hz, at or above {sm.FS / 2:.0f} Hz')
    folder = os.path.join(out_root, sid)
    os.makedirs(folder, exist_ok=True)
    shutil.copyfile(s_path, os.path.join(folder, 'structure.json'))
    for f in ('reference_model.py',):
        shutil.copyfile(os.path.join(os.path.dirname(HERE), f), os.path.join(folder, f))
    shutil.copyfile(os.path.join(HERE, 'shock_model.py'), os.path.join(folder, 'shock_model.py'))
    H = sm.frf(st)
    seed = int(rp['structure']['seed'])
    targets, checks = {}, {}
    for kind in kinds:
        ref_u, pts_u, fine_u, band, extra = unit_target(kind, st, seed)
        mask = sm.line_mask(band)
        d_u = ideal_drive(H, mask, ref_u)
        # the target is the reachable projection of the specification: the small-signal response at the controls to
        # the least-squares drive (the Random suite's targets are responses of the structure in service; here the
        # specification is projected so a linear structure can be controlled on every frame). The specification's
        # departure from it is the declared deviation.
        spec_pts = pts_u
        ref_u = np.fft.irfft(np.einsum('lcd,dl->cl', H, np.fft.rfft(d_u, axis=1)), n=sm.N, axis=1)
        pts_u = sm.srs(ref_u, sm.FS, sm.NAT516)
        fine_u = sm.srs(ref_u, sm.FS, sm.NAT516)
        dd = sm.error_db(pts_u, spec_pts)
        extra = dict(extra, declared_deviation_db=dict(min=float(dd.min()), max=float(dd.max()),
                     rms=float(np.sqrt(np.mean(dd ** 2))), what='SRS of the projected target over the specification'))
        if st.rules is not None:
            s, sizing = size_full_level(st, d_u)
        else:
            if linear_scale_from is None or kind not in linear_scale_from:
                raise SystemExit(f'{sid}: a linear structure takes its scale from its realistic member (pass it)')
            s, sizing = linear_scale_from[kind], dict(from_member='the realistic structure of the same seed and rig')
        # checks: finite play at full level and 6 dB over; the state reached
        y0, run0 = sm.play(st, d_u * s)
        y6, run6 = sm.play(st, d_u * s * 10 ** (6 / 20))
        ok = bool(np.all(np.isfinite(y0)) and np.all(np.isfinite(y6)))
        if not ok:
            raise SystemExit(f'REFUSED {sid}/{kind}: the ideal drive does not play to finite responses')
        pts, fine = pts_u * s, fine_u * s
        assert np.all(np.isfinite(pts)) and np.all(pts > 0)
        ref = ref_u * s
        e0 = sm.error_db(sm.srs(y0, sm.FS, sm.NAT516), pts)
        np.asarray(ref, '<f8').tofile(os.path.join(folder, f'reference_{kind}.f8'))
        tgt = dict(schema='hertzforge-benchmark-shock-target/0.2-draft', kind=kind, scale=s, units='m/s^2',
                   reference=dict(file=f'reference_{kind}.f8', sha256=sha(os.path.join(folder, f'reference_{kind}.f8')),
                                  shape=list(ref.shape), dtype='<f8', what='full-level waveform at each control'),
                   natural_hz=sm.NAT516.tolist(), q=sm.Q, points=pts.tolist(), fine_natural_hz=sm.NAT516.tolist(),
                   effective_duration_s=sm.effective_duration(ref).round(6).tolist() if kind == 'srs' else None,
                   fine_points=fine.tolist(), band_hz=list(band), peak_g=float(np.abs(ref).max() / 9.80665), **extra)
        put = os.path.join(folder, f'target_{kind}.json')
        json.dump(tgt, open(put, 'w'), indent=1)
        targets[kind] = dict(file=f'target_{kind}.json', sha256=sha(put), scale=s, peak_g=tgt['peak_g'])
        checks[kind] = dict(sizing=sizing, ideal_drive_full_level_srs_error_db=[float(e0.min()), float(e0.max())],
                            ideal_drive_rms_v=float(np.sqrt(np.mean((d_u * s) ** 2))),
                            state_at_full_level=state_at_peak(st, run0) if st.rules is not None else None,
                            state_6db_over=state_at_peak(st, run6) if st.rules is not None else None,
                            finite_at_full_and_6db_over=ok)
    meta = dict(schema=SCHEMA, status='draft, not released', format=FORMAT,
                set=dict(name='Hertzforge Open Benchmark Set', suite='MIMO Shock', version=FORMAT, tier='Frames'),
                evidence_level='simulator', stage=rp['stage'], effects=rp['effects'],
                structure=dict({k: rp['structure'][k] for k in ('id', 'seed', 'variant', 'level_word', 'cls', 'controls', 'drives', 'modes')},
                               file='structure.json', sha256=sha(os.path.join(folder, 'structure.json')),
                               same_as='the MIMO Random suite, version 0.2, folder ' + sid),
                grid=dict(fs_hz=sm.FS, n=sm.N, srs_points='twelfth-octave centres 1000 10^(k/40), k = -80 to 12: 10.0 to 1995.3 Hz (93), Method 516.8', fine_points='the same points (draft 0.1 kept a separate twelfth-octave reading; it is now the main one)', q=sm.Q),
                level_rule_in_time="reference_model.simulate_level: the structure's own rule, each mode's peak follower of "
                                   "its running amplitude (decay exp(-1) in decay_cycles periods), from rest at each pulse",
                given_to_the_law=['the small-signal frequency response at the controls (reference_model.discrete_frf at r = 0, '
                                  'the ideal identification of this draft)', 'the target', 'the ladder'],
                ladder=LADDER, tolerance=TOL, targets=targets,
                checks=dict(highest_reachable_hz=float(fmax.max()), simulator_bound_hz=sm.FS / 2, by_target=checks),
                made=dict(by='shock/make_shock_problem.py', when=time.strftime('%Y-%m-%d'), random_problem_record=rp['structure']['sha256']))
    json.dump(meta, open(os.path.join(folder, 'problem.json'), 'w'), indent=1)
    return folder, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--random', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--structures', required=True); ap.add_argument('--targets', default='srs,half_sine,sawtooth')
    a = ap.parse_args()
    kinds = a.targets.split(',')
    sids = a.structures.split(',')
    order = sorted(sids, key=lambda s: 0 if ('-realistic' in s or '-strong' in s) else 1)   # members first
    done = {}
    for sid in order:
        lin = None
        if not ('-realistic' in sid or '-strong' in sid):
            base, sd = sid.rsplit('_s', 1)
            mem = f'{base}-realistic_s{sd}'
            if mem not in done:
                _, done[mem] = make(a.random, a.out, mem, kinds)
            lin = {k: v['scale'] for k, v in done[mem]['targets'].items()}
        folder, meta = make(a.random, a.out, sid, kinds, lin)
        done[sid] = meta
        print(sid, {k: round(v['peak_g'], 2) for k, v in meta['targets'].items()},
              {k: (c['state_at_full_level'] or {}).get('highest_hz_reached') for k, c in meta['checks']['by_target'].items()}, flush=True)


if __name__ == '__main__':
    main()
