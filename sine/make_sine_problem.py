"""Make MIMO Sine problems (draft 0.1) on the MIMO Random suite's structures.

    python sine/make_sine_problem.py --random <folder of the Random set's problems> --out <problems folder> \
        --structures frame-6x6_s00,frame-12x6-strong_s01 [--workers 6] [--down-sweep yes] [--log make.log]
    python sine/make_sine_problem.py --random ... --out ... --design   # blocks A and D and the limit block (sine/plan.py)

A group is a rig (controls x drives, as the Random folder names it) and a seed: every structure of the design at that
rig and seed is made together, members first, because a linear structure takes the full level of its realistic
member and a single-effect member (`_soft`, `_hard`) that of its two-effect member at the same tier.

Each sine problem folder is named as the Random folder it is built on and holds that folder's `structure.json` byte
for byte (its SHA-256 is checked against the Random problem's own record), so a structure id means the same in all
three suites. What is new: the reference, the tests T1 to T5 (and the limit test L1 on the limit block), the full
level, and the case files the runner reads. FORMAT_SINE.md describes every file and field.

Checks at generation (a refusal stops the problem, the log says so and the run ends with a count): every mode's
highest reachable frequency is under FS / 2 (the simulator's bound; the Random study's item 5); the full-level ideal
drive and the same drive 6 dB higher play to finite responses through reference_model.simulate_level (a sweep at
8 octaves a minute over the band, and 3 s at the dwell frequency, each ramped in over 0.5 s); the reference is finite
and positive.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zlib

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sine_model as sm  # noqa: E402
rm = sm.rm

SCHEMA = 'hertzforge-benchmark-sine-problem/0.1-draft'
TEST_SCHEMA = 'hertzforge-benchmark-sine-test/0.1-draft'
CASE_SCHEMA = 'msnb-case/1'
REFERENCE_SCHEMA = 'msnb-reference/1'
CHECK_SWEEP_RATE = 8.0          # octaves a minute: the finite-play check's sweep (the whole band in 50 s)
CHECK_DWELL_S = 3.0
LIMIT_MIN_GAP_DB = 1.0          # the limit gauge's top peak must stand this far over its second
# the laws and the tests each accepts (the shipped laws' manifests, Plan 2026-11-25 section 1)
LAW_TESTS = {'matrix': ('T1', 'T2', 'T3', 'T4', 'T5', 'L1'), 'paired': ('T3', 'T4'), 'ratio': ('T3', 'T4'),
             'steepest_descent': ('T4',), 'secant': ('T4',)}
TEST_KIND = {'T1': 'sweep', 'T2': 'sweep', 'T5': 'sweep', 'L1': 'sweep', 'T3': 'stepped', 'T4': 'dwell'}


class Refused(Exception):
    pass


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def put_array(folder, name, a, dtype, units, what):
    path = os.path.join(folder, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.asarray(a).astype(dtype).tofile(path)
    return dict(file=name, sha256=sha(path), dtype=dtype, shape=list(np.shape(a)), units=units, what=what)


def put_json(folder, name, obj):
    path = os.path.join(folder, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(obj, fh, indent=1)
    return dict(file=name, sha256=sha(path))


def noise_seed(sid, test):
    """The noise draw of a structure and test, shared by every law on it (so laws are compared on the same draw)."""
    return zlib.crc32(f'{sid}/{test}'.encode()) & 0x7FFFFFFF


def member_of(sid):
    """The structure whose full level this one takes, or None when it is sized itself."""
    base, sd = sid.rsplit('_s', 1)
    if '-realistic' in base or '-strong' in base:
        for tag in ('_soft', '_hard'):
            if base.endswith(tag):
                return base[:-len(tag)] + '_s' + sd
        return None
    return f'{base}-realistic_s{sd}'


def peaks(y):
    """Indices of the local maxima of y."""
    return np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1


def limit_rule(st, f, Yh):
    """The limit block's channel and bound: for each held-out gauge, the peaks of its full-level small-signal response
    under the ideal drive; the gauge whose highest peak stands furthest over its second; the bound flat, at the
    geometric mean of the two (so the response passes it around one resonance only, at the small signal)."""
    best = None
    names = [st.outputs[i]['name'] for i in sm.gauges(st)]
    for g in range(Yh.shape[1]):
        a = np.abs(Yh[:, g])
        p = peaks(a)
        if p.size < 2:
            continue
        top = p[np.argsort(a[p])[::-1][:2]]
        gap = 20 * np.log10(a[top[0]] / a[top[1]])
        if best is None or gap > best['gap_db']:
            bound = float(np.sqrt(a[top[0]] * a[top[1]]))
            over = a > bound
            edges = np.flatnonzero(np.diff(over.astype(int)))
            best = dict(gauge=int(g), output=int(sm.gauges(st)[g]), name=names[g], gap_db=float(gap), bound_mps2_pk=bound,
                        top_peak_hz=float(f[top[0]]), top_peak_mps2_pk=float(a[top[0]]),
                        second_peak_hz=float(f[top[1]]), second_peak_mps2_pk=float(a[top[1]]),
                        bite_db=float(20 * np.log10(a[top[0]] / bound)), regions_over_bound=int((edges.size + 1) // 2),
                        span_over_bound_hz=[float(f[over].min()), float(f[over].max())])
    if best is None or best['gap_db'] < LIMIT_MIN_GAP_DB:
        raise Refused('no held-out gauge has a resonance standing %g dB over the rest' % LIMIT_MIN_GAP_DB)
    return best


def finite_play(st, fg, dg, s, f_dwell, d_dwell):
    """The finite-play check: the full-level ideal drive and 6 dB over, a fast sweep and a dwell, through
    simulate_level. Returns the record; raises Refused on a non-finite response."""
    rec = {}
    for tag, k in (('full_level', 1.0), ('6db_over', 10 ** (6 / 20))):
        D, _ = sm.sweep_drive(fg, dg, sm.BAND, CHECK_SWEEP_RATE, 'up', scale=k)
        run = sm.play(st, D)
        y = run.outputs
        run2 = sm.play(st, sm.tone_drive(d_dwell, f_dwell, CHECK_DWELL_S, scale=k))
        ok = bool(np.all(np.isfinite(y)) and np.all(np.isfinite(run2.outputs)))
        if not ok:
            raise Refused(f'the ideal drive at {tag} does not play to finite responses')
        r = None if st.reference_amplitude is None else run.peak_envelope / st.reference_amplitude
        rec[tag] = dict(finite=ok, control_peak_over_spec=float(np.abs(y).max() / (s * k)),
                        dwell_control_peak_over_spec=float(np.abs(run2.outputs).max() / (s * k)),
                        drive_peak_v=float(np.abs(D).max()),
                        peak_ratio_max=None if r is None else float(np.max(r[sm.article_modes(st)])))
    rec['what'] = (f'a log sweep up at {CHECK_SWEEP_RATE:g} octaves a minute over the band playing the ideal drive, and '
                   f'{CHECK_DWELL_S:g} s at the dwell frequency, from rest through reference_model.simulate_level, each ramped in '
                   f'over {sm.RAMP_S:g} s (raised cosine); peaks over the specification amplitude')
    return rec


def make(random_root, out_root, sid, scale_from=None, t5_why=None, limit_block=False, check_play=True):
    """Make one problem folder. scale_from: (scale, member id) for a structure that takes its member's full level.
    t5_why: None to build T5, else why it is not built (block A only, and only when the runner runs a down-only sweep)."""
    src = os.path.join(random_root, sid)
    with open(os.path.join(src, 'problem.json')) as fh:
        rp = json.load(fh)
    s_path = os.path.join(src, 'structure.json')
    if sha(s_path) != rp['structure']['sha256']:
        raise Refused('structure.json does not match its problem record')
    st = rm.load_structure(s_path)
    fmax = rm.max_frequency(st)
    if np.any(fmax >= sm.SIM_BOUND_HZ) or np.any(st.f_hz <= 0):
        raise Refused(f'a mode can reach {fmax.max():.1f} Hz, at or above {sm.SIM_BOUND_HZ:.0f} Hz')
    band = tuple(float(x) for x in rp['grid']['band_hz'])
    if band != sm.BAND:
        raise Refused(f'the Random problem\'s band {band} is not the suite\'s {sm.BAND}')
    folder = os.path.join(out_root, sid)
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    shutil.copyfile(s_path, os.path.join(folder, 'structure.json'))
    shutil.copyfile(os.path.join(os.path.dirname(HERE), 'reference_model.py'), os.path.join(folder, 'reference_model.py'))
    shutil.copyfile(os.path.join(HERE, 'sine_model.py'), os.path.join(folder, 'sine_model.py'))
    sc = rp['scoring']['stress_modal']
    os.makedirs(os.path.join(folder, 'scoring'))
    shutil.copyfile(os.path.join(src, sc['file']), os.path.join(folder, 'scoring', 'stress_modal.f8'))
    if sha(os.path.join(folder, 'scoring', 'stress_modal.f8')) != sc['sha256']:
        raise Refused('scoring/stress_modal.f8 does not match its problem record')

    nc, nd = len(sm.controls(st)), st.n_drives
    fs_ = sm.log_grid(band, sm.SIZE_PER_OCTAVE)
    d_u, R_u, H_u = sm.ideal(st, fs_, 1.0)
    am = np.flatnonzero(sm.article_modes(st))

    # ---- full level
    sizing = {}
    if scale_from is None:
        if st.rules is None:
            raise Refused('a linear structure takes its full level from its realistic member')
        want = float(np.median(np.abs(np.asarray(st.raw['level_targets']['frequency_change'])[am])))
        curves = sm.sweep_curves(st, am, fs_, d_u)
        s, got = sm.size_full_level(st, curves, want)
        sizing = dict(rule='sized', random_median_change=want, sine_median_effect=got)
    else:
        s, mem = scale_from
        sizing = dict(rule='from_member', member=mem)
        curves = sm.sweep_curves(st, am, fs_, d_u) if st.rules is not None else None
    state = {}
    if curves is not None:
        for tag, k in (('full_level', 1.0), ('6db_over', 2.0)):
            ms, r, ch, zf = sm.state_at(st, curves, s * k)
            fpk = st.f_hz[ms] * (1 + ch)
            state[tag] = dict(ratio_p10_p50_p90=[float(x) for x in np.percentile(r, [10, 50, 90])],
                              change_p10_p50_p90=[float(x) for x in np.percentile(ch, [10, 50, 90])],
                              median_abs_change=float(np.median(np.abs(ch))),
                              damping_factor_p10_p50_p90=[float(x) for x in np.percentile(zf, [10, 50, 90])],
                              largest_drop=float(ch.min()), largest_rise=float(ch.max()), highest_hz_reached=float(fpk.max()),
                              modes=int(ms.size))
        state['what'] = ('each article mode in band under a steady tone at the frequencies within 15 % of its small-signal '
                         'frequency (the resonance the sweep passes), the level rule\'s fixed point reached by a slow ramp '
                         'of the level from zero (sine_model.tone_ratio_curve); a quasi-steady figure: a sweep that '
                         'leaves a mode too little time stays under it')

    # ---- the reference on its grid, at full level
    fg = sm.log_grid(band, sm.REF_PER_OCTAVE)
    dg, Rg, Hg = sm.ideal(st, fg, s)
    amp, ph = np.abs(Rg), np.degrees(np.angle(Rg))
    if not (np.all(np.isfinite(amp)) and np.all(amp > 0) and np.all(np.isfinite(dg))):
        raise Refused('the reference or the ideal drive is not finite and positive')
    files = dict(
        reference_hz=put_array(folder, 'reference_hz.f8', fg, '<f8', 'Hz', 'the reference grid: 20 2^(k/384), then 2000'),
        reference_amplitude=put_array(folder, 'reference_amplitude.f8', amp, '<f8', 'm/s^2 peak',
                                      'the reference amplitude at each control at full level, (points, controls)'),
        reference_phase=put_array(folder, 'reference_phase_deg.f8', ph, '<f8', 'degrees, (-180, 180]',
                                  'the reference phase at each control relative to the tone, (points, controls)'),
        ideal_drive=put_array(folder, 'ideal_drive.c16', dg, '<c16', 'V peak',
                              'the least-squares drive of the small-signal model at full level, (points, drives); not given to the law'),
    )
    Yh = np.einsum('lgd,ld->lg', sm.frf(st, fg, sm.gauges(st)), dg)
    files['ideal_held_out'] = put_array(folder, 'scoring/ideal_held_out.c16', Yh, '<c16', 'm/s^2 peak',
                                        'the held-out gauges\' small-signal response to the ideal drive at full level, (points, gauges); the scorer\'s target there')
    files['stress_modal'] = dict(file='scoring/stress_modal.f8', sha256=sc['sha256'], dtype=sc.get('dtype', '<f8'), shape=sc['shape'],
                                 units='Pa per unit modal displacement', what='the Random problem\'s file, byte for byte')
    dev = 20 * np.log10(amp / s)
    gerr = sm.reference_grid_error(st, fg, amp, ph, s)
    mid = np.sqrt(fg[1:] * fg[:-1])
    _, Rm, _ = sm.ideal(st, mid, s)
    Ri = sm.interp_reference(fg, amp, ph, mid)
    abs_err = float(np.max(np.abs(Ri - Rm)) / s)
    okm = np.abs(Rm) > s * 10 ** (-6 / 20)
    db_err_6 = float(np.max(np.abs(20 * np.log10(np.abs(Ri[okm] / Rm[okm]))))) if okm.any() else None
    sv = np.linalg.svd(Hg, compute_uv=False)
    cond = sv[:, 0] / sv[:, -1]
    reference = dict(
        spec=dict(amplitude_mps2_pk=s, amplitude_g_pk=s / sm.G0, phase_deg=0.0, shape='flat in acceleration over the band',
                  polarity='in phase at every control'),
        projection=('the specification projected onto what the system reaches in its small-signal model: the response at '
                    'the controls to the least-squares drive (Tikhonov, %g of the largest singular value), MIMO Shock Default 3' % sm.LAM_REL),
        declared_deviation=dict(amplitude_db_min=float(dev.min()), amplitude_db_max=float(dev.max()),
                                amplitude_db_rms=float(np.sqrt(np.mean(dev ** 2))),
                                share_under_minus_6_db=float(np.mean(dev < -6)), share_under_minus_20_db=float(np.mean(dev < -20)),
                                phase_deg_max_abs=float(np.max(np.abs(ph))),
                                what='the reference against the specification, every grid point and control'),
        interpolation='amplitude log-log, phase linear against log f the shorter way round, held beyond the ends (the engine\'s)',
        grid_check=dict(max_abs_error_over_spec=abs_err, max_db_error_where_over_minus_6_db=db_err_6,
                        max_db_error_anywhere=gerr[0], max_deg_error_anywhere=gerr[1],
                        what='the interpolated reference at the grid midpoints against the projection computed there'),
        condition=dict(median=float(np.median(cond)), p95=float(np.percentile(cond, 95)), max=float(cond.max())),
        files={k: files[k] for k in ('reference_hz', 'reference_amplitude', 'reference_phase')})

    # ---- the tests
    f_res, prom, i_res = sm.first_resonance(st, fs_, H_u)
    m_res = sm.nearest_mode(st, f_res)
    step_s, settle_s, m_slow, tau_slow = sm.step_seconds(st, band)
    ladder_f = sm.step_ladder(band)
    n_dwell = sm.dwell_cycles(f_res, st.zeta[m_res])
    tests = {}
    common = dict(schema=TEST_SCHEMA, band_hz=list(band), ladder=sm.LADDER, judged='updates at full level (the last rung), '
                  'outside the settle window of a step or a dwell', reference='reference.json', noise=dict(relative=sm.NOISE_REL,
                  of='the full-level specification amplitude, on every channel read'))
    for t in ('T1', 'T2', 'T5'):
        rate = sm.SWEEP_RATE[t]
        tests[t] = dict(common, id=t, kind='sweep', built=True, spacing='log', rate_oct_per_min=rate, direction=sm.SWEEP_DIRECTION[t],
                        start_hz=band[0] if sm.SWEEP_DIRECTION[t] == 'up' else band[1], sweeps=1,
                        seconds=float(sm.sweep_seconds(band, rate)))
    if t5_why is not None:
        tests['T5'] = dict(schema=TEST_SCHEMA, id='T5', kind='sweep', built=False, why='not built: ' + t5_why,
                           would_be=dict(rate_oct_per_min=1.0, direction='down', band_hz=list(band)))
    tests['T3'] = dict(common, id='T3', kind='stepped', built=True, spacing='log', steps_per_octave=sm.STEPS_PER_OCTAVE,
                       frequencies_hz=[float(x) for x in ladder_f], step_s=step_s, seconds=float(step_s * ladder_f.size),
                       settle=dict(seconds=settle_s, from_='the first update of the step at full level'),
                       rule=dict(slowest_mode=m_slow, slowest_mode_hz=float(st.f_hz[m_slow]), slowest_mode_zeta=float(st.zeta[m_slow]),
                                 slowest_tau_s=tau_slow,
                                 what='every step held (%g + %g) time constants 1 / (zeta w) of the slowest mode in band, in '
                                      'seconds and the same at every step (the runner holds every step step_s), at least %d '
                                      'cycles of the lowest step; the first %g time constants of each step judged out '
                                      '(Derivations LD-5; the loop-speed rule keeps the loop at two of the mode\'s time '
                                      'constants or more, so three loop time constants settle it to 5 %%); seconds counts '
                                      'the held steps only, not the engine\'s own settle window before each'
                                      % (sm.STEP_SETTLE_TAU, sm.STEP_JUDGE_TAU, sm.STEP_MIN_CYCLES, sm.STEP_SETTLE_TAU)))
    tests['T3']['settle']['from'] = tests['T3']['settle'].pop('from_')
    pred = None
    if curves is not None and m_res in curves:
        ms, r, ch, zf = sm.state_at(st, {m_res: curves[m_res]}, s)
        fn_full = st.f_hz[m_res] * (1 + ch[0])
        pred = dict(ratio=float(r[0]), change=float(ch[0]), damping_factor=float(zf[0]),
                    offset_in_bandwidths=float((f_res - fn_full) / (st.zeta[m_res] * zf[0] * fn_full)),
                    what='the dwell mode at full level, quasi-steady (its resonance against the tone, in half-power half-bandwidths)')
    tests['T4'] = dict(common, id='T4', kind='dwell', built=True, dwell_hz=f_res, dwell_for=dict(cycles=n_dwell),
                       seconds=float(n_dwell / f_res),
                       settle=dict(cycles=int(np.ceil(sm.DWELL_SETTLE_TAU * sm.tau_cycles(st.zeta[m_res]))),
                                   seconds=float(np.ceil(sm.DWELL_SETTLE_TAU * sm.tau_cycles(st.zeta[m_res])) / f_res),
                                   from_='the first update of the dwell at full level'),
                       resonance=dict(prominence_db=prom, mode=m_res, mode_f0_hz=float(st.f_hz[m_res]), mode_zeta=float(st.zeta[m_res]),
                                      has_level_rule=bool(st.rules is not None and st.reference_amplitude[m_res] < 1e29),
                                      at_full_level=pred,
                                      what='the lowest peak in band of the largest singular value of the small-signal response '
                                           'at the controls standing 3 dB over its flanks (1/1536 octave grid)'),
                       rule=dict(what='(%g + %g) time constants of the dwell mode, at least %g s; the first %g judged out'
                                      % (sm.DWELL_SETTLE_TAU, sm.DWELL_JUDGE_TAU, sm.DWELL_MIN_S, sm.DWELL_SETTLE_TAU)))
    tests['T4']['settle']['from'] = tests['T4']['settle'].pop('from_')
    lim = None
    if limit_block:
        Yhs = np.einsum('lgd,ld->lg', sm.frf(st, fs_, sm.gauges(st)), d_u * s)
        lim = limit_rule(st, fs_, Yhs)
        tests['L1'] = dict(tests['T1'], id='L1', limit=dict(lim, form='flat in acceleration over the band',
                           what='one limit channel on a held-out gauge, the bound at the geometric mean of its two highest '
                                'full-level small-signal peaks under the ideal drive (it bites on one resonance at the small signal)'))
    test_files = {}
    for t, obj in tests.items():
        test_files[t] = dict(put_json(folder, f'test_{t}.json', obj), built=obj['built'], kind=obj['kind'])
    put_json(folder, 'reference.json', dict(schema='hertzforge-benchmark-sine-reference/0.1-draft', **reference))
    files['case_reference'] = write_case_reference(folder, st, fg, amp, ph)

    # ---- checks
    checks = dict(highest_reachable_hz=float(fmax.max()), simulator_bound_hz=sm.SIM_BOUND_HZ)
    if check_play:
        d_dwell = sm.ideal(st, np.array([f_res]), s)[0][0]
        checks['play'] = finite_play(st, fg, dg, s, f_res, d_dwell)
    vmax = float(np.abs(dg).max())
    checks['engine'] = dict(controls=nc, drives=nd, ideal_drive_peak_v=vmax, ideal_drive_norm_max_v=float(np.linalg.norm(dg, axis=1).max()),
                            within_channel_limits=bool(nc <= sm.ENGINE_MAX_CONTROLS and nd <= sm.ENGINE_MAX_DRIVES),
                            within_drive_ceiling=bool(vmax < sm.ENGINE_MAX_DRIVE_VPK),
                            what='the shipped engine\'s limits (16 controls, 8 drives, 10 V peak a drive); recorded, not refused')
    checks['state'] = state
    meta = dict(schema=SCHEMA, status='draft, not released', format=sm.FORMAT,
                set=dict(name='Hertzforge Open Benchmark Set', suite='MIMO Sine', version=sm.FORMAT, tier='Frames'),
                evidence_level='simulator', stage=rp['stage'], effects=rp['effects'],
                structure=dict({k: rp['structure'][k] for k in ('id', 'seed', 'variant', 'level_word', 'cls', 'controls', 'drives', 'modes')},
                               file='structure.json', sha256=sha(os.path.join(folder, 'structure.json')),
                               same_as='the MIMO Random suite, version 0.2, folder ' + sid),
                band_hz=list(band), band_from='the Random problem\'s grid.band_hz', fs_hz=sm.FS,
                level_rule_in_time="reference_model.simulate_level: the structure's own rule, each mode's peak follower of its "
                                   "running amplitude (decay exp(-1) in decay_cycles periods); under a steady tone the envelope "
                                   "is the modal amplitude times max(1, f / f_m) (sine_model)",
                full_level=dict(spec_amplitude_mps2_pk=s, spec_amplitude_g_pk=s / sm.G0, sizing=sizing,
                                what='the median over the article\'s modes in band of the size of change (drop plus rise) at the '
                                     'resonance the tone passes, quasi-steady, equal to the median |frequency change| of the '
                                     'Random problem\'s full level (level_targets); a linear structure takes its realistic '
                                     'member\'s, a single-effect structure its two-effect member\'s'),
                reference=dict(file='reference.json', **{k: reference[k] for k in ('spec', 'declared_deviation', 'condition')}),
                tests=test_files, files=files,
                given_to_the_law=['the controls\' measured responses and the drives (the engine\'s own identification)',
                                  'the reference', 'the test and its ladder', 'limit channels and their bounds (L1)'],
                kept_for_scoring=['held-out gauges and their ideal response', 'the modal stress matrix', 'the ideal drive'],
                service_state=rp.get('service_state'),
                checks=checks,
                made=dict(by='sine/make_sine_problem.py', when=time.strftime('%Y-%m-%d'), random_problem_record=rp['structure']['sha256']))
    put_json(folder, 'problem.json', meta)
    write_cases(folder, sid, meta, tests)
    return folder, meta


def g9(x):
    """A number as the runner writes one: 9 significant digits."""
    return float(f'{x:.9g}')


def write_case_reference(folder, st, fg, amp, ph):
    """The reference in the runner's own form (msnb-reference/1, FORMAT_RUN.md): the same grid, amplitudes and phases
    as the binary files, 9 significant digits, phases in (-180, 180]."""
    ph = np.where(ph <= -180.0, ph + 360.0, ph)
    body = dict(schema=REFERENCE_SCHEMA, controls=[st.outputs[i]['name'] for i in sm.controls(st)],
                frequency_hz=[g9(x) for x in fg], amplitude_m_s2_pk=[[g9(x) for x in row] for row in amp],
                phase_deg=[[g9(x) if g9(x) > -180.0 else 180.0 for x in row] for row in ph])
    path = os.path.join(folder, 'case_reference.json')
    with open(path, 'w') as fh:
        json.dump(body, fh, separators=(',', ':'))
    return dict(file='case_reference.json', sha256=sha(path), schema=REFERENCE_SCHEMA,
                what='the reference as the runner reads it (msnb-reference/1): the binary reference files at 9 significant digits')


def case_test(obj, t):
    if TEST_KIND[t] == 'sweep':
        return dict(type='sweep', band_hz=obj['band_hz'], rate_oct_per_min=obj['rate_oct_per_min'], direction=obj['direction'])
    if TEST_KIND[t] == 'stepped':
        return dict(type='stepped', band_hz=obj['band_hz'], steps_per_octave=obj['steps_per_octave'], step_s=obj['step_s'])
    return dict(type='dwell', freq_hz=obj['dwell_hz'], dwell_s=obj['seconds'])


def write_cases(folder, sid, meta, tests):
    """The runner's case files (msnb-case/1, FORMAT_RUN.md of the runner draft), one per law and test the law's
    manifest accepts and the problem built. Every path is relative to the problems folder (the runner's --root)."""
    st = rm.load_structure(folder)
    cr = meta['files']['case_reference']
    spec = meta['full_level']['spec_amplitude_mps2_pk']
    names = [st.outputs[i]['name'] for i in sm.controls(st)]
    for law, accepts in LAW_TESTS.items():
        for t in accepts:
            obj = tests.get(t)
            if obj is None or not obj.get('built'):
                continue
            case = dict(schema=CASE_SCHEMA, case_id=f'{sid}__{t}__{law}',
                        structure=dict(file=f'{sid}/structure.json', sha256=meta['structure']['sha256']),
                        controls=names, fs_hz=sm.FS, law=dict(id=law, params={}), test=case_test(obj, t),
                        reference=dict(file=f'{sid}/' + cr['file'], sha256=cr['sha256'], phase='auto'),
                        full_level_scale=1.0,
                        ladder=dict(levels_db=list(sm.LADDER['table_db']), rung=dict(sm.LADDER['per_rung'])),
                        noise=dict(rms_m_s2=g9(sm.NOISE_REL * spec), seed=noise_seed(sid, t)),
                        limits=[], protective='default')
            if t == 'L1':
                L = obj['limit']
                case['limits'] = [dict(output=L['name'], bound=dict(frequency_hz=list(obj['band_hz']),
                                                                    amplitude_m_s2_pk=[g9(L['bound_mps2_pk'])] * 2))]
            put_json(folder, f'cases/{t}__{law}.json', case)


# ---------------------------------------------------------------------------------------------- a group and the run
def order_members(sids):
    first = [s for s in sids if member_of(s) is None]
    return first + [s for s in sids if member_of(s) is not None]


def make_group(args):
    random_root, out_root, sids, limit_set, t5_set, t5_why, check_play = args
    lines = []
    scales = {}
    for sid in order_members(sids):
        t0 = time.time()
        mem = member_of(sid)
        try:
            frm = None
            if mem is not None:
                if mem not in scales:
                    raise Refused(f'its member {mem} was not made')
                frm = (scales[mem], mem)
            why = None if sid in t5_set else (t5_why or 'T5 is built on block A only')
            folder, meta = make(random_root, out_root, sid, frm, why, sid in limit_set, check_play)
            scales[sid] = meta['full_level']['spec_amplitude_mps2_pk']
            e = meta['checks']['engine']
            lines.append((sid, 'made', f"spec {meta['full_level']['spec_amplitude_g_pk']:.3g} g pk, ideal drive peak "
                          f"{e['ideal_drive_peak_v']:.3g} V, highest reachable {meta['checks']['highest_reachable_hz']:.1f} Hz, "
                          f"{len(os.listdir(os.path.join(folder, 'cases')))} cases, {time.time() - t0:.1f} s"))
        except Refused as ex:
            lines.append((sid, 'REFUSED', str(ex)))
        except Exception as ex:   # a fault in the maker is a refusal too, said as such
            lines.append((sid, 'REFUSED', f'maker fault: {type(ex).__name__}: {ex}'))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--random', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--structures', help='comma list (each with its member made first when needed)')
    ap.add_argument('--design', action='store_true', help='blocks A and D and the limit block (sine/plan.py)')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--down-sweep', choices=('yes', 'no'), default='yes',
                    help='yes: the runner runs a down-only sweep (its FORMAT_RUN.md: "A down-only sweep is the engine\'s own"), '
                         'so T5 is built on block A')
    ap.add_argument('--no-play', action='store_true', help='skip the finite-play check (tests only)')
    ap.add_argument('--log', default=None)
    a = ap.parse_args()
    import plan
    if a.design:
        sids, limit_set = plan.structures(), set(plan.limit_structures())
    else:
        sids, limit_set = a.structures.split(','), set(plan.limit_structures())
        extra = [member_of(s) for s in sids if member_of(s) and member_of(s) not in sids]
        sids = sorted(set(sids) | set(extra))
    groups = {}
    for s in sids:
        base, sd = s.rsplit('_s', 1)
        groups.setdefault((base.split('-')[1].split('_')[0], int(sd)), []).append(s)
    os.makedirs(a.out, exist_ok=True)
    log = open(a.log, 'a', buffering=1) if a.log else sys.stdout
    log.write(f"GENERATION START: {len(sids)} structures in {len(groups)} groups, {a.workers} workers, down sweep {a.down_sweep}, "
              f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    t5_set = set(plan.block_a()) if a.down_sweep == 'yes' else set()
    t5_why = None if a.down_sweep == 'yes' else 'the runner has not reported that the engine runs a down-only sweep'
    jobs = [(a.random, a.out, g, limit_set, t5_set, t5_why, not a.no_play) for _, g in sorted(groups.items())]
    made = refused = 0
    if a.workers > 1:
        from multiprocessing import get_context
        with get_context('spawn').Pool(a.workers) as pool:
            results = pool.imap_unordered(make_group, jobs)
            for lines in results:
                for sid, status, what in lines:
                    made += status == 'made'; refused += status != 'made'
                    log.write(f"{time.strftime('%H:%M:%S')} {status} {sid}: {what}\n")
    else:
        for j in jobs:
            for sid, status, what in make_group(j):
                made += status == 'made'; refused += status != 'made'
                log.write(f"{time.strftime('%H:%M:%S')} {status} {sid}: {what}\n")
    log.write(f"GENERATION END: {made} made, {refused} refused, of {len(sids)}, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return 0 if refused == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
