"""the study: stage every part 2 export as a format 0.2 problem folder and check it with the study's tools.

    PY=python; export PYTHONDONTWRITEBYTECODE=1
    nice -n 10 $PY stage2.py [--seeds 0,1] [--out $OBS_DATA/problems] [--jobs 4]

Each export is staged in its own process by code/make_problem.py (--export-dir). Then, per folder, with
the study's reference model (the folder's own reference_model.py):
  stager checks       frf parity, stress frf parity, runner plant equal, targets equal (the stager's own)
  rule at full level  reference_model.frequency_change and level_state at each mode's full-level amplitude ratio
                      (envelope over RMS x the lab fixed point's modal RMS / reference amplitude) against level_targets
  rule off = none     a level variant with every term zeroed has the none variant's FRF and modes exactly
  bound               reference_model.max_frequency under the Nyquist frequency
Writes stage2_summary.json beside the folders.
"""
import os, sys, json, argparse, subprocess, importlib.util
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); BM2 = os.path.normpath(os.path.join(HERE, '..'))
EXPORTS = os.path.join(os.environ.get('OBS_GEN_BULK', os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'exports/obs_gen2')), 'plants')


def load_rm(folder):
    spec = importlib.util.spec_from_file_location('rm_' + os.path.basename(folder), os.path.join(folder, 'reference_model.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def stage_one(args):
    exp, out = args
    seed, rig, variant = os.path.basename(exp).split('_', 2); ns, nc = rig.split('x')
    p = subprocess.run([sys.executable, os.path.join(BM2, 'make_problem.py'), '--seed', seed[1:], '--ns', ns, '--nc', nc, '--change', variant,
                        '--export-dir', exp, '--no-cases', '--out', out], capture_output=True, text=True)
    return exp, p.returncode, p.stdout.splitlines()[0] if p.returncode == 0 else (p.stdout + p.stderr)[-800:]


def check(folder, exp):
    rm = load_rm(folder); st = rm.load_structure(folder); prob = rm.load_json(os.path.join(folder, 'problem.json'))
    side = np.load(os.path.join(exp, 'side_v2.npz')); s = rm.load_json(os.path.join(folder, 'structure.json'))
    ck = prob['provenance']['checks']; out = dict(folder=os.path.basename(folder), stage=prob['stage'], effects=prob['effects'],
                                                  frf_parity=ck['frf_parity']['value'], stress_frf_parity=ck['stress_frf_parity']['value'],
                                                  runner_plant_equal=ck['runner_structure_equal'], target_banded_equal=ck['target_banded_equal'],
                                                  target_line_equal=ck['target_line_equal'],
                                                  eq_step={k: ck['equalized_step_residual'][k] for k in ('db_median', 'db_p95', 'db_max', 'phase_deg_p95')} if 'equalized_step_residual' in ck else None)
    if s['kind'] == 'jointed':
        ref = np.array([m['reference_amplitude'] for m in s['modes']]); el = np.arange(ref.size) >= 6
        k1 = side['lab_k1env'] if 'lab_k1env' in side.files else np.full(ref.size, float(s['level_targets']['envelope_over_rms']))
        k2 = side['lab_sine_over_rms'] if 'lab_sine_over_rms' in side.files else k1
        r1 = np.where(el, k1 * side['lab_sigma'] / ref, 0.0); r2 = np.where(el, k2 * side['lab_sigma'] / ref, 0.0)
        R = st.rules         # drop and damping at the mean envelope, the rise at the sine amplitude 2 sigma (V1, the study's ruling 2)
        import obs_gen2 as g
        zfull = np.array([m['zeta'] for m in s['modes']]) * (side['lab_zfac'] if 'lab_zfac' in side.files else 1.0)
        Wr = g.wpow(r1, zfull, np.array([m['f_hz'] for m in s['modes']])) if g.POWER_WEIGHTED else np.minimum(1, r1)
        fc = -R['drop'] * Wr + R['rise'] * np.minimum(R['rise_cap'], r2) ** R['rise_exponent']     # the spectrum's expectation (V2)
        tgt = np.array(s['level_targets']['frequency_change'])
        fz = rm.level_state(st, r1)
        out['rule_at_mean_envelope_vs_targets_max_abs'] = float(np.abs(rm.frequency_change(st, r1) - tgt).max())
        cap = np.array([m['level'].get('damping_cap', np.inf) for m in s['modes']])   # reference_model does not apply the cap yet
        z0 = np.array([m['zeta'] for m in s['modes']])
        zf = np.minimum(0.9, z0 * (1 + R['damping_gain'] * (Wr if g.POWER_WEIGHTED else np.minimum(cap, r1)))) / z0
        out['reference_model_damping_uncapped_vs_targets_max_abs'] = float(np.abs(np.asarray(fz[1]) / z0 - np.array(s['level_targets']['damping_factor'])).max())
        out['modes_with_damping_cap'] = int(np.isfinite(cap).sum())
        out.update(rule_vs_targets_max_abs=float(np.abs(fc - tgt).max()),
                   damping_vs_targets_max_abs=None if zf is None else float(np.abs(zf - np.array(s['level_targets']['damping_factor'])).max()),
                   max_frequency_hz=float(np.max(rm.max_frequency(st))), nyquist_hz=4096.0,
                   n_rising=int((tgt > 0).sum()), n_falling=int((tgt < 0).sum()),
                   rise_max=float(tgt.max()), drop_median_elastic=float(np.median(tgt[6:])))
    return out


def rule_off_equals_none(folder_level, folder_none):
    rm = load_rm(folder_level); a = rm.load_structure(folder_level); b = rm.load_structure(folder_none)
    sa, sb = rm.load_json(os.path.join(folder_level, 'structure.json')), rm.load_json(os.path.join(folder_none, 'structure.json'))
    for m in sa['modes']:
        m['level'] = dict(m['level'], drop=0.0, rise=0.0, damping_gain=0.0)
    same_modes = all(x['f_hz'] == y['f_hz'] and x['zeta'] == y['zeta'] and x['drive_shape'] == y['drive_shape'] and x['output_shape'] == y['output_shape']
                     for x, y in zip(sa['modes'], sb['modes'])) and len(sa['modes']) == len(sb['modes'])
    f = np.arange(20.0, 2000.0001, 0.5); Ha = rm.frf(a, f); Hb = rm.frf(b, f)
    st0 = rm.from_runner_structure(dict(rm.to_runner_structure(a), modes=[dict(mm, level=dict(mm['level'], drop=0.0, rise=0.0, damping_gain=0.0))
                                                                   for mm in rm.to_runner_structure(a)['modes']]))
    r = np.full(len(sa['modes']), 0.7)
    fz = rm.frequency_change(st0, r)
    return dict(same_modes=bool(same_modes), frf_max_rel=float(np.abs(Ha - Hb).max() / np.abs(Hb).max()),
                zeroed_rule_change_max_abs=float(np.abs(fz).max()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seeds', default='0,1'); ap.add_argument('--out', default=os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'problems'))
    ap.add_argument('--jobs', type=int, default=4); a = ap.parse_args()
    seeds = {f's{s}' for s in a.seeds.split(',')}
    exps = sorted(os.path.join(EXPORTS, d) for d in os.listdir(EXPORTS) if d.split('_')[0] in seeds)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(a.jobs) as ex:
        staged = list(ex.map(stage_one, [(e, a.out) for e in exps]))
    rows, by = [], {}
    for exp, rc, msg in staged:
        if rc != 0:
            rows.append(dict(export=exp, failed=msg)); print('FAILED', exp, msg); continue
        r = check(msg, exp); r['export'] = exp; rows.append(r); by[os.path.basename(exp)] = msg
        print(json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items() if k != 'export'}))
    off = {}
    for k, folder in by.items():
        sd, rig, var = k.split('_', 2)
        if var != 'none' and f'{sd}_{rig}_none' in by:
            off[k] = rule_off_equals_none(folder, by[f'{sd}_{rig}_none']) if var.startswith('realistic') else None
    off = {k: v for k, v in off.items() if v}
    for k, v in off.items():
        print('rule off vs none', k, v)
    json.dump(dict(rows=rows, rule_off_equals_none=off), open(os.path.join(a.out, 'stage2_summary.json'), 'w'), indent=1)
