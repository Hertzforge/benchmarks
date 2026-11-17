"""Stage a Frames problem folder of the Open Benchmark Set (DRAFT 0.2) from an export of the frame generator.

Version 0.2 (the study): writes format 0.2-draft (FORMAT.md). It takes the per-mode rule from the
export: a structure of case schema mm-x4-case/2 whose modes carry `level` objects is staged with those objects as they
are; a version 1 structure (one structure-wide `joint`) is staged with that rule restated on every mode, as
the 0.1 converter does. --export-dir names the export folder (the study writes them under
$OBS_DATA/exports/); without it the stopped study's structure folder is read, as in 0.1. The service
truth (unit A and B in service, the stress truth) still comes from the stopped study's context (x4_truth.context)
for the same seed and rig: when the reworked generator supplies its own, this is where it must be read.
Default output root: $OBS_DATA/problems_f_tests/staged.

    PY=python
    PYTHONDONTWRITEBYTECODE=1 $PY make_problem.py --seed 0 --ns 6 --nc 12 [--change none|realistic|strong]
                                                     [--analytic-run RUN_FOLDER] [--out ROOT]

--ns and --nc are the study's order (shakers, controls). The folder is named in the software's order:
ROOT/frame-<controls>x<drives>[-variant]_s<seed>/ , ROOT = $OBS_EARLIER_STUDY/benchmarks_draft (local, not synced).

It READS the study's structure folder ($X/plants/s<seed>_<ns>x<nc>_<change>/: structure, outputs, side arrays, references),
the study's context for the seed (x4_truth.context: the service responses and the stress truth, which the exports
do not hold) and the batch's case files ($X/cases_b/). It writes only into ROOT. Nothing of the study is changed.

Checks made while staging, recorded in problem.json under provenance.checks:
  frf parity            reference_model.frf against the study's FRF (x4_truth.frf times force_per_volt), every exported output
  stress frf parity     reference_model.stress_frf against the study's modal stress route
  runner structure          the structure object rebuilt from structure.json equals the study's structure file, value for value
  targets               the banded target expanded by the reference model, and the line target, equal the study's reference files byte for byte
  cases                 every batch case file of this case (default and opened passes) is rebuilt from the problem folder and compared
  simulator (optional)  the reference model against the software simulator's own analytic FRF kept in a full run folder

This file imports the study's modules; the reference model and the scorer do not.
"""
import sys
sys.dont_write_bytecode = True
import os, json, glob, hashlib, shutil, zlib, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, 'generator')      # the generator's modules
X4 = EXP
sys.path.insert(0, HERE)
import reference_model as rm

ROOT = os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'problems_f_tests/staged')
BULK = os.path.join(os.environ.get('OBS_EARLIER_STUDY', 'earlier-study'), 'x4')
SET_VERSION = '0.2-draft'
SCORECARD = 'mimo-metrics-report-card/0.1-provisional'
LIMITS_ID = 'frames-fair-run/0.1-draft'
GRID = dict(fs_hz=8192.0, nfft=16384, band_hz=[20.0, 2000.0])
LEVEL = {'none': None, 'realistic': dict(frequency_drop=0.01, damping_factor=1.5), 'strong': dict(frequency_drop=0.03, damping_factor=3.0)}
LEVEL_WORD = {'none': 'Linear', 'realistic': 'Realistic', 'strong': 'Harsh'}

# The limits of a fair run: the batch's case file, written out. RUN is x4_matrix.RUN with the batch's overrides
# (b_write_case): 64 frames a loop, autospectra record, protections by pass. Compared with the real case files below.
RUN = dict(levels_db=[-12.0, -9.0, -6.0, -3.0, 0.0], loops_at_full=30, record_loops=15, frames_per_loop=64,
           memory_loops=4.0, window='hann', overlap=0.5,
           identification=dict(frames=None, noise_frames=None, start_vrms=None, below_reference_db=None),
           noise_sigma=1e-4, noise_below_db=40.0, drive_ceiling_vrms=5.0, drive_pretest_max_vrms=1.0,
           alarm_db=None, abort_db=None, drive_seed=None, budget_fraction=None, keep_loop_matrices=False,
           response_record='autospectra', protections='default')
PASS = {'default': dict(protections='default', alarm_db=None, abort_db=None),
        'opened': dict(protections='opened', alarm_db=None, abort_db=None)}
TOKEN = {'default': '', 'opened': '_divopen'}
FRF_MODE = {'hold': {'kind': 'hold'}, 'update': {'kind': 'update', 'blend': 0.5}}


EXTRA_LEVEL = ()   # the study's fix: reference_model.py carries damping_cap (the study's notes); was tuple(k for k in ('damping_cap',) if k not in rm.LEVEL_FIELDS)   # spec fields reference_model.py does not know (yet):
                                 # the stager carries them itself. Empty once the reference model reads damping_cap (the study, part 2).


def without_extra(export_structure):
    """The structure with the level fields reference_model.to_runner_structure does not carry removed (for the equality check)."""
    if export_structure.get('kind') != 'jointed':
        return export_structure
    return dict(export_structure, modes=[dict(m, level={k: v for k, v in m['level'].items() if k not in EXTRA_LEVEL}) if m.get('level') else m for m in export_structure['modes']])


def put_back_extra(export_structure, st):
    """Copy the extra level fields of structure.json into a runner structure built by reference_model.to_runner_structure."""
    if export_structure.get('kind') != 'jointed':
        return export_structure
    for pm, sm in zip(export_structure['modes'], st.raw['modes']):
        for k in EXTRA_LEVEL:
            if sm.get('level') and k in sm['level'] and pm.get('level') is not None:
                pm['level'][k] = sm['level'][k]
    return export_structure


def as_case2(export_structure):
    """A jointed structure of mm-x4-case/1 (one structure-wide joint) written as mm-x4-case/2: the joint restated on every
    mode, joint keeping decay_cycles. A structure that already carries per-mode levels, or a modal structure, is returned as is."""
    if export_structure.get('kind') != 'jointed' or any(pm.get('level') is not None for pm in export_structure['modes']):
        return export_structure
    return dict(export_structure, modes=[dict(pm, level=rm.law_as_level(export_structure['joint'])) for pm in export_structure['modes']],
                joint=dict(decay_cycles=export_structure['joint']['decay_cycles']))


def structure_id(ns, nc, change):
    return f'frame-{nc}x{ns}' + ('' if change == 'none' else '-' + change)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def put_array(folder, name, a, dtype, units, what, seen='scorer'):
    a = np.ascontiguousarray(a).astype(dtype); path = os.path.join(folder, name); os.makedirs(os.path.dirname(path), exist_ok=True)
    a.tofile(path)
    return dict(file=name, dtype=dtype, shape=list(a.shape), units=units, what=what, seen_by=seen, sha256=sha(path))


def put_json(folder, name, obj, **kw):
    path = os.path.join(folder, name)
    with open(path, 'w') as fh:
        json.dump(obj, fh, **kw)
    return sha(path)


def limits(ns):
    """limits.json: the fair run in words and numbers, and the runner's own keys for it."""
    return dict(
        id=LIMITS_ID,
        grid=dict(GRID, df_hz=0.5, lo_line=40, n_lines=3961),
        identification=dict(rule='the product defaults', frames=max(64, 10 * ns), frames_rule='64, or 10 a drive if that is more',
                            noise_frames=32, start_vrms=0.01, below_reference_db=20.0),
        ladder_db=RUN['levels_db'], loops_per_step_below_full=1, loops_at_full=RUN['loops_at_full'], record_loops=RUN['record_loops'],
        estimator=dict(frames_per_loop=RUN['frames_per_loop'], window='hann', overlap=0.5, memory_loops=RUN['memory_loops'],
                       loop_seconds=66.0),
        model=dict(default={'kind': 'hold'}, an_entry_may_declare={'kind': 'update', 'blend': 0.5}),
        noise=dict(below_smallest_control_reference_db=RUN['noise_below_db'],
                   seed_rule='for the baselines: crc32 of the study case id, and 0x7fffffff; any 31-bit seed otherwise, recorded in the result'),
        drive=dict(ceiling_vrms=RUN['drive_ceiling_vrms'], pretest_max_vrms=RUN['drive_pretest_max_vrms'], law_time_budget='the product default'),
        tolerance_lines=dict(alarm_db=None, abort_db=None, why='the target cannot be reached; the tolerance is a scorecard line, not a run-time stop'),
        protections=dict(
            first_pass=dict(name='default', envelope_abort_lines=10, divergence_loops=5, divergence_floor_db=3.0, overload_fraction=0.9, settle_loops=1),
            second_pass=dict(name='opened', when='the first pass did not complete', same_noise_seed=True, envelope_abort_lines='unlimited',
                             divergence_loops=16, divergence_floor_db=1000.0, overload_fraction=1.0)),
        outcome=dict(rule='completed, or stopped with the guard, the loop and the reason (requirement R10)',
                     scored_if='completed, or stopped inside the record with at least 40 Welch frames', min_partial_frames=40),
        record=dict(response='autospectra of every output over the record', drive='the full drive spectral matrix over the record'),
        runner=dict(schema='mm-x4-case/1', run=RUN, passes=PASS, frf_mode=FRF_MODE,
                    note='run, with one pass and the law filled in, is the "run" object of the research runner\'s case file'))


def runner_case(folder, case_id, target, law, params, frf, pas, sim_seed):
    """The research runner's case file (mm-x4-case/1) for one entry on a staged problem, as a dict.

    structure and reference are given inline as the objects themselves (the runner also takes them by file name).
    """
    prob = rm.load_json(os.path.join(folder, 'problem.json')); st = rm.load_structure(folder); lim = rm.load_json(os.path.join(folder, 'limits.json'))
    run = dict(lim['runner']['run'], law=dict(id=law, params=params), frf_mode=lim['runner']['frf_mode'][frf], sim_seed=sim_seed)
    run.update(lim['runner']['passes'][pas])
    outputs = [dict(name=o['name'], role=o['role']) for o in st.outputs]
    n_phys = sum(o['role'] != 'monitor' for o in outputs)
    if prob['scorecard']['stress_route'] == 'played_modal':
        run['moments'] = dict(first=n_phys, count=len(outputs) - n_phys)
    g = prob['grid']
    export_structure = rm.to_runner_structure(st); step = st.raw['simulator'].get('step')
    if step is not None:
        export_structure['step'] = step
    put_back_extra(export_structure, st)
    schema = 'mm-x4-case/2' if step is not None else 'mm-x4-case/1'      # a step (the study part 2 exports) needs schema 2; the stopped study's cases stay /1
    return dict(schema=schema, case_id=case_id, grid=dict(fs_hz=g['fs_hz'], nfft=g['nfft'], band_hz=g['band_hz']),
                structure=export_structure, outputs=outputs, reference=dict(kind='sdm', n=prob['structure']['controls'],
                lo_line=g['lo_line'], n_lines=g['n_lines'], target=target), run=run)


class _ExportModal:
    """The lab modal model of a generator export, enough for x4_truth.frf: fn, zeta, Phi at the exported rows
    and drives (zeros elsewhere), h(f)."""
    def __init__(self, side):
        self.fn = np.asarray(side['fn'], float); self.zeta = np.asarray(side['zeta'], float); self.wn = 2 * np.pi * self.fn
        self.Phi = np.zeros((int(side['ndof']), self.fn.size)); self.Phi[np.asarray(side['rows'])] = side['Phi_rows']; self.Phi[np.asarray(side['shk'])] = side['Phi_shk']

    def h(self, f):
        w = 2 * np.pi * np.asarray(f)[None, :]; wn = self.wn[:, None]
        return 1.0 / (wn ** 2 - w ** 2 + 2j * self.zeta[:, None] * wn * w)


def g2_context(pd, side, NF):
    """the study part 2: the service truth and the lab model from the generator's export (side_v2.npz, ref_line.c16,
    geometry_v2.json) in the shape make() reads from x4_truth.context. Indices c, held, tips are into U = the rows."""
    from types import SimpleNamespace
    nc, nh, nt = int(side['n_control']), int(side['n_held']), int(side['n_tip']); n = nc + nh + nt
    c, held, tips = np.arange(nc), nc + np.arange(nh), nc + nh + np.arange(nt)
    GAU = np.zeros((n, NF)); GAU[c] = side['GAU_c']; GAU[held] = side['GAU_h']
    GBU = np.zeros((n, NF)); GBU[held] = side['GBU_h']; GBU[tips] = side['GBU_t']
    GA = np.fromfile(os.path.join(pd, 'ref_line.c16'), '<c16').reshape(NF, nc, nc)
    geo = rm.load_json(os.path.join(pd, 'geometry_v2.json'))
    fr = SimpleNamespace(xyz=[np.array(p) for p in geo['nodes']], el=geo['elements'],
                         lump={int(d['node']): (d['kg'], 0.0) for d in geo['lumped_masses']},
                         meta=dict(mounts=geo['mounts'], tips=geo['tips'], dims=geo['dims_m'], member=geo.get('member')))
    return dict(c=c, held=held, tips=tips, U=np.asarray(side['rows']), shk=np.asarray(side['shk']), mlab=_ExportModal(side), SB=np.asarray(side['SB']),
                GAU=GAU, GBU=GBU, GA=GA, truthB=dict(rms=side['truthB_rms'], d8=side['truthB_d8'], d4=side['truthB_d4']), S=dict(units={2: fr}))


def make(seed, ns, nc, change='none', out=ROOT, analytic_run=None, export_dir=None, cases=True):
    sys.path.insert(0, X4); sys.path.insert(0, EXP)
    import x4_truth as xt, x4_export as xe
    from x4_truth import F, NF
    g2 = bool(export_dir) and os.path.exists(os.path.join(export_dir, 'plant_v2.json'))     # a generator export
    if g2:
        pd, tag = export_dir, '_v2'; tier = change; cases = False
        side = np.load(os.path.join(pd, 'side_v2.npz')); ctx = g2_context(pd, side, NF)
    else:
        xe.register_tiers() if change != 'none' else None
        tier = xe.SERVICE_TIER[change]; ctx = xt.context(seed, ns, nc, tier)
        pd = export_dir or xe.plant_dir(seed, ns, nc, change); tag = '' if change == 'none' else '_m'
        side = np.load(os.path.join(pd, f'side{tag}.npz'))
    export_structure = rm.load_json(os.path.join(pd, f'plant{tag}.json')); outs = rm.load_json(os.path.join(pd, f'outputs{tag}.json'))
    sid = structure_id(ns, nc, change)
    folder = os.path.join(out, f'{sid}_s{seed:02d}'); os.makedirs(os.path.join(folder, 'scoring'), exist_ok=True)
    c, held, tips = ctx['c'], ctx['held'], ctx['tips']; n_c, n_h, n_t = c.size, held.size, tips.size; n_phys = n_c + n_h + n_t
    rows, shk = np.asarray(side['rows']), np.asarray(side['shk']); fr = ctx['S']['units'][2]; xyz = np.array(fr.xyz)
    assert np.array_equal(rows, ctx['U'][np.r_[c, held, tips]]) and np.array_equal(shk, ctx['shk'])
    assert np.array_equal(side['fn'], ctx['mlab'].fn) and np.array_equal(side['SB'], ctx['SB'])
    assert np.array_equal(side['truthB_rms'], ctx['truthB']['rms']) or change != 'none'
    where = lambda dof: dict(node=int(dof // 6), direction='xyz'[int(dof % 6)], xyz=[round(float(v), 6) for v in xyz[dof // 6]])

    # ---- structure.json
    outputs = []
    for i, o in enumerate(outs):
        if i < n_phys:
            grp = 'control' if i < n_c else 'gauge' if i < n_c + n_h else 'tip'
            outputs.append(dict(name=o['name'], role=o['role'], group=grp, **where(rows[i])))
        else:
            outputs.append(dict(name=o['name'], role='monitor', group='modal_monitor', mode=i - n_phys))
    zeta = np.asarray(side['zeta'], float); modes = []
    for m, pm in enumerate(export_structure['modes']):
        assert pm['f_hz'] == float(side['fn'][m])
        z = pm['zeta'] if 'zeta' in pm else float(zeta[m])
        assert z == float(zeta[m]) and ('q' not in pm or pm['q'] == float(1 / (2 * z)))
        mode = dict(f_hz=pm['f_hz'], zeta=z, drive_shape=pm['drive_shape'], output_shape=pm['output_shape'])
        if 'reference_amplitude' in pm:
            mode['reference_amplitude'] = pm['reference_amplitude']
        if export_structure['kind'] == 'jointed':      # the per-mode rule: the export's own, or its structure-wide joint restated
            mode['level'] = ({k: pm['level'][k] for k in rm.LEVEL_FIELDS + EXTRA_LEVEL if k in pm['level']} if pm.get('level') is not None
                             else rm.law_as_level(export_structure['joint']))
            rm.check_level(mode['level'], f'mode {m}')
        modes.append(mode)
    level_law = None; level_targets = None
    if export_structure['kind'] == 'jointed' and g2:     # the export states its own targets; joint carries decay_cycles (delta 0)
        level_law = dict(kind='jointed', **export_structure['joint'],
                         what='every mode carries its own rule (its level object, FORMAT.md); the envelope is a peak follower of the '
                              'mode\'s amplitude that decays by exp(-1) in decay_cycles periods of its small-signal frequency; the structure-wide '
                              'law is zero and only carries decay_cycles')
        level_targets = rm.load_json(os.path.join(pd, 'level_targets_v2.json'))
    elif export_structure['kind'] == 'jointed':
        level_law = dict(kind='jointed', decay_cycles=export_structure['joint']['decay_cycles'],
                         what='every mode carries its own rule (its level object, FORMAT.md); the envelope is a peak follower of the '
                              'mode\'s amplitude that decays by exp(-1) in decay_cycles periods of its small-signal frequency')
        tchange = side['target_change'] if 'target_change' in side.files else -np.asarray(side['target_drop'], float)
        afl = LEVEL.get(change)
        level_targets = dict(what='what each mode is meant to do at full level under the reference line drive: signed frequency change '
                                  '(fraction; negative a drop, positive a rise) and damping factor',
                             at_full_level=None if afl is None else dict(frequency_change=-afl['frequency_drop'], damping_factor=afl['damping_factor']),
                             frequency_change=[float(v) for v in tchange],
                             damping_factor=[float(v) for v in side['target_zfac']],
                             envelope_over_rms=float(dict(zip(side['law_keys'].tolist(), side['law_vals'].tolist()))['env_over_rms']))
    sim = {k: export_structure[k] for k in ('feedthrough', 'residual', 'transport_delay_samples')}
    sim.update({k: export_structure.get(k) for k in ('softening', 'noisy_outputs', 'discretization')} if export_structure['kind'] == 'modal' else {'output_delays_s': export_structure['output_delays_s']})
    if export_structure.get('step') is not None:
        sim['step'] = export_structure['step']      # the simulator's step (the study); reference_model.to_runner_structure does not carry it, runner_case adds it
    structure = dict(
        schema=rm.SCHEMA_STRUCTURE, structure_id=sid, seed=seed, variant=change, kind=export_structure['kind'],
        what='A modal model: every mode has a natural frequency, a damping ratio, a drive shape (per volt) and an output shape. '
             'Every output is an acceleration. See reference_model.py for the formulas.',
        units=dict(drive='V', output='m/s^2', output_spectral_density='(m/s^2)^2/Hz', frequency='Hz', stress='Pa', force_per_volt='N/V'),
        n_modes=len(modes), n_drives=int(ns), n_outputs=len(outputs), force_per_volt=float(side['force_per_volt']),
        rigid_modes=6, rigid_modes_what='the first six modes are the article on its soft suspension; they do not change with level',
        drives=[dict(name=f'd{j + 1:02d}', **where(shk[j])) for j in range(ns)],
        outputs=outputs, level_law=level_law, level_targets=level_targets, simulator=sim, modes=modes)
    s_sha = put_json(folder, 'structure.json', structure)
    shutil.copyfile(os.path.join(HERE, 'reference_model.py'), os.path.join(folder, 'reference_model.py'))
    st = rm.load_structure(folder); checks = {}

    # ---- geometry.json (for figures; the generator frame_model.py is the full definition)
    geometry = dict(what='Unit B, the article on the shakers: nodes (m), beam elements, lumped masses. For figures. The seeded generator is the definition.',
                    nodes=[[round(float(v), 6) for v in p] for p in xyz],
                    elements=[dict(i=int(e['i']), j=int(e['j']), group=e['group']) for e in fr.el],
                    lumped_masses=[dict(node=int(n), kg=float(m)) for n, (m, _) in sorted(fr.lump.items())],
                    mounts=[int(n) for n in fr.meta['mounts']], tips=[int(n) for n in fr.meta['tips']],
                    dims_m=[float(v) for v in fr.meta['dims']])
    g_sha = put_json(folder, 'geometry.json', geometry)

    # ---- check: reference model against the study's FRF (per volt), physical outputs; and the stress route
    fpv = float(side['force_per_volt']); phys = np.arange(n_phys)
    Href = xt.frf(ctx['mlab'], rows, shk) * fpv; H = rm.frf(st, F, phys)
    checks['frf_parity'] = dict(what='max |H_reference_model - H_study| / max |H_study|, every physical output, drive and line',
                                value=float(np.abs(H - Href).max() / np.abs(Href).max()), lines=int(NF), outputs=int(n_phys), drives=int(ns))
    Sref = np.einsum('rm,mf,sm->frs', ctx['SB'], ctx['mlab'].h(F), ctx['mlab'].Phi[shk], optimize=True) * fpv
    Ss = rm.stress_frf(st, ctx['SB'], F)
    checks['stress_frf_parity'] = dict(what='the same for fibre stress per volt', value=float(np.abs(Ss - Sref).max() / np.abs(Sref).max()), rows=int(ctx['SB'].shape[0]))
    plain = without_extra({k: v for k, v in as_case2(export_structure).items() if k != 'step'})
    checks['runner_structure_equal'] = bool(rm.to_runner_structure(st) == plain)
    step = st.raw['simulator'].get('step')
    if step is not None and step.get('kind') == 'equalized':     # the study's ruling after V1: record the equalized step's residual
        Hd = rm.discrete_frf(st, F, GRID['fs_hz'], phys); dly = int(step.get('half_taps', 2))
        Hc = H * np.exp(-2j * np.pi * F * dly / GRID['fs_hz'])[:, None, None]
        keep = np.abs(Hc) >= 0.01 * np.abs(Hc).max(0, keepdims=True)                  # points above 1 percent of the element's peak
        db = np.abs(20 * np.log10(np.abs(Hd) / np.abs(Hc)))[keep]; ph = np.abs(np.degrees(np.angle(Hd / Hc)))[keep]
        checks['equalized_step_residual'] = dict(
            what='reference_model.discrete_frf (the equalized step) against the continuous model delayed half_taps samples, physical outputs, '
                 '20 to 2000 Hz, points above 1 percent of each element\'s peak; dB of the magnitude ratio and phase in degrees',
            step=step, db_median=float(np.median(db)), db_p95=float(np.percentile(db, 95)), db_max=float(db.max()),
            phase_deg_p95=float(np.percentile(ph, 95)), points=int(keep.sum()))
    if plain != as_case2(export_structure):
        checks['runner_structure_equal_what'] = 'compared without the export\'s step key (structure.json keeps it in simulator; runner_case puts it back)'
    if as_case2(export_structure) is not export_structure:
        checks['runner_structure_equal_what'] = 'the export is mm-x4-case/1; compared with its joint restated on every mode (mm-x4-case/2)'
    if analytic_run:      # the software simulator's own analytic FRF of the same structure, from a full run folder
        Ha = np.fromfile(os.path.join(analytic_run, 'frf_plant_analytic.c16'), '<c16').reshape(NF, -1, ns)
        n = min(Ha.shape[1], n_phys); delay = export_structure.get('discretization', {}).get('delay', 0) if export_structure['kind'] == 'modal' else 0
        Hd = H[:, :n] * np.exp(-2j * np.pi * F * delay / GRID['fs_hz'])[:, None, None]
        mag = 20 * np.log10(np.abs(Ha[:, :n]) / np.abs(H[:, :n]))
        checks['simulator_analytic'] = dict(what='software simulator analytic FRF (frf_plant_analytic.c16) against the reference model; the simulator delays every path by `delay` samples',
                                            run=analytic_run, delay_samples=delay, magnitude_db_median=float(np.median(mag)),
                                            magnitude_db_p95_abs=float(np.percentile(np.abs(mag), 95)), magnitude_db_max_abs=float(np.abs(mag).max()),
                                            complex_rel_max_after_delay=float(np.abs(Ha[:, :n] - Hd).max() / np.abs(Hd).max()))

    # ---- targets
    B3, bol, bidx = rm.band_matrix(F, 3); assert np.array_equal(B3, xt.B3) and np.array_equal(bol, xt.BO3)
    t = ctx['GAU'][c] @ B3.T; rs = np.load(os.path.join(pd, 'ref_side.npz')); assert np.array_equal(t, rs['t'])
    centre = 1000.0 * 2.0 ** (bidx / 3.0)
    bands = [dict(index=int(k), centre_hz=float(centre[i]), lo_hz=float(centre[i] * 2 ** (-1 / 6)), hi_hz=float(centre[i] * 2 ** (1 / 6)),
                  first_line_hz=float(F[bol == i][0]), last_line_hz=float(F[bol == i][-1]), lines=int((bol == i).sum())) for i, k in enumerate(bidx)]
    tb = dict(what='The specification: unit A in service at the controls, each control autospectrum averaged over the lines of each one-third-octave band. '
                   'No cross terms are specified. A controller that needs a full matrix holds each value flat across its band (reference_model.expand_banded).',
              units='(m/s^2)^2/Hz, one-sided', per_octave=3, centre_hz=1000.0, band_rule='line f belongs to band round(3 log2(f / 1000))',
              controls=[o['name'] for o in outputs[:n_c]], bands=bands, values=[[float(v) for v in r] for r in t])
    tb_sha = put_json(folder, 'target_banded.json', tb)
    Gb = rm.expand_banded(np.array(tb['values']), bol)
    ref_b = np.fromfile(os.path.join(pd, 'ref_banded.c16'), '<c16'); ref_l = np.fromfile(os.path.join(pd, 'ref_line.c16'), '<c16')
    checks['target_banded_equal'] = bool(np.array_equal(Gb.astype('<c16').ravel(), ref_b))
    GA = (ctx['GA'] + xt.H_(ctx['GA'])) / 2
    checks['target_line_equal'] = bool(np.array_equal(GA.astype('<c16').ravel(), ref_l))
    files = {}
    files['target_line'] = put_array(folder, 'target_line.c16', GA, '<c16', '(m/s^2)^2/Hz, one-sided',
                                     'unit A in service: the full spectral matrix at the controls, (lines, controls, controls). The line target, and the scorer\'s source for what the system can reach',
                                     seen='law (problem .../line) and scorer')
    assert files['target_line']['sha256'] == sha(os.path.join(pd, 'ref_line.c16'))

    # ---- scoring truth
    files['service_a_psd'] = put_array(folder, 'scoring/service_a_psd.f8', np.concatenate([ctx['GAU'][c], ctx['GAU'][held]]), '<f8', '(m/s^2)^2/Hz',
                                       'unit A in service: autospectra at the controls, then at the held-out gauges, (channels, lines). The report card\'s reference')
    files['service_b_psd'] = put_array(folder, 'scoring/service_b_psd.f8', np.concatenate([ctx['GBU'][held], ctx['GBU'][tips]]), '<f8', '(m/s^2)^2/Hz',
                                       'unit B in service: autospectra at the held-out gauges, then at the component tips, (channels, lines). The response spectra\'s reference')
    tr = ctx['truthB']
    files['stress_truth'] = put_array(folder, 'scoring/stress_truth.f8', np.stack([tr['rms'], tr['d8'], tr['d4']]), '<f8', 'Pa; damage rates in Pa^b Hz',
                                      'unit B in service, per beam element (worst of its 8 fibre points): RMS stress, damage rate with exponent 8, damage rate with exponent 4, (3, elements)')
    files['stress_modal'] = put_array(folder, 'scoring/stress_modal.f8', ctx['SB'], '<f8', 'Pa per unit modal displacement',
                                      'the article on the shakers: fibre stress per unit modal displacement, (8 x elements, modes); rows are element by element, 8 fibre points each')
    if int(side['n_modal']) if 'n_modal' in side.files else 0:
        files['modal_monitor_scale'] = put_array(folder, 'scoring/modal_monitor_scale.f8', side['modal_scale'], '<f8', '1',
                                                 'monitor output m is this scale times the acceleration of mode m, (modes)')

    # ---- limits, problem
    lim = limits(ns); l_sha = put_json(folder, 'limits.json', lim, indent=1)
    cls = 'square' if ns == nc else 'more controls than drives'
    route = 'played_modal' if export_structure['kind'] == 'jointed' else 'drive'
    if export_structure['kind'] == 'jointed':
        R = st.rules
        stage, effects = 2, [e for e, on in (('softening', bool((R['drop'] > 0).any())), ('hardening', bool((R['rise'] > 0).any()))) if on]
    else:
        stage, effects = 1, []
    problem = dict(
        schema=rm.SCHEMA_PROBLEM, status='Version 0.2 of the set (format 0.2-draft), DOI 10.5281/zenodo.24310742.',
        set=dict(name='Hertzforge Open Benchmark Set', version=SET_VERSION, suite='MIMO Random', tier='Frames'),
        evidence_level='simulator', stage=stage, effects=effects,
        stage_what='1: the structure does not change with level; 2: it does (its modes soften, harden, or both)',
        structure=dict(id=sid, seed=seed, variant=change, level_word=LEVEL_WORD.get(change, change), cls=cls, controls=int(nc), drives=int(ns),
                       modes=len(modes), elements=int(ctx['SB'].shape[0] // 8), file='structure.json', sha256=s_sha,
                       reference_model='reference_model.py', geometry='geometry.json', geometry_sha256=g_sha),
        problems={f'{sid}/banded': dict(target='banded'), f'{sid}/line': dict(target='line')},
        grid=lim['grid'],
        roles=dict(control=int(n_c), held_out=dict(gauge=int(n_h), tip=int(n_t)), monitor=len(outputs) - n_phys,
                   what='control: the law sees it and it has a target. held_out: recorded, never given to the law; gauges have a target (unit A in service), '
                        'tips have a response-spectrum reference (unit B in service). monitor: a scoring device (one per mode on a structure that changes with level).'),
        given_to_the_law=['the frequency response at the controls, by identification under the limits', 'the target at the controls', 'the limits'],
        kept_for_scoring=['held-out gauges and tips, and their service responses', 'the stress of unit B in service', 'the modal stress matrix'],
        targets=dict(banded=dict(file='target_banded.json', sha256=tb_sha, seen_by='law'), line=files.pop('target_line')),
        limits=dict(id=LIMITS_ID, file='limits.json', sha256=l_sha),
        scorecard=dict(version=SCORECARD, scorer='scorer.py', stress_route=route,
                       stress_route_what={'drive': 'stress and gauge responses from the played drive through the structure (it does not change with level)',
                                          'played_modal': 'stress from the modal monitors\' displacement band moments, gauges from the played outputs'}[route],
                       starting_values=dict(level_db=1.5, grades_db=dict(A=3.0, B=4.5, C=6.0), over_test_alarm_db=6.0, under_db=-3.0, provisional=True)),
        scoring=files, baselines=dict(file='baselines.json', status='not included in this release'),
        provenance=dict(study='the set\'s generator (code/generator/obs_gen2.py)' if g2 else 'an earlier study', study_case=f's{seed}_{ns}x{nc}_{change}',
                        study_order='shakers x controls', service_tier=tier, service_truth='fixed point (obs_gen2.py)' if g2 else 'x4_truth.context',
                        runner_commit='bccb6754 with the study runner' if g2 else '473f08f6', generator_export=f'generator export {os.path.basename(os.path.normpath(pd))}', generator='code/make_problem.py (0.2)',
                        export_schema='mm-x4-case/2' if as_case2(export_structure) is export_structure and export_structure['kind'] == 'jointed' else 'mm-x4-case/1', checks=checks))
    ss = os.path.join(pd, 'service_state_v2.json')
    if g2 and os.path.exists(ss):     # the study's ruling 2 (part 4): the state in service, beside which the scorer sets the state as played
        problem['service_state'] = rm.load_json(ss)
    put_json(folder, 'problem.json', problem, indent=1)

    # ---- check: the batch's case files for this case, rebuilt from the staged folder
    ok = bad = 0; diffs = []
    for cp in (sorted(glob.glob(os.path.join(BULK, 'cases_b', f's{seed}_{ns}x{nc}_{change}_*.json'))) if cases else []):
        case = rm.load_json(cp); cid = os.path.splitext(os.path.basename(cp))[0]      # the file's name is the design's id (a few early files hold an older id inside)
        base = cid[:-len('_divopen')] if cid.endswith('_divopen') else cid
        parts = base.split(f's{seed}_{ns}x{nc}_{change}_', 1)[1]
        if cid.endswith(('_prod', '_seed')) or '_f8' in cid or '_L120' in cid:
            continue          # the study's own arms: the product's lines, seed-only runs, 8 frames, 120 loops
        frf = parts.rsplit('_', 1)[1]; pas = 'opened' if cid.endswith('_divopen') else 'default'
        ref = next(r for r in ('converted_law', 'converted', 'banded', 'line') if parts.rsplit('_', 1)[0].endswith('_' + r))
        mine = runner_case(folder, cid, ref, case['run']['law']['id'], case['run']['law']['params'], frf, pas, zlib.crc32(base.encode()) & 0x7fffffff)
        same = (mine['run'] == case['run'] and mine['grid'] == case['grid'] and mine['outputs'] == case['outputs'] and mine['schema'] == case['schema']
                and mine['structure'] == as_case2(rm.load_json(case['plant_file'])) and os.path.basename(case['reference_file']) == f'ref_{ref}.json')
        ok += same; bad += not same
        if not same:
            diffs.append(cid)
    checks['cases'] = dict(what='batch case files of this case (default and opened passes, 64 frames, 30 loops) rebuilt from the staged folder: run block, grid, outputs and structure equal',
                           compared=ok + bad, equal=ok, different=diffs[:10])
    problem['provenance']['checks'] = checks; put_json(folder, 'problem.json', problem, indent=1)
    return folder, checks


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ns', type=int, default=6)
    ap.add_argument('--nc', type=int, default=12); ap.add_argument('--change', default='none', help='none, realistic, strong, or a tier name of the export')
    ap.add_argument('--export-dir', default=None, help='the export folder (plant[_m].json, outputs[_m].json, side[_m].npz, ref_*.c16)')
    ap.add_argument('--no-cases', action='store_true', help='skip rebuilding the stopped study\'s batch case files')
    ap.add_argument('--analytic-run', default=None); ap.add_argument('--out', default=ROOT); a = ap.parse_args()
    folder, checks = make(a.seed, a.ns, a.nc, a.change, a.out, a.analytic_run, a.export_dir, not a.no_cases)
    print(folder); print(json.dumps(checks, indent=1))
    print('files:', ', '.join(sorted(os.path.relpath(os.path.join(r, f), folder) for r, _, fs in os.walk(folder) for f in fs)))
