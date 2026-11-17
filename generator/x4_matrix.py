"""an earlier study run matrix: a resumable job list for the closed-loop law comparison.

    PY=python
    $PY x4_matrix.py list  [filters]     # counts by status, and the commands for each job not yet finished
    $PY x4_matrix.py write [filters]     # export plants and references where missing, write the case files
    $PY x4_matrix.py score [filters]     # score every finished run not yet scored (x4_truth.score), one card each
    filters: --seeds 0,1 (0 to 19; 20 to 29 need --audit) --rigs 6x12,6x6 --changes none,strong --laws pinv_closed,band_average
             --settings default,tuned --frf hold,update --refs banded,converted,line --limit N

A job is law x setting x reference x FRF mode x rig x plant change x seed.

Rules of the study design (owner's review, 2026-11-14):
  Reference   band_average runs on the banded specification, its native form (it controls band averages; the
              line reference would give it the same band targets, the converted reference would not be the
              specification). The narrowband laws (pinv_closed, pinv_open, shape_svd, buzz_match, min_drive) run
              on the converted narrowband reference and on the traditional line-by-line reference. They are
              not given the banded specification as a diagonal matrix: that asks for zero coherence between
              controls, which the specification does not say. buzz_match and min_drive use only the
              autospectra of the reference they are given and build their own cross terms.
  Setting     every law at its documented defaults and at the one tuned setting of the study's notes.
  Seeds       0 to 19 are the study. 20 to 29 are RESERVED FOR THE INDEPENDENT AUDIT: the study must not run
              or look at them. This script refuses them unless --audit is given (for the auditor only).
  Classes     square rigs (6x6, 12x12) and non-square rigs (3x12, 6x12, 6x24, 12x24) are separate classes
              in every table. Every job and card carries cls; never pool the two.

Status comes from files
only, so the list can be run again at any time: a case is written when cases/<id>.json exists,
finished when runs/<id>/manifest.json exists (the runner writes it last and refuses to overwrite),
scored when cards/<id>.json exists. Everything is under
$OBS_EARLIER_STUDY/x4/ (local, not synced).
"""
import os, sys, json, zlib, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

BULK = os.environ.get('OBS_GEN_BULK', os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'exports/x4_port'))   # the study: was $OBS_EARLIER_STUDY/x4 (read only)
RUNNER = os.environ.get('OBS_RUNNER', 'x4_runner')
PY = 'python'
GRID = dict(fs_hz=8192.0, nfft=16384, band_hz=[20.0, 2000.0])     # must equal x4_export.GRID
SEEDS = tuple(range(20))             # the study
AUDIT_SEEDS = tuple(range(20, 30))    # reserved for the independent audit; not run or looked at by the study
RIGS = ((3, 12), (6, 12), (6, 24), (12, 24), (6, 6), (12, 12))    # (shakers, controls); the last two are square
CHANGES = ('none', 'realistic', 'strong')
FRF = {'hold': {'kind': 'hold'}, 'update': {'kind': 'update', 'blend': 0.5}}
LAWS = ('pinv_closed', 'pinv_open', 'shape_svd', 'buzz_match', 'min_drive', 'band_average')
# which reference each law gets (see Rules above for why)
REFS = {'band_average': ('banded',), 'pinv_closed': ('converted', 'line'), 'pinv_open': ('converted', 'line'),
        'shape_svd': ('converted', 'line'), 'buzz_match': ('converted', 'line'), 'min_drive': ('converted', 'line')}
ARMS = [(law, st, ref) for law in LAWS for st in ('default', 'tuned') for ref in REFS[law]]
cls_of = lambda ns, nc: 'square' if ns == nc else 'nonsquare'
CLASSES = ('nonsquare', 'square')
RUN = dict(levels_db=[-12.0, -9.0, -6.0, -3.0, 0.0], loops_at_full=30, record_loops=15, frames_per_loop=8,
           memory_loops=4.0, window='hann', overlap=0.5,
           identification=dict(frames=None, noise_frames=None, start_vrms=None, below_reference_db=None),
           noise_sigma=1e-4, noise_below_db=40.0, drive_ceiling_vrms=5.0, drive_pretest_max_vrms=1.0,
           alarm_db=None, abort_db=None, drive_seed=None, budget_fraction=None, keep_loop_matrices=False)


def tuned(law, ns):
    """One tuned setting per law (the study's notes)."""
    return {'pinv_closed': {'weighting': 'reference', 'convergence': 0.25},
            'pinv_open': {'regularization': 'tikhonov', 'regularization_amount': 1e-2, 'weighting': 'reference'},
            'shape_svd': {'selection': 'count', 'shapes': max(1, ns // 2)},
            'buzz_match': {'coherence_floor': 0.3},
            'min_drive': {'tradeoff': 'fixed', 'tradeoff_value': 1e-6},
            'band_average': {'oct_order': 3, 'drive_reduction': 0.70}}[law]


def law_params(law, setting, frf, ns):
    p = tuned(law, ns) if setting == 'tuned' else {}
    if law == 'band_average' and frf == 'update':
        p = dict(p, use_tf_update=True)    # the law's own switch; without it the service's update does nothing for this law
    return p


def jobs(a=None):
    sel = lambda name, default, conv=str: [conv(v) for v in getattr(a, name).split(',')] if a is not None and getattr(a, name) else list(default)
    seeds = sel('seeds', SEEDS, int); rigs = [tuple(int(v) for v in r.split('x')) for r in sel('rigs', [f'{s}x{c}' for s, c in RIGS])]
    changes, laws = sel('changes', CHANGES), sel('laws', LAWS); settings, frfs = sel('settings', ('default', 'tuned')), sel('frf', FRF)
    refs = sel('refs', ('banded', 'converted', 'line')); out = []
    held_back = [v for v in seeds if v in AUDIT_SEEDS]
    if held_back and not (a is not None and a.audit):
        sys.exit(f'seeds {held_back} are reserved for the independent audit; the study does not run or look at them')
    if a is not None and a.audit and any(v not in AUDIT_SEEDS for v in seeds):
        sys.exit('--audit is for seeds 20 to 29 only')
    for seed in seeds:
        for ns, nc in rigs:
            for change in changes:
                for law, st, ref in ARMS:
                    for frf in frfs:
                        if law in laws and st in settings and ref in refs:
                            cid = f's{seed}_{ns}x{nc}_{change}_{law}_{st}_{ref}_{frf}'
                            out.append(dict(id=cid, seed=seed, ns=ns, nc=nc, cls=cls_of(ns, nc), change=change, law=law, setting=st, ref=ref, frf=frf))
    return out[:a.limit] if a is not None and a.limit else out


plant_dir = lambda j: os.path.join(BULK, 'plants', f's{j["seed"]}_{j["ns"]}x{j["nc"]}_{j["change"]}')
case_path = lambda j: os.path.join(BULK, 'cases', j['id'] + '.json')
RUNS = os.environ.get('OBS_GEN_RUNS', os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'runs/x4_port'))   # the study: was an earlier run folder (read only)  # run folders are bulk data and stay here (join, 2026-11-14)
run_dir = lambda j: os.path.join(RUNS, j['id'])
card_path = lambda j: os.path.join(BULK, 'cards', j['id'] + '.json')
tier_of = lambda change: 'linear' if change == 'none' else change


def status(j):
    if os.path.exists(card_path(j)):
        return 'scored'
    if os.path.exists(os.path.join(run_dir(j), 'manifest.json')):
        return 'finished'
    return 'written' if os.path.exists(case_path(j)) else 'new'


def commands(j):
    pd, st, cmds = plant_dir(j), status(j), []
    rig = f'--seed {j["seed"]} --ns {j["ns"]} --nc {j["nc"]} --change {j["change"]}'
    if st == 'new':
        if not os.path.exists(os.path.join(pd, 'plant.json')):
            cmds.append(f'{PY} {HERE}/x4_export.py {rig}')
        if not os.path.exists(os.path.join(pd, f'ref_{j["ref"]}.json')):
            cmds.append(f'{PY} {HERE}/x4_reference.py {rig} --tier {tier_of(j["change"])}')
        cmds.append(f'{PY} {HERE}/x4_matrix.py write --seeds {j["seed"]} --rigs {j["ns"]}x{j["nc"]} --changes {j["change"]}')
    if st in ('new', 'written'):
        cmds.append(f'{RUNNER} {case_path(j)} --out {run_dir(j)}')
    if st != 'scored':
        cmds.append(f'{PY} {HERE}/x4_matrix.py score --seeds {j["seed"]} --rigs {j["ns"]}x{j["nc"]} --changes {j["change"]} --laws {j["law"]}')
    return cmds


def write(js):
    import x4_export as xe, x4_reference as xr, x4_truth as xt
    assert xe.GRID == GRID; os.makedirs(os.path.join(BULK, 'cases'), exist_ok=True); n = 0
    for j in js:
        if status(j) != 'new':
            continue
        pd = plant_dir(j)
        if not os.path.exists(os.path.join(pd, 'plant.json')):
            xe.write_plant(pd, *xe.build(j['seed'], j['ns'], j['nc'], j['change']))
        if not os.path.exists(os.path.join(pd, f'ref_{j["ref"]}.json')):
            xr.export(j['seed'], j['ns'], j['nc'], tier_of(j['change']), j['change'])
        case = dict(schema='mm-x4-case/1', case_id=j['id'], grid=GRID, plant_file=os.path.join(pd, 'plant.json'),
                    outputs=json.load(open(os.path.join(pd, 'outputs.json'))),
                    reference_file=os.path.join(pd, f'ref_{j["ref"]}.json'),
                    run=dict(RUN, law=dict(id=j['law'], params=law_params(j['law'], j['setting'], j['frf'], j['ns'])),
                             frf_mode=FRF[j['frf']], sim_seed=zlib.crc32(j['id'].encode()) & 0x7fffffff))
        json.dump(case, open(case_path(j), 'w'), indent=1); n += 1
    print(f'{n} case files written to {os.path.join(BULK, "cases")}')


def played_state(j):
    """(lab_fn, lab_zeta, note): the lab modal state as played, where the run folder gives it."""
    if j['change'] == 'none':
        return None, None, 'linear plant'
    side = np.load(os.path.join(plant_dir(j), 'side.npz')); log = os.path.join(run_dir(j), 'softening_log.csv')
    plant = json.load(open(os.path.join(plant_dir(j), 'plant.json')))
    if os.path.exists(log) and plant.get('softening'):
        rows = np.atleast_2d(np.loadtxt(log, delimiter=',', skiprows=0 if open(log).read(1).isdigit() else 1))
        fn = side['fn'].copy(); tail = rows[rows[:, 0] >= rows[-1, 0] - 0.5 * (rows[-1, 0] - rows[0, 0])]
        fn[plant['softening']['modes']] = tail[:, 1:].mean(0)
        return fn, None, 'frequencies from softening_log.csv, mean over the last half of the log'
    return None, None, 'NOT AS PLAYED: no modal state log for this plant kind; stress is through the low-level linear model'


def score(js):
    import x4_export as xe, x4_truth as xt
    os.makedirs(os.path.join(BULK, 'cards'), exist_ok=True); n = 0
    for j in js:
        if status(j) != 'finished':
            continue
        win = json.load(open(os.path.join(run_dir(j), 'record_window.json')))      # R, 2026-11-14 (join request 3)
        if not win.get('frames_drive'):
            man = json.load(open(os.path.join(run_dir(j), 'manifest.json')))
            print(f'{j["id"]} [{j["cls"]}]: NOT SCORED, the record window is empty; trip: {man["outcome"].get("trip")}', flush=True)
            continue
        g = float(np.load(os.path.join(plant_dir(j), 'side.npz'))['force_per_volt']); D, f = xe.read_run(run_dir(j), g)
        fn, zeta, note = played_state(j)
        r = xt.score(D, j['seed'], j['ns'], j['nc'], f=f, tier=tier_of(j['change']), lab_fn=fn, lab_zeta=zeta)
        man = json.load(open(os.path.join(run_dir(j), 'manifest.json')))
        r.update(job=j, cls=j['cls'], stress_model=note, outcome=man.get('outcome'))
        json.dump(r, open(card_path(j), 'w'), indent=1); n += 1
        print(f'{j["id"]} [{j["cls"]}]: stress rms {r["stress"]["rms"]:.2f} dB, bias {r["stress"]["bias"]:+.2f}, within 3 dB {r["stress"]["in3"]:.2f}', flush=True)
    print(f'{n} runs scored')


# ======================================================================================
# Agent R, 2026-11-14: the batch driver (additions; list and write above are unchanged; score now skips a
# run with an empty record window, the join's request 3).
#
#     $PY x4_matrix.py run --design --jobs 12 [--blocks A,B1] [--half] [--seeds 0,1] [--rigs ..] [--changes ..] [--laws ..]
#     $PY x4_matrix.py run --jobs N [filters] [--mode census|both|open|product] [--frames 64] [--loops-full 30] [--record-loops 15]
#     $PY x4_matrix.py status [--design ...| filters ...]          # batch counts by class
#     $PY x4_matrix.py plant [filters] [--stress modal|fibres]     # export monitor plants and references where missing
#     common: [--monitors auto|always|never] [--stress modal|fibres] [--redo] [--keep-all]
#
# --design runs the cut of the study's notes as x4_analyze.design() lists it (blocks A to F2), seeds in
# rising order. Ids are the design's: <job id>[_f<frames>][_L<loops>] and then nothing (the default-protections
# pass), _divopen (the pass with the protections opened) or _prod (the default protections plus the product's
# alarm and abort lines). 64 frames a loop and 30 full-level loops carry no token.
#
# Passes and cards. Every pass that runs writes its own card cards/<id>.json, whatever happened: scored if the
# run completed (or stopped inside its record window with enough frames), else scored false with the trip record
# (card key "trip": guard, loop, control or drive, error).
#   mode census   the default pass; if it does not complete, the opened pass (same sim_seed), marked as such and
#                 carrying default_trip and twin_of. The design's seeds 0 to 4.
#   mode both     the default pass and the opened pass whatever happens (the design's block T twins).
#   mode open     the opened pass only. The design's seeds 5 to 19 and its long and 8-frame blocks.
#   mode product  the default protections with the product setup's lines (alarm 3 dB, abort 6 dB), no rerun.
# Default protections: Protections::default (config.rs 297), no alarm or abort lines (the study's RUN).
# Opened: hertzforge-mimo-random/examples/mimo_study.rs 419-425 (no lines, abort-line count unlimited, divergence
# 16 loops and 1000 dB, overload fraction 1.0). A task is finished when every card its mode calls for exists;
# finished tasks are skipped, so the command can be stopped and started again.
#
# Reference converted_law (the design's): H_id D_first H_id^H from the band_average law's own first solve on the
# same case. Before any job that needs it, the driver makes it with a seed-only run of the block A band_average
# job of that case (same plant file, same sim_seed, runner key stop_after seed) and x4_reference.export_converted_law.
#   --monitors auto  monitor outputs only on plants that change with level; the unchanged plant is scored from
#                    the played drive through the linear model (verified equal to 0.016 dB, the study's notes)
#   --stress modal   stress from the modal monitors' displacement band moments (plant_m, 122 extra outputs)
#   --stress fibres  stress from one monitor output per fibre stress row (plant_b, 576 extra outputs, slower)
# After scoring, each run folder is cut to KEEP (the autospectra as float32); x4_analyze.py harvest reads only
# files that are kept, so it can run at any time: $PY x4_analyze.py harvest --runs an earlier run folder/b.noindex
RUNS_B = os.path.join(RUNS, 'b.noindex')                 # .noindex: Spotlight does not index it
CASES_B = os.path.join(BULK, 'cases_b')
CHANGES_B = ('none', 'realistic', 'strong', 'realistic_soft', 'strong_soft')
LINES = {'off': dict(alarm_db=None, abort_db=None), 'product': dict(alarm_db=3.0, abort_db=6.0)}   # setup.rs 67-68
OPENED = dict(protections='opened', alarm_db=None, abort_db=None)                                 # mimo_study.rs 419-425
KEEP = ('manifest.json', 'loops.csv', 'events.txt', 'record_window.json', 'law_manifest.txt', 'softening_log.csv',
        'drive_sdm_played.c16', 'response_psd_played.f4', 'response_moments_played.f8')
KEEP_BAND_AVERAGE = ('drive_sdm_first.c16',)             # the band_average law's first solve, for the auditor
MIN_PARTIAL_FRAMES = 40      # a partial record window is scored only with at least this many Welch frames
FRAMES_DEFAULT = 64          # the product default (config.rs 26)
TAG = {'fibres': 'b', 'modal': 'm'}
PASSES = {'census': ('default', 'opened'), 'both': ('default', 'opened'), 'open': ('opened',), 'product': ('product',)}
TOKEN = {'default': '', 'opened': '_divopen', 'product': '_prod'}
PROT = {'default': 'default', 'opened': 'open', 'product': 'product'}      # the design's and the analysis's word for each pass
BLOCK_ORDER = ('A', 'B1', 'B2', 'B3', 'C', 'T', 'P', 'E', 'F1', 'F2')
base_id = lambda j: f's{j["seed"]}_{j["ns"]}x{j["nc"]}_{j["change"]}_{j["law"]}_{j["setting"]}_{j["ref"]}_{j["frf"]}'


def _dump(obj, path, **kw):
    tmp = f'{path}.tmp{os.getpid()}'; json.dump(obj, open(tmp, 'w'), **kw); os.replace(tmp, path)


def b_set(a, **over):
    """The settings of one task as a plain dict (it goes to the workers)."""
    st = dict(frames=a.frames, loops_full=a.loops_full or RUN['loops_at_full'], record_loops=a.record_loops or RUN['record_loops'],
              mode=a.mode, monitors=a.monitors, stress=a.stress, keep_all=a.keep_all, block=None)
    st.update(over); return st


def b_id(j, st, pas='default'):
    return (base_id(j) + (f'_f{st["frames"]}' if st['frames'] != FRAMES_DEFAULT else '')
            + (f'_L{st["loops_full"]}' if st['loops_full'] != RUN['loops_at_full'] else '')
            + (f'_r{st["record_loops"]}' if st['record_loops'] != RUN['record_loops'] else '') + TOKEN[pas])


def b_route(j, st):
    """None (score from the drive), 'modal' or 'fibres'."""
    return st['stress'] if st['monitors'] == 'always' or (st['monitors'] == 'auto' and j['change'] != 'none') else None


b_tag = lambda j, st: TAG[b_route(j, st)] if b_route(j, st) else ''
b_run_dir = lambda j, st, pas: os.path.join(RUNS_B, b_id(j, st, pas))
b_case_path = lambda j, st, pas: os.path.join(CASES_B, b_id(j, st, pas) + '.json')
b_card_path = lambda j, st, pas: os.path.join(BULK, 'cards', b_id(j, st, pas) + '.json')


def _card(path):
    try:
        c = json.load(open(path)); return c if 'batch' in c else None
    except (OSError, ValueError):
        return None


def b_done(j, st):
    cards = [_card(b_card_path(j, st, pas)) for pas in PASSES[st['mode']]]
    if st['mode'] == 'census':
        return cards[0] is not None and (cards[0]['outcome']['completed'] or cards[1] is not None)
    return all(c is not None for c in cards)


def b_tasks_design(a):
    """The cut of the study's notes as tasks (job, settings), seeds rising. A default job and its opened twin are
    one task: census where the twin is conditional, both where the design asks for the twin outright (block T)."""
    import x4_analyze as xa
    sel = lambda name, conv=str: {conv(v) for v in getattr(a, name).split(',')} if getattr(a, name) else None
    seeds, rigs, changes, laws, blocks = sel('seeds', int), sel('rigs'), sel('changes'), sel('laws'), sel('blocks')
    half = {'A': 10, 'B1': 10, 'B2': 5, 'C': 5}; groups = {}
    for d in xa.design():
        if d['seed'] in AUDIT_SEEDS:
            sys.exit('the design holds an audit seed; refused')
        if ((seeds and d['seed'] not in seeds) or (rigs and f'{d["ns"]}x{d["nc"]}' not in rigs) or (changes and d['change'] not in changes)
                or (laws and d['law'] not in laws) or (a.half and d['seed'] >= half.get(d['block'], 99))):
            continue
        key = (base_id(d), d['frames_per_loop'], d['loops_at_full']); g = groups.setdefault(key, dict(d=d, prots={}))
        g['prots'].setdefault(d['prot'], []).append((d['cond'], d['block']))
    out = []; first = lambda bl: min(bl, key=BLOCK_ORDER.index)
    for (bid, frames, loops), g in groups.items():
        d = g['d']; j = {k: d[k] for k in ('seed', 'ns', 'nc', 'cls', 'change', 'law', 'setting', 'ref', 'frf')}; j['id'] = bid; pr = g['prots']
        twin = [bl for c, bl in pr.get('open', []) if not c]         # opened runs the design asks for outright
        add = lambda mode, bl: out.append((j, b_set(a, frames=frames, loops_full=loops, mode=mode, block=first(bl)))) if not blocks or set(bl) & blocks else None
        if 'default' in pr:
            add('both' if twin else 'census', [bl for _, bl in pr['default']] + twin)
        elif 'open' in pr:
            add('open', [bl for _, bl in pr['open']])
        if 'product' in pr:
            add('product', [bl for _, bl in pr['product']])
    out.sort(key=lambda t: (t[0]['seed'], BLOCK_ORDER.index(t[1]['block']), t[0]['ns'], t[0]['nc'], t[0]['change'], t[0]['id'], t[1]['mode']))
    return out[:a.limit] if a.limit else out


def b_plant(key):
    """Export the plant (tag b or m, or none) and the references for one (seed, ns, nc, change, tag), where missing."""
    seed, ns, nc, change, tag = key
    import x4_export as xe, x4_reference as xr
    pd = xe.plant_dir(seed, ns, nc, change); made = []
    if tag and not os.path.exists(os.path.join(pd, f'side_{tag}.npz')):
        xe.write_plant_b(pd, *xe.build_b(seed, ns, nc, change, stress=(tag == 'b'), modal=True if tag == 'm' else None), tag=tag); made.append(f'plant_{tag}')
    if not tag and not os.path.exists(os.path.join(pd, 'plant.json')):
        assert change == 'none'; xe.write_plant(pd, *xe.build(seed, ns, nc, 'none')); made.append('plant')
    if not all(b_ref_ok(pd, r) for r in ('banded', 'converted', 'line')):
        xe.register_tiers(); xr.export(seed, ns, nc, xe.SERVICE_TIER[change], change); made.append('references')
    return key, made


def b_ref_ok(pd, r):
    """The reference object exists and names its matrix file by a path that exists (join, Problem 1)."""
    try:
        return os.path.exists(json.load(open(os.path.join(pd, f'ref_{r}.json')))['file'])
    except (OSError, ValueError, KeyError):
        return False


def b_seed_job(seed, ns, nc, change):
    """The block A band_average job of a case: the run whose first solve makes ref_converted_law."""
    j = dict(seed=seed, ns=ns, nc=nc, cls=cls_of(ns, nc), change=change, law='band_average', setting='default', ref='banded', frf='hold')
    j['id'] = base_id(j); return j


def b_convlaw(key):
    """ref_converted_law for one case, where missing: a seed-only run of the block A band_average job (same plant
    file and sim_seed as that job, so the same identification and the same first solve), then H_id D_first H_id^H."""
    import subprocess
    import x4_reference as xr
    seed, ns, nc, change, tag, st = key; j = b_seed_job(seed, ns, nc, change); pd = plant_dir(j)
    if b_ref_ok(pd, 'converted_law'):
        return key[:5], None
    case = json.load(open(b_write_case(j, st, 'opened'))); case['case_id'] = j['id'] + '_seed'; case['run']['stop_after'] = 'seed'
    case['run'].pop('moments', None); cp = os.path.join(CASES_B, case['case_id'] + '.json'); _dump(case, cp)
    run = os.path.join(RUNS_B, case['case_id'])
    pr = subprocess.run([RUNNER, cp, '--out', run, '--overwrite'], capture_output=True, text=True)
    if pr.returncode != 0:
        raise RuntimeError(f'seed-only run of {case["case_id"]} failed (exit {pr.returncode}): {(pr.stdout + pr.stderr).strip()[-400:]}')
    chk = xr.export_converted_law(seed, ns, nc, change, run)
    for name in os.listdir(run):
        if name not in KEEP + KEEP_BAND_AVERAGE + ('frf_identified.c16',) and os.path.isfile(os.path.join(run, name)):
            os.remove(os.path.join(run, name))
    return key[:5], chk


def b_write_case(j, st, pas):
    """The case file of one pass. Every pass and variant of a job has the same sim_seed (from the bare job id)."""
    pd = plant_dir(j); os.makedirs(CASES_B, exist_ok=True); route = b_route(j, st); tag = TAG[route] if route else ''
    assert route or j['change'] == 'none', 'a plant that changes with level is scored from its monitors'
    plant_file = os.path.join(pd, f'plant_{tag}.json' if tag else 'plant.json')
    outputs = json.load(open(os.path.join(pd, f'outputs_{tag}.json' if tag else 'outputs.json')))
    run = dict(RUN, law=dict(id=j['law'], params=law_params(j['law'], j['setting'], j['frf'], j['ns'])), frf_mode=FRF[j['frf']],
               sim_seed=zlib.crc32(base_id(j).encode()) & 0x7fffffff, response_record='autospectra', protections='default',
               frames_per_loop=st['frames'], loops_at_full=st['loops_full'], record_loops=min(st['record_loops'], st['loops_full']))
    run.update(OPENED if pas == 'opened' else LINES['product' if pas == 'product' else 'off'])
    if route == 'modal':
        n_phys = sum(o['role'] != 'monitor' for o in outputs); run['moments'] = dict(first=n_phys, count=len(outputs) - n_phys)
    case = dict(schema='mm-x4-case/1', case_id=b_id(j, st, pas), grid=GRID, plant_file=plant_file, outputs=outputs,
                reference_file=os.path.join(pd, f'ref_{j["ref"]}.json'), run=run)
    _dump(case, b_case_path(j, st, pas)); return b_case_path(j, st, pas)


def b_plant_state(j, D, psd, side, f, run):
    """What the plant did at full level: per elastic mode, the realized frequency drop and damping factor."""
    import x4_truth as xt
    out = dict(kind=str(side['kind']))
    el = np.arange(side['fn'].size) >= 6; td, tz = np.asarray(side['target_drop']), np.asarray(side['target_zfac'])
    if int(side['n_modal']):
        ms = xt.modal_state(D, psd, side, f); ok = ms['ok'] & el & (2 * ms['zeta0'] * ms['f0'] >= 2.0)   # resolved: half-power width of 4 lines
        drop = 1 - ms['fn'] / ms['f0']; zf = ms['zeta'] / ms['zeta0']; q = lambda v: [float(np.percentile(v, p)) for p in (10, 50, 90)]
        out.update(modes_fitted=int(ok.sum()), drop_pct_p10_p50_p90=q(100 * drop[ok]), zfac_p10_p50_p90=q(zf[ok]),
                   target_drop_pct_p10_p50_p90=q(100 * td[ok]), target_zfac_p10_p50_p90=q(tz[ok]),
                   drop_over_target_p50=float(np.median(drop[ok] / td[ok])) if (td[ok] > 0).all() else None,
                   zrise_over_target_p50=float(np.median((zf[ok] - 1) / (tz[ok] - 1))) if (tz[ok] > 1).all() else None,
                   fn=[round(float(v), 4) for v in np.nan_to_num(ms['fn'])], zeta=[round(float(v), 6) for v in np.nan_to_num(ms['zeta'])])
    log = os.path.join(run, 'softening_log.csv')
    if os.path.exists(log) and os.path.getsize(log):
        rows = np.atleast_2d(np.loadtxt(log, delimiter=',')); man = json.load(open(os.path.join(run, 'manifest.json')))
        loop_s = man['timing']['simulated_loop_s']; tail = rows[rows[:, 0] >= rows[-1, 0] - man['outcome']['record_loops_recorded'] * loop_s]
        d = 1 - tail[:, 1:].mean(0) / side['fn'][el]
        out.update(log_drop_pct_p10_p50_p90=[float(np.percentile(100 * d, p)) for p in (10, 50, 90)],
                   log_drop_over_target_p50=float(np.median(d / td[el])), log_note='softening_log.csv, mean over the record window')
    return out


def b_prune(run, keep_all=False, keep=KEEP):
    """Cut a run folder to what is kept; the autospectra go to float32. Returns the bytes left."""
    p8 = os.path.join(run, 'response_psd_played.f8')
    if os.path.exists(p8) and not keep_all:
        np.fromfile(p8, '<f8').astype('<f4').tofile(os.path.join(run, 'response_psd_played.f4'))
    total = 0
    for name in os.listdir(run):
        path = os.path.join(run, name)
        if not keep_all and name not in keep:
            os.remove(path) if os.path.isfile(path) else None
        elif os.path.isfile(path):
            total += os.path.getsize(path)
    return total


def b_trip(oc):
    """A tripped run's record for its card: which guard, when, on which control or drive, how large."""
    t = oc.get('trip_record')
    if not t:
        return dict(guard='run_error' if oc.get('run_error') else 'stopped_early', words=oc.get('run_error'), loop=oc.get('loops_total'))
    fl = t.get('fields') or {}
    return dict(guard=t['kind'], loop=t['at_loop'], level_db=t['at_level_db'], at_full_level=t['at_full_level'],
                full_level_loops_done=t['full_level_loops_done'], control=t.get('control'), control_name=t.get('control_name'),
                drive=t.get('drive'), error_db=fl.get('error_db', fl.get('worst_db')), lines=fl.get('abort_lines', fl.get('lines')),
                consecutive_loops=fl.get('consecutive_loops'), words=t['words'], fields=fl)


def b_score(j, st, run, oc):
    """Score one finished run folder. (card core, extra keys, note, scored). Not scored: the core holds None."""
    import x4_export as xe, x4_truth as xt
    tier = xe.SERVICE_TIER[j['change']]; xe.register_tiers(); route = b_route(j, st); pd = plant_dir(j)
    core = dict(seed=j['seed'], ns=j['ns'], nc=j['nc'], tier=tier, stress=None, ers=None, card=None)
    extra = dict(stress_route=None, stress_drive=None, card_drive=None, structure_state=None)
    ok = oc['completed'] or (oc['record_window_status'] == 'partial' and oc['record_frames_drive'] >= MIN_PARTIAL_FRAMES)
    if not ok:
        return core, extra, 'NOT SCORED: the run did not reach a record window', False
    if route:
        side = np.load(os.path.join(pd, f'side_{TAG[route]}.npz')); D, psd, f = xe.read_run_b(run, float(side['force_per_volt']))
        core = xt.score_played(D, psd, side, j['seed'], j['ns'], j['nc'], f=f, tier=tier, moments=xe.read_moments(run))
        lin = xt.score(D, j['seed'], j['ns'], j['nc'], f=f, tier=tier)       # the drive route, for the size of its error on this plant
        extra = dict(stress_route=route, stress_drive=lin['stress'], card_drive=lin['card'],
                     structure_state=b_plant_state(j, D, psd, side, f, run) if j['change'] != 'none' else None)
        note = ('as played: fibre stress from the modal monitors\' displacement band moments' if route == 'modal' else
                'as played: fibre stress from the fibre stress monitor outputs') + ', gauges from the played outputs'
    else:
        g = float(np.load(os.path.join(pd, 'side.npz'))['force_per_volt']); D, f = xe.read_run(run, g)
        core = xt.score(D, j['seed'], j['ns'], j['nc'], f=f, tier=tier); extra['stress_route'] = 'drive'
        note = 'linear plant: the played drive through the linear lab model'
    return core, extra, note, True


def b_one(args):
    """Run, score and prune one task: the passes its mode calls for, one card per pass. Returns a small summary; never raises."""
    import subprocess, time, traceback
    j, st = args; t0 = time.time(); summ = dict(id=b_id(j, st, PASSES[st['mode']][0]), cls=j['cls'], mode=st['mode'], passes=[]); first_trip = None; kept = 0
    try:
        for pas in PASSES[st['mode']]:
            case = b_write_case(j, st, pas); run = b_run_dir(j, st, pas); t1 = time.time()
            pr = subprocess.run([RUNNER, case, '--out', run, '--overwrite'], capture_output=True, text=True)
            if pr.returncode not in (0, 1):
                raise RuntimeError(f'runner refused {case} (exit {pr.returncode}): {pr.stderr.strip()[-400:]}')
            man = json.load(open(os.path.join(run, 'manifest.json'))); oc = man['outcome']; t2 = time.time()
            core, extra, note, scored = b_score(j, st, run, oc); t3 = time.time()
            kb = b_prune(run, st['keep_all'], KEEP + (KEEP_BAND_AVERAGE if j['law'] == 'band_average' else ())); kept += kb
            trip = None if oc['completed'] else b_trip(oc); twin = pas == 'opened' and len(PASSES[st['mode']]) == 2
            card = dict(core, job=dict(j, id=b_id(j, st, pas), base_id=base_id(j), prot=PROT[pas], protections=pas, loops_at_full=st['loops_full'],
                                       frames_per_loop=st['frames'], record_loops=st['record_loops'], block=st['block'],
                                       **({'lines': 'product'} if pas == 'product' else {})),
                        cls=j['cls'], stress_model=note, outcome=oc, scored=scored, trip=trip, **extra,
                        default_trip=first_trip if twin else None, twin_of=b_id(j, st, 'default') if twin else None,
                        batch=dict(version=3, protections=pas, mode=st['mode'], lines='product' if pas == 'product' else 'off', frames=st['frames'],
                                   loops_full=st['loops_full'], record_loops=st['record_loops'], stress_route=b_route(j, st) or 'drive',
                                   completed=oc['completed'], record_window_status=oc['record_window_status'],
                                   opened_twin=b_id(j, st, 'opened') if st['mode'] == 'both' or (st['mode'] == 'census' and pas == 'default' and not oc['completed']) else None,
                                   runner_wall_s=man.get('wall_s'), wall_engage_s=man['timing']['wall_engage_s'],
                                   wall_loop_median_s=man['timing']['wall_loop_median_s'], simulated_loop_s=man['timing']['simulated_loop_s'],
                                   score_wall_s=t3 - t2, wall_s=time.time() - t1, kept_bytes=kb, run_dir=run))
            cp = b_card_path(j, st, pas); os.makedirs(os.path.dirname(cp), exist_ok=True)
            if os.path.exists(cp) and _card(cp) is None:        # a card of the same name from before the batch (the join's): set aside, not lost
                os.makedirs(os.path.join(BULK, 'cards', '_pre_batch'), exist_ok=True); os.replace(cp, os.path.join(BULK, 'cards', '_pre_batch', os.path.basename(cp)))
            _dump(card, cp, indent=1)
            summ['passes'].append(dict(protections=pas, completed=oc['completed'], scored=scored, guard=trip['guard'] if trip else None,
                                       stress_rms=core['stress']['rms'] if scored else None, wall_s=time.time() - t1, runner_wall_s=man.get('wall_s')))
            if pas == 'default':
                first_trip = trip
            if st['mode'] == 'census' and oc['completed']:
                break
        summ.update(ok=True, wall_s=time.time() - t0, kept_bytes=kept)
    except Exception as e:      # the batch goes on; the task stays unfinished and is tried again next time
        summ.update(ok=False, error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:], wall_s=time.time() - t0)
    return summ


def b_tasks(js, a):
    return b_tasks_design(a) if a.design else [(j, b_set(a)) for j in js]


def b_run(js, a):
    import time, multiprocessing as mp
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[v] = '1'      # one BLAS thread per worker; the workers are the parallelism
    tasks = b_tasks(js, a); bad = sorted({j['change'] for j, _ in tasks} - set(CHANGES_B))
    if bad:
        sys.exit(f'unknown plant change {bad}; the batch knows {CHANGES_B}')
    if any(j['seed'] in AUDIT_SEEDS for j, _ in tasks) and not a.audit:
        sys.exit('audit seeds are refused without --audit')
    n = max(1, min(12, a.jobs)); todo = [t for t in tasks if a.redo or not b_done(*t)]
    print(f'{len(tasks)} tasks selected ({"the design" if a.design else "filters"}), {len(tasks) - len(todo)} finished already, {len(todo)} to run on {n} workers', flush=True)
    if not todo:
        return
    os.makedirs(RUNS_B, exist_ok=True); t0 = time.time(); ctx = mp.get_context('spawn')
    plants = sorted({(j['seed'], j['ns'], j['nc'], j['change'], b_tag(j, st)) for j, st in todo})
    seeds_ = {}
    for j, st in todo:
        if j['ref'] == 'converted_law':
            seeds_[(j['seed'], j['ns'], j['nc'], j['change'], b_tag(j, st))] = dict(st, frames=FRAMES_DEFAULT, loops_full=RUN['loops_at_full'], mode='open')
    with ctx.Pool(n) as pool:
        for key, made in pool.imap_unordered(b_plant, plants):
            if made:
                print(f'  exported s{key[0]}_{key[1]}x{key[2]}_{key[3]}: {", ".join(made)}', flush=True)
        for key, chk in pool.imap_unordered(b_convlaw, [k + (st,) for k, st in sorted(seeds_.items())]):
            if chk:
                print(f'  ref_converted_law s{key[0]}_{key[1]}x{key[2]}_{key[3]}: band means against the banded targets, one-third octave '
                      f'{chk["b3_rms_db"]:.2f} dB RMS (bias {chk["b3_bias_db"]:+.2f})', flush=True)
        t1 = time.time(); print(f'plants and references ready in {t1 - t0:.0f} s', flush=True); done = []
        for r in pool.imap_unordered(b_one, todo):
            done.append(r)
            if r['ok']:
                txt = '; '.join(f'{p["protections"]}: ' + ('' if p['completed'] else f'tripped ({p["guard"]})' + (', partial window, ' if p['scored'] else ''))
                                + (f'stress rms {p["stress_rms"]:.2f} dB' if p['scored'] else '') + f', {p["wall_s"]:.0f} s' for p in r['passes'])
                print(f'[{len(done)}/{len(todo)}] {r["id"]} [{r["cls"]}, {r["mode"]}] {txt}', flush=True)
            else:
                print(f'[{len(done)}/{len(todo)}] {r["id"]} FAILED: {r["error"]}', flush=True)
                open(os.path.join(RUNS_B, 'errors.log'), 'a').write(json.dumps(r) + '\n')
    wall = time.time() - t1; good = [r for r in done if r['ok']]
    if good:
        guards = {}; comp = [p['runner_wall_s'] for r in good for p in r['passes'] if p['completed']]; trip = [p['runner_wall_s'] for r in good for p in r['passes'] if not p['completed']]
        for r in good:
            for p in r['passes']:
                if p['guard']:
                    guards[p['guard']] = guards.get(p['guard'], 0) + 1
        print(f'{len(good)} tasks ({sum(len(r["passes"]) for r in good)} runs) in {wall:.0f} s on {n} workers: {3600 * len(good) / wall:.0f} tasks an hour; '
              f'{np.mean([r["wall_s"] for r in good]):.1f} s a task in its worker; a completed run {np.mean(comp) if comp else float("nan"):.0f} s, '
              f'a tripped run {np.mean(trip) if trip else float("nan"):.0f} s; {np.mean([r["kept_bytes"] for r in good]) / 1e6:.1f} MB kept a task; '
              f'runs that did not complete, by guard: {guards}; {sum(not r["passes"][-1]["scored"] for r in good)} tasks with no scored run; '
              f'{len(done) - len(good)} failed (errors.log)', flush=True)


def b_status(js, a):
    count = {c: {} for c in CLASSES}
    for j, st in b_tasks(js, a):
        cs = {pas: _card(b_card_path(j, st, pas)) for pas in PASSES[st['mode']]}; first = cs[PASSES[st['mode']][0]]
        if first is None:
            k = 'not run'
        elif st['mode'] in ('open', 'product'):
            k = f'{st["mode"]}: ' + ('scored' if first['scored'] else f'not scored ({(first["trip"] or {}).get("guard")})')
        elif first['outcome']['completed'] and st['mode'] == 'census':
            k = 'scored at defaults'
        else:
            o = cs['opened']; t = 'completed' if first['outcome']['completed'] else f'tripped ({first["trip"]["guard"]})'
            k = f'default pass {t}, ' + ('opened twin not run' if o is None else 'opened twin scored' if o['scored'] else 'opened twin not scored')
        count[j['cls']][k] = count[j['cls']].get(k, 0) + 1
    for c in CLASSES:   # the two classes are never pooled
        print(f'# {c}: {sum(count[c].values())} tasks: ' + (', '.join(f'{v} {k}' for k, v in sorted(count[c].items())) or 'none'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('list', 'write', 'score', 'plant', 'run', 'status'))
    for k in ('seeds', 'rigs', 'changes', 'laws', 'settings', 'frf', 'refs'):
        ap.add_argument('--' + k, default='')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--quiet', action='store_true')
    ap.add_argument('--audit', action='store_true', help='seeds 20 to 29, for the independent auditor only')
    ap.add_argument('--jobs', type=int, default=4, help='run: worker processes, 1 to 12')
    ap.add_argument('--design', action='store_true', help='run, status: the cut of the study\'s notes (x4_analyze.design), not the filters\' matrix')
    ap.add_argument('--blocks', default='', help='with --design: blocks to run, for example A,B1')
    ap.add_argument('--half', action='store_true', help='with --design: the half cut (seeds 0 to 9 of A and B1, 0 to 4 of B2 and C, the small blocks whole)')
    ap.add_argument('--mode', default='census', choices=tuple(PASSES), help='run without --design: which passes (see the comment above)')
    ap.add_argument('--frames', type=int, default=FRAMES_DEFAULT, help='run without --design: frames per loop (product default 64)')
    ap.add_argument('--loops-full', type=int, default=0, help='run without --design: loops at full level (default RUN, 30)')
    ap.add_argument('--record-loops', type=int, default=0, help='run: the last this many full-level loops are recorded (default RUN, 15)')
    ap.add_argument('--monitors', default='auto', choices=('auto', 'always', 'never'), help='run: monitor outputs for stress as played')
    ap.add_argument('--stress', default='modal', choices=tuple(TAG), help='run: which monitors carry the stress')
    ap.add_argument('--redo', action='store_true', help='run: also tasks that are already finished')
    ap.add_argument('--keep-all', action='store_true', help='run: do not cut the run folders')
    a = ap.parse_args(); js = [] if a.design and a.cmd in ('run', 'status') else jobs(a)
    if a.cmd == 'write':
        write(js)
    elif a.cmd == 'score':
        score(js)
    elif a.cmd == 'plant':
        st = b_set(a)
        for key in sorted({(j['seed'], j['ns'], j['nc'], j['change'], b_tag(j, st)) for j in js}):
            print(*b_plant(key), flush=True)
    elif a.cmd == 'run':
        b_run(js, a)
    elif a.cmd == 'status':
        b_status(js, a)
    else:
        count = {c: {} for c in CLASSES}
        for j in js:
            st = status(j); count[j['cls']][st] = count[j['cls']].get(st, 0) + 1
            if st != 'scored' and not a.quiet:
                print(f'# {j["id"]} [{j["cls"]}, {st}]'); print('\n'.join(commands(j)))
        for c in CLASSES:   # the two classes are never pooled
            print(f'# {c}: {sum(count[c].values())} jobs: ' + (', '.join(f'{v} {k}' for k, v in count[c].items()) or 'none'))
