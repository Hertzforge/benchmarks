"""an earlier study exporter: the lab plant of unit B for a seed and rig, as the modal data the runner reads.

    PY=python
    $PY x4_export.py --seed 0 --ns 6 --nc 12 [--change none|realistic|strong] [--kind modal|jointed]

Writes, under $OBS_EARLIER_STUDY/x4/plants/s<seed>_<ns>x<nc>_<change>/ :
  plant.json      the "plant" object of X4_PLANT_SCHEMA.md (mm-x4-case/1), kind modal or jointed
  outputs.json    the "outputs" list: controls first, then held-out gauges (h..), then component tips (t..)
  side.npz        what stays on the Python side: fn, zeta, drive and output shapes, the modal stress
                  matrix SB, unit B's service stress truth, index lists, force_per_volt
The plant's drive unit is volts: drive_shape = force_per_volt x mode shape at the shaker DOF, so a
recorded drive matrix in V^2/Hz times force_per_volt^2 is N^2/Hz, the unit x4_truth.score expects.
All file writing is in write_plant() so the format can be retargeted in one place.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import x4_truth as xt
from x4_truth import F, DF, NF

BULK = os.environ.get('OBS_GEN_BULK', os.path.join(os.environ.get('OBS_DATA', 'obs-data'), 'exports/x4_port'))   # the study: was $OBS_EARLIER_STUDY/x4 (read only)
GRID = dict(fs_hz=8192.0, nfft=16384, band_hz=[20.0, 2000.0])   # 0.5 Hz lines, lines 40 to 4000 = the study grid F
CHANGE = {'none': None, 'realistic': (0.01, 1.5), 'strong': (0.03, 3.0)}   # (frequency drop, damping factor) at full level, as an earlier study
LINE_DRIVE_VRMS = 0.5     # the reference line drive's largest shaker sits at this many volts


def plant_dir(seed, ns, nc, change):
    return os.path.join(BULK, 'plants', f's{seed}_{ns}x{nc}_{change}')


def modal_rms(ctx, D):
    """RMS modal displacement of each lab mode under a drive matrix D (nf, ns, ns) in N^2/Hz."""
    md = ctx['mlab']; A = (md.Phi[ctx['shk']].T[None] @ xt.factor(D)) * md.h(F).T[:, :, None]
    return np.sqrt((np.abs(A) ** 2).sum((0, 2)) * DF)


def build(seed, ns, nc, change='none', kind=None, tier_ctx=None):
    """The plant object, the outputs list and the side arrays, as plain Python and numpy."""
    ctx = tier_ctx or xt.context(seed, ns, nc); md = ctx['mlab']; shk = ctx['shk']
    c, held, tips = ctx['c'], ctx['held'], ctx['tips']; rows = ctx['U'][np.r_[c, held, tips]]
    D = xt.line_drive(ctx); drms = np.sqrt(np.real(np.einsum('fii->i', D)) * DF)
    g = float(drms.max() / LINE_DRIVE_VRMS); qrms = modal_rms(ctx, D); nm = md.fn.size
    kind = kind or ('modal' if change == 'none' else 'jointed'); elastic = np.arange(nm) >= 6
    outputs = [dict(name=f'c{i + 1:02d}', role='control') for i in range(c.size)] + \
              [dict(name=f'h{i + 1:02d}', role='held_out') for i in range(held.size)] + \
              [dict(name=f't{i + 1:02d}', role='held_out') for i in range(tips.size)]
    dshape = md.Phi[shk].T * g; oshape = md.Phi[rows].T    # (nm, ns), (nm, n_out)
    base = dict(n_outputs=int(rows.size), n_drives=int(ns), feedthrough=0.0, residual=[], transport_delay_samples=0)
    if kind == 'modal':
        modes = [dict(f_hz=float(md.fn[m]), q=float(1 / (2 * md.zeta[m])), drive_shape=dshape[m].tolist(),
                      output_shape=oshape[m].tolist(), response='acceleration') for m in range(nm)]
        soft = None
        if CHANGE[change]:   # every elastic mode sits d low at full level (fraction r^2/(1+r^2), r = 1 at the reference)
            soft = dict(modes=np.flatnonzero(elastic).tolist(), reference_rms=qrms[elastic].tolist(),
                        fraction=2 * CHANGE[change][0], smoothing=0.25, read_samples=GRID['nfft'] // 2)
        plant = dict(kind='modal', **base, modes=modes, softening=soft, noisy_outputs=None,
                     discretization=dict(kind='fitted', band_hz=2200.0, taps=9, delay=4))
    else:
        d, gz = CHANGE[change] or (0.0, 1.0)   # frequency falls d and damping rises to gz times at the reference amplitude
        modes = [dict(f_hz=float(md.fn[m]), zeta=float(md.zeta[m]), drive_shape=dshape[m].tolist(),
                      output_shape=oshape[m].tolist(),
                      reference_amplitude=float(np.sqrt(2) * qrms[m]) if elastic[m] else 1e30) for m in range(nm)]
        joint = dict(delta=d, damping_gain=gz - 1.0, freq_exponent=1.0, damping_exponent=1.0, decay_cycles=5.0) if CHANGE[change] else None
        plant = dict(kind='jointed', **base, modes=modes, joint=joint, output_delays_s=[])
    side = dict(fn=md.fn, zeta=md.zeta, drive_shape=dshape, output_shape=oshape, SB=ctx['SB'], rows=rows, shk=shk,
                n_control=c.size, n_held=held.size, n_tip=tips.size, force_per_volt=g, line_drive_rms_n=drms,
                modal_rms_line_drive=qrms, truthB_rms=ctx['truthB']['rms'], seed=seed, ns=ns, nc=nc, change=change, kind=kind)
    return plant, outputs, side


def write_plant(folder, plant, outputs, side):
    """The only place that knows the file format (X4_PLANT_SCHEMA.md, first issue of 2026-11-14)."""
    os.makedirs(folder, exist_ok=True)
    json.dump(plant, open(os.path.join(folder, 'plant.json'), 'w'))
    json.dump(outputs, open(os.path.join(folder, 'outputs.json'), 'w'), indent=1)
    np.savez(os.path.join(folder, 'side.npz'), **side)
    return folder


def check(plant, side, ctx):
    """The exported modal data rebuilds the study's FRF (per volt) at every exported output."""
    w = 2j * np.pi * F; H = np.zeros((NF, plant['n_outputs'], plant['n_drives']), complex)
    for m in plant['modes']:
        wn = 2 * np.pi * m['f_hz']; z = m['zeta'] if 'zeta' in m else 1 / (2 * m['q'])
        H += (w ** 2 / (w ** 2 + 2 * z * wn * w + wn ** 2))[:, None, None] * np.outer(m['output_shape'], m['drive_shape'])[None]
    Href = xt.frf(ctx['mlab'], side['rows'], side['shk']) * side['force_per_volt']
    return float(np.abs(H - Href).max() / np.abs(Href).max())


def read_run(run_folder, force_per_volt):
    """A run folder's played drive matrix as (D in N^2/Hz, f). Shapes come from the run's manifest."""
    f = np.fromfile(os.path.join(run_folder, 'freqs_hz.f8'), '<f8')
    D = np.fromfile(os.path.join(run_folder, 'drive_sdm_played.c16'), '<c16'); nd = int(round(np.sqrt(D.size / f.size)))
    return D.reshape(f.size, nd, nd) * force_per_volt ** 2, f


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ns', type=int, default=6)
    ap.add_argument('--nc', type=int, default=12); ap.add_argument('--change', default='none', choices=list(CHANGE))
    ap.add_argument('--kind', default=None, choices=('modal', 'jointed')); a = ap.parse_args()
    ctx = xt.context(a.seed, a.ns, a.nc); plant, outputs, side = build(a.seed, a.ns, a.nc, a.change, a.kind, ctx)
    folder = write_plant(plant_dir(a.seed, a.ns, a.nc, a.change), plant, outputs, side)
    print(f'{folder}: kind {plant["kind"]}, {len(plant["modes"])} modes ({plant["modes"][6]["f_hz"]:.1f} to {plant["modes"][-1]["f_hz"]:.1f} Hz elastic), '
          f'{plant["n_outputs"]} outputs ({side["n_control"]} control, {side["n_held"]} held-out, {side["n_tip"]} tip), {plant["n_drives"]} drives, '
          f'force_per_volt {side["force_per_volt"]:.2f} N/V, FRF rebuild error {check(plant, side, ctx):.1e} (relative)')


# ======================================================================================
# Agent R, 2026-11-14: plants with monitor outputs (stress as played) and plants that
# change with level. Everything above is unchanged; these are additions.
#
#     $PY x4_matrix.py plant --seeds 0 --rigs 6x12 --changes none,realistic,strong,realistic_soft,strong_soft
#
# writes plant_b.json, outputs_b.json and side_b.npz into the plant folder of each (build_b, write_plant_b).
#
# Stress monitors. Fibre stress row r is sigma_r = sum_m SB[r, m] q_m (q modal displacement).
# The plant gets one extra output per row, a_r = scale_r * sum_m SB[r, m] q_m'' (each mode's
# output_shape entry is scale_r * SB[r, m], response acceleration), so for stationary response
# G_sigma_r(f) = G_a_r(f) / scale_r^2 / (2 pi f)^4 whatever the modes' frequencies and damping did.
# Modal monitors (changed plants): one output per mode, a_m = scale_m * q_m''.
# Every monitor is scaled so its RMS under the reference line drive is MON_LEVEL times the median
# control RMS, which puts the simulator's measurement noise and float32 rounding far below it.
#
# Plants that change with level. Targets per lab mode m at full level, as an earlier study form A:
#     drop_m = d * k1_m,   damping factor_m = 1 + (g - 1) * k1_m     (elastic modes; none for the 6 suspension modes)
# k1 from x3.mode_factors(mlab, elastic, seed, unit 2, 'A', 'lab') (lognormal, median 1, on a 1/12-octave
# field, the same field the service model of the same unit uses). an earlier study took the damping factor from its
# second field k2 (correlation 0.7 with k1); one joint law per plant cannot carry two fields, so here the
# damping rise follows k1.
#   jointed (jointed.rs), both changes:  f = f0 (1 - delta min(1, r)),  zeta = zeta0 (1 + gain r),  r = A_m / ref_m
#       delta = JOINT_HEADROOM * d * max(k1),  gain = (g - 1) * delta / d,  both exponents 1,  decay_cycles 5
#       ref_m = ENV_OVER_RMS * qrms_m * delta / (d * k1_m)
#     so at envelope A_m = ENV_OVER_RMS * qrms_m (qrms_m: the mode's RMS displacement under the reference line
#     drive on the linear plant at full level) the mode has dropped d k1_m and its damping is 1 + (g-1) k1_m
#     times; both go on linearly with amplitude up to JOINT_HEADROOM times the largest target.
#   softening (fitted.rs), frequency only, Q holds:  f = f0 (1 - fraction r^2 / (1 + r^2)),  r = rms_m / ref_m
#       fraction = 2 * d * max(k1),  ref_m = qrms_m * sqrt((1 - x_m) / x_m),  x_m = d * k1_m / fraction
#     so at rms_m = qrms_m the mode has dropped d k1_m. smoothing 0.25, one read a second.
MON_LEVEL = 1.0e6
JOINT_HEADROOM = 1.5
ENV_OVER_RMS = float(np.sqrt(2))     # the joint envelope (a peak follower) over the RMS modal displacement; see the study's notes
# Calibrated per tier on seed 0, rig 6 x 12, pinv_closed, line reference (the study's notes): with sqrt(2) the realistic tier
# played 1.32 times its frequency target and 1.39 times its damping rise, the strong tier 1.00 and 1.06.
ENV_OVER_RMS_TIER = {'realistic': 1.9, 'strong': 1.45}
TIERS_B = {'none': None, 'realistic': (0.01, 1.5, 'jointed'), 'strong': (0.03, 3.0, 'jointed'),
           'realistic_soft': (0.01, 1.0, 'modal'), 'strong_soft': (0.03, 1.0, 'modal')}
SERVICE_TIER = {'none': 'linear', 'realistic': 'realistic', 'strong': 'strong',
                'realistic_soft': 'realistic_soft', 'strong_soft': 'strong_soft'}


def register_tiers():
    """Frequency-only service tiers for the softening plant, added to an earlier study's table at run time (x3 file unchanged)."""
    import x3_banded_spec_study as x3
    x3.TIERS.setdefault('realistic_soft', (0.01, 1.0, 'A')); x3.TIERS.setdefault('strong_soft', (0.03, 1.0, 'A'))
    return x3


def level_targets(seed, md, change):
    """(drop, zfac, k1): each lab mode's frequency drop and damping factor asked for at full level."""
    nm = md.fn.size; el = (np.arange(nm) >= 6).astype(float)
    if TIERS_B[change] is None:
        return np.zeros(nm), np.ones(nm), el
    d, g, _ = TIERS_B[change]; x3 = register_tiers(); k1, _ = x3.mode_factors(md, el, seed, 2, 'A', 'lab')
    return d * k1, 1 + (g - 1) * k1, k1


def accel_cov(ctx, D):
    """Covariance of the lab modes' accelerations (nm, nm) under drive matrix D (N^2/Hz), linear plant."""
    md = ctx['mlab']; A = (md.Phi[ctx['shk']].T[None] @ xt.factor(D)) * (md.h(F) * -xt.W ** 2).T[:, :, None]
    A2 = A.transpose(1, 0, 2).reshape(md.fn.size, -1); return (A2.real @ A2.real.T + A2.imag @ A2.imag.T) * DF


def build_b(seed, ns, nc, change='none', stress=True, modal=None, ctx=None, env_over_rms=None):
    """The plant with monitor outputs and, for a tier, the level law. (plant, outputs, side)."""
    ctx = ctx or xt.context(seed, ns, nc); md = ctx['mlab']; nm = md.fn.size; elastic = np.arange(nm) >= 6
    plant, outputs, side = build(seed, ns, nc, 'none', 'modal', ctx)        # the linear plant, physical outputs
    n_phys = plant['n_outputs']; qrms = side['modal_rms_line_drive']; modal = (change != 'none') if modal is None else modal
    kappa = ENV_OVER_RMS_TIER.get(change, ENV_OVER_RMS) if env_over_rms is None else float(env_over_rms)
    drop, zfac, k1 = level_targets(seed, md, change); kind = 'modal'; law = {}
    # monitors
    C = accel_cov(ctx, xt.line_drive(ctx)); osh = side['output_shape']          # (nm, n_phys)
    ctrl_rms = float(np.median(np.sqrt(np.einsum('mc,mn,nc->c', osh[:, :nc], C, osh[:, :nc]))))
    SB = ctx['SB']; s_scale = np.zeros(0); m_scale = np.zeros(0); extra = []
    if stress:
        s_scale = MON_LEVEL * ctrl_rms / np.sqrt(np.einsum('rm,mn,rn->r', SB, C, SB))
        extra.append((SB * s_scale[:, None]).T)                                  # (nm, n_stress)
        outputs = outputs + [dict(name=f's{r:04d}', role='monitor') for r in range(SB.shape[0])]
    if modal:
        m_scale = MON_LEVEL * ctrl_rms / np.sqrt(np.diag(C)); extra.append(np.diag(m_scale))
        outputs = outputs + [dict(name=f'm{m:03d}', role='monitor') for m in range(nm)]
    full = np.hstack([osh] + extra); plant['n_outputs'] = int(full.shape[1])
    for m in range(nm):
        plant['modes'][m]['output_shape'] = full[m].tolist()
    plant['noisy_outputs'] = int(n_phys)                 # the monitors read clean (modal plant; the jointed plant has no such switch)
    if TIERS_B[change] is not None:
        d, g, kind = TIERS_B[change]; kmax = float(k1[elastic].max())
        if kind == 'modal':
            frac = 2 * d * kmax; x = np.where(elastic, d * k1 / frac, 0.5)
            plant['softening'] = dict(modes=np.flatnonzero(elastic).tolist(), fraction=frac, smoothing=0.25, read_samples=GRID['nfft'] // 2,
                                      reference_rms=(qrms * np.sqrt((1 - x) / x))[elastic].tolist())
            law = dict(fraction=frac)
        else:
            delta = JOINT_HEADROOM * d * kmax; gain = (g - 1) * delta / d
            ref = np.where(elastic, kappa * qrms * delta / np.maximum(d * k1, 1e-300), 1e30)
            modes = [dict(f_hz=float(md.fn[m]), zeta=float(md.zeta[m]), drive_shape=plant['modes'][m]['drive_shape'],
                          output_shape=plant['modes'][m]['output_shape'], reference_amplitude=float(ref[m])) for m in range(nm)]
            plant = dict(kind='jointed', n_outputs=plant['n_outputs'], n_drives=plant['n_drives'], feedthrough=0.0, residual=[],
                         transport_delay_samples=0, modes=modes, output_delays_s=[],
                         joint=dict(delta=delta, damping_gain=gain, freq_exponent=1.0, damping_exponent=1.0, decay_cycles=5.0))
            law = dict(delta=delta, damping_gain=gain, env_over_rms=kappa)
    side = dict(side, change=change, kind=kind, n_phys=n_phys, n_stress=int(s_scale.size), n_modal=int(m_scale.size),
                stress_scale=s_scale, modal_scale=m_scale, target_drop=drop, target_zfac=zfac, k1=k1,
                law_keys=np.array(sorted(law)), law_vals=np.array([law[k] for k in sorted(law)]), mon_level=MON_LEVEL, ctrl_rms=ctrl_rms)
    return plant, outputs, side


def write_plant_b(folder, plant, outputs, side, tag='b'):
    """tag 'b': fibre stress monitors (and modal monitors on a changed plant). tag 'm': modal monitors only (the moments route)."""
    os.makedirs(folder, exist_ok=True)
    for name, obj in ((f'plant_{tag}.json', plant), (f'outputs_{tag}.json', outputs)):
        tmp = os.path.join(folder, name + f'.tmp{os.getpid()}'); json.dump(obj, open(tmp, 'w')); os.replace(tmp, os.path.join(folder, name))
    tmp = os.path.join(folder, f'side_{tag}.tmp{os.getpid()}.npz'); np.savez(tmp, **side); os.replace(tmp, os.path.join(folder, f'side_{tag}.npz'))
    return folder


def read_moments(run_folder):
    """(2, n, n) displacement band moments of the run's moment block (response_moments_played.f8), or None."""
    path = os.path.join(run_folder, 'response_moments_played.f8')
    if not os.path.exists(path):
        return None
    M = np.fromfile(path, '<f8'); n = int(round(np.sqrt(M.size / 2))); return M.reshape(2, n, n)


def read_run_b(run_folder, force_per_volt):
    """(D in N^2/Hz, psd (lines, outputs), f): the played drive matrix and output autospectra of a run folder."""
    f = np.fromfile(os.path.join(run_folder, 'freqs_hz.f8'), '<f8'); D, _ = read_run(run_folder, force_per_volt)
    p4 = os.path.join(run_folder, 'response_psd_played.f4')
    psd = np.fromfile(p4, '<f4').astype(float) if os.path.exists(p4) else np.fromfile(os.path.join(run_folder, 'response_psd_played.f8'), '<f8')
    return D, psd.reshape(f.size, -1), f

