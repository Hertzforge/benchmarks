"""The MIMO Sine suite, draft 0.1: tests, the reference, the level rule under a tone, the tolerances. NumPy only.

The structure is a MIMO Random structure (`structure.json` of version 0.2 of the set, byte for byte) and the level
rule is the one `reference_model.simulate_level` steps: per sample, each mode's amplitude sqrt(q^2 + (v/w)^2) feeds
a peak follower that decays by exp(-1) in `decay_cycles` periods of the mode's small-signal frequency, and the next
sample's frequency and damping come from that envelope through the mode's `level` object. Nothing in the rule is
new for sine.

Under a steady tone at frequency f the modal displacement of mode m is q_m(t) = Q_m cos(2 pi f t + phi), so
sqrt(q^2 + (v/w)^2) runs between Q_m and Q_m f / f_m (w = 2 pi f_m, the mode's frequency at its envelope) and the
peak follower holds Q_m max(1, f / f_m): at the mode's own resonance the envelope is the modal amplitude itself. The
amplitude ratio r_m = Q_m max(1, f / f_m) / reference_amplitude_m then sets f_m and zeta_m, which set Q_m:

    Q_m = |F_m(f)| / |w_m(r)^2 - w^2 + 2j zeta_m(r) w_m(r) w|,     F_m = drive_shape_m . d(f)  (the modal force)

The tone's steady state is the fixed point of that loop (`tone_ratio_curve`), solved per mode: the modes are
independent in modal coordinates, the outputs only add them. A slow ramp of the level from zero follows the
smallest fixed point; that is the branch used here. `check_tone_state` holds the prediction to simulate_level.

Units: drives in volts peak, outputs in m/s^2 peak, frequency in Hz. Amplitudes are peak values throughout.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import reference_model as rm  # noqa: E402

FORMAT = '0.1-draft'
FS = 8192.0                    # the Random suite's sample rate; its structures are checked against FS / 2
SIM_BOUND_HZ = FS / 2.0
BAND = (20.0, 2000.0)          # every Random v0.2 problem's band (problem.json grid.band_hz), read at generation
REF_PER_OCTAVE = 384           # the reference grid: 1/384 octave (checked by the midpoint interpolation error)
SIZE_PER_OCTAVE = 1536         # the sizing and peak-finding grid: 1/1536 octave
LAM_REL = 1e-3                 # Tikhonov regularization of the least-squares drive, relative to the largest singular value
G0 = 9.80665                   # m/s^2 per g (the engine's profiles are in g peak; the conversion is the runner's, at its edge)
DRIVE_HEADROOM_DB = 6.0       # each case's drive ceiling over its ideal full-level drive peak (owner, 2026-11-25)
ENGINE_MAX_DRIVE_VPK = 10.0    # the shipped MIMO Sine engine's ceiling on a drive (hertzforge-mimo-sine limits.rs, origin/main fa084b49)
ENGINE_MAX_CONTROLS = 16       # and on the counts of controls and drives (the same file)
ENGINE_MAX_DRIVES = 8

# ---------------------------------------------------------------------------------------------- the tests
SWEEP_RATE = {'T1': 1.0, 'T2': 4.0, 'T5': 1.0}          # octaves a minute
SWEEP_DIRECTION = {'T1': 'up', 'T2': 'up', 'T5': 'down'}
STEPS_PER_OCTAVE = 12                                    # T3: geometric steps lo 2^(k/12), the top edge exactly
STEP_SETTLE_TAU = 6.0          # T3: a step is held 6 time constants to settle (loop held at >= 2 tau_m by the loop-speed
STEP_JUDGE_TAU = 4.0           # rule, three loop time constants to 5 %) and 4 more to judge: 10 tau_m of the slowest mode in band,
STEP_MIN_CYCLES = 20           # in seconds, the same at every step; at least 20 cycles of the lowest step (the engine's floor)
DWELL_SETTLE_TAU = 20.0        # T4: 20 time constants of the dwell mode to settle at full level (ten loop time constants at
DWELL_JUDGE_TAU = 10.0         # the loop-speed rule's 2 tau_m), 10 more to judge
DWELL_MIN_S = 10.0             # and at least 10 s (1,000 updates at a 10 ms update)
# the engine's default level table (LadderConfig: -12, -9, -6, -3, 0 dB) with the runner's default rung (2 s): a one-level
# run on a realistic structure aborts at the end of the settle window (Did_Not_Work DNW-A2), so every test climbs in 3 dB rungs
LADDER = dict(sysid_level_db=-20.0, start_db=-12.0, table_db=[-12.0, -9.0, -6.0, -3.0, 0.0], per_rung=dict(seconds=2.0),
              full_for=dict(sweeps=1, what='sweeps: one pass; stepped and dwell: until the schedule ends'),
              what='the engine ladder in its own form (LadderConfig); judged at the last rung (full level) only')
NOISE_REL = 1e-3               # measurement noise on every read channel, relative to the full-level specification amplitude

# ---------------------------------------------------------------------------------------------- tolerances (read)
# amplitude bands as ratios of the measured over the specified amplitude, (low, high); see FORMAT_SINE.md, "Tolerances"
TOL = {
    'iec_check_point': dict(below_500=(0.75, 1.25), above_500=(0.50, 1.50), clause='IEC 60068-2-6:2007 4.1.4.2',
                            status='read', what='each control as a check point: +/-25 % up to 500 Hz, +/-50 % above'),
    'iec_reference_point': dict(all=(0.85, 1.15), clause='IEC 60068-2-6:2007 4.1.4.1 with 4.2.1 and 4.2.1.1', status='read',
                                what='the composite (arithmetic average of the check points, 4.2.1.1) as the fictitious '
                                     'reference point of multipoint control (4.2.1): +/-15 %'),
    'mil_composite': dict(all=(0.90, 1.10), clause='MIL-STD-810H Method 514.8 Table 514.8-III, footnote 1', status='read',
                          what='composite control (the average of the control channels): +/-10 %'),
    'mil_average_individual': dict(below_500=(0.75, 1.25), above_500=(0.50, 1.50), clause='MIL-STD-810H Method 514.8 '
                                   'Table 514.8-III (multi-point control, average control)', status='read',
                                   what='each control channel under average control: +/-25 % below 500 Hz, +/-50 % above'),
    'mil_maxi_individual': dict(below_500=(0.75, 1.10), above_500=(0.50, 1.10), clause='MIL-STD-810H Method 514.8 '
                                'Table 514.8-III (multi-point control, maxi control)', status='read',
                                what='each control channel under maxi control: +10 %/-25 % below 500 Hz, +10 %/-50 % above'),
}
FREQ_TOL = {'iec_sweep': dict(clause='IEC 60068-2-6:2007 4.1.5.1', status='read',
                              what='+/-1 Hz from 5 to 50 Hz, +/-2 % above 50 Hz (and +/-0.05 Hz to 0.25 Hz, +/-20 % 0.25 to 5 Hz)'),
            'iec_fixed': dict(clause='IEC 60068-2-6:2007 4.1.5.2 a)', status='read', what='fixed frequency +/-2 %'),
            'mil': dict(clause='MIL-STD-810H Method 514.8 Table 514.8-III', status='read', what='frequency +/-0.1 %')}
RATE_TOL = dict(fraction=0.10, clause='IEC 60068-2-6:2007 4.1.6 (one octave a minute +/-10 %); MIL-STD-810H Method 514.8 '
                'Table 514.8-III (+/-10 % of the stated rate)', status='read')
DURATION_TOL = dict(fraction=0.05, clause='MIL-STD-810H Method 514.8 Table 514.8-III (test time duration +/-5 %)', status='read')
PHASE_ANALOGY_DEG = 10.0       # reported, not judged: no standard read sets a phase tolerance for several exciters


def iec_freq_tol_hz(f_hz):
    """IEC 60068-2-6 4.1.5.1, endurance by sweeping, the tolerance in Hz at f."""
    f = np.asarray(f_hz, float)
    return np.where(f <= 0.25, 0.05, np.where(f <= 5.0, 0.2 * f, np.where(f <= 50.0, 1.0, 0.02 * f)))


def band_at(tol, f_hz):
    """The (low, high) amplitude ratio band of tolerance `tol` at each frequency: arrays (lines,)."""
    f = np.asarray(f_hz, float)
    if 'all' in tol:
        lo, hi = tol['all']
        return np.full(f.shape, lo), np.full(f.shape, hi)
    b, a = tol['below_500'], tol['above_500']
    return np.where(f <= 500.0, b[0], a[0]), np.where(f <= 500.0, b[1], a[1])


# ---------------------------------------------------------------------------------------------- grids
def log_grid(band=BAND, per_octave=REF_PER_OCTAVE):
    """lo 2^(k / per_octave) below hi, then hi exactly (the engine's own step-ladder rule)."""
    lo, hi = band
    n = int(np.ceil(np.log2(hi / lo) * per_octave - 1e-9))
    f = lo * 2.0 ** (np.arange(n) / per_octave)
    return np.append(f[f < hi * (1 - 1e-12)], hi)


def step_ladder(band=BAND, steps_per_octave=STEPS_PER_OCTAVE):
    return log_grid(band, steps_per_octave)


def sweep_seconds(band, rate_oct_min):
    return 60.0 * np.log2(band[1] / band[0]) / rate_oct_min


# ---------------------------------------------------------------------------------------------- the structure
def controls(st):
    return st.rows('control')


def gauges(st):
    return st.rows('held_out', 'gauge')


def article_modes(st, band=BAND):
    """Modes that carry a level rule (not the suspension's) with a small-signal frequency in `band`. A linear
    structure: the modes in band that are not its suspension (the first `rigid_modes`)."""
    inb = (st.f_hz >= band[0]) & (st.f_hz <= band[1])
    if st.rules is None:
        k = np.arange(st.n_modes) >= int(st.raw.get('rigid_modes', 0))
        return inb & k
    return (st.reference_amplitude < 1e29) & inb


def effect_size(st, r):
    """Each mode's size of change at amplitude ratio r, both effects counted: drop min(1,r)^p + rise min(cap,r)^p
    (MIMO Shock's measure, shock_model.effect_size)."""
    R = st.rules
    r = np.broadcast_to(np.asarray(r, float), st.f_hz.shape)
    return R['drop'] * np.minimum(1.0, r) ** R['drop_exponent'] + R['rise'] * np.minimum(R['rise_cap'], r) ** R['rise_exponent']


def mode_state(st, m, r):
    """(f_hz, zeta) of mode m at amplitude ratios r (array): the level rule of reference_model.level_state, one mode."""
    r = np.asarray(r, float)
    if st.rules is None:
        return np.full(r.shape, st.f_hz[m]), np.full(r.shape, st.zeta[m])
    R = {k: v[m] for k, v in st.rules.items()}
    ff = 1.0 - R['drop'] * np.minimum(1.0, r) ** R['drop_exponent'] + R['rise'] * np.minimum(R['rise_cap'], r) ** R['rise_exponent']
    z = np.minimum(rm.MAX_DAMPING, st.zeta[m] * (1.0 + R['damping_gain'] * np.minimum(R['damping_cap'], r) ** R['damping_exponent']))
    return st.f_hz[m] * ff, z


# ---------------------------------------------------------------------------------------------- the reference
def regularized_inverse(H, lam_rel=LAM_REL):
    """Tikhonov inverse per line, (lines, drives, controls): the regularization lam_rel times the line's largest
    singular value (MIMO Shock's regularized_inverse without the band mask)."""
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    mu = lam_rel * s[:, :1]
    with np.errstate(divide='ignore', invalid='ignore'):
        f = s / (s ** 2 + mu ** 2)
    f[~np.isfinite(f)] = 0.0
    return np.einsum('lji,lj,lkj->lik', Vh.conj(), f, U.conj())


def frf(st, f_hz, rows):
    with np.errstate(all='ignore'):
        return rm.frf(st, f_hz, rows=rows)


def ideal(st, f_hz, amplitude=1.0):
    """The specification (amplitude at every control, in phase) projected onto what the system reaches in its
    small-signal model. Returns (d (lines, drives) volts peak: the least-squares drive; R (lines, controls): its
    response at the controls, the reference; H (lines, controls, drives))."""
    H = frf(st, f_hz, controls(st))
    G = regularized_inverse(H)
    spec = np.full((len(f_hz), H.shape[1]), float(amplitude), complex)
    d = np.einsum('ldc,lc->ld', G, spec)
    R = np.einsum('lcd,ld->lc', H, d)
    return d, R, H


def interp_reference(f_grid, amp, phase_deg, f_hz):
    """The engine's interpolation of a reference given on a grid: amplitude log-log, phase linear against log f
    (taking the shorter way round between neighbouring points), held beyond the ends. Returns complex (lines, controls)."""
    lf, lq = np.log(f_grid), np.log(np.clip(f_hz, f_grid[0], f_grid[-1]))
    out_a = np.empty((len(lq), amp.shape[1]))
    out_p = np.empty_like(out_a)
    ph = np.unwrap(np.radians(phase_deg), axis=0)
    for c in range(amp.shape[1]):
        out_a[:, c] = np.exp(np.interp(lq, lf, np.log(amp[:, c])))
        out_p[:, c] = np.interp(lq, lf, ph[:, c])
    return out_a * np.exp(1j * out_p)


def reference_grid_error(st, f_grid, amp, phase_deg, amplitude):
    """The reference at the geometric midpoints of the grid, interpolated as the engine does, against the projection
    computed there: (largest |dB|, largest |degrees|)."""
    mid = np.sqrt(f_grid[1:] * f_grid[:-1])
    _, Rm, _ = ideal(st, mid, amplitude)
    Ri = interp_reference(f_grid, amp, phase_deg, mid)
    e = Ri / Rm
    return float(np.max(np.abs(20 * np.log10(np.abs(e))))), float(np.max(np.abs(np.degrees(np.angle(e)))))


# ---------------------------------------------------------------------------------------------- the level rule under a tone
R_GRID = np.logspace(-5, 3, 1601)          # amplitude ratios at which the fixed-point curves are tabulated


def tone_ratio_curve(st, m, f_hz, F_unit):
    """Mode m under a steady tone: M(r), the smallest specification amplitude (m/s^2 peak, the scale of the unit
    drive) at which the tone at one of the frequencies f_hz holds the mode's envelope ratio at r or above, on R_GRID.
    F_unit (lines,) is the modal force of the unit drive at f_hz. The reached ratio at scale s is then
    r_m(s) = M^{-1}(s) (`ratio_at`), the largest over the frequencies: the resonance the tone passes."""
    fn, ze = mode_state(st, m, R_GRID)                                    # (r,)
    w = 2 * np.pi * np.asarray(f_hz, float)[:, None]
    wn = 2 * np.pi * fn[None, :]
    h = 1.0 / np.abs(wn ** 2 - w ** 2 + 2j * ze[None, :] * wn * w)          # (lines, r)
    env = np.maximum(1.0, w / wn)
    A = np.abs(F_unit)[:, None] * h * env                                   # modal envelope per unit scale
    a_ref = st.reference_amplitude[m] if st.reference_amplitude is not None else 1.0
    with np.errstate(divide='ignore'):
        s_of_r = R_GRID[None, :] * a_ref / A                                # the scale that holds ratio r at f
    S = np.maximum.accumulate(s_of_r, axis=1)                              # ramped from zero: the first crossing
    return S.min(axis=0)                                                    # (r,), nondecreasing


def ratio_at(M, s):
    """r_m(s) from M (one mode's curve on R_GRID), for scales s (array)."""
    s = np.atleast_1d(np.asarray(s, float))
    lr = np.log(R_GRID)
    with np.errstate(divide='ignore'):
        lm = np.log(np.maximum(M, 1e-300))
    lm = np.maximum.accumulate(lm)
    out = np.exp(np.interp(np.log(s), lm, lr, left=lr[0], right=lr[-1]))
    return out


def sweep_curves(st, modes, f_size, d_unit, window=0.15):
    """For each mode in `modes`, its curve M over the tone frequencies within +/-window of its small-signal
    frequency (and in band). d_unit (lines, drives) on f_size."""
    F = d_unit @ st.drive_shape.T                                         # (lines, modes): modal force
    out = {}
    for m in modes:
        sel = (f_size >= st.f_hz[m] * (1 - window)) & (f_size <= st.f_hz[m] * (1 + window))
        if not sel.any():
            sel = np.argmin(np.abs(f_size - st.f_hz[m])) == np.arange(f_size.size)
        out[int(m)] = tone_ratio_curve(st, m, f_size[sel], F[sel, m])
    return out


def state_at(st, curves, s):
    """Per mode in curves: (ratio, signed frequency change, damping factor) reached at scale s."""
    ms = np.array(sorted(curves), int)
    r = np.array([ratio_at(curves[m], s)[0] for m in ms])
    rr = np.zeros(st.n_modes)
    rr[ms] = r
    f, z = rm.level_state(st, rr)
    return ms, r, (f / st.f_hz - 1.0)[ms], (z / st.zeta)[ms]


def size_full_level(st, curves, want):
    """The scale s (m/s^2 peak) at which the median over `curves`' modes of effect_size at the reached ratio equals
    `want` (the median |frequency change| of the Random suite's full level). Returns (s, achieved median)."""
    ms = np.array(sorted(curves), int)

    def med(s):
        rr = np.zeros(st.n_modes)
        rr[ms] = [ratio_at(curves[m], s)[0] for m in ms]
        return float(np.median(effect_size(st, rr)[ms]))
    lo, hi = 1e-6, 1e8
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if med(mid) < want else (lo, mid)
        if hi / lo < 1 + 1e-10:
            break
    s = float(np.sqrt(lo * hi))
    return s, med(s)


# ---------------------------------------------------------------------------------------------- resonances and time constants
def first_resonance(st, f_size, H=None, prominence_db=3.0):
    """The first identified resonance in band: the lowest local maximum of the largest singular value of the
    small-signal response at the controls standing at least prominence_db over the deeper of its two flanking
    minima (each flank searched to the next higher point or the band edge). Returns (f_hz, prominence_db, index)."""
    if H is None:
        H = frf(st, f_size, controls(st))
    sv = 20 * np.log10(np.linalg.svd(H, compute_uv=False)[:, 0])
    n = sv.size
    for i in range(1, n - 1):
        if not (sv[i] > sv[i - 1] and sv[i] >= sv[i + 1]):
            continue
        j = i - 1
        lo_l = sv[i]
        while j >= 0 and sv[j] <= sv[i]:
            lo_l = min(lo_l, sv[j]); j -= 1
        k = i + 1
        lo_r = sv[i]
        while k < n and sv[k] <= sv[i]:
            lo_r = min(lo_r, sv[k]); k += 1
        prom = sv[i] - max(lo_l, lo_r)
        if prom >= prominence_db:
            return float(f_size[i]), float(prom), i
    raise ValueError('no resonance in band stands out by %g dB' % prominence_db)


def nearest_mode(st, f_hz):
    """The in-band mode nearest f in units of its half-power bandwidth."""
    d = np.abs(st.f_hz - f_hz) / (st.zeta * st.f_hz)
    d[~((st.f_hz >= BAND[0] * 0.8) & (st.f_hz <= BAND[1] * 1.2))] = np.inf
    return int(np.argmin(d))


def tau_cycles(zeta):
    """A mode's time constant 1 / (zeta w) in cycles of its own frequency: 1 / (2 pi zeta) (Derivations LD-5)."""
    return 1.0 / (2 * np.pi * np.asarray(zeta, float))


def step_seconds(st, band=BAND):
    """T3: seconds a step, one number for the whole ladder (the runner's stepped test holds every step `step_s`):
    (STEP_SETTLE_TAU + STEP_JUDGE_TAU) time constants 1 / (zeta w) of the slowest mode in band, at least STEP_MIN_CYCLES
    cycles of the band's lowest step. Returns (step_s, settle_s, the slowest mode, its time constant in seconds)."""
    inb = np.flatnonzero((st.f_hz >= band[0]) & (st.f_hz <= band[1]))
    tau = 1.0 / (st.zeta[inb] * 2 * np.pi * st.f_hz[inb])
    k = int(np.argmax(tau))
    step = max(STEP_MIN_CYCLES / band[0], (STEP_SETTLE_TAU + STEP_JUDGE_TAU) * float(tau[k]))
    settle = step * STEP_SETTLE_TAU / (STEP_SETTLE_TAU + STEP_JUDGE_TAU)
    return float(step), float(settle), int(inb[k]), float(tau[k])


def dwell_cycles(f_hz, zeta):
    """T4: cycles of the dwell: (DWELL_SETTLE_TAU + DWELL_JUDGE_TAU) time constants of the dwell mode, at least
    DWELL_MIN_S seconds."""
    n = int(np.ceil((DWELL_SETTLE_TAU + DWELL_JUDGE_TAU) * tau_cycles(zeta)))
    return max(n, int(np.ceil(DWELL_MIN_S * f_hz)))


# ---------------------------------------------------------------------------------------------- tones in time
RAMP_S = 0.5                   # the checks' raised-cosine ramp-in (a drive switched on from rest rings every mode)


def zeta_from_frequency(st, m, f_obs):
    """The damping ratio mode m has when the level rule puts it at frequency f_obs: the rule's state at the smallest
    amplitude ratio whose size of change reaches the observed one (the branch a slow rise from rest follows); the
    small-signal value for a linear structure or a mode that has not moved."""
    f_obs = np.atleast_1d(np.asarray(f_obs, float))
    if st.rules is None or st.reference_amplitude[m] > 1e29:
        return np.full(f_obs.shape, st.zeta[m])
    fr, zr = mode_state(st, m, R_GRID)
    size = np.maximum.accumulate(np.abs(fr / st.f_hz[m] - 1.0))
    want = np.abs(f_obs / st.f_hz[m] - 1.0)
    k = np.minimum(np.searchsorted(size, want, side='left'), R_GRID.size - 1)
    out = zr[k]
    out[want <= 1e-12] = st.zeta[m]
    return out


def ramp_in(n, fs=FS, seconds=RAMP_S):
    t = np.arange(n) / fs
    return np.where(t < seconds, 0.5 - 0.5 * np.cos(np.pi * t / seconds), 1.0) if seconds > 0 else np.ones(n)


def sweep_drive(f_grid, d_grid, band, rate_oct_min, direction='up', fs=FS, scale=1.0, t_max=None, ramp_s=RAMP_S):
    """Drive samples (samples, drives) of a logarithmic sweep playing at every instant the steady drive d(f) of the
    grid (complex, volts peak), interpolated in log f, ramped in over ramp_s; and the instantaneous frequency."""
    T = sweep_seconds(band, rate_oct_min)
    if t_max is not None:
        T = min(T, t_max)
    t = np.arange(int(T * fs)) / fs
    k = rate_oct_min / 60.0 * np.log(2.0)
    f0 = band[0] if direction == 'up' else band[1]
    sgn = 1.0 if direction == 'up' else -1.0
    f = f0 * np.exp(sgn * k * t)
    theta = 2 * np.pi * f0 * (np.exp(sgn * k * t) - 1.0) / (sgn * k)
    lf = np.log(f_grid)
    lq = np.log(f)
    D = np.empty((t.size, d_grid.shape[1]))
    ph = np.exp(1j * theta)
    for j in range(d_grid.shape[1]):
        re = np.interp(lq, lf, d_grid[:, j].real)
        im = np.interp(lq, lf, d_grid[:, j].imag)
        D[:, j] = np.real((re + 1j * im) * ph) * scale
    return D * ramp_in(t.size, fs, ramp_s)[:, None], f


def tone_drive(d, f_hz, seconds, fs=FS, scale=1.0, ramp_s=RAMP_S):
    t = np.arange(int(seconds * fs)) / fs
    return np.real(np.asarray(d)[None, :] * np.exp(2j * np.pi * f_hz * t)[:, None]) * scale * ramp_in(t.size, fs, ramp_s)[:, None]


def play(st, drive, rows=None, keep_modes=False):
    with np.errstate(all='ignore'):
        run = rm.simulate_level(st, drive, FS, rows=controls(st) if rows is None else rows, keep_modes=keep_modes)
    return run


def check_tone_state(st, f_hz, d, seconds, curves_mode=None):
    """Play a steady tone (drive d, complex volts peak) for `seconds` from rest through simulate_level and return each
    mode's final envelope ratio (envelope / reference_amplitude)."""
    run = play(st, tone_drive(d, f_hz, seconds))
    return run.envelope / st.reference_amplitude, run
