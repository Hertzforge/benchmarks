"""The MIMO Shock suite, draft 0.2: pulses, targets, the SRS and the level rule in time. NumPy only.

The structure is a MIMO Random structure (`structure.json` of version 0.2 of the set, byte for byte) and the level
rule is the one `reference_model.simulate_level` steps: per sample, each mode's amplitude sqrt(q^2 + (v/w)^2) feeds
a peak follower that decays by exp(-1) in `decay_cycles` periods of the mode's small-signal frequency, and the next
sample's frequency and damping come from that envelope through the mode's `level` object. Nothing in the rule is
new for shock. Under a steady random drive it is the Random suite's rule (the same function on the same file); in a
pulse the envelope follows the mode's running amplitude, so the structure softens or hardens as it responds and
recovers over about `decay_cycles` periods after the response dies down. Each pulse starts from rest.

A pulse is one record of N samples at FS. A target is the full-level reference waveform at every control and its
SRS (maximax absolute acceleration, Q = 10) at the twelfth-octave centres 10 to 1995 Hz (Method 516.8); the ladder plays it at a level (dB) as the
reference times 10^(level / 20).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import reference_model as rm  # noqa: E402

FS = 8192.0                     # the Random suite's sample rate; its structures are checked against FS / 2
N = 16384                       # two seconds a record (a 1 s record let the inverse drives wrap; see Design.md)
Q = 10.0
BAND_SRS = (50.0, 2000.0)       # the SRS target's span (the shock record's: third octaves 50 Hz to 2 kHz)
BAND_CLASSICAL_LO = 5.0         # classical pulses carry their content (and the compensation) down to a few Hz
K3 = np.arange(-13, 4)          # third-octave points 1000 2^(k/3): 49.6 Hz to 2000 Hz, 17 points
NAT3 = 1000.0 * 2.0 ** (K3 / 3.0)
K12 = np.arange(-52, 13)        # twelfth-octave points over the same span (the record's fine reading), 65 points
NAT12 = 1000.0 * 2.0 ** (K12 / 12.0)
K516 = np.arange(-80, 13)       # draft 0.2: twelfth-octave centres 1000 10^(k/40) (base ten), 10.0 Hz to 1995.3 Hz, 93 points
NAT516 = 1000.0 * 10.0 ** (K516 / 40.0)
SRS_INNER_DB = (-1.5, 3.0)      # Method 516.8: -1.5/+3 dB on at least 90 % of the points, the rest within -3/+6 dB
SRS_INNER_SHARE = 0.9
SRS_OUTER_DB = (-3.0, 6.0)
TE_CREST_FACTOR = 3.0           # effective duration TE (Method 516.8 Annex A 1.3.2): the minimum time holding every
                                # magnitude above APk / CF, CF 3 when the specification gives none (draft 0.2 used 10 %)
T_PULSE = 0.6                   # the SRS pulse's wavelets start here (s): room before it for the drive's precursor
TE = 0.06                       # the SRS pulse's effective duration (s)
T_CLASSICAL = 0.011             # classical pulse duration (s), the 516 functional shock's 11 ms
T_CENTRE = 0.9                 # classical pulse centre (s); the compensation pulses sit either side
COMP_FACTOR = 5.0               # each compensation half-sine lasts this times the pulse duration


# ---------------------------------------------------------------------------------------------- SRS
def srs(x, fs, natural_hz, q=Q, scale=1.0):
    """Maximax absolute-acceleration SRS of each row of x (rows, samples), Smallwood's ramp-invariant filter, from
    rest, over the record. (rows, points), divided by `scale`. Uses scipy's lfilter when present (the same filter)."""
    x = np.atleast_2d(np.asarray(x, float))
    nat = np.asarray(natural_hz, float)
    zeta = 1.0 / (2.0 * q)
    dt = 1.0 / fs
    out = np.empty((x.shape[0], nat.size))
    try:
        from scipy.signal import lfilter
    except ImportError:
        lfilter = None
    coefs = []
    for fn in nat:
        wn = 2 * np.pi * fn
        wd = wn * np.sqrt(1 - zeta * zeta)
        e = np.exp(-zeta * wn * dt)
        kk = wd * dt
        c, s = e * np.cos(kk), e * np.sin(kk)
        sp = s / kk
        coefs.append(([1 - sp, 2 * (sp - c), e * e - sp], [1.0, -2 * c, e * e]))
    if lfilter is not None:
        for k, (b, a) in enumerate(coefs):
            out[:, k] = np.max(np.abs(lfilter(b, a, x, axis=1)), axis=1)
        return out / scale
    B = np.array([c[0] for c in coefs]).T[:, None, :]          # (3, 1, points)
    A = np.array([c[1] for c in coefs]).T[:, None, :]
    y1 = np.zeros((x.shape[0], nat.size)); y2 = np.zeros_like(y1); x1 = np.zeros((x.shape[0], 1)); x2 = np.zeros_like(x1)
    peak = np.zeros_like(y1)
    for t in range(x.shape[1]):
        x0 = x[:, t:t + 1]
        y0 = B[0] * x0 + B[1] * x1 + B[2] * x2 - A[1] * y1 - A[2] * y2
        np.maximum(peak, np.abs(y0), out=peak)
        y2, y1, x2, x1 = y1, y0, x1, x0
    return peak / scale


def error_db(measured, target):
    m, t = np.asarray(measured, float), np.asarray(target, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where((m > 0) & (t > 0), 20 * np.log10(m / t), -400.0)


def excess_516(e):
    """Per control (rows of SRS errors in dB), how far its errors are outside Method 516.8's SRS tolerance, dB: the
    larger of (a) the inner-band excess of the point at the 90 % share (every point ranked by its excess over
    -1.5/+3 dB; at least 90 % must be inside) and (b) the largest excess over -3/+6 dB. At or below 0 is inside;
    negative is the margin."""
    e = np.atleast_2d(np.asarray(e, float))
    lo, hi = SRS_INNER_DB
    d_in = np.sort(np.maximum(e - hi, lo - e), axis=1)
    k = int(np.ceil(SRS_INNER_SHARE * e.shape[1] - 1e-9)) - 1
    d_out = np.max(np.maximum(e - SRS_OUTER_DB[1], SRS_OUTER_DB[0] - e), axis=1)
    return np.maximum(d_in[:, k], d_out)


def within_516(e):
    return bool(np.all(excess_516(e) <= 0.0))


def effective_duration(x, fs=FS, crest_factor=TE_CREST_FACTOR):
    """TE of each row (s), Method 516.8 Annex A 1.3.2: the minimum time containing every time-history magnitude
    above APk / CF (APk the row's peak magnitude, CF the crest factor, 3 unless given): first to last such sample."""
    x = np.atleast_2d(np.asarray(x, float))
    out = np.empty(x.shape[0])
    for i, r in enumerate(np.abs(x)):
        k = np.flatnonzero(r > r.max() / crest_factor)
        out[i] = (k[-1] - k[0] + 1) / fs if k.size else 0.0
    return out


# ---------------------------------------------------------------------------------------------- pulses
def _half_sine(t, centre, dur):
    u = (t - (centre - dur / 2)) / dur
    return np.where((u >= 0) & (u <= 1), np.sin(np.pi * np.clip(u, 0, 1)), 0.0)


def classical(shape, peak=1.0, dur=T_CLASSICAL, centre=T_CENTRE, fs=FS, n=N):
    """A classical pulse with a pre- and a post-pulse half-sine of COMP_FACTOR times its duration, their amplitudes
    solved so the velocity and the displacement return to zero at the end (a shaker cannot hold a velocity change).
    shape: 'half_sine' or 'sawtooth' (terminal peak, 10 % return time). Returns (waveform, ideal pulse alone,
    window (start, end) of the pulse proper in seconds)."""
    t = np.arange(n) / fs
    if shape == 'half_sine':
        main = _half_sine(t, centre, dur)
        a0, a1 = centre - dur / 2, centre + dur / 2
    elif shape == 'sawtooth':
        a0 = centre - dur / 2
        rise = np.clip((t - a0) / dur, 0, None)
        ret = 0.1 * dur
        main = np.where(t < a0, 0.0, np.where(t <= a0 + dur, rise, np.clip(1 - (t - a0 - dur) / ret, 0, None)))
        a1 = a0 + dur + ret
    else:
        raise ValueError(shape)
    main = peak * main
    tc = COMP_FACTOR * dur
    pre = _half_sine(t, a0 - tc / 2, tc)
    post = _half_sine(t, a1 + tc / 2, tc)
    dt = 1.0 / fs
    def vx(a):                                            # velocity and displacement at the end
        v = np.cumsum(a) * dt
        return np.array([v[-1], np.sum(v) * dt])
    M = np.column_stack([vx(pre), vx(post)])
    c = np.linalg.solve(M, -vx(main))
    return main + c[0] * pre + c[1] * post, main, (a0, a1)


def wavelet(t, f, n_half, t0):
    tw = t - t0
    d = n_half / (2 * f)
    m = (tw >= 0) & (tw <= d)
    w = np.zeros_like(t)
    w[m] = np.sin(2 * np.pi * f * tw[m] / n_half) * np.sin(2 * np.pi * f * tw[m])
    return w


def srs_shape(nat):
    """The target SRS shape over the points: +6 dB an octave from 50 to 100 Hz, flat above (the shock record's twist
    target's shape, its 50 g flat from 100 Hz), one at 100 Hz and above."""
    nat = np.asarray(nat, float)
    return np.minimum(1.0, nat / 100.0)


def synth_srs_pulse(seed, fs=FS, n=N, iters=30):
    """One acceleration waveform whose SRS on the twelfth-octave points follows srs_shape (unit flat level):
    Smallwood wavelets at each twelfth-octave point from 40 Hz, odd half-cycle counts, signs and delays inside TE,
    drawn from `seed`; amplitudes by the amplitude iteration on the waveform's SRS (the record's synthesis, x2)."""
    rng = np.random.default_rng([20261121, seed])
    t = np.arange(n) / fs
    fw = 1000.0 * 2.0 ** (np.arange(-56, 13) / 12.0)
    B = np.zeros((fw.size, n))
    for k, f in enumerate(fw):
        nmax = int(np.clip(2 * f * TE * 0.7, 3, 41))
        nh = 2 * int(rng.integers(1, (nmax - 1) // 2 + 1)) + 1
        d = nh / (2 * f)
        B[k] = rng.choice((-1.0, 1.0)) * wavelet(t, f, nh, T_PULSE + rng.uniform(0, max(TE - d, 0)))
    want = srs_shape(fw)
    A = want / 4
    best = (np.inf, A)
    for _ in range(iters):
        M = srs(A @ B, fs, fw)[0]
        e = np.sqrt(np.mean(error_db(M, want) ** 2))
        if e < best[0]:
            best = (e, A.copy())
        A = A * np.clip(want / M, 0.5, 2.0)
    return best[1] @ B, float(best[0])


# ---------------------------------------------------------------------------------------------- the structure
def controls(st):
    return st.rows('control')


def frf(st, fs=FS, n=N, r=None):
    """Small-signal frequency response of the stepped structure at the controls on the record's lines,
    (n/2 + 1, controls, drives), Y = H D under numpy's rfft sign."""
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return rm.discrete_frf(st, f, fs, rows=controls(st), r=r)


def play(st, drives, fs=FS, rows=None):
    """One pulse from rest through the level rule in time: drives (drives, samples), volts. Returns (responses
    (rows, samples), the LevelRun). rows default: the controls."""
    run = rm.simulate_level(st, np.asarray(drives, float).T, fs, rows=controls(st) if rows is None else rows)
    return run.outputs.T, run


def article_modes(st, band=(20.0, 2000.0)):
    """Modes that carry a level rule (not the suspension's) with a small-signal frequency in `band`."""
    if st.rules is None:
        return np.zeros(st.n_modes, bool)
    return (st.reference_amplitude < 1e29) & (st.f_hz >= band[0]) & (st.f_hz <= band[1])


def effect_size(st, r):
    """Each mode's size of change at amplitude ratio r, both effects counted: drop min(1,r)^p + rise min(cap,r)^p."""
    R = st.rules
    r = np.broadcast_to(np.asarray(r, float), st.f_hz.shape)
    return R['drop'] * np.minimum(1.0, r) ** R['drop_exponent'] + R['rise'] * np.minimum(R['rise_cap'], r) ** R['rise_exponent']


def regularized_inverse(H, mask, lam_rel=1e-3):
    """Tikhonov inverse per line, the regularization lam_rel times the line's largest singular value; zero off mask."""
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    mu = lam_rel * s[:, :1]
    with np.errstate(divide='ignore', invalid='ignore'):
        f = s / (s ** 2 + mu ** 2)
    f[~np.isfinite(f)] = 0.0
    G = np.einsum('lji,lj,lkj->lik', Vh.conj(), f, U.conj())
    G[~mask] = 0
    return G


def line_mask(band, fs=FS, n=N):
    f = np.fft.rfftfreq(n, 1.0 / fs)
    return (f >= band[0]) & (f <= band[1])
