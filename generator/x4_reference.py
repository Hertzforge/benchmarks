"""an earlier study references for a seed and rig, as the runner's "sdm" reference files.

    PY=python
    $PY x4_reference.py --seed 0 --ns 6 --nc 12 [--tier linear] [--change none] [--frf-run RUN_FOLDER]

Writes into the plant folder of x4_export (same seed, rig and --change), each as a raw '<c16' file of
shape (3961, nc, nc) on lines 40 to 4000 of the 0.5 Hz grid, with a small JSON object beside it:
  ref_banded      the banded specification: unit A in service at the controls, one-third-octave band
                  means of each autospectrum, flat across each band, no cross terms
  ref_converted   the converted narrowband reference H_id D_b H_id^H, D_b one flat drive matrix per
                  one-third-octave band fitted to the banded specification (a1.band_law_v log fit,
                  polished by x3.band_fit), H_id the exact linear lab FRF or a run's identified FRF
  ref_line        the traditional line-by-line reference: unit A's full service matrix at the controls
and ref_side.npz with the band targets t (nc, nb), the band of each line and D_b.
Added 2026-11-14 (R), each by its own function, the three above untouched:
  ref_converted_id   as ref_converted, with a run's identified FRF as H_id (export_identified)
  ref_converted_law  H_id D_first H_id^H: the band_average law's own first solve through the identified FRF
                     (export_converted_law; $PY x4_reference.py --seed 0 --ns 6 --nc 12 --change none --law-run RUN_FOLDER)
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import x4_truth as xt
import x4_export as xe
from x4_truth import F, NF, B3, BO3, H_

LO_LINE = int(round(F[0] / (xe.GRID['fs_hz'] / xe.GRID['nfft'])))


def banded(ctx):
    """(t, G): band targets (nc, nb) and the diagonal reference (nf, nc, nc), flat in each band."""
    c = ctx['c']; t = ctx['GAU'][c] @ B3.T; G = np.zeros((NF, c.size, c.size), complex)
    i = np.arange(c.size); G[:, i, i] = t[:, BO3].T
    return t, G


def converted(ctx, t, Hid=None):
    """(G, Db): H_id D_b H_id^H and the band drive matrix per line (nf, ns, ns), N^2/Hz."""
    import x3_banded_spec_study as x3
    import a1_audit as a1
    c = ctx['c']; Hlin = xt.frf(ctx['mlab'], ctx['U'][c], ctx['shk']); Hid = Hlin if Hid is None else Hid
    Fd3, _ = a1.band_law_v(Hlin.transpose(1, 2, 0), ctx['GAU'][c], N=3, mode='log', npass=6)
    Db = x3.band_fit(Hid, t, x3.db_from_v(Fd3), 3)
    return Hid @ Db @ H_(Hid), Db


def identified(run_folder, nc, ns, force_per_volt):
    """A run's identified FRF (controls x drives per volt) as per newton on the study grid."""
    f = np.fromfile(os.path.join(run_folder, 'freqs_hz.f8'), '<f8')
    H = np.fromfile(os.path.join(run_folder, 'frf_identified.c16'), '<c16').reshape(f.size, nc, ns)
    assert f.size == NF and np.allclose(f, F), 'the run is not on the study grid'
    return H / force_per_volt


def write_reference(folder, name, G):
    """The only place that knows the reference file format (X4_PLANT_SCHEMA.md, kind sdm)."""
    os.makedirs(folder, exist_ok=True); G = (G + H_(G)) / 2
    G.astype('<c16').tofile(os.path.join(folder, name + '.c16'))
    json.dump(dict(kind='sdm', n=int(G.shape[1]), lo_line=LO_LINE, n_lines=int(G.shape[0]), file=os.path.abspath(os.path.join(folder, name + '.c16'))),
              open(os.path.join(folder, name + '.json'), 'w'))


def export(seed, ns, nc, tier='linear', change='none', frf_run=None):
    ctx = xt.context(seed, ns, nc, tier); folder = xe.plant_dir(seed, ns, nc, change); os.makedirs(folder, exist_ok=True)
    t, Gb = banded(ctx); Hid = None
    if frf_run:
        Hid = identified(frf_run, nc, ns, float(np.load(os.path.join(folder, 'side.npz'))['force_per_volt']))
    Gc, Db = converted(ctx, t, Hid)
    for name, G in (('ref_banded', Gb), ('ref_converted', Gc), ('ref_line', ctx['GA'])):
        write_reference(folder, name, G)
    np.savez(os.path.join(folder, 'ref_side.npz'), t=t, band_of_line=BO3, Db=Db.astype(np.complex64), tier=tier,
             hid='run:' + frf_run if frf_run else 'exact linear')
    # checks: band means of the converted reference against the banded targets; reachability of each reference
    gc = np.real(np.einsum('fii->if', Gc)); eb = xt.db10(gc @ B3.T, t)
    Hc = xt.frf(ctx['mlab'], ctx['U'][ctx['c']], ctx['shk']); P = Hc @ xt.x1.pinv_trunc(Hc)
    def reach(G):
        g = np.real(np.einsum('fii->if', G)); gp = np.real(np.einsum('fii->if', P @ G @ H_(P)))
        return float(np.sqrt(np.mean(xt.db10(gp @ B3.T, g @ B3.T) ** 2)))
    print(f'{folder}: lines {LO_LINE} to {LO_LINE + NF - 1}, {nc} controls, {t.shape[1]} one-third-octave bands')
    print(f'  converted against banded targets, band means: rms {np.sqrt(np.mean(eb ** 2)):.3f} dB, worst {np.abs(eb).max():.3f} dB')
    print(f'  declared deviation (P G P against G, one-third octave rms): banded {reach(Gb):.3f}, converted {reach(Gc):.3f}, line {reach(ctx["GA"]):.3f} dB')
    return folder


def export_identified(seed, ns, nc, tier='linear', change='none', frf_run=None, name='ref_converted_id'):
    """R, 2026-11-14 (join request 6): the converted reference built from a run's IDENTIFIED FRF, as a real test
    would build it, written beside the others as ref_converted_id (.c16, .json). export() and its three files are
    untouched. Returns (folder, band-mean error of the new reference against the banded targets, dB RMS)."""
    ctx = xt.context(seed, ns, nc, tier); folder = xe.plant_dir(seed, ns, nc, change); t, _ = banded(ctx)
    Hid = identified(frf_run, nc, ns, float(np.load(os.path.join(folder, 'side.npz'))['force_per_volt']))
    Gc, Db = converted(ctx, t, Hid); write_reference(folder, name, Gc)
    eb = xt.db10(np.real(np.einsum('fii->if', Gc)) @ B3.T, t); return folder, float(np.sqrt(np.mean(eb ** 2)))


def export_converted_law(seed, ns, nc, change, law_run, tier=None, name='ref_converted_law'):
    """R, 2026-11-14 (design, "converted_law"): the narrowband reference H_id D_first H_id^H, where D_first is the
    drive matrix the band_average law commanded on its first solve (its seed call on the identification FRF, before any
    gain correction; the runner's drive_sdm_first.c16) and H_id the identified FRF of the same run (frf_identified.c16).
    Both are per volt, so the product is in output units squared per Hz. Written beside the other references as
    ref_converted_law (.c16, .json) with ref_converted_law_side.npz (D_first in V^2/Hz, where it came from, checks).
    Returns a dict of checks: band-mean error against the banded targets in one-third and one-sixth octaves, the number
    of flat stretches of D_first along frequency (the law's bands: 41 at one-sixth octave on 20 to 2000 Hz), and the
    reference's band means against ref_converted (the decibel fit).
    """
    folder = xe.plant_dir(seed, ns, nc, change); f = np.fromfile(os.path.join(law_run, 'freqs_hz.f8'), '<f8')
    assert f.size == NF and np.allclose(f, F), 'the run is not on the study grid'
    man = json.load(open(os.path.join(law_run, 'manifest.json'))); assert man['law']['id'] == 'band_average', 'the first solve must be the band_average law\'s'
    H = np.fromfile(os.path.join(law_run, 'frf_identified.c16'), '<c16').reshape(NF, nc, ns)
    D = np.fromfile(os.path.join(law_run, 'drive_sdm_first.c16'), '<c16').reshape(NF, ns, ns)
    G = H @ D @ H_(H); write_reference(folder, name, G)
    g = np.real(np.einsum('fii->if', G)); d = np.real(np.einsum('fii->if', D)); out = dict(folder=folder, law_run=law_run)
    B6 = xt.x1.BANDS[6]
    ref_b = np.fromfile(os.path.join(folder, 'ref_banded.c16'), '<c16').reshape(NF, nc, nc); t = np.real(np.einsum('fii->if', ref_b))
    for tag, B in (('b3', B3), ('b6', B6)):
        e = xt.db10(g @ B.T, t @ B.T); out[f'{tag}_rms_db'] = float(np.sqrt(np.mean(e ** 2))); out[f'{tag}_bias_db'] = float(e.mean())
    steps = np.flatnonzero(np.abs(np.diff(D, axis=0)).max((1, 2)) > 1e-9 * np.abs(D).max())      # the law's solve is flat inside each of its bands
    out['drive_plateaus'] = int(steps.size + 1); out['drive_vrms'] = [float(v) for v in np.sqrt(d.sum(1) * xt.DF)]
    pc = os.path.join(folder, 'ref_converted.c16')
    if os.path.exists(pc):
        gc = np.real(np.einsum('fii->if', np.fromfile(pc, '<c16').reshape(NF, nc, nc))); e = xt.db10(g @ B3.T, gc @ B3.T)
        out['vs_converted_b3_rms_db'] = float(np.sqrt(np.mean(e ** 2))); out['vs_converted_b3_bias_db'] = float(e.mean())
    np.savez(os.path.join(folder, name + '_side.npz'), D_first=D.astype(np.complex64), law_run=law_run, case_id=man['case_id'],
             law_params=json.dumps(man['law'].get('params_given')), checks=json.dumps(out))
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--ns', type=int, default=6)
    ap.add_argument('--nc', type=int, default=12); ap.add_argument('--tier', default='linear'); ap.add_argument('--change', default='none')
    ap.add_argument('--frf-run', default=None)
    ap.add_argument('--law-run', default=None, help='a band_average run folder: write ref_converted_law from its first solve, and nothing else')
    a = ap.parse_args()
    if a.law_run:
        print(json.dumps(export_converted_law(a.seed, a.ns, a.nc, a.change, a.law_run), indent=1))
    else:
        export(a.seed, a.ns, a.nc, a.tier, a.change, a.frf_run)
