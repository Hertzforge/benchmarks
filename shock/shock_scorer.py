"""The MIMO Shock scorer, draft 0.2: one run's report card under several readings. NumPy only.

    card = score_run(run)          # run: the dict harness.run returns (or its saved JSON)

No reading is the verdict; each is reported with its margin and, on a miss, the criterion it broke and by how much
(the acceptance proposal's rule: margins and pass rates under several readings, never one binary verdict).

  strict    the shock record's run criteria 1 to 7 (mshock_gym.harness.score, R6-RUN): 1 probes and the first
            ladder pulse at most +3 dB over their own level; 2 no pulse below full level more than 3 dB over the
            full-level target (absolute); 3 the first full-level pulse within +3 dB; 4 the third full-level pulse
            inside the 516.8 tolerance (below); 5 the run completed and its last pulse inside it; 6 no identification
            excitation over +3 dB absolute (none in this suite: true; the key keeps the record's name); 7 no full-level
            pulse more than 3 dB over. Criteria 1, 2, 3 and 7 are over-test guards on the worst point, as the record
            has them; 4 and 5 were "within 3 dB at every point" in draft 0.1
  ends      the last full-level pulse inside Method 516.8's SRS tolerance on every control: -1.5/+3 dB on at least 90 %
            of its points and every point within -3/+6 dB (margin: shock_model.excess_516, negative inside)
  ends_pm3  secondary, draft 0.1's ends reading: every point of every control within -3/+3 dB
  level     the acceptance proposal's Level reading on the last full-level pulse: the mean SRS error over controls
            and points within 1.5 dB, and no point over the +6 dB limit line
  exposure  the largest absolute exposure of the run, any pulse or probe, against the full-level target: within
            +3 dB (an over-test of the article whatever the level it was meant for)
  card      the report card on the last full-level pulse: level (mean dB), pattern (RMS dB; grade A to 3, B to 4.5,
            C to 6), the worst point and the share of points over +6 dB, the twelfth-octave over
  classical (half-sine, sawtooth) on the last full-level pulse, the share of controls inside +/-20 % of the peak over
            the pulse window, with the duration within +/-10 %, the velocity change within +/-20 %, the compensation
            inside +/-20 % of the peak, the sawtooth's post-pulse slope under 10 % of TD, and all of them (516.8)
  duration  (SRS target) the share of controls whose effective duration is within 0.8 to 1.2 of the target's, on the
            last full-level pulse
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shock_model as sm  # noqa: E402

SCORER_VERSION = 'shock-0.2-draft'
TOL = 3.0
LEVEL_DB = 1.5
LIMIT_DB = 6.0
GRADES = (('A', 3.0), ('B', 4.5), ('C', 6.0))


def grade(v):
    for g, t in GRADES:
        if v <= t:
            return g
    return 'D'


def score_run(run):
    R = run['records']
    probes = [r for r in R if r['kind'] == 'probe']
    ladder = [r for r in R if r['kind'] == 'pulse']
    full = [r for r in ladder if abs(r['level_db']) < 1e-9]
    first = ladder[0] if ladder else None
    nan = float('nan')
    f1 = full[0]['over_db'] if full else nan
    x516 = lambda r: float(sm.excess_516(r['errors_db']).max())
    f3 = x516(full[2]) if len(full) >= 3 else nan
    last = x516(R[-1]) if R and run['end'] == 'completed' else nan
    below = [r for r in R if r['level_db'] < 0]
    c1v = max([r['over_db'] for r in probes] + ([first['over_db']] if first else [np.inf]))
    c2v = max([r['abs_db'] for r in below], default=-np.inf)
    c7v = max([r['over_db'] for r in full], default=-np.inf)
    crit = {
        '1_probes_and_first_within_3_over': (c1v, c1v <= TOL),
        '2_no_pulse_below_full_over_3_abs': (c2v, c2v <= TOL),
        '3_f1_within_3_over': (f1, bool(full) and f1 <= TOL),
        '4_f3_within_516': (f3, len(full) >= 3 and f3 <= 0.0),
        '5_completed_last_within_516': (last, run['end'] == 'completed' and last <= 0.0),
        '6_no_shot_over_3_abs': (-np.inf, True),
        '7_no_full_level_pulse_over_3': (c7v, c7v <= TOL),
    }
    lim = lambda k: 0.0 if '516' in k else TOL
    broke = {k: (None if not np.isfinite(v) else round(float(v - lim(k)), 3)) for k, (v, ok) in crit.items() if not ok}
    out = dict(scorer=SCORER_VERSION, problem=run['problem'], kind=run['kind'], law=run['law'], end=run['end'],
               pulses=len(R), full_level_pulses=len(full))
    out['strict'] = dict(passed=not broke, broke_db_over_limit=broke,
                         criteria={k: bool(ok) for k, (v, ok) in crit.items()})
    lf = full[-1] if full else None
    if lf is None:
        out['ends'] = dict(passed=False, why='no full-level pulse')
        out['ends_pm3'] = dict(passed=False, why='no full-level pulse')
        out['level'] = dict(passed=False, why='no full-level pulse')
        out['card'] = None
    else:
        e = np.asarray(lf['errors_db'])
        xc = sm.excess_516(e)
        out['ends'] = dict(passed=bool(np.all(xc <= 0.0)), excess_db=round(float(xc.max()), 3),
                           margin_db=round(-float(xc.max()), 3), controls_out=int(np.sum(xc > 0.0)))
        out['ends_pm3'] = dict(passed=bool(np.all(np.abs(e) <= TOL)), worst_db=float(np.abs(e).max()),
                               margin_db=round(TOL - float(np.abs(e).max()), 3),
                               controls_out=int(np.sum(np.any(np.abs(e) > TOL, axis=1))))
        mean = float(np.mean(e))
        out['level'] = dict(passed=bool(abs(mean) <= LEVEL_DB and e.max() <= LIMIT_DB), mean_db=round(mean, 3),
                            mean_margin_db=round(LEVEL_DB - abs(mean), 3), over_limit_db=round(float(e.max() - LIMIT_DB), 3),
                            broke=[w for w, b in (('mean', abs(mean) > LEVEL_DB), ('limit line', e.max() > LIMIT_DB)) if b])
        rms = float(np.sqrt(np.mean(e ** 2)))
        out['card'] = dict(level_db=round(mean, 3), pattern_rms_db=round(rms, 3), pattern_grade=grade(rms),
                           worst_over_db=round(float(e.max()), 3), worst_under_db=round(float(e.min()), 3),
                           share_over_6db=round(float(np.mean(e > LIMIT_DB)), 4), over12_db=round(lf['over12_db'], 3),
                           drive_rms_v=lf['drive_rms'], state=lf.get('state'))
        if lf.get('classical'):
            c = lf['classical']
            b, v = np.asarray(c['band_pct']), np.asarray(c['velocity_pct'])
            du, cp = np.asarray(c['duration_pct']), np.asarray(c['compensation_pct'])
            ps = np.asarray(c['post_slope_pct']) if 'post_slope_pct' in c else None
            ok = [b <= 20.0, np.abs(du) <= 10.0, np.abs(v) <= 20.0, cp <= 20.0] + ([ps < 10.0] if ps is not None else [])
            out['classical'] = dict(share_in_band_20pct=round(float(np.mean(ok[0])), 4),
                                    share_duration_10pct=round(float(np.mean(ok[1])), 4),
                                    share_velocity_20pct=round(float(np.mean(ok[2])), 4),
                                    share_compensation_20pct=round(float(np.mean(ok[3])), 4),
                                    share_post_slope_10pct=round(float(np.mean(ok[4])), 4) if ps is not None else None,
                                    share_all_516=round(float(np.mean(np.all(ok, axis=0))), 4),
                                    worst_band_pct=round(float(b.max()), 2), worst_velocity_pct=round(float(np.abs(v).max()), 2),
                                    worst_duration_pct=round(float(np.abs(du).max()), 2),
                                    worst_compensation_pct=round(float(cp.max()), 2))
        if lf.get('te_ratio'):
            te = np.asarray(lf['te_ratio'])
            out['duration'] = dict(share_te_0p8_1p2=round(float(np.mean((te >= 0.8) & (te <= 1.2))), 4),
                                   te_ratio_min=round(float(te.min()), 3), te_ratio_max=round(float(te.max()), 3))
    ex = max([r['abs_db'] for r in R], default=nan)
    out['exposure'] = dict(passed=bool(R) and ex <= TOL, largest_abs_db=round(ex, 3) if R else None,
                           margin_db=round(TOL - ex, 3) if R else None,
                           where=(lambda r: dict(index=r['index'], kind=r['kind'], level_db=r['level_db']))(max(R, key=lambda r: r['abs_db'])) if R else None)
    out['first_pulse_over_db'] = round(first['over_db'], 3) if first else None
    return out
