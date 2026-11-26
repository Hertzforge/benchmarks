"""Tests of the MIMO Sine suite (draft 0.1): python3 -m unittest discover -s sine/tests -v

Needs the MIMO Random v0.2 problems: set MSNB_RANDOM_PROBLEMS to the folder holding them (the tests that need them
are skipped otherwise). Every check here is shown to fail on a case built to break it."""
import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SINE = os.path.dirname(HERE)
sys.path.insert(0, SINE)
sys.path.insert(0, HERE)
import sine_model as sm  # noqa: E402
import make_sine_problem as mk  # noqa: E402
import sine_scorer as sc  # noqa: E402
import publication_check as pub  # noqa: E402
import case_schema  # noqa: E402
import batch  # noqa: E402
import plan  # noqa: E402
import synth  # noqa: E402

RANDOM = os.environ.get('MSNB_RANDOM_PROBLEMS')
NEEDS = unittest.skipUnless(RANDOM and os.path.isdir(RANDOM or ''), 'MSNB_RANDOM_PROBLEMS is not set')
SID = 'frame-6x6-realistic_s00'
LIN = 'frame-6x6_s00'
TMP = None


def setUpModule():
    global TMP
    if not (RANDOM and os.path.isdir(RANDOM)):
        return
    TMP = tempfile.mkdtemp(prefix='msnb-b-test-')
    os.makedirs(os.path.join(TMP, 'problems'))
    with np.errstate(all='ignore'):
        _, meta = mk.make(RANDOM, os.path.join(TMP, 'problems'), SID, None, None, True, check_play=False)
        mk.make(RANDOM, os.path.join(TMP, 'problems'), LIN, (meta['full_level']['spec_amplitude_mps2_pk'], SID), None,
                False, check_play=False)


def tearDownModule():
    if TMP:
        shutil.rmtree(TMP, ignore_errors=True)


def P(sid=SID):
    return os.path.join(TMP, 'problems', sid)



# ------------------------------------------------------------------------------------------------ the plan
class TestPlan(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(plan.block_a()), 300)
        self.assertEqual(len(plan.block_d()), 80)
        self.assertEqual(len(plan.limit_structures()), 40)
        self.assertEqual(len(set(plan.structures())), 380)
        self.assertEqual(len(plan.jobs()[0]), 300 * 11 + 80 * 10 + 40)
        # the Python version: K3 and matrix on T1, T2 (380 structures), T5 (block A, 300) and L1 (the limit block, 40)
        py = plan.jobs(version='python')[0]
        self.assertEqual(len(py), 2 * (380 + 380 + 300 + 40))
        self.assertEqual({(j['problem'], j['test']) for j in py if j['law'] == 'k3'},
                         {(j['problem'], j['test']) for j in py if j['law'] == 'matrix'})
        self.assertFalse({j['test'] for j in py} & {'T3', 'T4'})
        hf = plan.jobs()[0]
        self.assertNotIn('k3', {j['law'] for j in hf})

    def test_no_held_back_seed(self):
        seeds = {int(s.rsplit('_s', 1)[1]) for s in plan.structures()}
        self.assertTrue(seeds <= set(plan.RANDOM_STUDY_SEEDS))


# ------------------------------------------------------------------------------------------------ generation
@NEEDS
class TestGeneration(unittest.TestCase):
    def test_structure_byte_for_byte(self):
        a = open(os.path.join(RANDOM, SID, 'structure.json'), 'rb').read()
        b = open(os.path.join(P(), 'structure.json'), 'rb').read()
        self.assertEqual(a, b)
        meta = json.load(open(os.path.join(P(), 'problem.json')))
        rp = json.load(open(os.path.join(RANDOM, SID, 'problem.json')))
        self.assertEqual(meta['structure']['sha256'], rp['structure']['sha256'])

    def _copy(self, tweak):
        root = tempfile.mkdtemp(dir=TMP)
        shutil.copytree(os.path.join(RANDOM, SID), os.path.join(root, SID))
        tweak(os.path.join(root, SID))
        return root

    def test_refuses_a_changed_structure(self):
        def tweak(d):
            p = os.path.join(d, 'structure.json')
            b = bytearray(open(p, 'rb').read()); b[-2] = ord(' ') if b[-2] != ord(' ') else ord('\n')
            open(p, 'wb').write(bytes(b))
        root = self._copy(tweak)
        with self.assertRaisesRegex(mk.Refused, 'does not match'):
            mk.make(root, os.path.join(root, 'out'), SID, None, None, False, check_play=False)

    def test_refuses_a_mode_over_the_bound(self):
        def tweak(d):
            p = os.path.join(d, 'structure.json')
            s = json.load(open(p))
            s['modes'][-1]['level']['rise'] = 10.0
            json.dump(s, open(p, 'w'))
            q = os.path.join(d, 'problem.json')
            rp = json.load(open(q))
            rp['structure']['sha256'] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
            json.dump(rp, open(q, 'w'))
        root = self._copy(tweak)
        with self.assertRaisesRegex(mk.Refused, 'can reach'):
            mk.make(root, os.path.join(root, 'out'), SID, None, None, False, check_play=False)

    def test_full_level_meets_the_random_median(self):
        meta = json.load(open(os.path.join(P(), 'problem.json')))
        sz = meta['full_level']['sizing']
        self.assertEqual(sz['rule'], 'sized')
        self.assertAlmostEqual(sz['sine_median_effect'], sz['random_median_change'], delta=1e-3 * sz['random_median_change'])
        lin = json.load(open(os.path.join(P(LIN), 'problem.json')))
        self.assertEqual(lin['full_level']['spec_amplitude_mps2_pk'], meta['full_level']['spec_amplitude_mps2_pk'])

    def test_cases_carry_the_drive_gain_and_ceiling(self):
        # every case: gain = the problem's ideal full-level drive peak, so that peak is 1 V in engine volts; ceiling
        # 6 dB over it. A case without the gain (the runner's default, 1) would put the ideal drive at its own peak
        for sid in (SID, LIN):
            meta = json.load(open(os.path.join(P(sid), 'problem.json')))
            dg = np.fromfile(os.path.join(P(sid), meta['files']['ideal_drive']['file']), dtype='<c16')
            for f in os.listdir(os.path.join(P(sid), 'cases')):
                d = json.load(open(os.path.join(P(sid), 'cases', f)))['drives']
                self.assertAlmostEqual(np.abs(dg).max() / d['gain'], 1.0, places=7, msg=f)
                self.assertAlmostEqual(20 * np.log10(d['ceiling_vpk']), sm.DRIVE_HEADROOM_DB, places=6, msg=f)
        bad = json.load(open(os.path.join(P(), 'cases', 'T1__matrix.json')))
        bad['drives']['gain'] = 0.0
        self.assertNotEqual(case_schema.check(bad), [])

    def test_cases_meet_the_runner_schema(self):
        for sid in (SID, LIN):
            for f in os.listdir(os.path.join(P(sid), 'cases')):
                self.assertEqual(case_schema.check(json.load(open(os.path.join(P(sid), 'cases', f)))), [], f)
        names = sorted(os.listdir(os.path.join(P(), 'cases')))
        self.assertEqual(len(names), 12)                      # block A and the limit block: 6 + 2 + 2 + 1 + 1
        self.assertIn('L1__matrix.json', names)

    def test_schema_check_fails_bad_cases(self):
        good = json.load(open(os.path.join(P(), 'cases', 'T1__matrix.json')))
        for bad in (dict(good, extra=1), dict(good, structure=dict(file='/abs/structure.json')),
                    dict(good, test=dict(type='sweep', band_hz=[20, 2000])), dict(good, law=dict(id='other')),
                    dict(good, structure=dict(file='../x/structure.json'))):
            self.assertNotEqual(case_schema.check(bad), [])

    def test_reference_file_matches_the_arrays(self):
        cr = json.load(open(os.path.join(P(), 'case_reference.json')))
        pr = sc.Problem(P())
        a = np.asarray(cr['amplitude_m_s2_pk'])
        self.assertEqual(a.shape, pr.amp.shape)
        self.assertLess(np.max(np.abs(a / pr.amp - 1)), 1e-8)
        ph = np.asarray(cr['phase_deg'])
        self.assertTrue(np.all(ph > -180) and np.all(ph <= 180))
        self.assertLess(np.max(np.abs(np.angle(np.exp(1j * np.radians(ph - pr.phase))))), 1e-6)

    def test_ideal_files_match_the_model(self):
        pr = sc.Problem(P())
        k = np.arange(0, pr.f_grid.size, 53)
        d, y = pr.ideal_at(pr.f_grid[k])
        self.assertLess(np.max(np.abs(d - pr.d_ideal[k])), 1e-9 * np.abs(pr.d_ideal).max())
        self.assertLess(np.max(np.abs(y - pr.y_ideal_h[k])), 1e-9 * np.abs(pr.y_ideal_h).max())

    def test_reference_is_the_projection(self):
        pr = sc.Problem(P())
        f = pr.f_grid[::97]
        _, R, _ = sm.ideal(pr.st, f, pr.spec)
        self.assertLess(np.max(np.abs(pr.reference_at(f) - R)) / pr.spec, 1e-9)

    def test_t5_built_on_block_a_only(self):
        t5 = json.load(open(os.path.join(P(), 'test_T5.json')))
        self.assertTrue(t5['built'])
        root = tempfile.mkdtemp(dir=TMP)
        mk.make(RANDOM, root, 'frame-6x6-realistic_soft_s00', (1.0, SID), 'T5 is built on block A only', False, check_play=False)
        t5 = json.load(open(os.path.join(root, 'frame-6x6-realistic_soft_s00', 'test_T5.json')))
        self.assertFalse(t5['built'])
        self.assertNotIn('T5__matrix.json', os.listdir(os.path.join(root, 'frame-6x6-realistic_soft_s00', 'cases')))

    def test_step_and_dwell_rules(self):
        pr = sc.Problem(P())
        st = pr.st
        t3, t4 = pr.tests['T3'], pr.tests['T4']
        inb = (st.f_hz >= 20) & (st.f_hz <= 2000)
        tau = np.max(1 / (2 * np.pi * st.zeta[inb] * st.f_hz[inb]))
        self.assertAlmostEqual(t3['step_s'], max(1.0, 10 * tau), places=9)
        m = t4['resonance']['mode']
        n = max(np.ceil(30 / (2 * np.pi * st.zeta[m])), np.ceil(10 * t4['dwell_hz']))
        self.assertAlmostEqual(t4['seconds'], n / t4['dwell_hz'], places=9)


# ------------------------------------------------------------------------------------------------ the scorer
@NEEDS
class TestScorer(unittest.TestCase):
    def score(self, test, sid=SID, **kw):
        d = tempfile.mkdtemp(dir=TMP)
        r = synth.make_run(P(sid), test, d, **kw)
        with np.errstate(all='ignore'):
            return sc.score(P(sid), d)

    def test_pass_on_a_perfect_run(self):
        for t in ('T1', 'T3', 'T4', 'T5'):
            c = self.score(t)
            self.assertTrue(c['scored'])
            for k in sm.TOL:
                self.assertGreater(c['amplitude'][k]['share_inside'], 0.999, (t, k))
                self.assertLess(c['amplitude'][k]['worst_excursion_db'], 0, (t, k))
            self.assertLess(c['phase']['p95_abs_deg'], 1e-6)
            self.assertLess(abs(c['resonances']['worst_db']), 1e-6)
            self.assertTrue(c['practice']['line_1_level']['within'])
            self.assertEqual(c['practice']['line_2_pattern']['controls']['grade'], 'A')
            self.assertFalse(c['practice']['line_3_over_test']['alarm'])
            self.assertEqual(c['completion']['outcome'], 'completed')
            self.assertGreater(c['completion']['coverage'], 0.99)
            self.assertLess(abs(c['drive']['effort_db_max']), 1e-6)
            self.assertLess(np.max(np.abs(c['stress']['error_db_p10_p50_p90'])), 1e-6)

    def test_amplitude_fails_out_of_band(self):
        c = self.score('T1', gain_db=lambda f: 4.0)              # +58 %: outside every band read
        for k in sm.TOL:
            self.assertLess(c['amplitude'][k]['share_inside'], 0.01, k)
        c = self.score('T1', gain_db=lambda f: 4.0 if 600 < f < 700 else 0.0)
        a = c['amplitude']['iec_check_point']
        self.assertLess(a['share_inside'], 0.99)
        self.assertTrue(600 < a['at_hz'] < 700)

    def test_reads_a_whole_bin_row_given_as_null(self):
        # the runner writes a bin with no judged update as null for the whole row, not as a row of nulls: both forms
        # must read the same
        d = tempfile.mkdtemp(dir=TMP)
        synth.make_run(P(), 'T1', d)
        r = json.load(open(os.path.join(d, 'run.json')))
        h = r['history']
        n = len(h['bin_error_db_min'][0])
        for k in range(5):
            h['bin_error_db_min'][k] = [None] * n
            h['bin_error_db_max'][k] = [None] * n
        json.dump(r, open(os.path.join(d, 'run.json'), 'w'))
        with np.errstate(all='ignore'):
            want = sc.score(P(), d)
        for k in range(5):
            h['bin_error_db_min'][k] = None
            h['bin_error_db_max'][k] = None
        json.dump(r, open(os.path.join(d, 'run.json'), 'w'))
        with np.errstate(all='ignore'):
            got = sc.score(P(), d)
        self.assertEqual(got['amplitude'], want['amplitude'])

    def test_amplitude_finds_a_bin_extreme_between_points(self):
        c = self.score('T1', bin_bump_db=(1234.0, 5.0))
        a = c['amplitude']['iec_check_point']
        self.assertGreater(a['worst_excursion_db'], 0)
        self.assertAlmostEqual(a['at_hz'], 1234.0, delta=1234.0 * 0.02)

    def test_phase_is_reported_not_judged(self):
        c = self.score('T4', phase_deg=30.0)
        self.assertFalse(c['phase']['judged'])
        self.assertAlmostEqual(c['phase']['p95_abs_deg'], 30.0, places=6)
        self.assertEqual(c['phase']['share_inside_10_deg_analogy'], 0.0)

    def test_over_test_through_a_resonance(self):
        st = sc.Problem(P()).st
        m = int(np.flatnonzero(sm.article_modes(st))[5])
        fm = st.f_hz[m]
        c = self.score('T1', gain_db=lambda f: 8.0 if abs(np.log2(f / fm)) < 1 / 80 else 0.0)
        r = c['resonances']
        self.assertAlmostEqual(r['worst_db'], 8.0, places=6)
        self.assertEqual(r['worst_mode'] in (m,) or abs(np.log2(st.f_hz[r['worst_mode']] / fm)) < 0.05, True)

    def test_practice_lines_fail(self):
        c = self.score('T1', gain_db=lambda f: 2.0, drive_scale=10 ** (2 / 20))     # every channel 2 dB high
        self.assertFalse(c['practice']['line_1_level']['within'])
        c = self.score('T1', gain_db=lambda f: 1.0, drive_scale=10 ** (1 / 20))
        self.assertTrue(c['practice']['line_1_level']['within'])
        c = self.score('T1', gain_db=lambda f: 7.0 if 250 < f < 500 else 0.0)     # two whole 1/3-octave bands
        self.assertTrue(c['practice']['line_3_over_test']['alarm'])
        c = self.score('T1', gain_db=lambda f: 5.0)
        self.assertEqual(c['practice']['line_2_pattern']['controls']['grade'], 'C')

    def test_held_out_and_stress_follow_the_drive(self):
        c = self.score('T4', drive_scale=2.0)
        self.assertAlmostEqual(c['practice']['line_2_pattern']['held_out']['summed_rms_db'], 20 * np.log10(2), places=6)
        self.assertAlmostEqual(c['stress']['error_db_p10_p50_p90'][1], 20 * np.log10(2), places=6)
        self.assertAlmostEqual(c['drive']['effort_db_p50'], 20 * np.log10(2), places=6)

    def test_the_drive_gain_turns_engine_volts_into_the_structures_input(self):
        # a run file in engine volts with its gain reads as the same drive given in the structure's volts; read without
        # the gain (the file's settings removed) the held-out gauges, the stress and the effort fall by the gain
        same = self.score('T4', drive_scale=2.0, drive_gain=5.0)
        ref = self.score('T4', drive_scale=2.0)
        for path in (('practice', 'line_2_pattern', 'held_out', 'summed_rms_db'), ('drive', 'effort_db_p50'),
                     ('stress', 'error_db_p10_p50_p90', 1)):
            a, b = same, ref
            for k in path:
                a, b = a[k], b[k]
            self.assertAlmostEqual(a, b, places=6, msg=path)
        self.assertAlmostEqual(same['drive']['largest_drive_v'] * 5.0, ref['drive']['largest_drive_v'], places=6)
        d = tempfile.mkdtemp(dir=TMP)
        synth.make_run(P(), 'T4', d, drive_scale=2.0, drive_gain=5.0)
        r = json.load(open(os.path.join(d, 'run.json')))
        del r['settings']
        json.dump(r, open(os.path.join(d, 'run.json'), 'w'))
        with np.errstate(all='ignore'):
            wrong = sc.score(P(), d)
        self.assertAlmostEqual(ref['drive']['effort_db_p50'] - wrong['drive']['effort_db_p50'], 20 * np.log10(5.0), places=6)

    def test_completion_reads_an_early_end(self):
        c = self.score('T1', stop_fraction=0.5, outcome_class='stopped_by_protective_default')
        self.assertEqual(c['completion']['outcome'], 'stopped_by_protective_default')
        self.assertEqual(c['completion']['guard'], 'control_abort')
        self.assertLess(c['completion']['coverage'], 0.51)
        c = self.score('T3', stop_fraction=0.25, outcome_class='aborted')
        self.assertLess(c['completion']['coverage'], 0.26)

    def test_coverage_counts_from_where_full_level_began(self):
        # the engine runs the lower rungs on the test's own clock: a whole run covers what follows them
        for t in ('T1', 'T3'):
            c = self.score(t, ladder_in_test=True)
            self.assertGreater(c['completion']['coverage'], 0.99, t)
            self.assertGreater(c['completion']['ladder_share'], 0.0, t)
        # and an early end still reads as one
        c = self.score('T1', ladder_in_test=True, stop_fraction=0.5, outcome_class='stopped_by_protective_default')
        self.assertLess(c['completion']['coverage'], 0.6)

    def test_nothing_judged_is_not_scored(self):
        c = self.score('T1', all_unjudged=True)
        self.assertFalse(c['scored'])

    def test_frequency_tolerances_fail(self):
        c = self.score('T1', rate_factor=1.3)
        self.assertEqual(c['frequency']['share_of_sweep_within_rate_tolerance'], 0.0)
        self.assertFalse(c['frequency']['duration_within_5_pct'])
        c = self.score('T4', dwell_offset=0.03)
        self.assertFalse(c['frequency']['within_iec_fixed_2_pct'])
        c = self.score('T4', dwell_offset=0.005)
        self.assertTrue(c['frequency']['within_iec_fixed_2_pct'])
        self.assertFalse(c['frequency']['within_mil_0_1_pct'])

    def test_state_as_tested(self):
        c = self.score('T1', mode_shift=-0.02)
        self.assertAlmostEqual(c['state']['as_tested']['median_abs_change'], 0.02, places=9)
        c = self.score('T1', sid=LIN)
        self.assertIsNone(c['state'])

    def test_limit_reading(self):
        c = self.score('L1', limit_over_db=2.0)
        self.assertAlmostEqual(c['limit']['worst_over_bound_db'], 2.0, places=6)
        self.assertGreater(c['limit']['share_over_bound'], 0.99)
        c = self.score('L1', limit_over_db=-1.0)
        self.assertLess(c['limit']['worst_over_bound_db'], 0)

    def test_zeta_from_frequency_inverts_the_rule(self):
        st = sc.Problem(P()).st
        m = int(np.flatnonzero(sm.article_modes(st))[3])
        for r in (0.1, 0.5, 2.0):
            f, z = sm.mode_state(st, m, np.array([r]))
            self.assertAlmostEqual(sm.zeta_from_frequency(st, m, f)[0] / z[0], 1.0, delta=0.02)


# ------------------------------------------------------------------------------------------------ the batch
@NEEDS
class TestBatch(unittest.TestCase):
    def run_batch(self, name, **kw):
        out = tempfile.mkdtemp(dir=TMP)
        argv = ['--problems', os.path.join(TMP, 'problems'), '--out', out, '--name', name, '--runner',
                os.path.join(HERE, 'fake_runner.py'), '--only', LIN, '--laws', 'matrix', '--workers', '2']
        for k, v in kw.items():
            argv += ['--' + k.replace('_', '-'), str(v)]
        with contextlib.redirect_stdout(io.StringIO()):
            batch.main(argv)
        return out, open(os.path.join(out, name, 'batch.log')).read().splitlines()

    def test_end_line_and_code_hash(self):
        os.environ['FAKE_REFUSE'] = f'{LIN}__T2__matrix'
        os.environ['FAKE_CRASH'] = f'{LIN}__T3__matrix'
        try:
            out, log = self.run_batch('b1')
        finally:
            del os.environ['FAKE_REFUSE'], os.environ['FAKE_CRASH']
        self.assertTrue(log[-1].endswith('BATCH END b1: 3 done, 2 failed, 0 skipped'), log[-1])
        card = json.load(open(os.path.join(out, 'b1', LIN, 'T1__matrix', 'card.json')))
        self.assertEqual(card['batch']['code_sha256'], batch.code_hash())
        info = json.load(open(os.path.join(out, 'b1', LIN, 'T1__matrix', 'batch.json')))
        self.assertEqual(info['code_sha256'], batch.code_hash())
        for line in log:                                  # no outcome in the log lines
            self.assertNotIn('completed', line)
            self.assertNotIn('abort', line)
        # continued: the scored jobs are skipped, the failed ones run again
        argv = ['--problems', os.path.join(TMP, 'problems'), '--out', out, '--name', 'b1', '--runner',
                os.path.join(HERE, 'fake_runner.py'), '--only', LIN, '--laws', 'matrix', '--workers', '2']
        with contextlib.redirect_stdout(io.StringIO()):
            batch.main(argv)
        log = open(os.path.join(out, 'b1', 'batch.log')).read().splitlines()
        self.assertTrue(log[-1].endswith('BATCH END b1: 2 done, 0 failed, 3 skipped'), log[-1])

    def test_a_run_s_processor_time_and_memory_are_recorded(self):
        os.environ['FAKE_SPIN'] = f'{LIN}__T1__matrix'
        try:
            out, log = self.run_batch('b_use', tests='T1,T2')
        finally:
            del os.environ['FAKE_SPIN']
        spun = json.load(open(os.path.join(out, 'b_use', LIN, 'T1__matrix', 'batch.json')))
        plain = json.load(open(os.path.join(out, 'b_use', LIN, 'T2__matrix', 'batch.json')))
        self.assertGreaterEqual(spun['user_s'] + spun['sys_s'], 0.5)
        self.assertLess(plain['user_s'] + plain['sys_s'], spun['user_s'] + spun['sys_s'] - 0.3)
        self.assertGreater(spun['max_rss_bytes'], 5e6)
        self.assertLess(spun['max_rss_bytes'], 5e9)

    def test_a_run_past_its_time_is_stopped(self):
        os.environ['FAKE_HANG'] = f'{LIN}__T1__matrix'
        t0 = time.time()
        try:
            out, log = self.run_batch('b_hang', tests='T1', job_timeout=2)
        finally:
            del os.environ['FAKE_HANG']
        self.assertLess(time.time() - t0, 30)
        info = json.load(open(os.path.join(out, 'b_hang', LIN, 'T1__matrix', 'batch.json')))
        self.assertEqual(info['failed'], 'timed out after 2 s')
        self.assertIsNone(info['runner_exit'])
        self.assertTrue(log[-1].endswith('BATCH END b_hang: 0 done, 1 failed, 0 skipped'), log[-1])

    def test_case_set_and_job_list(self):
        out = tempfile.mkdtemp(dir=TMP)
        cases = os.path.join(out, 'record')
        os.makedirs(os.path.join(cases, LIN))
        c = json.load(open(os.path.join(TMP, 'problems', LIN, 'cases', 'T1__matrix.json')))
        c['protective'] = 'widened'
        json.dump(c, open(os.path.join(cases, LIN, 'T1__matrix.json'), 'w'))
        jobs = os.path.join(out, 'jobs.json')
        json.dump([dict(problem=LIN, test='T1', law='matrix'), dict(problem=LIN, test='T4', law='matrix')], open(jobs, 'w'))
        argv = ['--problems', os.path.join(TMP, 'problems'), '--out', out, '--name', 'b3', '--runner',
                os.path.join(HERE, 'fake_runner.py'), '--workers', '2', '--cases', cases, '--jobs', jobs]
        with contextlib.redirect_stdout(io.StringIO()):
            batch.main(argv)
        log = open(os.path.join(out, 'b3', 'batch.log')).read().splitlines()
        # two jobs listed: T1 from the set, T4 has no case file in the set and fails (never the problem's own case)
        self.assertTrue(log[-1].endswith('BATCH END b3: 1 done, 1 failed, 0 skipped'), log[-1])
        self.assertTrue(any('failed frame-6x6_s00__T4__matrix: no case file in the set' in x for x in log))
        info = json.load(open(os.path.join(out, 'b3', LIN, 'T1__matrix', 'batch.json')))
        self.assertEqual(info['case_sha256'], batch.sha256(os.path.join(cases, LIN, 'T1__matrix.json')))
        self.assertNotEqual(info['case_sha256'], batch.sha256(os.path.join(TMP, 'problems', LIN, 'cases', 'T1__matrix.json')))
        self.assertEqual(info['case_file'], f'{os.path.basename(out)}/record/{LIN}/T1__matrix.json')

    def test_free_disk_stop(self):
        out, log = self.run_batch('b2', min_free_gb=1e12)
        self.assertTrue(any('BATCH STOP b2: free disk' in x for x in log))
        self.assertTrue(log[-1].endswith('BATCH END b2: 0 done, 0 failed, 5 skipped'), log[-1])

    def test_free_disk_is_read_on_root_too(self):
        real = batch.shutil.disk_usage

        def fake(path):
            u = real(path)
            return u._replace(free=10 * 10 ** 9) if os.path.abspath(path) == '/' else u._replace(free=10 ** 15)
        batch.shutil.disk_usage = fake
        try:
            self.assertAlmostEqual(batch.free_gb(TMP), 10.0)
            out, log = self.run_batch('b4', min_free_gb=50)
        finally:
            batch.shutil.disk_usage = real
        self.assertTrue(any('BATCH STOP b4: free disk 10.0 GB under 50 GB' in x for x in log), log)

    def test_workers_setting(self):
        self.assertEqual((batch.DEFAULT_WORKERS, batch.MAX_WORKERS, batch.BIG_SYSTEMS, batch.BIG_AT_ONCE),
                         (16, 18, ('24x12',), 4))
        out, log = self.run_batch('b5', workers=40)
        self.assertIn('BATCH START b5: 5 jobs, 18 workers, at most 4 on 24x12 at once', log[0])

    def test_interleave(self):
        jobs = [dict(problem=f'frame-12x6_s{i:02d}', test='T1', law='matrix') for i in range(80)]
        jobs += [dict(problem=f'frame-24x12_s{i:02d}', test='T1', law='matrix') for i in range(20)]
        out = batch.interleave(jobs, lambda j: batch.system_of(j['problem']) == '24x12')
        self.assertEqual(sorted(map(str, out)), sorted(map(str, jobs)))
        pos = [i for i, j in enumerate(out) if '24x12' in j['problem']]
        self.assertLessEqual(max(np.diff(pos)), 6)                         # one in five, evenly
        self.assertGreaterEqual(min(np.diff(pos)), 4)
        self.assertLessEqual(pos[0], 5)
        self.assertGreaterEqual(pos[-1], 90)
        self.assertEqual([j for j in out if '24x12' in j['problem']], jobs[80:])      # each group keeps its order
        self.assertEqual([j for j in out if '24x12' not in j['problem']], jobs[:80])

    def test_at_most_four_big_at_once(self):
        import threading
        import time as _t
        a = argparse.Namespace(out=tempfile.mkdtemp(dir=TMP), name='b6', runner=os.path.join(HERE, 'fake_runner.py'),
                               workers=10, big_systems='24x12', big_at_once=4)
        b = batch.Batch(a)
        lock, now, most, ran = threading.Lock(), [0], [0], []

        def job(j):
            big = '24x12' in j['problem']
            with lock:
                ran.append(j['problem'])
                if big:
                    now[0] += 1
                    most[0] = max(most[0], now[0])
            _t.sleep(0.05 if big else 0.005)
            if big:
                with lock:
                    now[0] -= 1
        b.job = job
        jobs = [dict(problem=f'frame-24x12_s{i:02d}', test='T1', law='matrix') for i in range(12)]
        jobs += [dict(problem=f'frame-6x6_s{i:02d}', test='T1', law='matrix') for i in range(30)]
        b.run(jobs)
        self.assertEqual(b.order, batch.interleave(jobs, b.is_big))       # run() spreads the big jobs
        self.assertNotEqual(b.order, jobs)
        self.assertEqual(sorted(ran), sorted(j['problem'] for j in jobs))
        self.assertEqual(most[0], 4)
        self.assertEqual(b.most_big_at_once, 4)


# ------------------------------------------------------------------------------------------------ publication
class TestPublication(unittest.TestCase):
    def test_finds_each_kind(self):
        d = tempfile.mkdtemp()
        try:
            word = 'Pl' + 'ant'                          # spelt in pieces so this file passes the check itself
            open(os.path.join(d, 'a.json'), 'w').write('{"what": "the %s under test"}\n' % word)
            open(os.path.join(d, 'b.json'), 'w').write(json.dumps({'p': os.path.expanduser('~') + '/x'}) + '\n')
            open(os.path.join(d, 'c.f8'), 'wb').write(np.zeros(3).tobytes() + ('/' + 'Users/someone/x').encode())
            open(os.path.join(d, 'd.md'), 'w').write('~' + '/dev/thing\n')
            open(os.path.join(d, 'ok.json'), 'w').write('{"what": "the system under test, planted nothing"}\n')
            n, found = pub.run([d])
            self.assertEqual(n, 5)
            kinds = sorted(os.path.basename(f.split(': ')[0]) for f in found)
            for k in ('a.json', 'b.json', 'c.f8', 'd.md'):
                self.assertIn(k, kinds)
            self.assertNotIn('ok.json', kinds)
        finally:
            shutil.rmtree(d)

    def test_a_drive_prefix_in_an_array_counts_only_as_a_path(self):
        # three bytes 'C:' and a backslash turn up by chance in floating-point arrays; a path written into one is found
        d = tempfile.mkdtemp()
        try:
            drive = 'C' + ':' + chr(92)
            open(os.path.join(d, 'chance.f8'), 'wb').write(np.zeros(3).tobytes() + drive.encode() + bytes([0, 200, 7, 1]) + np.ones(3).tobytes())
            open(os.path.join(d, 'path.f8'), 'wb').write(np.zeros(3).tobytes() + (drive + 'Users' + chr(92) + 'someone').encode())
            n, found = pub.run([d])
            kinds = sorted(os.path.basename(f.split(': ')[0]) for f in found)
            self.assertEqual(kinds, ['path.f8'])
        finally:
            shutil.rmtree(d)

    def test_listed_name_by_hash(self):
        names = frozenset({pub.phrase_hash('Example Widget Works')})
        self.assertTrue(pub.check_text('x', 'made by example widget works ltd', names))
        self.assertFalse(pub.check_text('x', 'made by example works', names))
        self.assertEqual(pub.phrase_hash('a  B-c'), pub.phrase_hash('a b c'))

    def test_suite_code_is_clean(self):
        n, found = pub.run([SINE])
        self.assertEqual(found, [])



# ------------------------------------------------------------------------------------------------ the launchd wrapper
class TestBatchAgent(unittest.TestCase):
    def setUp(self):
        import batch_agent
        self.ba = batch_agent
        self.d = tempfile.mkdtemp(prefix='msnb-agent-test-')
        self.addCleanup(shutil.rmtree, self.d, True)
        self.sets = os.path.join(self.d, 'sets.json')
        json.dump(dict(sets=[dict(name='study_x', problems='/p', cases='/c', jobs_file='/j.json')]), open(self.sets, 'w'))
        self.runner = os.path.join(self.d, 'runner')
        open(self.runner, 'w').write('')

    def args(self, *extra):
        return ['--sets', self.sets, '--set', 'study_x', '--runner', self.runner, '--out', self.d] + list(extra)

    def test_the_plist(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.ba.main(['plist'] + self.args())
        import plistlib
        pl = plistlib.load(open(os.path.join(self.d, 'study_x', 'hf.msnb.study_x.plist'), 'rb'))
        self.assertEqual(pl['ProcessType'], 'Interactive')
        prog = pl['ProgramArguments']
        self.assertEqual(prog[prog.index('--workers') + 1], '16')
        self.assertEqual(prog[prog.index('--min-free-gb') + 1], '50')
        self.assertEqual(prog[prog.index('--jobs') + 1], '/j.json')
        for k in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            self.assertEqual(pl['EnvironmentVariables'][k], '1')

    def test_a_network_folder_is_refused(self):
        orig = self.ba.fs_type
        self.ba.fs_type = lambda path: 'smbfs' if os.path.realpath(path).startswith(os.path.realpath(self.d)) else 'apfs'
        try:
            with self.assertRaises(SystemExit) as e:
                self.ba.cmd_start(self.ba.argparse.Namespace(sets=self.sets, set='study_x', runner=self.runner, out=self.d,
                                                            workers=16, jobs=None, label=None))
            self.assertIn('network file system', str(e.exception))
        finally:
            self.ba.fs_type = orig

    def test_under_fifty_gb_free_is_refused(self):
        orig_fs, orig_du = self.ba.fs_type, self.ba.shutil.disk_usage
        self.ba.fs_type = lambda path: 'apfs'
        self.ba.shutil.disk_usage = lambda p: orig_du(p)._replace(free=int(49.9e9))
        try:
            with self.assertRaises(SystemExit) as e:
                self.ba.cmd_start(self.ba.argparse.Namespace(sets=self.sets, set='study_x', runner=self.runner, out=self.d,
                                                            workers=16, jobs=None, label=None))
            self.assertIn('under 50 GB', str(e.exception))
        finally:
            self.ba.fs_type, self.ba.shutil.disk_usage = orig_fs, orig_du

    def test_the_mount_of_a_path(self):
        table = ('/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)\n'
                 '/dev/disk3s5 on /srv/data (apfs, local, journaled, nobrowse)\n'
                 '//u@10.0.0.2/home on /srv/data/net/share (smbfs, nodev, nosuid, mounted by u)\n')
        self.assertEqual(self.ba.fs_type('/srv/data/net/share/a/b', table), 'smbfs')
        self.assertEqual(self.ba.fs_type('/srv/data/net/share', table), 'smbfs')
        self.assertEqual(self.ba.fs_type('/srv/data/net/shared', table), 'apfs')
        self.assertEqual(self.ba.fs_type('/usr/bin', table), 'apfs')
        self.assertEqual(self.ba.fs_type('/'), self.ba.fs_type(os.path.join('/', 'nonexistent_x', 'y')))
        self.assertNotIn(self.ba.fs_type(self.d), self.ba.NETWORK_FS)


# ------------------------------------------------------------------------------------------------ the Python harness
@NEEDS
class TestHarness(unittest.TestCase):
    def run_case(self, case_path, out):
        from pyloop import harness as hz
        with np.errstate(all='ignore'), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return hz.main([case_path, '--root', os.path.join(TMP, 'problems'), '--out', out])

    def test_a_linear_dwell_completes_and_the_scorer_reads_it(self):
        out = tempfile.mkdtemp(dir=TMP)
        code = self.run_case(os.path.join(P(LIN), 'cases', 'T4__matrix.json'), out)
        self.assertEqual(code, 0)
        with np.errstate(all='ignore'):
            c = sc.score(P(LIN), out)
        self.assertTrue(c['scored'])
        self.assertEqual(c['completion']['outcome'], 'completed')
        self.assertGreater(c['completion']['coverage'], 0.99)
        self.assertGreater(c['amplitude']['iec_check_point']['share_inside'], 0.99)

    def test_the_tracked_response_is_the_structure_times_the_drive(self):
        # at the end of the linear dwell the tracked controls are H(f) times the drive in force (noise 1e-3); the
        # conjugate convention is shown to fail the same check
        import reference_model as rm
        out = tempfile.mkdtemp(dir=TMP)
        self.run_case(os.path.join(P(LIN), 'cases', 'T4__matrix.json'), out)
        r = json.load(open(os.path.join(out, 'run.json')))
        h = r['history']
        st = rm.load_structure(P(LIN))
        names = [o['name'] for o in st.outputs]
        rows = [names.index(x) for x in r['structure']['controls']]
        C = np.array(h['control_re'][-1]) + 1j * np.array(h['control_im'][-1])
        D = np.array(h['drive_re'][-2]) + 1j * np.array(h['drive_im'][-2])
        from pyloop.fitted import FittedModal
        H = FittedModal(st, rows, sm.FS).frf([h['frequency_hz'][-1]])[0]     # the linear tier as the simulator steps it
        # the drive in the file is engine volts; the structure was played it times the case's gain
        g = r['settings']['drive_gain']
        case = json.load(open(os.path.join(P(LIN), 'cases', 'T4__matrix.json')))
        self.assertEqual(g, case['drives']['gain'])
        self.assertGreater(abs(np.log10(g)), 0.05)              # the check below can see the gain
        good = np.linalg.norm(C - g * H @ D) / np.linalg.norm(C)
        bad = np.linalg.norm(C - g * H.conj() @ D) / np.linalg.norm(C)
        no_gain = np.linalg.norm(C - H @ D) / np.linalg.norm(C)
        self.assertLess(good, 0.02)
        self.assertGreater(bad, 0.1)
        self.assertGreater(no_gain, 0.1)
        # the ideal full-level drive peak is 1 V in engine volts, so the judged drive sits near it, under the ceiling
        self.assertLess(np.abs(D).max(), case['drives']['ceiling_vpk'])

    def test_an_engine_identification_file_reads_back_in_the_harness_convention(self):
        # an engine-style run folder (FORMAT_RUN.md: law_identification.f8, the response per engine volt turned by
        # exp(j w turn / fs)) written from the harness's own exact response: read back it must equal that response;
        # the file read with the turn's sign wrong must not
        from pyloop import harness as hz
        case = hz.Case(os.path.join(P(LIN), 'cases', 'T4__matrix.json'), os.path.join(TMP, 'problems'))
        idn = hz.identify(case)
        turn = 37.0
        g, H, T = idn.grid_hz, idn.H0, idn.T0
        rot = np.exp(1j * 2 * np.pi * g * turn / case.fs)[:, None, None]
        Hf = (H * rot).reshape(g.size, -1)
        n, m = H.shape[1], H.shape[2]
        cols = [g[:, None], Hf.real, Hf.imag, np.zeros((g.size, 0)), np.zeros((g.size, 0)), idn.coh, np.zeros((g.size, 0)),
                np.ones((g.size, 1))]
        d = tempfile.mkdtemp(dir=TMP)
        a = np.ascontiguousarray(np.concatenate(cols, 1), dtype='<f8')
        a.tofile(os.path.join(d, 'law_identification.f8'))
        sha = hashlib.sha256(open(os.path.join(d, 'law_identification.f8'), 'rb').read()).hexdigest()

        def write(t):
            run = dict(structure=dict(sha256=case.raw['structure']['sha256'], fs_hz=case.fs, controls=case.raw['controls'],
                                      limits=[], drives=case.drive_names),
                       identification=dict(source='engine', ideal_vs_measured=dict(delay_turn_samples=t), lag_frames=lag,
                                           law_given=dict(file='law_identification.f8', sha256=sha, lines=int(g.size),
                                                          controls=n, limits=0, drives=m)))
            json.dump(run, open(os.path.join(d, 'run.json'), 'w'))
            return hz.identification_from_run(case, d)
        lag = 301.0
        got = write(turn)
        self.assertLess(np.abs(got.H0 - H).max() / np.abs(H).max(), 1e-12)
        # the settle window's structure term on the file read back is the engine's on the file as written (core.rs
        # settle_s: 3 x its group delay + lag / fs); without the added term it is not
        from pyloop.ident import Identification
        written = Identification(g, H * rot, idn.T0, idn.coh, idn.coh_limit, idn.trust, {})
        for f in (0.5 * (g[0] + g[-1]), g[0] + 0.3 * (g[-1] - g[0])):
            engine = 3.0 * written.group_delay_at(f, 2.0) + lag / case.fs
            here = 3.0 * got.level['as_written'].group_delay_at(f, 2.0) + got.level['lag_frames'] / case.fs
            self.assertAlmostEqual(here, engine, delta=1e-12)
            # read from the turned-back response, the window is not the engine's
            self.assertGreater(abs(3.0 * got.group_delay_at(f, 2.0) + lag / case.fs - engine), 1e-3, 'the check cannot fail')
        self.assertTrue(np.all(got.trust))
        wrong = write(-turn)
        self.assertGreater(np.abs(wrong.H0 - H).max() / np.abs(H).max(), 0.1)
        # a file of another structure is refused
        run = json.load(open(os.path.join(d, 'run.json')))
        run['structure']['sha256'] = '0' * 64
        json.dump(run, open(os.path.join(d, 'run.json'), 'w'))
        with self.assertRaises(hz.Refused):
            hz.identification_from_run(case, d)

    def test_the_linear_tier_is_stepped_as_the_simulator_steps_it(self):
        # the simulator's fitted filters (hertzforge-hal-sim fitted.rs) match the continuous response delayed `delay`
        # samples to well under 0.01 dB in band; the reference model's triangle hold, which the harness used before,
        # does not; stepping a tone through the filters gives their own response
        import reference_model as rm
        from pyloop.fitted import FittedModal
        st = rm.load_structure(P(LIN))
        rows = [i for i, o in enumerate(st.outputs) if o['role'] == 'control']
        f = np.geomspace(20.0, 2000.0, 400)
        fm = FittedModal(st, rows, sm.FS)
        with np.errstate(all='ignore'):
            Hc = rm.frf(st, f, rows=rows) * np.exp(-2j * np.pi * f * fm.delay / sm.FS)[:, None, None]
            Ht = rm.discrete_frf(st, f, sm.FS, rows=rows) * np.exp(-2j * np.pi * f * fm.delay / sm.FS)[:, None, None]

        def worst(Hx):
            return float((np.abs(Hx - Hc).reshape(f.size, -1).max(1) / np.abs(Hc).reshape(f.size, -1).max(1)).max())
        self.assertLess(worst(fm.frf(f)), 1e-3)
        self.assertGreater(worst(Ht), 1e-2)
        f0, D = 1234.5, np.array([1.0, 0.5j, -0.3, 0.2, 0.0, 0.1])
        t = np.arange(int(2 * sm.FS)) / sm.FS
        d = (D[None, :] * np.exp(2j * np.pi * f0 * t)[:, None]).real
        y = np.concatenate([fm.step(d[i:i + 82]) for i in range(0, t.size, 82)])
        seg = slice(t.size - 8192, t.size)
        got = 2 * (y[seg] * np.exp(-2j * np.pi * f0 * t[seg])[:, None]).mean(0)
        want = fm.frf([f0])[0] @ D
        self.assertLess(np.abs(got - want).max() / np.abs(want).max(), 1e-6)

    def test_the_pipeline_is_recorded_and_the_switch_leaves_it_out(self):
        out_on, out_off = tempfile.mkdtemp(dir=TMP), tempfile.mkdtemp(dir=TMP)
        case = os.path.join(P(LIN), 'cases', 'T4__matrix.json')
        from pyloop import harness as hz
        with np.errstate(all='ignore'), contextlib.redirect_stdout(io.StringIO()):
            hz.main([case, '--root', os.path.join(TMP, 'problems'), '--out', out_on])
            hz.main([case, '--root', os.path.join(TMP, 'problems'), '--out', out_off, '--no-pipeline'])
        on, off = (json.load(open(os.path.join(d, 'run.json'))) for d in (out_on, out_off))
        pp = on['identification']['pipeline']
        self.assertTrue(on['options']['pipeline'])
        self.assertEqual(pp['pipeline_frames'], 264)
        self.assertEqual(pp['lag_frames'] - pp['pipeline_frames'], pp['turn_samples'])
        self.assertAlmostEqual(pp['transport_delay_s'], pp['lag_frames'] / 8192.0, places=15)
        self.assertFalse(off['options']['pipeline'])
        self.assertIsNone(off['identification']['pipeline'])
        self.assertNotEqual(on['history']['drive_re'], off['history']['drive_re'], 'the check cannot fail')

    def test_refuses_a_case_whose_structure_changed(self):
        case = json.load(open(os.path.join(P(LIN), 'cases', 'T4__matrix.json')))
        case['structure']['sha256'] = '0' * 64
        path = os.path.join(TMP, 'bad_case.json')
        json.dump(case, open(path, 'w'))
        out = os.path.join(TMP, 'refused_run')
        self.assertEqual(self.run_case(path, out), 2)
        self.assertFalse(os.path.exists(out))


class TestPipeline(unittest.TestCase):
    """The runner's pipeline in the harness (Package F): each check is shown to fail on a case built to break it."""

    def case(self, test='T4'):
        from pyloop import harness as hz
        return hz.Case(os.path.join(P(LIN), 'cases', f'{test}__matrix.json'), os.path.join(TMP, 'problems'))

    def test_the_generator_is_splitmix64(self):
        # Vigna's splitmix64 from state 0, its published first four outputs; the engine's uniform is the top 53 bits
        from pyloop import harness as hz
        want = [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F, 0xF88BB8A8724C81EC]
        got = [int(u * 2 ** 53) for u in hz.splitmix_uniform(0, 4)]
        self.assertEqual(got, [z >> 11 for z in want])
        self.assertNotEqual([int(u * 2 ** 53) for u in hz.splitmix_uniform(1, 4)], got, 'the check cannot fail')

    def test_the_restated_lag_is_the_shots_own_cross_correlation(self):
        # the frequency-domain restatement against measure_lag done literally: the shot's first realization played
        # twice from rest through the stepped structure behind the pipeline, its second copy cross-correlated with the
        # drive (ident.rs measure_lag). On the dwell's narrow band the drives' cross-terms decide the peak: the
        # restatement that leaves them out (each drive alone) gives another lag, and so does a wrong pipeline
        from pyloop import harness as hz
        from pyloop.fitted import FittedModal
        for test in ('T4', 'T1'):
            case = self.case(test)
            fs, n = case.fs, hz.record_len_for(case.fs)
            self.assertEqual(n, 32768)
            k = np.arange(n // 2 + 1)
            f = k * fs / n
            on = np.flatnonzero((k > 0) & (k < n // 2) & (f >= case.ident_band[0]) & (f <= case.ident_band[1]))
            m = case.st.n_drives
            X = np.zeros((m, n // 2 + 1), complex)
            X[:, on] = np.exp(2j * np.pi * hz.splitmix_uniform(hz.IDENT_SEED, m * on.size).reshape(m, on.size))
            w = np.fft.irfft(X, n=n, axis=1).T
            w = np.concatenate([w, w])                                         # the rows as the engine writes them
            x = np.concatenate([np.zeros((hz.PIPELINE_FRAMES, m)), w])[:2 * n]   # what reaches the structure
            fm = FittedModal(case.st, case.controls, fs)
            with np.errstate(all='ignore'):
                y = np.concatenate([fm.step(case.gain * x[i:i + 4096]) for i in range(0, 2 * n, 4096)])
            xs, ys = np.fft.rfft(w[n:].T, axis=1), np.fft.rfft(y[n:].T, axis=1)
            s = sum(np.abs(np.fft.irfft(ys[i][None] * np.conj(xs), n=n, axis=1)).sum(0) for i in range(ys.shape[0]))
            literal = int(np.argmax(s[:n // 2]))
            with np.errstate(all='ignore'):
                self.assertEqual(hz.engine_lag(case), literal, test)
                self.assertNotEqual(hz.engine_lag(case, pipeline_frames=hz.PIPELINE_FRAMES - 10), literal,
                                    'the check cannot fail')
            if test == 'T4':
                Hl = hz.small_signal_frf(case, f[on])
                alone = np.zeros(n)
                for i in range(Hl.shape[1]):
                    for j in range(m):
                        Y = np.zeros(n // 2 + 1, complex)
                        Y[on] = Hl[:, i, j] * np.exp(-2j * np.pi * f[on] * hz.PIPELINE_FRAMES / fs)
                        alone += np.abs(np.fft.irfft(Y, n=n))
                self.assertNotEqual(int(np.argmax(alone[:n // 2])), literal, 'the cross-terms are not needed here')

    def test_the_station_plays_the_drive_late_and_reads_c_as_h_d(self):
        # with the pipeline the structure takes the written drive PIPELINE_FRAMES samples late, and a steady reading is
        # still H D on the exact response; the station without the retarded phase would read H D exp(-j w P)
        from pyloop import harness as hz
        case = self.case()
        f0 = 101.3
        D = np.array([1.0, 0.5j, -0.3, 0.2, 0.0, 0.1]) * 0.05
        got = {}
        for P_ in (0, hz.PIPELINE_FRAMES):
            sta = hz.Station(case, P_)
            sta.set_bandwidth(f0)
            seen = []
            step0 = sta.fitted.step
            sta.fitted.step = lambda u, s0=step0: (seen.append(u.copy()), s0(u))[1]
            written = []
            with np.errstate(all='ignore'):
                for _ in range(600):
                    C, _, d = sta.play(D, D, f0)
                    written.append(d)
            x, w = np.concatenate(seen) / case.gain, np.concatenate(written)
            self.assertEqual(float(np.abs(x[:P_]).max()) if P_ else 0.0, 0.0)
            self.assertLess(np.abs(x[P_:] - w[:w.shape[0] - P_]).max(), 1e-12)
            got[P_] = C
            if P_:
                self.assertGreater(np.abs(x[P_ - 1:] - w[:w.shape[0] - P_ + 1]).max(), 1e-3, 'the check cannot fail')
        H = hz.small_signal_frf(case, [f0])[0][:len(case.controls)]
        want = H @ D
        err = lambda C: np.linalg.norm(C - want) / np.linalg.norm(want)  # noqa: E731
        self.assertLess(err(got[hz.PIPELINE_FRAMES]), 0.02)
        self.assertLess(err(got[0]), 0.02)
        unretarded = want * np.exp(-2j * np.pi * f0 * hz.PIPELINE_FRAMES / case.fs)
        self.assertGreater(np.linalg.norm(got[hz.PIPELINE_FRAMES] - unretarded) / np.linalg.norm(want), 0.5,
                           'the check cannot fail')

    def test_the_law_is_told_the_engines_loop_delay(self):
        # the law's transport delay is lag_frames / fs (service.rs make_law_setup) and its group delay is read on the
        # response with the lag taken out (law-math speed.rs: the identification's group delay + the transport)
        from pyloop import harness as hz
        from pyloop.ident import Identification
        case = self.case()
        idn = hz.identify(case)
        L = 335
        s = hz.make_setup(case, idn, L)
        self.assertEqual(s.transport_delay_s, L / case.fs)
        self.assertEqual(hz.make_setup(case, idn).transport_delay_s, 0.0)
        g = idn.grid_hz
        turned = Identification(g, idn.H0 * np.exp(2j * np.pi * g * (L - 264) / case.fs)[:, None, None], idn.T0, idn.coh,
                                idn.coh_limit, idn.trust, {})
        import copy
        x = copy.copy(idn)
        x.gd_turn, x._gd = (L - 264, case.fs), None
        fq = case.test['freq_hz']
        self.assertAlmostEqual(x.group_delay_at(fq, 2.0), turned.group_delay_at(fq, 2.0), delta=1e-15)
        self.assertGreater(abs(idn.group_delay_at(fq, 2.0) - turned.group_delay_at(fq, 2.0)), 1e-3, 'the check cannot fail')

    def test_an_engine_files_turn_must_be_its_lag_less_the_pipeline(self):
        from pyloop import harness as hz
        case = self.case()
        idn = hz.identify(case)
        idn.level.update(lag_frames=300.0, frames_ahead=8, delay_turn_samples=36.0)
        P_, L, _ = hz.pipeline_for(case, idn, idn)
        self.assertEqual((P_, L), (264, 300))
        idn.level['delay_turn_samples'] = 35.0
        with self.assertRaises(hz.Refused):
            hz.pipeline_for(case, idn, idn)


class TestCandidates(unittest.TestCase):
    """The research round's candidates in the harness (Package F): each check is shown to fail on a case built to
    break it."""

    def write_case(self, test='T1', law='k2f', params=None, rung=None):
        c = json.load(open(os.path.join(P(LIN), 'cases', f'{test}__matrix.json')))
        c['law'] = dict(id=law, params=params or {})
        c['case_id'] = c['case_id'].replace('__matrix', '__' + law)
        if rung is not None:
            c['ladder']['rung'] = rung
        path = os.path.join(tempfile.mkdtemp(dir=TMP), 'case.json')
        json.dump(c, open(path, 'w'))
        return path

    def case(self, **kw):
        from pyloop import harness as hz
        return hz.Case(self.write_case(**kw), os.path.join(TMP, 'problems'))

    def test_a_candidate_takes_only_its_tests_and_parameters(self):
        from pyloop import harness as hz
        self.assertEqual(self.case(test='T1', law='k2f').law_params['expected_drop'], hz.K2F_EXPECTED_DROP)
        self.assertEqual(self.case(test='T4', law='k5', params=dict(dither=0.03)).law_params['dither'], 0.03)
        with self.assertRaises(hz.Refused):
            self.case(test='T4', law='k2f')
        with self.assertRaises(hz.Refused):
            self.case(test='T1', law='k5')
        with self.assertRaises(hz.Refused):
            self.case(test='T1', law='k2f', params=dict(no_such=1))
        with self.assertRaises(hz.Refused):          # a shipped law still takes no parameters
            self.case(test='T1', law='matrix', params=dict(eps=1e-2))

    def test_a_rung_of_whole_sweeps_bounces_one_pass_a_rung(self):
        # five rungs of one sweep each: up at -12 dB, down at -9, up at -6, down at -3, up at 0, then done; the 2 s
        # rungs of the suite's case reach full level within the first pass
        from pyloop import harness as hz
        case = self.case(test='T1', law='k4', rung={'sweeps': 1})
        sch = hz.Schedule(case)
        dt = case.dt
        seen, k = [], 0
        while not sch.done and k < 10 ** 6:
            f = sch.frequency()
            lv = sch.level_db()
            if not seen or seen[-1][0] != lv:
                seen.append([lv, f, f])
            seen[-1][2] = f
            sch.step(dt)
            k += 1
        lo, hi = case.test['band_hz']
        self.assertEqual([x[0] for x in seen], [-12.0, -9.0, -6.0, -3.0, 0.0])
        ups = [x[2] > x[1] for x in seen]
        self.assertEqual(ups, [True, False, True, False, True])
        self.assertAlmostEqual(k * dt, 5 * hz.test_length_s(case.test), delta=5 * dt)
        self.assertAlmostEqual(case.max_test_s, 1.5 * 5 * hz.test_length_s(case.test) + 60.0, places=6)
        sec = hz.Schedule(self.case(test='T1', law='k4'))
        for _ in range(int(round(9.0 / dt))):
            sec.step(dt)
        self.assertEqual(sec.level_db(), 0.0, 'the check cannot fail')
        self.assertLess(sec.frequency(), lo * 1.2)

    def test_the_modal_fit_switch(self):
        from pyloop import harness as hz
        from pyloop import modal
        case = self.case(test='T4', law='k5')          # the dwell's narrow band: a quick modal fit
        idn = hz.identify(case)
        setup = hz.make_setup(case, idn)
        setup.band = case.ident_band                    # the dwell's identification band, around its resonance
        with np.errstate(all='ignore'):
            a = modal.modal_form(setup)
            modal.MODAL_FITS['test_other'] = lambda ident, band: modal.ModalForm(ident, [], [], np.zeros((0, 6, 6)), 'x')
            try:
                b = modal.modal_form(setup, fit='test_other')
                law = modal.FollowedPoles(setup, modal_fit='test_other')
            finally:
                del modal.MODAL_FITS['test_other']
        self.assertGreater(a.M, 0)
        self.assertEqual(b.M, 0)
        self.assertEqual(law.mf.M, 0)
        with self.assertRaises(ValueError):
            modal.modal_form(setup, fit='no_such_fit')

    def test_k4s_fitted_model_is_the_identification_at_low_level(self):
        from pyloop import harness as hz
        from pyloop.k4_ladder import LevelLadder
        case = self.case(test='T4', law='k5')          # the dwell's narrow band: a quick modal fit
        idn = hz.identify(case)
        setup = hz.make_setup(case, idn)
        setup.band = case.ident_band                    # the dwell's identification band, around its resonance
        with np.errstate(all='ignore'):
            law = LevelLadder(setup)
        M = law.M
        self.assertGreater(M.f0.size, 0)
        for f in np.geomspace(setup.band[0], setup.band[1], 7):
            want = M.mf.H0_at(f)
            got = M.H(f, M.f0, M.z0)
            self.assertLess(np.abs(got - want).max() / np.abs(want).max(), 1e-10)
            moved = M.H(f, M.f0 * 0.97, M.z0)
            if np.any(np.abs(f - M.f0) < 3 * 2 * M.z0 * M.f0):
                self.assertGreater(np.abs(moved - want).max() / np.abs(want).max(), 1e-3, 'the check cannot fail')
        # each residue rank one: the dominant singular pair of the fitted residue
        r = 0
        s = np.linalg.svd(M.mf.A[r], compute_uv=False)
        self.assertAlmostEqual(np.linalg.norm(np.outer(M.Phi[r], M.Gam[r])), s[0], delta=1e-9 * s[0])

    def test_k5s_period_reads_the_engines_loop_delays(self):
        from pyloop import harness as hz
        from pyloop.k5_rls_dither import RLSDither
        case = self.case(test='T4', law='k5')
        idn = hz.identify(case)
        f = case.test['freq_hz']
        s0 = hz.make_setup(case, idn)
        s0.tracker_delay_factor = 1.0 / np.pi
        del s0.tracker_delay_factor
        testbed = int(np.ceil((2.0 / (2 * np.pi * s0.tracking_bw(f)) + idn.group_delay_at(f)) / s0.dt_c))
        self.assertEqual(RLSDither(s0)._period(f), max(testbed, 2))
        s1 = hz.make_setup(case, idn, 335)
        want = int(np.ceil((hz.TRACKER_DELAY_FACTOR / s1.tracking_bw(f) + idn.group_delay_at(f) + 335 / case.fs)
                           / s1.dt_c))
        self.assertEqual(RLSDither(s1)._period(f), max(want, 2))
        self.assertNotEqual(RLSDither(s1)._period(f), RLSDither(s0)._period(f), 'the check cannot fail')



def _synth_setup(idn, delay_samples):
    from pyloop.interface import Setup
    import synth_modal as sy
    n, m = idn.H0.shape[1], idn.H0.shape[2]
    s = Setup(n=n, m=m, p=0, dt_c=0.01, fs=sy.FS, band=sy.BAND, drive_ceiling=np.full(m, 2.0), ident=idn,
              ref_grid=np.ones((idn.grid_hz.size, n), complex), phase_specified=True,
              limit_grid=np.zeros((idn.grid_hz.size, 0)), alarm_db=1.5, abort_db=20.0, shape='tall' if n > m else 'square',
              tracking=dict(mode='proportional', pct=10.0, max_hz=100.0))
    s.tracker_delay_factor, s.transport_delay_s = 0.625, (264 + (delay_samples or 0)) / sy.FS
    if delay_samples is not None:
        idn.block_delay = (delay_samples, sy.FS)
    return s


class TestModalFitM7d(unittest.TestCase):
    """Package C's M7d and M9d in the harness, with the harness's known delay (Package G): each check is shown to fail
    on a case built to break it."""

    def fit(self, which='m7d', delay=3.0, tau_given=None):
        from pyloop import modal_fits as MF
        import synth_modal as sy
        idn, tru, tau = sy.identification(delay_samples=delay)
        t = tau if tau_given is None else tau_given
        with np.errstate(all='ignore'):
            out = getattr(MF, which)(idn, sy.BAND, t, sub_bands=MF.octave_sub_bands(sy.BAND))
        return out, tru, tau

    def test_the_port_is_package_cs_fit(self):
        # Package C's own code wrote the vector on the same identifications (experiments/G/make_m7d_vector.py): the
        # synthetic one at noise 1e-3 and at 0.1 (where the clustering's MAC and share thresholds bind)
        from pyloop import modal_fits as MF
        import synth_modal as sy
        with open(os.path.join(HERE, 'data', 'm7d_parity.json')) as fh:
            v = json.load(fh)
        idq, _, tq = sy.identification(noise=v['noisy']['noise'])
        for which, vv in (('m7d', v), ('m9d', v), ('m7d', v['noisy']), ('m9d', v['noisy'])):
            if vv is v:
                (f, z, A), _, tau = self.fit(which)
            else:
                tau = tq
                with np.errstate(all='ignore'):
                    f, z, A = getattr(MF, which)(idq, sy.BAND, tq, sub_bands=MF.octave_sub_bands(sy.BAND))
            self.assertAlmostEqual(tau, vv['tau_s'], places=15)
            w = vv[which]
            Aw = np.array(w['A_re']) + 1j * np.array(w['A_im'])
            self.assertEqual(f.size, len(w['f']), which)
            self.assertLess(np.max(np.abs(f - w['f']) / np.array(w['f'])), 1e-9, which)
            self.assertLess(np.max(np.abs(z - w['zeta']) / np.array(w['zeta'])), 1e-7, which)
            self.assertLess(np.linalg.norm(A - Aw) / np.linalg.norm(Aw), 1e-7, which)

    @NEEDS
    def test_the_port_is_package_cs_fit_on_a_frame_band(self):
        # near-repeated modes of a frame: the harness's identification of frame-6x6-realistic_s00, 250 to 500 Hz, the
        # block's delay given; Package C's own code wrote the vector (experiments/G/make_m7d_vector.py)
        from pyloop import harness as hz
        from pyloop import modal_fits as MF
        with open(os.path.join(HERE, 'data', 'm7d_parity.json')) as fh:
            v = json.load(fh)['frame']
        self.assertEqual(v['problem'], SID)
        case = hz.Case(os.path.join(P(SID), 'cases', 'T1__matrix.json'), os.path.join(TMP, 'problems'))
        idn = hz.identify(case)
        tau = hz.block_delay_samples(case) / case.fs
        self.assertAlmostEqual(tau, v['tau_s'], places=15)
        band = tuple(v['band'])
        for which in ('m7d', 'm9d'):
            with np.errstate(all='ignore'):
                f, z, A = getattr(MF, which)(idn, band, tau, sub_bands=MF.octave_sub_bands(band))
            w = v[which]
            Aw = np.array(w['A_re']) + 1j * np.array(w['A_im'])
            self.assertEqual(f.size, len(w['f']), which)
            self.assertLess(np.max(np.abs(f - w['f']) / np.array(w['f'])), 1e-9, which)
            self.assertLess(np.max(np.abs(z - w['zeta']) / np.array(w['zeta'])), 1e-7, which)
            self.assertLess(np.linalg.norm(A - Aw) / np.linalg.norm(Aw), 1e-7, which)

    def test_the_known_delay_is_taken_out_and_put_back(self):
        (f, z, A), (ft, zt, At), tau = self.fit('m7d')
        self.assertEqual(f.size, ft.size)
        j = [int(np.argmin(np.abs(f - x))) for x in ft]
        turn = np.exp(-2j * np.pi * ft * tau)[:, None, None]
        err = [np.linalg.norm(A[j[k]] - At[k] * turn[k]) / np.linalg.norm(At[k]) for k in range(ft.size)]
        self.assertLess(max(err), 0.02)
        # the same fit not told the delay: a real rank-one residue cannot hold the turn
        (f0, z0, A0), _, _ = self.fit('m7d', tau_given=0.0)
        j0 = [int(np.argmin(np.abs(f0 - x))) for x in ft] if f0.size else []
        err0 = [np.linalg.norm(A0[j0[k]] - At[k] * turn[k]) / np.linalg.norm(At[k]) for k in range(ft.size)] if f0.size else [1.0]
        self.assertGreater(max(err0), 0.10, 'the check cannot fail')

    def test_the_fit_reads_the_harness_known_delay_and_refuses_without_it(self):
        from pyloop import modal
        import synth_modal as sy
        idn, _, _ = sy.identification(delay_samples=3.0)
        with self.assertRaises(ValueError):
            modal.MODAL_FITS['m7d'](idn, sy.BAND)
        s = _synth_setup(idn, 3.0)
        with np.errstate(all='ignore'):
            mf = modal.modal_form(s, fit='m7d')
        self.assertIn('known delay 3 samples', mf.source)
        self.assertAlmostEqual(mf.delay_s, 3.0 / sy.FS, places=15)
        self.assertEqual(mf.M, len(sy.MODES))

    def test_the_cache_keys_the_known_delay(self):
        from pyloop import modal
        import synth_modal as sy
        idn, (ft, _, _), _ = sy.identification(delay_samples=3.0)
        with np.errstate(all='ignore'):
            a = modal.modal_form(_synth_setup(idn, 3.0), fit='m7d')
            idn2 = sy.identification(delay_samples=3.0)[0]
            b = modal.modal_form(_synth_setup(idn2, 2.0), fit='m7d')
        self.assertTrue(np.array_equal(a.ident.H0, b.ident.H0), 'the check cannot fail')
        self.assertNotEqual(a.source, b.source)
        self.assertGreater(np.abs(a.A - b.A).max(), 1e-3 * np.abs(a.A).max())

    @NEEDS
    def test_the_harness_gives_the_law_the_blocks_delay_not_the_lag(self):
        from pyloop import harness as hz
        for sid, want in ((SID, 2.0), (LIN, 4.0)):
            c = json.load(open(os.path.join(P(sid), 'cases', 'T1__matrix.json')))
            path = os.path.join(tempfile.mkdtemp(dir=TMP), 'case.json')
            json.dump(c, open(path, 'w'))
            case = hz.Case(path, os.path.join(TMP, 'problems'))
            self.assertEqual(hz.block_delay_samples(case), want, sid)
            idn = hz.identify(case)
            P_, L, _ = hz.pipeline_for(case, idn, idn)
            self.assertNotEqual(L - P_, want, 'the check cannot fail: the lag less the pipeline is not the delay')
        # a structure whose controls carry their own output delays has no single delay
        case.st.raw = dict(case.st.raw, simulator=dict(case.st.raw['simulator'], output_delays_s=[1e-4] * 64))
        st_kind = case.st.kind
        case.st.kind = 'jointed'
        try:
            self.assertIsNone(hz.block_delay_samples(case))
        finally:
            case.st.kind = st_kind

    def test_k3_and_k4_default_to_m7d(self):
        from pyloop import harness as hz
        self.assertEqual(hz.CANDIDATES['k3'][1]['modal_fit'], 'm7d')
        self.assertEqual(hz.CANDIDATES['k4'][1]['modal_fit'], 'm7d')
        self.assertEqual(hz.CANDIDATES['k2f'][1]['modal_fit'], 'current')      # the brief changes K3 and K4 only
        from pyloop import modal
        self.assertIn('m7d', modal.MODAL_FITS)
        self.assertIn('m9d', modal.MODAL_FITS)
        self.assertIn('modal_fit', hz.CANDIDATES['k3'][4])

    @NEEDS
    def test_a_k3_case_takes_m7d_unless_it_sets_a_fit(self):
        tc = TestCandidates()
        tc.write_case = TestCandidates.write_case.__get__(tc)
        from pyloop import harness as hz
        c = hz.Case(tc.write_case(test='T1', law='k3'), os.path.join(TMP, 'problems'))
        self.assertEqual(c.law_params['modal_fit'], 'm7d')
        c = hz.Case(tc.write_case(test='T1', law='k4', params=dict(modal_fit='m9d')), os.path.join(TMP, 'problems'))
        self.assertEqual(c.law_params['modal_fit'], 'm9d')

    def test_the_law_built_on_it_and_the_run_file_record(self):
        from pyloop import harness as hz
        from pyloop.modal import FollowedPoles
        from pyloop.k4_ladder import LevelLadder
        import synth_modal as sy
        for cls in (FollowedPoles, LevelLadder):
            idn, _, _ = sy.identification(delay_samples=3.0)
            with np.errstate(all='ignore'):
                law = cls(_synth_setup(idn, 3.0), modal_fit='m7d')
            r = hz.modal_form_record(law)
            self.assertEqual(r['modes'], len(sy.MODES), cls.__name__)
            self.assertIn('m7d', r['source'])
            self.assertAlmostEqual(r['delay_s'], 3.0 / sy.FS, places=15)
        self.assertIsNone(hz.modal_form_record(object()))


if __name__ == '__main__':
    unittest.main()
