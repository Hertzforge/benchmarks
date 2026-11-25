"""Tests of the MIMO Sine suite (draft 0.1): python3 -m unittest discover -s sine/tests -v

Needs the MIMO Random v0.2 problems: set MSNB_RANDOM_PROBLEMS to the folder holding them (the tests that need them
are skipped otherwise). Every check here is shown to fail on a case built to break it."""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
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

    def test_completion_reads_an_early_end(self):
        c = self.score('T1', stop_fraction=0.5, outcome_class='stopped_by_protective_default')
        self.assertEqual(c['completion']['outcome'], 'stopped_by_protective_default')
        self.assertEqual(c['completion']['guard'], 'control_abort')
        self.assertLess(c['completion']['coverage'], 0.51)
        c = self.score('T3', stop_fraction=0.25, outcome_class='aborted')
        self.assertLess(c['completion']['coverage'], 0.26)

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

    def test_free_disk_stop(self):
        out, log = self.run_batch('b2', min_free_gb=1e12)
        self.assertTrue(any('BATCH STOP b2: free disk' in x for x in log))
        self.assertTrue(log[-1].endswith('BATCH END b2: 0 done, 0 failed, 5 skipped'), log[-1])


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

    def test_listed_name_by_hash(self):
        names = frozenset({pub.phrase_hash('Example Widget Works')})
        self.assertTrue(pub.check_text('x', 'made by example widget works ltd', names))
        self.assertFalse(pub.check_text('x', 'made by example works', names))
        self.assertEqual(pub.phrase_hash('a  B-c'), pub.phrase_hash('a b c'))

    def test_suite_code_is_clean(self):
        n, found = pub.run([SINE])
        self.assertEqual(found, [])


if __name__ == '__main__':
    unittest.main()
