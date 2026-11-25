"""Tests of sine/record_jobs.py on a small synthetic problems folder.

    python3 -W ignore -m unittest sine.tests.test_record_jobs     (from the repository root)
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import plan  # noqa: E402
import record_jobs as rj  # noqa: E402


def make_problem(root, sid, tests):
    d = os.path.join(root, sid)
    os.makedirs(os.path.join(d, 'cases'))
    json.dump(dict(tests={t: dict(built=True) for t in tests}), open(os.path.join(d, 'problem.json'), 'w'))
    for law, accepts in plan.LAW_TESTS.items():
        for t in accepts:
            if t in tests:
                case = dict(case_id=f'{sid}__{t}__{law}', law=dict(id=law, params={}), reference=dict(phase='auto'),
                            noise=dict(rms_m_s2=0.01, seed=12345), drives=dict(gain=2.5, ceiling_vpk=1.99526231),
                            protective='default')
                json.dump(case, open(os.path.join(d, 'cases', f'{t}__{law}.json'), 'w'))


class TestRecordJobs(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        study, held = os.path.join(self.d, 'problems'), os.path.join(self.d, 'heldout', 'problems')
        for s in (0, 1, 2):
            for t in plan.TIERS:
                for r in ('6x6', '12x6', '12x3'):
                    tests = ['T1', 'T2', 'T3', 'T4', 'T5'] + (['L1'] if t and r != '12x3' else [])
                    make_problem(study, plan.sid(r, t, s), tests)
        for s in (90, 91):
            for t in plan.TIERS:
                for r in ('6x6', '12x6'):
                    make_problem(held, plan.sid(r, t, s), ['T1', 'T3', 'T4'])
        json.dump(dict(seeds=[90, 91]), open(os.path.join(self.d, 'heldout', 'heldout_seeds.json'), 'w'))
        self.base = os.path.join(self.d, 'jobs')
        with contextlib.redirect_stdout(io.StringIO()):
            rj.main(['--problems', study, '--heldout', held, '--base', self.base])
        self.sets = {s['name']: s for s in json.load(open(os.path.join(self.base, 'sets.json')))['sets']}

    def tearDown(self):
        shutil.rmtree(self.d)

    def case(self, name, sid, t, law):
        return json.load(open(os.path.join(self.base, name, 'cases', sid, f'{t}__{law}.json')))

    def test_counts(self):
        # study: 27 structures; matrix on T1 to T5 (and L1 on the 12 jointed 6x6 and 12x6), paired 2, ratio 2, steepest 1,
        # secant 1: 12 x 12 + 15 x 11
        self.assertEqual(self.sets['study']['jobs'], 12 * 12 + 15 * 11)
        self.assertEqual(self.sets['study_defaults']['structures'], 12)        # 12x6 and 6x6, three tiers, seeds 0 and 1
        self.assertEqual(self.sets['study_repeat_1']['jobs'], 20)
        self.assertEqual(self.sets['audit_defaults']['structures'], 12)
        # T1, T3, T4 built: frame-6x6-realistic_s90 matrix 3, paired 2, ratio 2, steepest 1, secant 1; frame-12x6-strong_s91
        # matrix T1, T3, T4 and the four other laws' T4
        self.assertEqual(self.sets['audit_repeat_2']['jobs'], 9 + 7)

    def test_settings(self):
        sid = plan.sid('6x6', 'realistic', 0)
        c = self.case('study', sid, 'T4', 'secant')
        self.assertEqual((c['protective'], c['reference']['phase'], c['drives']['ceiling_vpk']), ('widened', 'specified', 1.99526231))
        c = self.case('study', sid, 'T3', 'ratio')
        self.assertEqual((c['protective'], c['reference']['phase']), ('widened', 'auto'))
        c = self.case('study_defaults', sid, 'T4', 'matrix')
        self.assertEqual(c['protective'], 'default')
        self.assertNotIn('ceiling_vpk', c['drives'])
        self.assertEqual(c['drives']['gain'], 2.5)
        seeds = {self.case(f'study_repeat_{k}', sid, 'T4', 'matrix')['noise']['seed'] for k in (1, 2, 3)}
        ids = {self.case(f'study_repeat_{k}', sid, 'T4', 'matrix')['identification_seed'] for k in (1, 2, 3)}
        self.assertEqual(len(seeds | {12345}), 4)
        self.assertEqual(len(ids), 3)
        r = self.case('study_repeat_1', sid, 'T4', 'paired')
        self.assertEqual((r['protective'], r['noise']['seed']), ('widened', self.case('study_repeat_1', sid, 'T4', 'matrix')['noise']['seed']))
        self.assertFalse(os.path.exists(os.path.join(self.base, 'study_defaults', 'cases', plan.sid('12x3', 'strong', 0))))

    def test_problem_cases_untouched(self):
        c = json.load(open(os.path.join(self.d, 'problems', plan.sid('6x6', 'realistic', 0), 'cases', 'T4__secant.json')))
        self.assertEqual((c['protective'], c['reference']['phase']), ('default', 'auto'))

    def test_never_overwritten(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
            rj.main(['--problems', os.path.join(self.d, 'problems'), '--heldout', os.path.join(self.d, 'heldout', 'problems'),
                     '--base', self.base])


if __name__ == '__main__':
    unittest.main()
