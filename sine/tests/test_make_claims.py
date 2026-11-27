"""Tests of sine/make_claims.py on synthetic readings built to give known claims.

    python3 -W ignore -m unittest sine.tests.test_make_claims     (from the repository root)
"""
import csv
import filecmp
import json
import os
import random
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import make_claims as mc  # noqa: E402

BASE = dict(iec_share=0.8, worst_over_db=1.0, completion=1.0)


def sid(system, kind, seed):
    tier, _, eff = kind.partition('_')
    return f'frame-{system}' + ('' if tier == 'linear' else f'-{tier}') + (f'_{eff}' if eff else '') + f'_s{seed:02d}'


def row(system, kind, seed, test, law, scored=True, exc=-1.0, version='hertzforge', **vals):
    r = dict(version=version, structure=sid(system, kind, seed), system=system, kind=kind, seed=seed, test=test, law=law,
             outcome='completed', scored=int(scored))
    r.update(BASE)
    r.update(vals)
    for k in mc.LINES:
        r[f'excursion_{k}'] = exc if scored else 'unscored'
    if not scored:
        r['worst_over_db'] = ''
        r['iec_share'] = 0.0
    return r


def pair_cell(system, kind, test, a, b, diffs, reading='worst_over_db', seeds=None, version='hertzforge', vb=None):
    """Rows for two laws on a cell whose difference a - b of one reading is diffs (one value a structure); b's rows in
    version vb (default: the same version)."""
    seeds = seeds or list(range(len(diffs)))
    out = []
    for s, d in zip(seeds, diffs):
        out.append(row(system, kind, s, test, b, version=vb or version, **{reading: BASE[reading]}))
        out.append(row(system, kind, s, test, a, version=version, **{reading: BASE[reading] + d}))
    return out


def run(cmd, **kw):
    args = [cmd]
    for k, v in kw.items():
        args += [f'--{k}', v]
    return mc.main(args)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.d)

    def claims(self, rows, name='readings.csv', sub=''):
        os.makedirs(os.path.join(self.d, sub), exist_ok=True)
        rp = os.path.join(self.d, sub, name)
        mc.write_csv(rp, rows)
        out = os.path.join(self.d, sub, 'claims.json')
        if os.path.exists(out):
            os.remove(out)
        run('claims', readings=rp, out=out)
        return json.load(open(out)), out

    def kinds(self, doc, kind, **match):
        return [c for c in doc['claims'] if c['kind'] == kind and all(c.get(k) == v for k, v in match.items())]


class TestPairRule(Tmp):
    def test_clear_claim(self):
        diffs = [-1.0] * 13 + [0.0] * 4 + [0.5] * 3
        doc, _ = self.claims(pair_cell('12x6', 'realistic', 'T4', 'matrix', 'secant', diffs))
        c = self.kinds(doc, 'pair', reading='worst_over_db')
        self.assertEqual(len(c), 1)
        self.assertEqual((c[0]['law'], c[0]['against'], c[0]['side']), ('matrix', 'secant', 'lower'))
        self.assertEqual((c[0]['study']['lower'], c[0]['study']['higher'], c[0]['study']['n']), (13, 3, 20))
        self.assertAlmostEqual(c[0]['study']['median'], -1.0)

    def test_twelve_of_twenty_is_enough(self):
        doc, _ = self.claims(pair_cell('12x6', 'strong', 'T3', 'matrix', 'paired', [0.5] * 12 + [-0.5] * 4 + [0.0] * 4))
        self.assertEqual(len(self.kinds(doc, 'pair', reading='worst_over_db')), 1)

    def test_eleven_of_twenty_is_not(self):
        # the median is +0.5, but only eleven of twenty are beyond 0.2 dB on its side (0.15 is under the threshold)
        doc, _ = self.claims(pair_cell('12x6', 'strong', 'T3', 'matrix', 'paired', [0.5] * 11 + [0.15] * 5 + [-0.5] * 4))
        self.assertEqual(self.kinds(doc, 'pair', reading='worst_over_db'), [])

    def test_five_on_the_other_side_is_too_many(self):
        doc, _ = self.claims(pair_cell('12x6', 'strong', 'T3', 'matrix', 'paired', [0.5] * 15 + [-0.5] * 5))
        self.assertEqual(self.kinds(doc, 'pair', reading='worst_over_db'), [])

    def test_median_at_the_threshold_is_not_beyond(self):
        doc, _ = self.claims(pair_cell('6x6', 'realistic', 'T4', 'ratio', 'secant', [0.2] * 20))
        self.assertEqual(self.kinds(doc, 'pair', reading='worst_over_db'), [])
        doc, _ = self.claims(pair_cell('6x6', 'realistic', 'T4', 'ratio', 'secant', [0.2001] * 20))
        self.assertEqual(len(self.kinds(doc, 'pair', reading='worst_over_db')), 1)

    def test_exactly_at_the_threshold(self):
        self.assertFalse(mc.pair_stats([0.2] * 20, 0.2)['claim'])
        self.assertTrue(mc.pair_stats([0.25] * 20, 0.2)['claim'])
        self.assertFalse(mc.pair_stats([0.25] * 11 + [0.2] * 9, 0.2)['claim'])   # 0.2 itself is not beyond
        self.assertTrue(mc.pair_stats([-0.25] * 16 + [0.25] * 4, 0.2)['claim'])               # 16 lower, 4 higher

    def test_share_threshold(self):
        doc, _ = self.claims(pair_cell('6x6', 'strong', 'T4', 'matrix', 'paired', [0.049] * 20, reading='iec_share'))
        self.assertEqual(self.kinds(doc, 'pair', reading='iec_share'), [])
        doc, _ = self.claims(pair_cell('6x6', 'strong', 'T4', 'matrix', 'paired', [0.051] * 20, reading='iec_share'))
        c = self.kinds(doc, 'pair', reading='iec_share')
        self.assertEqual(len(c), 1)
        self.assertEqual(c[0]['side'], 'higher')
        # the same 0.051 in dB is not a claim
        doc, _ = self.claims(pair_cell('6x6', 'strong', 'T4', 'matrix', 'paired', [0.051] * 20))
        self.assertEqual(self.kinds(doc, 'pair', reading='worst_over_db'), [])

    def test_ten_structure_cell(self):
        doc, _ = self.claims(pair_cell('6x6', 'realistic_soft', 'T4', 'matrix', 'paired', [-1.0] * 6 + [1.0] * 2 + [0.0] * 2))
        self.assertEqual(len(self.kinds(doc, 'pair', reading='worst_over_db')), 1)
        doc, _ = self.claims(pair_cell('6x6', 'realistic_soft', 'T4', 'matrix', 'paired', [-1.0] * 6 + [1.0] * 3 + [0.0]))
        self.assertEqual(self.kinds(doc, 'pair', reading='worst_over_db'), [])

    def test_missing_reading_counts_on_neither_side(self):
        rows = pair_cell('12x6', 'strong', 'T4', 'matrix', 'secant', [-1.0] * 14 + [0.0] * 6)
        # four of the fourteen unscored for secant: their difference is missing, so 10 of 20 remain on the side (10 of the
        # 16 with a reading would be enough; the count is against the cell's 20)
        k = 0
        for r in rows:
            if r['law'] == 'secant' and r['seed'] < 4:
                r.update(row('12x6', 'strong', r['seed'], 'T4', 'secant', scored=False))
                k += 1
        self.assertEqual(k, 4)
        doc, _ = self.claims(rows)
        self.assertEqual(self.kinds(doc, 'pair', reading='worst_over_db'), [])

    def test_unordered_pairs_once_in_law_order(self):
        rows = pair_cell('6x6', 'strong', 'T4', 'secant', 'matrix', [1.0] * 20)
        doc, _ = self.claims(rows)
        c = self.kinds(doc, 'pair', reading='worst_over_db')
        self.assertEqual(len(c), 1)
        self.assertEqual((c[0]['law'], c[0]['against'], c[0]['side']), ('matrix', 'secant', 'lower'))

    def test_laws_on_different_tests_are_not_paired(self):
        rows = [row('6x6', 'strong', s, 'T1', 'matrix', worst_over_db=5.0) for s in range(20)]
        rows += [row('6x6', 'strong', s, 'T4', 'secant') for s in range(20)]
        doc, _ = self.claims(rows)
        self.assertEqual(self.kinds(doc, 'pair'), [])


class TestLevelsAndEffects(Tmp):
    def test_levels(self):
        rows = [row('12x3', 'linear', s, 'T1', 'matrix', exc=-0.5 if s < 11 else 2.0) for s in range(20)]
        rows += [row('12x3', 'linear', s, 'T1', 'paired', exc=-0.5) if s < 9 else row('12x3', 'linear', s, 'T1', 'paired', scored=False)
                 for s in range(20)]
        doc, _ = self.claims(rows)
        lv = self.kinds(doc, 'level')
        self.assertEqual(len(lv), 2 * len(mc.LINES))
        self.assertEqual({c['side'] for c in lv if c['law'] == 'matrix'}, {'inside'})
        p = [c for c in lv if c['law'] == 'paired']
        self.assertEqual({c['side'] for c in p}, {'outside'})          # 11 of 20 unscored: the median is outside
        self.assertEqual(p[0]['study']['unscored'], 11)
        self.assertIsNone(p[0]['study']['median'])

    def test_effect_pooled_over_laws_names_no_law(self):
        rows = []
        for law in ('matrix', 'paired'):
            for s in range(20):
                rows.append(row('6x6', 'linear', s, 'T3', law))
                rows.append(row('6x6', 'realistic', s, 'T3', law, worst_over_db=BASE['worst_over_db'] + 2.0))
                rows.append(row('6x6', 'strong', s, 'T3', law, worst_over_db=BASE['worst_over_db'] + 2.0))
        doc, _ = self.claims(rows)
        e = self.kinds(doc, 'effect', reading='worst_over_db')
        self.assertEqual({(c['structure_kind'], c['against']) for c in e}, {('realistic', 'linear'), ('strong', 'linear')})
        self.assertTrue(all('law' not in c for c in e))
        self.assertEqual(e[0]['study']['n'], 40)
        for c in e:
            self.assertNotIn('matrix', c['text'])
            self.assertNotIn('paired', c['text'])


class TestCheck(Tmp):
    def verdicts(self, study_rows, audit_rows):
        doc, cp = self.claims(study_rows)
        ap = os.path.join(self.d, 'audit.csv')
        mc.write_csv(ap, audit_rows)
        out = os.path.join(self.d, 'verdicts.csv')
        run('check', claims=cp, readings=ap, out=out)
        return {r['id']: r for r in csv.DictReader(open(out))}, doc

    def one(self, audit_diffs):
        study = pair_cell('12x6', 'realistic', 'T4', 'matrix', 'secant', [-1.0] * 5 + [-1.5] * 10 + [-2.0] * 5)
        audit = pair_cell('12x6', 'realistic', 'T4', 'matrix', 'secant', audit_diffs, seeds=list(range(90, 90 + len(audit_diffs))))
        v, doc = self.verdicts(study, audit)
        cid = self.kinds(doc, 'pair', reading='worst_over_db')[0]['id']
        return v[cid]['verdict']

    def test_holds_weaker_does_not_hold(self):
        self.assertEqual(self.one([-1.5] * 10), 'holds')
        self.assertEqual(self.one([-0.6] * 10), 'weaker')
        self.assertEqual(self.one([-0.15] * 10), 'does not hold')       # under the threshold
        self.assertEqual(self.one([+1.5] * 10), 'does not hold')        # the other sign

    def test_not_tested_and_late(self):
        study = pair_cell('12x6', 'realistic', 'T4', 'matrix', 'secant', [-1.0] * 20)
        doc, cp = self.claims(study)
        run('add', claims=cp, text='a claim in words', stamp='2026-11-25 12:00:00 MST')
        ap = os.path.join(self.d, 'audit.csv')
        mc.write_csv(ap, pair_cell('6x6', 'strong', 'T4', 'matrix', 'secant', [-1.0] * 10))
        out = os.path.join(self.d, 'v.csv')
        run('check', claims=cp, readings=ap, out=out)
        v = list(csv.DictReader(open(out)))
        self.assertTrue(all(r['verdict'] == 'not tested' for r in v if r['kind'] != 'late'))
        late = [r for r in v if r['kind'] == 'late']
        self.assertEqual(late[0]['verdict'], 'not judged')
        self.assertEqual(json.load(open(cp))['late'][0]['written'], '2026-11-25 12:00:00 MST')

    def test_level_verdicts(self):
        study = [row('6x6', 'strong', s, 'T1', 'matrix', exc=-1.0 - 0.1 * s) for s in range(20)]
        for audit_exc, want in ((-1.5, 'holds'), (-0.2, 'weaker'), (0.5, 'does not hold')):
            audit = [row('6x6', 'strong', s, 'T1', 'matrix', exc=audit_exc) for s in range(90, 100)]
            v, doc = self.verdicts(study, audit)
            for c in self.kinds(doc, 'level'):
                self.assertEqual(v[c['id']]['verdict'], want, (audit_exc, c['line']))


class TestDeterminism(Tmp):
    def study(self):
        rnd = random.Random(7)
        rows = []
        for system in ('12x6', '6x6'):
            for kind in ('linear', 'realistic', 'strong'):
                for s in range(20):
                    for test, laws in (('T3', ('matrix', 'paired', 'ratio')), ('T4', mc.LAW_ORDER)):
                        for law in laws:
                            rows.append(row(system, kind, s, test, law, exc=rnd.uniform(-3, 2),
                                            iec_share=rnd.random(), worst_over_db=rnd.gauss(1, 2), completion=rnd.choice((1.0, rnd.random()))))
        return rows

    def test_identical_inputs_identical_outputs(self):
        rows = self.study()
        _, a = self.claims(rows, sub='one')
        shuffled = list(rows)
        random.Random(3).shuffle(shuffled)
        _, b = self.claims(shuffled, sub='two')
        # the readings files differ in row order only, so their SHA-256 differs; everything else must be equal
        da, db = json.load(open(a)), json.load(open(b))
        da['readings'].pop('sha256'); db['readings'].pop('sha256')
        self.assertEqual(da, db)
        _, c = self.claims(rows, sub='three')
        self.assertTrue(filecmp.cmp(a, c, shallow=False))       # the same readings file: byte-identical claims
        self.assertGreater(len(da['claims']), 0)

    def test_never_rewritten(self):
        rows = self.study()
        _, a = self.claims(rows)
        with self.assertRaises(SystemExit):
            run('claims', readings=os.path.join(self.d, 'readings.csv'), out=a)


class TestReadingsFromCards(Tmp):
    def card(self, case_id, outcome='completed', scored=True, share=0.9, worst=1.25, coverage=0.5, exc=-0.3,
             version='hertzforge'):
        c = dict(run=dict(case_id=case_id, version=version), completion=dict(outcome=outcome, coverage=coverage),
                 scored=scored)
        if scored:
            c['amplitude'] = {k: dict(share_inside=share, worst_excursion_db=exc) for k in mc.LINES}
            c['resonances'] = dict(worst_db=worst)
        return c

    def test_rows(self):
        runs = os.path.join(self.d, 'runs')
        cards = {'frame-12x6-realistic_s32/T4__matrix': self.card('frame-12x6-realistic_s32__T4__matrix', coverage=1.0),
                 'frame-6x6-strong_hard_s03/T1__matrix': self.card('frame-6x6-strong_hard_s03__T1__matrix', outcome='aborted'),
                 'frame-24x12_s01/T3__paired': self.card('frame-24x12_s01__T3__paired', outcome='aborted', scored=False, coverage=0.0)}
        for p, c in cards.items():
            os.makedirs(os.path.join(runs, p))
            json.dump(c, open(os.path.join(runs, p, 'card.json'), 'w'))
        out = os.path.join(self.d, 'r.csv')
        run('readings', runs=runs, out=out)
        rows = {r['structure']: r for r in mc.load_readings(out)}
        a = rows['frame-12x6-realistic_s32']
        self.assertEqual((a['system'], a['kind'], a['seed'], a['completion'], a['iec_share']), ('12x6', 'realistic', 32, 1.0, 0.9))
        b = rows['frame-6x6-strong_hard_s03']
        self.assertEqual((b['kind'], b['completion']), ('strong_hard', 0.5))
        self.assertAlmostEqual(b['iec_share'], 0.45)            # share inside x coverage for an aborted run
        self.assertEqual(b['worst_over_db'], 1.25)
        u = rows['frame-24x12_s01']
        self.assertEqual((u['kind'], u['iec_share'], u['completion']), ('linear', 0.0, 0.0))
        self.assertNotEqual(u['worst_over_db'], u['worst_over_db'])   # NaN: missing
        self.assertEqual(u['excursion_iec_check_point'], float('inf'))

    def test_two_cards_for_one_run_refused(self):
        runs = os.path.join(self.d, 'runs')
        for b in ('a', 'b'):
            os.makedirs(os.path.join(runs, b))
            json.dump(self.card('frame-6x6_s00__T4__matrix'), open(os.path.join(runs, b, 'card.json'), 'w'))
        with self.assertRaises(SystemExit):
            run('readings', runs=runs, out=os.path.join(self.d, 'r.csv'))

    def test_one_run_in_each_version_is_two_runs(self):
        runs = os.path.join(self.d, 'runs')
        for b, v in (('a', 'hertzforge'), ('b', 'python')):
            os.makedirs(os.path.join(runs, b))
            json.dump(self.card('frame-6x6_s00__T1__matrix', version=v), open(os.path.join(runs, b, 'card.json'), 'w'))
        out = os.path.join(self.d, 'r.csv')
        run('readings', runs=runs, out=out)
        self.assertEqual(sorted(r['version'] for r in mc.load_readings(out)), ['hertzforge', 'python'])

    def test_a_card_without_its_version_is_refused(self):
        runs = os.path.join(self.d, 'runs', 'a')
        os.makedirs(runs)
        c = self.card('frame-6x6_s00__T1__matrix')
        del c['run']['version']
        json.dump(c, open(os.path.join(runs, 'card.json'), 'w'))
        with self.assertRaises(ValueError):
            run('readings', runs=os.path.dirname(runs), out=os.path.join(self.d, 'r.csv'))


class TestVersions(Tmp):
    """K3 runs in the Python version only: it is paired with the Python matrix, never with the Hertzforge one, and its
    claims are marked as admitted after the screen."""

    def rows(self):
        clear = [-1.0] * 16 + [0.0] * 4
        rows = pair_cell('6x6', 'realistic', 'T1', 'matrix', 'k3', clear, version='python')      # Python matrix - K3
        # the Hertzforge matrix on the same cell, far from both (a cross-version pair would be a claim)
        rows += [row('6x6', 'realistic', s, 'T1', 'matrix', version='hertzforge', worst_over_db=BASE['worst_over_db'] + 5.0)
                 for s in range(20)]
        return rows

    def test_k3_pairs_with_the_python_matrix_only(self):
        doc, _ = self.claims(self.rows())
        pairs = self.kinds(doc, 'pair')
        self.assertEqual([(c['version'], c['law'], c['against'], c['reading']) for c in pairs],
                         [('python', 'matrix', 'k3', 'worst_over_db')])
        self.assertTrue(pairs[0]['text'].startswith('Python version. '))
        for c in doc['claims']:
            self.assertFalse(c['kind'] == 'pair' and c['version'] == 'hertzforge', c)
        # levels: one cell a version, the Hertzforge matrix's apart from the Python matrix's
        lv = {(c['version'], c['law']) for c in self.kinds(doc, 'level')}
        self.assertEqual(lv, {('python', 'matrix'), ('python', 'k3'), ('hertzforge', 'matrix')})

    def test_k3_claims_are_marked_admitted_after_the_screen(self):
        doc, _ = self.claims(self.rows())
        for c in doc['claims']:
            names = {c.get('law'), c.get('against')}
            self.assertEqual('admitted_after_screen' in c, 'k3' in names, c['id'])
            if 'k3' in names:
                self.assertIn('Include K3', c['admitted_after_screen'])
                self.assertIn('admitted to the study after the screen', c['text'])

    def test_effects_stay_within_a_version(self):
        rows = []
        for law in ('matrix', 'paired'):
            rows += pair_cell('12x6', 'strong', 'T4', law, 'x', [0.0] * 20)
        rows = [r for r in rows if r['law'] != 'x']
        lin = [dict(r, structure=sid('12x6', 'linear', r['seed']), kind='linear', worst_over_db=r['worst_over_db'] - 2.0)
               for r in rows]
        py = [dict(r, version='python') for r in lin]          # a Python linear cell must not pair with a Hertzforge one
        doc, _ = self.claims(rows + lin + py)
        eff = self.kinds(doc, 'effect')
        self.assertTrue(eff)
        self.assertEqual({c['version'] for c in eff}, {'hertzforge'})
        # with a Python strong cell as well, each version has its own effect, over its own laws
        pys = [dict(r, version='python', worst_over_db=r['worst_over_db'] + 1.0) for r in rows]
        doc, _ = self.claims(rows + lin + py + pys, sub='both')
        eff = {(c['version'], c['study']['median']) for c in self.kinds(doc, 'effect', reading='worst_over_db')}
        self.assertEqual(eff, {('hertzforge', 2.0), ('python', 3.0)})

    def test_the_audit_reads_the_claims_version(self):
        doc, cp = self.claims(self.rows())
        audit = [dict(r) for r in self.rows()]
        for r in audit:                                         # the Hertzforge matrix moved onto K3: no effect on the claim
            if r['version'] == 'hertzforge':
                r['worst_over_db'] = BASE['worst_over_db'] + 1.0
        ap = os.path.join(self.d, 'audit.csv')
        mc.write_csv(ap, audit)
        out = os.path.join(self.d, 'v.csv')
        run('check', claims=cp, readings=ap, out=out)
        v = {r['id']: r for r in csv.DictReader(open(out))}
        pid = self.kinds(doc, 'pair')[0]['id']
        self.assertEqual(v[pid]['verdict'], 'holds')


if __name__ == '__main__':
    unittest.main()
