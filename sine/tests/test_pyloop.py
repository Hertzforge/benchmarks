"""The suite's Python laws held to the shipped Rust laws' parity vectors (tests/data, copied from the Hertzforge
repository): python3 -m unittest discover -s sine/tests -v. Each check is shown to fail on a law it must not match."""
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from pyloop import parity  # noqa: E402
from pyloop import tone as T  # noqa: E402
import json  # noqa: E402

DATA = os.path.join(HERE, 'data')


class TestParity(unittest.TestCase):
    def test_every_law_and_schedule_matches_the_vectors(self):
        with np.errstate(all='ignore'):
            res = parity.check_all(DATA)
        self.assertEqual(len(res), 30)
        for k, (worst, n) in res.items():
            self.assertLess(worst, 1e-9, k)

    def test_a_different_law_does_not_match(self):
        lp = json.load(open(os.path.join(DATA, 'laws_parity.json')))
        mp = json.load(open(os.path.join(DATA, 'matrix_parity.json')))
        st, setup = parity.build(mp['lin4x4'])
        steps = T.steps(40.0, 56.0, 12.0, 0.4)
        cls, prm = parity.LAWS['secant_plain']
        with np.errstate(all='ignore'):
            w, _ = parity.compare(parity.go(st, setup, steps, cls, prm), lp['secant_factor/steps'])
        self.assertGreater(w, 0.1)
        cls, prm = parity.LAWS['paired']
        with np.errstate(all='ignore'):
            w, _ = parity.compare(parity.go(st, setup, steps, cls, prm), lp['steepest_descent/steps'])
        self.assertGreater(w, 0.1)

    def test_the_shipped_matrix_update_differs_where_the_vectors_cannot_see(self):
        # the vectors' reference is flat, so the shipped matrix's backward carry and seen reference are no-ops there;
        # on a sloped reference the two updates part
        from pyloop.controller import Config, Guards, run
        from pyloop.matrix_ref import RefMatrix
        from pyloop.matrix_shipped import ShippedMatrix
        mp = json.load(open(os.path.join(DATA, 'matrix_parity.json')))
        st, setup = parity.build(mp['lin4x4'])
        g = setup.ident.grid_hz
        setup.ref_grid = ((g / 60.0) ** 2)[:, None] * np.ones((1, st.n)) + 0j
        gd = Guards(change_caps=False, ceiling_on=False, notch=False, divergence_on=False, abort_db=None,
                    limit_abort_db=None, first_drive_db=300.0)
        cfg = Config(fs=setup.fs, dt_c=setup.dt_c, tracking=setup.tracking, fast=True, guards=gd)
        out = {}
        for cls in (RefMatrix, ShippedMatrix):
            with np.errstate(all='ignore'):
                r = run(st, T.Tone([T.sweep(60.0, 85.0, 1.0)], setup.fs), cls, {}, setup, cfg,
                        lambda f: (f / 60.0) ** 2 * np.ones(st.n, complex), record_ratio=False)
            out[cls.__name__] = np.asarray(r.arrays['D'])
        a, b = out['RefMatrix'], out['ShippedMatrix']
        self.assertEqual(a.shape, b.shape)
        self.assertGreater(np.max(np.linalg.norm(a - b, axis=1) / np.linalg.norm(a, axis=1)), 1e-3)


if __name__ == '__main__':
    unittest.main()
