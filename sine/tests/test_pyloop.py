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


class TestPairedFirstDrive(unittest.TestCase):
    """The crate's `paired` engage (be51b666): the pairs' own rule run on the identification to the drive it settles
    to, not the plain start of each reference over its own path."""

    def law(self, H):
        from types import SimpleNamespace
        from pyloop.paired_ref import PairedRef
        ident = SimpleNamespace(H0_at=lambda f: H)
        return PairedRef(SimpleNamespace(n=H.shape[0], m=H.shape[1], ident=ident))

    def test_the_first_drive_meets_every_control_on_a_coupled_pair(self):
        H = np.array([[1.0, 0.3 + 0.1j], [0.25 - 0.05j, 0.8]])
        R = np.array([2.0, 1.5], complex)
        from types import SimpleNamespace
        D = self.law(H).engage(SimpleNamespace(f=50.0, R=R)).D
        err = np.abs(20 * np.log10(np.abs(H @ D) / np.abs(R)))
        self.assertLess(err.max(), 0.01)
        plain = np.abs(R) / np.abs(np.diag(H))
        self.assertGreater(np.abs(20 * np.log10(np.abs(H @ plain) / np.abs(R))).max(), 1.0, 'the check cannot fail')


class TestTimeLimit(unittest.TestCase):
    def test_a_stepped_sines_time_limit_is_the_runners(self):
        # the runner's max_test_s for frame-6x6 s00 T3 (20 to 2000 Hz, 12 steps an octave, 5 rungs of 2 s), read from
        # its run file
        from pyloop.harness import test_length_s
        t = {'type': 'stepped', 'band_hz': [20.0, 2000.0], 'steps_per_octave': 12, 'step_s': 1.842905623287886}
        got = 1.5 * (test_length_s(t) + 4 * 2.0) + 60.0
        self.assertAlmostEqual(got, 420.51012706994237, places=9)
        from pyloop.harness import step_grid
        self.assertNotAlmostEqual(1.5 * (len(step_grid(t)) * t['step_s'] + 8.0) + 60.0, 420.51012706994237, places=3,
                                  msg='the check cannot fail')


if __name__ == '__main__':
    unittest.main()
