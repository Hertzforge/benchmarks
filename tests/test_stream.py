"""simulate_level played in pieces (state carried) gives the samples of one call (the study, part 3).

Run with run_tests.sh. Catches a lost modal state, envelope, drive history (transport delay, equalizer taps) or output
history (fractional output delays) at a piece boundary.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import reference_model as rm
from test_equalized import three_mode, EQ
from test_stepper import FS


@pytest.mark.parametrize('step', [None, EQ])
def test_pieces_equal_one_call(step):
    st = three_mode(step, extras=True, rules=True)
    d = np.random.default_rng(4).standard_normal((9000, 2)) * 150.0
    one = rm.simulate_level(st, d, FS, keep_modes=True)
    assert one.peak_ratio(st).max() > 1.0
    outs, envs, state = [], [], None
    for a, b in [(0, 1), (1, 7), (7, 2500), (2500, 2501), (2501, 9000)]:
        run = rm.simulate_level(st, d[a:b], FS, keep_modes=True, state=state)
        outs.append(run.outputs); envs.append(run.env); state = run.state
    y, e = np.vstack(outs), np.vstack(envs)
    assert np.abs(y - one.outputs).max() <= 1e-12 * np.abs(one.outputs).max()
    assert np.abs(e - one.env).max() <= 1e-12 * np.abs(one.env).max()
    assert np.array_equal(state['peak'], one.peak_envelope) or np.allclose(state['peak'], one.peak_envelope, rtol=1e-12, atol=0)
