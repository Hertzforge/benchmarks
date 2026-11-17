"""The generator's frequency bound check (set version 0.2): the audit's structure 24 at the strong size is refused,
and none of the set's 388 problems is."""
import glob
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import frequency_bound as fb

ROOT = os.environ.get('OBS_SET') or os.path.normpath(os.path.join(HERE, '..', '..', '..'))   # the set's root
FIXTURE = os.path.join(HERE, 'structure24_strong_modes.json')


def test_structure_24_strong_is_refused():
    fx = json.load(open(FIXTURE))
    assert len(fx['structures']) == 5
    for name, modes in fx['structures'].items():
        with pytest.raises(ValueError, match='bound'):
            fb.check_modes(modes, name)


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, 'problems')), reason="the set's problems are not here (set OBS_SET)")
def test_none_of_the_388_is_refused():
    folders = sorted(glob.glob(os.path.join(ROOT, 'problems', 'frame-*')))
    assert len(folders) == 388
    top = 0.0
    for f in folders:
        st = json.load(open(os.path.join(f, 'structure.json')))
        top = max(top, fb.check_modes(st['modes'], os.path.basename(f)))
    assert top < fb.SIM_BOUND_HZ


def test_a_linear_mode_uses_its_own_frequency():
    assert fb.highest_frequency(dict(f_hz=100.0)) == 100.0
    assert fb.highest_frequency(dict(f_hz=100.0, level=dict(rise=0.5, rise_cap=2.0, rise_exponent=1.0))) == 200.0
