"""The simulator's frequency bound, checked when a structure is generated (set version 0.2).

The simulator refuses a structure one of whose modes can reach the Nyquist frequency of the set's sample rate,
4,096 Hz at 8,192 samples a second. A mode's highest possible frequency is f0 (1 + rise rise_cap^rise_exponent)
(reference_model.max_frequency). The generator as the study ran it did not check this, and one structure of the
audit (structure 24 at the strong size) reached 4,118.6 Hz and was refused by the simulator at run time. This check
makes generation fail instead.
"""

SIM_BOUND_HZ = 4096.0


def highest_frequency(mode):
    """The highest frequency a mode can reach, Hz: its own level rule's ceiling, or its small-signal frequency."""
    lv = mode.get('level')
    if not lv:
        return float(mode['f_hz'])
    return float(mode['f_hz']) * (1.0 + float(lv['rise']) * float(lv['rise_cap']) ** float(lv['rise_exponent']))


def check_modes(modes, what='the structure', bound_hz=SIM_BOUND_HZ):
    """Raise ValueError when any mode can reach bound_hz; return the highest reachable frequency otherwise."""
    top = max(highest_frequency(m) for m in modes)
    if top >= bound_hz:
        worst = max(range(len(modes)), key=lambda k: highest_frequency(modes[k]))
        raise ValueError(f'{what}: mode {worst} can rise to {top:.1f} Hz, at or above the simulator\'s bound of '
                         f'{bound_hz:.0f} Hz; the simulator would refuse it, so it is not generated')
    return top
