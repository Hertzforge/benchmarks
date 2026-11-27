"""Reference of the crate's `paired` law (each control paired with drive c; amplitude by the ratio
raised to the loop-speed gain; phase kept; engage from the pair's own path)."""
import numpy as np
from .interface import Law, TickOut
from .laws_baselines import loop_gain

FIRST_DRIVE_ROUNDS = 200          # the crate's FIRST_DRIVE_ROUNDS

class PairedRef(Law):
    name = 'paired'
    def __init__(self, setup, g=0.5, gain_rule='bandwidth', beta=0.5):
        super().__init__(setup); self.g, self.rule, self.beta = g, gain_rule, beta
        self.k = min(setup.n, setup.m)
    def engage(self, x):
        """The first drive (the crate's `engage`, since its be51b666): each pair's reference over its own path, then
        the pairs' own update run on the identification at the tone (C = H D, each drive times the square root of
        its control's wanted over predicted size) until every control is within 0.01 dB or FIRST_DRIVE_ROUNDS
        rounds have run; the plain start is kept if the rounds leave a control further off than it."""
        H = self.setup.ident.H0_at(x.f); D = np.zeros(self.setup.m, complex)
        r = np.abs(x.R[:self.k])
        for c in range(self.k):
            D[c] = r[c] / max(abs(H[c, c]), 1e-300)

        def worst(d):
            y = np.abs(H[:self.k] @ d)
            return float(np.max(np.abs(20.0 * np.log10(np.maximum(y, 1e-300) / np.maximum(r, 1e-300)))))

        plain = D.copy()
        d = plain.copy()
        for _ in range(FIRST_DRIVE_ROUNDS):
            if worst(d) < 0.01:
                break
            y = np.abs(H[:self.k] @ d)
            v = d.copy()
            for c in range(self.k):
                if y[c] > 1e-300:
                    v[c] = d[c] * np.sqrt(r[c] / y[c])
            d = v
        if np.all(np.isfinite(d)) and worst(d) < worst(plain):
            D = d
        return TickOut(D)
    def tick(self, x):
        g = 0.0 if x.hold == 'engage' else loop_gain(self.setup, x.f, self.g, self.rule, self.beta)
        D = np.zeros(self.setup.m, complex)
        for c in range(self.k):
            a = abs(x.C[c]); ratio = abs(x.R[c]) / a if a > 1e-300 else 1.0
            D[c] = x.D_in_force[c] * ratio ** g
        return TickOut(D)
