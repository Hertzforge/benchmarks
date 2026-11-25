"""Reference of the crate's `paired` law (each control paired with drive c; amplitude by the ratio
raised to the loop-speed gain; phase kept; engage from the pair's own path)."""
import numpy as np
from .interface import Law, TickOut
from .laws_baselines import loop_gain

class PairedRef(Law):
    name = 'paired'
    def __init__(self, setup, g=0.5, gain_rule='bandwidth', beta=0.5):
        super().__init__(setup); self.g, self.rule, self.beta = g, gain_rule, beta
        self.k = min(setup.n, setup.m)
    def engage(self, x):
        H = self.setup.ident.H0_at(x.f); D = np.zeros(self.setup.m, complex)
        for c in range(self.k):
            D[c] = abs(x.R[c]) / max(abs(H[c, c]), 1e-300)
        return TickOut(D)
    def tick(self, x):
        g = 0.0 if x.hold == 'engage' else loop_gain(self.setup, x.f, self.g, self.rule, self.beta)
        D = np.zeros(self.setup.m, complex)
        for c in range(self.k):
            a = abs(x.C[c]); ratio = abs(x.R[c]) / a if a > 1e-300 else 1.0
            D[c] = x.D_in_force[c] * ratio ** g
        return TickOut(D)
