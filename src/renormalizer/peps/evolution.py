"""First/second-order real- or imaginary-time PEPS simple update."""

import numpy as np
from scipy.linalg import expm

from .peps import positive_int, retained_rank


class SimpleUpdate:
    """Evolve a copy of a PEPS with diagonal bond environments.

    ``max_bond`` controls the PEPS bonds. Measurements use a separate ``chi``.
    Imaginary time rescales local tensors/weights for stability and preserves
    the state only up to a global scalar. Real time retains that scalar.
    The input Hamiltonian is treated as fixed for the lifetime of this object.
    """

    def __init__(self, state, hamiltonian, max_bond, cutoff=1e-12, imag=True, order=2):
        hamiltonian.check_state(state)
        self.max_bond = positive_int(max_bond, "max_bond")
        retained_rank(np.ones(1), None, cutoff)
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.state = state.copy()
        self.hamiltonian = hamiltonian
        self.cutoff, self.imag, self.order = cutoff, bool(imag), order
        self.nsteps, self.time = 0, 0.0
        self.history = []
        self._gate_key, self._gates = None, None

    def step(self, dt):
        """One Lie/Strang step; returns the largest local discarded weight."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        key = (float(dt), self.imag, self.order)
        if key != self._gate_key:
            factor = -dt * (1.0 if self.imag else 1j) / self.order
            self._gates = [(where, expm(factor * mat)) for where, mat in self.hamiltonian.terms()]
            self._gate_key = key
        gates = self._gates if self.order == 1 else self._gates + self._gates[::-1]
        discarded = 0.0
        for where, gate in gates:
            if len(where) == 1:
                self.state.apply_one_site_gate(gate, where[0], renormalize=self.imag)
            else:
                err = self.state.apply_two_site_gate(gate, *where, max_bond=self.max_bond,
                                                     cutoff=self.cutoff, renormalize=self.imag)
                discarded = max(discarded, err)
        self.nsteps += 1
        self.time += dt
        return discarded

    def evolve(self, nsteps, dt, chi=None, measure_every=None):
        """Run steps and append optional energy measurements; return the state.

        No convergence is inferred automatically. Each history entry includes
        the maximum local discarded weight since the last measurement.
        """
        nsteps = positive_int(nsteps, "nsteps")
        if measure_every is not None:
            measure_every = positive_int(measure_every, "measure_every")
        discarded = 0.0
        for i in range(nsteps):
            discarded = max(discarded, self.step(dt))
            if measure_every is not None and ((i + 1) % measure_every == 0 or i == nsteps - 1):
                self.history.append({"step": self.nsteps, "time": self.time,
                                     "energy": self.hamiltonian.energy(self.state, chi),
                                     "max_discarded_weight": discarded})
                discarded = 0.0
        return self.state
