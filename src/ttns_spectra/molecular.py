"""CH3CN finite DVR operators, with the original rounded force/dipole data."""
import hashlib
import re
from collections import Counter
from types import SimpleNamespace
import numpy as np
from scipy.special import roots_hermite
from renormalizer import Model, Op
from renormalizer.model.basis import BasisSet
from renormalizer.utils import constant
from .paths import DATA
from ._backend.potential_ch3cn import get_potential_energy_CH3CN
CHEB = DATA / 'molecular'
CH_DIMS = [9,7,9,9,9,9,7,7,9,9,27,27]
CH_UNIT = 1000*constant.cm2au
class SavedDVR(BasisSet):
    is_phonon = True

    def __init__(self, dof, n):
        super().__init__(dof, int(n), [0]*int(n))
        x = roots_hermite(n)[0]
        t = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                t[i, j] = ((-1)**(i-j)*(2/(x[i]-x[j])**2-.5) if i != j
                           else (4*n-1-2*x[i]**2)/6)*.5
        self.matrices = {'I': np.eye(n), 't': t, 'q': np.diag(x)}
        self.matrices.update({f'q^{k}': np.diag(x**k) for k in (2, 3, 4)})

    def op_mat(self, op):
        if isinstance(op, str):
            op = Op(op, self.dof)
        result = np.eye(self.nbas)
        for symbol in op.symbol.split():
            result = result @ self.matrices[symbol]
        return result*op.factor

    def copy(self, new_dof):
        return type(self)(new_dof, self.nbas)


def ch_dipoles():
    lines = (CHEB/'ch3cn/vib.out').read_text().splitlines()
    end = next((i for i, line in enumerate(lines) if 'Analysis of possible Fermi resonances' in line), len(lines))
    start = [i for i, line in enumerate(lines[:end]) if 'IR Intensities' in line][-1]
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?'
    pattern = re.compile(r'^\s*\d+\s+' + r'\s+'.join([f'({number})']*3)
                         + r'\s+\(\s*' + r'\s+'.join([f'({number})']*3) + r'\s*\)\s*$')
    rows = [list(map(float, m.groups())) for line in lines[start:end] if (m := pattern.match(line))]
    if len(rows) < 12:
        raise ValueError('Missing CH3CN transition moments')
    mapping = [9, 8, 5, 2, 10, 11, 6, 7, 3, 4, 0, 1]
    moments = np.asarray(rows[-12:])[mapping, 3:]
    return {axis: [Op('q', f'v{i}', factor=float(np.round(value, 12)))
                   for i, value in enumerate(moments[:, a]) if abs(value) > 1e-14]
            for a, axis in enumerate('xyz')}


def ch3cn():
    """Construct the twelve-mode model; binary reference tensors load separately."""
    w, cubic, quartic = get_potential_energy_CH3CN()
    basis = [SavedDVR(f'v{i}', n) for i, n in enumerate(CH_DIMS)]
    terms = []
    for i, omega in enumerate(w):
        terms.extend([Op('t', f'v{i}', factor=float(np.round(omega, 8))*CH_UNIT),
                      Op('q^2', f'v{i}', factor=float(np.round(.5*omega, 8))*CH_UNIT)])
    for tensor in (cubic, quartic):
        for indices in zip(*np.nonzero(tensor)):
            counts = Counter(indices)
            symbols = ['q' if count == 1 else f'q^{count}' for count in counts.values()]
            terms.append(Op(' '.join(symbols), [f'v{i}' for i in counts],
                            factor=float(np.round(tensor[indices], 8))*CH_UNIT))
    dipole_terms = ch_dipoles()
    width = 156.1923983763416*CH_UNIT
    return SimpleNamespace(model=Model(basis, terms), ground=None,
        dipole_terms=dipole_terms, energy=9.837407426040597*CH_UNIT,
        width=width, center_h=width*1.050616953525222,
        metadata=dict(molecule='CH3CN', modes=12,
            dipole_convention='legacy Tx/Ty/Tz as q coefficients, no sqrt(2) change'))
