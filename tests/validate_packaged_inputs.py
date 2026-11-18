"""Check the installed molecular input and construct the complete CPU model."""
import hashlib
import json
import numpy as np
from ttns_spectra.paths import DATA
from ttns_spectra.molecular import CH_DIMS, ch3cn, ch_dipoles

path = DATA / 'molecular/ch3cn/vib.out'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '047f19b153d23dab2c8a0cfdbc60d310da0530fd96a82a57a595ffd5a41a3c73'
dipoles = ch_dipoles()
assert set(dipoles) == set('xyz')
assert all(terms and all(np.isfinite(op.factor) for op in terms) for terms in dipoles.values())
problem = ch3cn()
assert [basis.nbas for basis in problem.model.basis] == CH_DIMS
assert len(problem.model.ham_terms) > 24
assert all(np.isfinite(op.factor) for op in problem.model.ham_terms)
print(json.dumps(dict(status='passed',modes=len(CH_DIMS),hamiltonian_terms=len(problem.model.ham_terms),dipole_terms={a:len(v) for a,v in dipoles.items()})))
