"""TTNS tangent Chebyshev/Burg and conventional coCheb spectroscopy.

GPU modules load only when requested; saved-figure replay works on CPU.
"""
from .maxent import burg_maximum_entropy, extend_moments, reconstruct_spectrum
from .chebyshev import (chebyshev_moments, tangent_chebyshev,
    tangent_chebyshev_1site, tangent_chebyshev_2site, estimate_tangent_bounds)
from .workflow import (publication_settings, load_reference, prepare_reference,
    run_tangent, reproduce, reproduce_ir, reproduce_emission)

__version__ = '0.1.0'


def cocheb(*args, **kwargs):
    """Lazy GPU entry point; see ttns_spectra.conventional.cocheb and docs/API.md."""
    from .conventional import cocheb as implementation
    return implementation(*args, **kwargs)
