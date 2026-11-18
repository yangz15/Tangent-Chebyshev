"""Create the single user-facing notebook, with exactly two calculation blocks."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
nb.metadata.update(kernelspec=dict(name='python3',display_name='Python 3',language='python'))
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
nb.cells=[
md('''# TTNS Chebyshev spectroscopy

One-site and enlarged two-site tangent Chebyshev propagation, Burg spectral
maximum entropy, and optimized conventional coCheb. Functions live in
`src/ttns_spectra`; this notebook contains example calls, following the layout
of [Restart-Lanczos-MPS](https://github.com/yangz15/Restart-Lanczos-MPS).

The two blocks reproduce the selected **CH3CN IR (D16)** and **298 K aggregate
thermal emission (D32)** comparisons. Fresh mode regenerates all three response
curves from the bundled prepared states. It reuses the published endpoint scales
and high-accuracy reference curves. Setup and thermal preparation are excluded
from response timings. Runtime varies with hardware and load.

`MODE="replay"` rebuilds figures on CPU from archived arrays and timings; it
does not claim a fresh simulation. The delivered notebook records its actual mode.
'''),
md('''## Main functions

| Function | Definition and purpose |
|---|---|
| `load_reference` / `prepare_reference` | Load checked states / run DMRG or purified thermal preparation |
| `tangent_chebyshev_1site` | Fixed one-site projector, matrix-free recurrence, doubled moments |
| `tangent_chebyshev_2site` | Complete one-site space plus both-end-discarded edge sectors |
| `estimate_tangent_bounds` | Separate matrix-free endpoint setup |
| `burg_maximum_entropy` / `extend_moments` | Covariance/Levinson maximum entropy and scalar continuation |
| `reconstruct_spectrum` | Gaussian quadrature or rational Lorentzian reconstruction |
| `cocheb` | Whole-target variational compression, actual S/H, canonical reduction |
| `run_tangent` | Model-level tangent + Burg calculation, with `two_site=True/False` |
| `reproduce_ir` / `reproduce_emission` | The complete two comparison blocks below |

Signatures, units, returns and examples are in [docs/API.md](docs/API.md).
The recurrence and Burg equations are in [docs/ALGORITHMS.md](docs/ALGORITHMS.md).
"Berg" is interpreted as **Burg's spectral entropy-rate formulation**, solved
from cosine covariances, not the time-series forward/backward residual estimator.
'''),
code('''from pathlib import Path
import os
import sys

ROOT = Path.cwd()
if not (ROOT / "src" / "ttns_spectra").exists():
    raise RuntimeError("Start this notebook from the extracted package root")
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("RENO_GPU", "0")
os.environ.setdefault("RENO_LOG_LEVEL", "30")

from IPython.display import display, Image, Markdown
from ttns_spectra import reproduce_ir, reproduce_emission, publication_settings

MODE = os.environ.get("TTNS_RUN_MODE", "fresh")
OUTPUT = ROOT / "outputs"
print("Execution mode:", MODE)
print("Slurm job:", os.environ.get("SLURM_JOB_ID", "local workstation"))
print("IR settings:", publication_settings("IR"))
print("Emission settings:", publication_settings("fast"))'''),
md('''## Block 1 — CH3CN infrared spectrum and runtime

Binary TTNS D16; sum of x/y/z. One-site TaChe: 250 projected actions / 500
computed moments. coCheb: 500 compressed vectors (499 recurrence updates).
One-site TDVP: dt=320 au, 175 steps, T=56000 au. Gaussian sigma=10 cm^-1.
The independent reference is the saved MPS D50 two-site TDVP curve.
'''),
code('''ir = reproduce_ir(mode=MODE, output=OUTPUT)
display(Image(filename=str(Path(ir["figure"]).with_suffix(".png"))))
display(Markdown("| Method | Response seconds | Published seconds | Relative spectrum difference |\\n"
                 "|---|---:|---:|---:|\\n" + "\\n".join(
    f"| {r['method']} | {r['seconds']:.3f} | {r['published_seconds']:.3f} | {r['relative_difference_from_publication']:.3e} |"
    for r in ir["rows"])))'''),
md('''## Block 2 — Aggregate thermal emission and runtime

Open L10 aggregate, 298 K, J1=700 cm^-1, J2=87.5 cm^-1. Fast vibration:
1400 cm^-1, Huang–Rhys S=1, five local levels; binary TTNS D32. One-site
TaChe: 75 projected actions / 150 moments; coCheb: 150 vectors. One-site
TDVP: scaled dt=0.5, 60 steps, T=30. Lorentzian HWHM=0.1 scaled units
(140 cm^-1). The independent reference is the saved D64 chain two-site TDVP.
'''),
code('''emission = reproduce_emission(mode=MODE, output=OUTPUT)
display(Image(filename=str(Path(emission["figure"]).with_suffix(".png"))))
display(Markdown("| Method | Response seconds | Published seconds | Relative spectrum difference |\\n"
                 "|---|---:|---:|---:|\\n" + "\\n".join(
    f"| {r['method']} | {r['seconds']:.3f} | {r['published_seconds']:.3f} | {r['relative_difference_from_publication']:.3e} |"
    for r in emission["rows"])))'''),
md('''## Using the two-site method

The requested two PDFs compare one-site methods. To run the enlarged response
space with the same prepared aggregate, replace its TaChe call with:

```python
from ttns_spectra import load_reference, run_tangent
problem = load_reference("fast")
two_site = run_tangent(problem, "response", case="fast", actions=75, two_site=True)
```

`tangent_chebyshev_2site` exposes the lower-level operator API. No missing helper
definitions or old project checkout are needed. Dense validation covers both
one-site and two-site spaces, charged and neutral sources, projector action,
Hermiticity, nesting, source projection and moments.

Equal input coefficient counts are **not equal accuracy**. Extra Burg moments
are predictions, not new Hamiltonian applications. The supplied coCheb is the
optimized exact-S/H version selected in the existing study; "best" is not a
claim of global optimality. Detailed numerical evidence appears in `outputs`
and [docs/VALIDATION.md](docs/VALIDATION.md).
''')]
nbf.write(nb,root/'main.ipynb')
print('Wrote',root/'main.ipynb')
