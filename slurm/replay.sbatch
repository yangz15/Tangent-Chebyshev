#!/usr/bin/env bash
#SBATCH --job-name=ttns-replay
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm/replay-%j.out
#SBATCH --error=slurm/replay-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -c 'from ttns_spectra import reproduce_ir, reproduce_emission; reproduce_ir(mode="replay", output="outputs/replay"); reproduce_emission(mode="replay", output="outputs/replay")'
