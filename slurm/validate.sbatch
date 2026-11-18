#!/usr/bin/env bash
#SBATCH --job-name=ttns-validation
#SBATCH --partition=4V100,4A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/validate-%j.out
#SBATCH --error=slurm/validate-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 RENO_GPU=0 RENO_REQUIRE_GPU=1 RENO_LOG_LEVEL=30
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TTNS_GPU_TEST=1
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -u tests/validate_core.py
"$PYTHON_BIN" -u tests/dense_oracle.py
"$PYTHON_BIN" -u tests/validate_bounds.py
"$PYTHON_BIN" -u tests/validate_two_site_models.py
