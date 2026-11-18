#!/usr/bin/env bash
#SBATCH --job-name=ttns-notebook
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:V100-SXM2:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/notebook-%j.out
#SBATCH --error=slurm/notebook-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1 RENO_GPU=0 RENO_REQUIRE_GPU=1 RENO_LOG_LEVEL=30
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export TTNS_RUN_MODE=fresh
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" -u scripts/execute_notebook.py
