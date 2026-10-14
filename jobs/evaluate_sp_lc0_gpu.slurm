#!/usr/bin/env bash
# evaluate_sp for Lc0 on a GPU node: one array task per block of positions,
# one GPU each.
#
# Submit from the repository root (create logs/ first):
#   sbatch jobs/evaluate_sp_lc0_gpu.slurm
#
# With POSITIONS_PER_TASK=40 the 960 positions need array indices 0-23.
# Finished positions are skipped, so resubmitting the same job resumes it.
# Adjust --time and POSITIONS_PER_TASK once the pilot has measured search
# times on the GPU.

#SBATCH --job-name=c960-evaluate-sp-lc0
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-23
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/evaluate_sp_%x_%A_%a.out

set -euo pipefail

ENGINE="${ENGINE:-lc0}"
POSITIONS_PER_TASK="${POSITIONS_PER_TASK:-40}"

# Run-time environment: the CUDA libraries Lc0 was built against, and the
# project's Python environment. Override with --export=ALL,MODULES="...".
MODULES="${MODULES:-CUDA/12.2.2}"
# Module and conda scripts can reference unset variables, so relax `set -u`.
set +u
module load ${MODULES}
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-chess960}"
set -u

START=$(( SLURM_ARRAY_TASK_ID * POSITIONS_PER_TASK ))
END=$(( START + POSITIONS_PER_TASK - 1 ))
if (( START > 959 )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no positions; exiting."
    exit 0
fi
(( END > 959 )) && END=959

echo "engine=${ENGINE} positions=${START}-${END} host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true
python code/evaluate_sp.py run --engine "${ENGINE}" --positions "${START}-${END}"
