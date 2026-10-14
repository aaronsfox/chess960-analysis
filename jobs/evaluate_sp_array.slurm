#!/usr/bin/env bash
# evaluate_sp on a SLURM cluster: one array task per block of positions, one CPU each.
#
# For the CPU engines (Stockfish, Caissa, PlentyChess); Lc0 uses
# jobs/evaluate_sp_lc0_gpu.slurm. Submit from the repository root, e.g.
#   sbatch --export=ALL,ENGINE=stockfish jobs/evaluate_sp_array.slurm
#   sbatch --export=ALL,ENGINE=stockfish,MULTIPV=5 jobs/evaluate_sp_array.slurm
#
# With POSITIONS_PER_TASK=40 the 960 positions need array indices 0-23.
# Stockfish takes ~4 minutes per position (budgets up to 1e8 at ~0.5M
# nodes/s), so a task of 40 positions takes ~2.5 hours.
# Finished positions are skipped, so resubmitting the same job resumes it.
#
# Adjust --time per engine once the pilot has measured search times.

#SBATCH --job-name=c960-evaluate-sp
#SBATCH --array=0-23
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/evaluate_sp_%x_%A_%a.out

set -euo pipefail

ENGINE="${ENGINE:?set ENGINE, e.g. --export=ALL,ENGINE=stockfish}"
MULTIPV="${MULTIPV:-1}"
POSITIONS_PER_TASK="${POSITIONS_PER_TASK:-40}"

# Run-time environment: any modules the engine binaries need (e.g. the GCC
# they were built with, if they report GLIBCXX errors), and the project's
# Python environment. Set with --export=ALL,MODULES="GCC/10.2.0".
MODULES="${MODULES:-}"
# Module and conda scripts can reference unset variables, so relax `set -u`.
set +u
if [[ -n "${MODULES}" ]]; then
    module load ${MODULES}
fi
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

echo "engine=${ENGINE} multipv=${MULTIPV} positions=${START}-${END} host=$(hostname)"
python code/evaluate_sp.py run --engine "${ENGINE}" --multipv "${MULTIPV}" \
    --positions "${START}-${END}"
