#!/usr/bin/env bash
# Aim 1 on a SLURM cluster: one array task per block of positions, one CPU each.
#
# Submit from the repository root, e.g.
#   sbatch --export=ALL,ENGINE=stockfish jobs/aim1_array.slurm
#   sbatch --export=ALL,ENGINE=stockfish,MULTIPV=5 jobs/aim1_array.slurm
#
# With POSITIONS_PER_TASK=5 the 960 positions need array indices 0-191.
# Finished positions are skipped, so resubmitting the same job resumes it.
#
# Adjust --time per engine once the pilot has measured search times.

#SBATCH --job-name=c960-aim1
#SBATCH --array=0-191
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/aim1_%x_%A_%a.out

set -euo pipefail

ENGINE="${ENGINE:?set ENGINE, e.g. --export=ALL,ENGINE=stockfish}"
MULTIPV="${MULTIPV:-1}"
POSITIONS_PER_TASK="${POSITIONS_PER_TASK:-5}"

# Site-specific environment. Replace with your cluster's modules, e.g.
#   module load python/3.12
# and activate the project environment:
#   source .venv/bin/activate

START=$(( SLURM_ARRAY_TASK_ID * POSITIONS_PER_TASK ))
END=$(( START + POSITIONS_PER_TASK - 1 ))
if (( START > 959 )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no positions; exiting."
    exit 0
fi
(( END > 959 )) && END=959

echo "engine=${ENGINE} multipv=${MULTIPV} positions=${START}-${END} host=$(hostname)"
python code/aim1_evaluate.py run --engine "${ENGINE}" --multipv "${MULTIPV}" \
    --positions "${START}-${END}"
