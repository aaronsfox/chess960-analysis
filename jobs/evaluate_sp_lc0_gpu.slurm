#!/usr/bin/env bash
# evaluate_sp for Lc0 on a GPU node: one array task per block of positions,
# one GPU each, several Lc0 processes sharing that GPU.
#
# Submit from the repository root (create logs/ first):
#   sbatch jobs/evaluate_sp_lc0_gpu.slurm
#   sbatch --array=0-7%4 jobs/evaluate_sp_lc0_gpu.slurm      # at most 4 GPUs at once
#
# One single-threaded Lc0 search keeps an A100 mostly idle (~5k nodes/s), so
# each task runs PROCS_PER_GPU independent Lc0 processes on its GPU, each on
# its own share of the task's positions. Every search is still a fresh,
# single-threaded, node-limited search, so results don't depend on how many
# processes share the GPU; only the recorded times (time_s, nps,
# search_wall_s) do.
#
# Defaults: POSITIONS_PER_TASK=120 with PROCS_PER_GPU=4, so array 0-7 covers
# the 960 positions with 8 GPUs, 30 positions per process. At ~3.5 min per
# position on an unshared A100 that is ~2 hours per task if sharing costs
# nothing, and longer if the processes slow each other down. If you change
# POSITIONS_PER_TASK, change --array to match (0 to 960/POSITIONS_PER_TASK - 1).
# Finished positions are skipped, so resubmitting the same job resumes it.

#SBATCH --job-name=c960-evaluate-sp-lc0
#SBATCH --partition=dgx
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/evaluate_sp_%x_%A_%a.out

set -euo pipefail

ENGINE="${ENGINE:-lc0}"
POSITIONS_PER_TASK="${POSITIONS_PER_TASK:-120}"
PROCS_PER_GPU="${PROCS_PER_GPU:-4}"

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

echo "engine=${ENGINE} positions=${START}-${END} processes=${PROCS_PER_GPU} host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# Deal the positions out in turn (process 0 gets START, START+P, ...), so each
# process gets a similar mix. Each process logs to its own file.
LOG_BASE="logs/evaluate_sp_${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
pids=()
for (( p = 0; p < PROCS_PER_GPU; p++ )); do
    positions=$(seq -s, $(( START + p )) "${PROCS_PER_GPU}" "${END}")
    [[ -z "${positions}" ]] && continue
    echo "process ${p}: positions ${positions} (log ${LOG_BASE}_p${p}.log)"
    python code/evaluate_sp.py run --engine "${ENGINE}" --positions "${positions}" \
        > "${LOG_BASE}_p${p}.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
if (( status != 0 )); then
    echo "at least one Lc0 process failed; see ${LOG_BASE}_p*.log" >&2
fi
exit "${status}"
