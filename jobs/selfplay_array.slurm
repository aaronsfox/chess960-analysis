#!/usr/bin/env bash
# Engine self-play on a SLURM cluster: one array task per block of positions,
# each task running several games at once (one game per physical core).
#
# Build fastchess once on a login node first:
#   bash tools/fastchess/build_linux.sh
# Submit from the repository root (create logs/ first), choosing the engine:
#   sbatch --export=ALL,ENGINE=stockfish jobs/selfplay_array.slurm
#   sbatch --export=ALL,ENGINE=caissa    jobs/selfplay_array.slurm
# then, once every task has finished:
#   python code/selfplay.py collect
#
# Pilot (the 30 positions in config/pilot_positions.txt, one per task):
#   sbatch --array=0-29 --time=01:00:00 \
#       --export=ALL,ENGINE=stockfish,POSITIONS=@config/pilot_positions.txt,POSITIONS_PER_TASK=1 \
#       jobs/selfplay_array.slurm
#
# Settings come from config/selfplay.toml (time control, games per line,
# adjudication). At 10+0.1 a game takes very roughly 25-35 s of one core, so
# a position's 500 games take about 15 minutes on 16 cores and a block of 10
# positions about 2.5-3 hours; the pilot measures it (the log prints each
# position's time). With POSITIONS_PER_TASK=10 the 960 positions need array
# indices 0-95; if you change it, change --array to match.
# Finished positions are skipped, so resubmitting the same job resumes it.
#
# Timing matters for time-control games, so each game gets a whole physical
# core (--hint=nomultithread) and fastchess runs as many games at once as
# the task has CPUs. Don't run other work inside these allocations.

#SBATCH --job-name=c960-selfplay
#SBATCH --array=0-95
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/selfplay_%A_%a.out

set -euo pipefail

ENGINE="${ENGINE:?set ENGINE, e.g. --export=ALL,ENGINE=stockfish}"
POSITIONS_PER_TASK="${POSITIONS_PER_TASK:-10}"
CONCURRENCY="${CONCURRENCY:-${SLURM_CPUS_PER_TASK:-1}}"

# Any modules the engine binary needs at run time, e.g. MODULES="GCC/10.2.0".
MODULES="${MODULES:-}"
set +u
if [[ -n "${MODULES}" ]]; then
    module load ${MODULES}
fi
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-chess960}"
set -u

# The positions for this task: block i of the list (default: all 960).
TASK_POSITIONS=$(python - "${POSITIONS:-0-959}" "${SLURM_ARRAY_TASK_ID}" "${POSITIONS_PER_TASK}" <<'PY'
import sys
sys.path.insert(0, "code")
from utils import parse_position_spec
ids = parse_position_spec(sys.argv[1])
task, size = int(sys.argv[2]), int(sys.argv[3])
print(",".join(str(i) for i in ids[task * size:(task + 1) * size]))
PY
)
if [[ -z "${TASK_POSITIONS}" ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no positions; exiting."
    exit 0
fi

# fastchess's scratch files (openings, in-progress PGN) go to node-local
# storage when the cluster provides it.
WORK_DIR="${TMPDIR:-/tmp}/c960-selfplay-${SLURM_JOB_ID}"
mkdir -p "${WORK_DIR}"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "self-play engine=${ENGINE} positions=${TASK_POSITIONS} concurrency=${CONCURRENCY} host=$(hostname)"
python code/selfplay.py run --engine "${ENGINE}" --positions "${TASK_POSITIONS}" \
    --concurrency "${CONCURRENCY}" --work-dir "${WORK_DIR}"
