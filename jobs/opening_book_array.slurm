#!/usr/bin/env bash
# Build the self-play opening book on a SLURM cluster: one array task per
# block of positions, one CPU each.
#
# Submit from the repository root (create logs/ first):
#   sbatch jobs/opening_book_array.slurm
# then, once every task has finished:
#   python code/opening_book_generator.py collect
#
# For a list of positions instead of all 960 (e.g. the pilot), set POSITIONS
# to anything --positions accepts; task i then takes the i-th block of
# POSITIONS_PER_TASK positions from the list:
#   sbatch --array=0-29 --export=ALL,POSITIONS=@config/pilot_positions.txt,POSITIONS_PER_TASK=1 \
#       jobs/opening_book_array.slurm
#
# Settings come from config/opening_book.toml. With POSITIONS_PER_TASK=10
# the 960 positions need array indices 0-95. Expect very roughly 10 minutes
# per position with the default search budgets (the pilot measures it; the
# log prints each position's time). If you change POSITIONS_PER_TASK, change
# --array to match (0 to 960/POSITIONS_PER_TASK - 1).
# Finished positions are skipped, so resubmitting the same job resumes it.

#SBATCH --job-name=c960-opening-book
#SBATCH --array=0-95
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --output=logs/opening_book_%A_%a.out

set -euo pipefail

POSITIONS_PER_TASK="${POSITIONS_PER_TASK:-10}"

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
TASK_POSITIONS=$(python - "${POSITIONS:-0-959}" "${SLURM_ARRAY_TASK_ID}" "${POSITIONS_PER_TASK}" <<'EOF'
import sys
sys.path.insert(0, "code")
from utils import parse_position_spec
ids = parse_position_spec(sys.argv[1])
task, size = int(sys.argv[2]), int(sys.argv[3])
print(",".join(str(i) for i in ids[task * size:(task + 1) * size]))
EOF
)
if [[ -z "${TASK_POSITIONS}" ]]; then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no positions; exiting."
    exit 0
fi

echo "opening book positions=${TASK_POSITIONS} host=$(hostname)"
python code/opening_book_generator.py build --positions "${TASK_POSITIONS}"
