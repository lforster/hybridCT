#!/bin/bash
#SBATCH --job-name=mimo40_compare
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$REPO" && ./run_pipeline_40m.sh compare
