#!/bin/bash
#SBATCH --job-name=mimo40_post
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# deploy -> recalibrate -> 4 UQ evaluations -> calibrated merged 30 h predictions
set -euo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$REPO"
for s in deploy recal uq merge; do ./run_pipeline_40m.sh "$s"; done
