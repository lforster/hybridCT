#!/bin/bash
#SBATCH --job-name=mimo40_prep
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Build the data cache of one network: sbatch --export=ALL,TAG=nadir slurm/prep.sbatch
set -euo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$MIMO"
python pyct/train_cloud.py -y "config/sweep_40m/config_${TAG:?}_mimo_sweep.yaml" --prepare-data
