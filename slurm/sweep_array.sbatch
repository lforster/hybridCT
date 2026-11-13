#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH -G 1
#SBATCH --job-name=mimo40_sweep
#SBATCH --array=0-39
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x_%A_%a.out



# One sweep trial per array task: sbatch --export=ALL,TAG=nadir slurm/sweep_array.sbatch
# Near the time limit Lightning saves hpc_ckpt_*.ckpt and requeues the task; the rerun resumes it.
# Finished trials are skipped, so resubmitting the array is safe.
set -euo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$MIMO"
srun python pyct/train_cloud.py -y "config/sweep_40m/config_${TAG:?}_mimo_sweep.yaml" \
    --sweep local --trial-index "${SLURM_ARRAY_TASK_ID}"





