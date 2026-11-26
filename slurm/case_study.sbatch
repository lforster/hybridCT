#!/bin/bash
#SBATCH --job-name=mimo40_case
#SBATCH --array=0-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Four case studies in parallel (CPU only; SART runs one reconstruction per allocated core).
set -euo pipefail
# PIXI_ENV unset -> default env (same packages as cuda); env.sh sets CONDA_OVERRIDE_CUDA for CPU nodes
source "${REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
CFGS=(single_cal_truthmask single_cal_hullmask multi_cal_truthmask multi_cal_hullmask)
cd "$REPO/reconstruction"
# finished case studies (same config, inputs not newer) are skipped; FORCE=1 redoes them
python cloud_case_study_uq.py -y "config/case_study_40m_${CFGS[$SLURM_ARRAY_TASK_ID]}.yaml" \
    $([ "${FORCE:-0}" = 1 ] || echo --skip-done)
