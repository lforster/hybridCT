# Sourced by every slurm/*.sbatch script. Edit for your cluster.
# module load cuda/12.1            # if needed
# source ~/miniforge3/etc/profile.d/conda.sh && conda activate hybridct

CONDA_ENV=${CONDA_ENV:-hybridct}
set +u                              # conda's activate scripts reference unset variables
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"   # makes `conda activate` work in a non-interactive batch shell
else
  source ~/miniforge3/etc/profile.d/conda.sh   # or ~/miniconda3/..., ~/anaconda3/..., or `module load anaconda`
fi
conda activate "$CONDA_ENV"
set -u
echo "[env] python: $(command -v python)  env: ${CONDA_DEFAULT_ENV:-none}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export WANDB_MODE=${WANDB_MODE:-offline}   # compute nodes often have no internet
export MPLBACKEND=Agg
REPO=${REPO:-$SLURM_SUBMIT_DIR}
MIMO=$REPO/emulator/mimo
mkdir -p "$REPO/logs"

