# Sourced by every slurm/*.sbatch script. Edit for your cluster.
# module load cuda/12.x            # only if your cluster needs it

# --- Locations ---------------------------------------------------------------
REPO=${REPO:-$SLURM_SUBMIT_DIR}                  # hybridCT top directory (holds pixi.toml)
PIXI=${PIXI:-$HOME/.pixi/bin/pixi}
if [ ! -f "$REPO/pixi.toml" ]; then
  echo "[env] no pixi.toml in REPO=$REPO. Submit from the hybridCT top directory or export REPO=/path/to/hybridCT"
  exit 1
fi
if [ ! -x "$PIXI" ]; then
  echo "[env] pixi not found at $PIXI (set PIXI=/path/to/pixi)"
  exit 1
fi

# Keep pixi's cache on node-local disk, not NFS/Lustre/GPFS
export PIXI_CACHE_DIR=${PIXI_CACHE_DIR:-${TMPDIR:-/tmp}/pixi-cache-$USER}

# --- Activate the pixi environment (no install on compute nodes) --------------
# PIXI_ENV unset -> the manifest's default environment; otherwise e.g. cuda, cpu.
# Install it once on the login node first:
#   cd $REPO && CONDA_OVERRIDE_CUDA=12 ~/.pixi/bin/pixi install -e <env>
set +u                                            # activation scripts reference unset variables
ENV_ARGS=()
[ -n "${PIXI_ENV:-}" ] && ENV_ARGS=(-e "$PIXI_ENV")
# The cuda/default envs are locked only for the 'linux-64-cuda' platform, which
# requires the __cuda virtual package. CPU-only compute nodes (no NVIDIA driver)
# don't expose __cuda, so pixi reports "unsupported-platform". Overriding it is
# safe: the package set is identical and torch simply reports CUDA unavailable.
case "${PIXI_ENV:-default}" in
  cpu|full-cpu) ;;
  *) export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12}" ;;
esac
HOOK=$("$PIXI" shell-hook "${ENV_ARGS[@]}" --manifest-path "$REPO/pixi.toml" --as-is) || {
  echo "[env] 'pixi shell-hook' failed (see message above)"
  exit 1
}
eval "$HOOK"
unset HOOK
set -u

# --as-is does not check that the environment exists; fail loudly instead of
# silently falling back to the system python
if [ ! -x "${CONDA_PREFIX:-/nonexistent}/bin/python" ] || \
   [ "$(command -v python)" != "$CONDA_PREFIX/bin/python" ]; then
  echo "[env] pixi env '${PIXI_ENVIRONMENT_NAME:-?}' is not installed at ${CONDA_PREFIX:-?}"
  echo "      On the login node run: cd $REPO && CONDA_OVERRIDE_CUDA=12 $PIXI install -e ${PIXI_ENVIRONMENT_NAME:-default}"
  exit 1
fi
echo "[env] pixi: $(command -v python)  (${PIXI_ENVIRONMENT_NAME})"

# --- Runtime settings ---------------------------------------------------------
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export WANDB_MODE=${WANDB_MODE:-offline}          # compute nodes often have no internet
export MPLBACKEND=Agg
MIMO=$REPO/emulator/mimo
mkdir -p "$REPO/logs"
