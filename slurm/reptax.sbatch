#!/bin/bash
#SBATCH --job-name=mimo40_reptax
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# Rep_Taxonomy representation analyses of the deployed 40 m networks (after deploy + recal).
#   export   GPU: features of the 10 networks (+ random-weight controls) on a shared pixel sample
#   configs  Rep_Taxonomy configs + run_reptax.sh
#   analyze  paired views (cross-angle, single vs multi, probes, OOD, members, controls), geo ID, spectra
#   summary  figures and tables
# STAGES overrides the list, e.g. sbatch --export=ALL,STAGES="analyze summary",RT_DIR=$HOME/Rep_Taxonomy slurm/reptax.sbatch
# The CPU stages also run without a GPU: sbatch --gres=none --export=ALL,STAGES="analyze summary" ...
set -euo pipefail
source "${REPO:-$SLURM_SUBMIT_DIR}/slurm/env.sh"
cd "$REPO"
export RT_DIR=${RT_DIR:-$HOME/Rep_Taxonomy}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
for s in ${STAGES:-export configs analyze summary}; do ./run_pipeline_40m.sh "reptax_$s"; done
