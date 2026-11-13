#!/bin/bash -l
# Submit the whole 40 m pipeline with dependencies (run from the repository root):
#   ./slurm/submit_all.sh                 # everything
#   ./slurm/submit_all.sh --skip-prep     # caches already built
# Add cluster options (partition, account, QoS) via SBATCH_* env vars, e.g.
#   SBATCH_PARTITION=gpu SBATCH_ACCOUNT=myproj ./slurm/submit_all.sh
set -euo pipefail



REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"; mkdir -p logs
TAGS=(n705 n60 n456 n261 nadir 261 456 60 705 multiangle)
[ -f emulator/mimo/config/sweep_40m/config_nadir_mimo_sweep.yaml ] || ./run_pipeline_40m.sh configs
sub() { sbatch --parsable --export=ALL,REPO="$REPO" "$@"; }
sweep_ids=()
for t in "${TAGS[@]}"; do
  dep=()
  if [ "${1:-}" != "--skip-prep" ]; then
    p=$(sub --export=ALL,REPO="$REPO",TAG="$t" slurm/prep.sbatch); dep=(--dependency=afterok:$p)
  fi
  s=$(sub "${dep[@]}" --export=ALL,REPO="$REPO",TAG="$t" slurm/sweep_array.sbatch)
  sweep_ids+=("$s"); echo "$t: sweep $s ${dep[*]}"
done
# afterany: a few failed trials must not block deployment (select_and_deploy ignores them)
post=$(sub --dependency=afterany:$(IFS=:; echo "${sweep_ids[*]}") slurm/post_gpu.sbatch)
case=$(sub --dependency=afterok:$post slurm/case_study.sbatch)
cmp=$(sub --dependency=afterok:$case slurm/compare.sbatch)
echo "post $post -> case $case -> compare $cmp"
