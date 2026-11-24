#!/usr/bin/env bash
# 40 m MIMO sweep -> deploy -> recalibrate -> UQ evaluation -> case study, in order.
# Usage (from the repository root):
#   ./run_pipeline_40m.sh configs      # 1  write the 10 sweep configs
#   ./run_pipeline_40m.sh prep         # 2  build the data caches (run once, before any sweep job)
#   ./run_pipeline_40m.sh sweep [N]    # 3  local sweeps, N trials per network (default 40, sequential)
#   ./run_pipeline_40m.sh deploy       # 4  select the best trial per network and deploy it
#   ./run_pipeline_40m.sh recal        # 5  fit calibration.json per deployed model
#   ./run_pipeline_40m.sh uq           # 6  four UQ evaluations (single/multi x raw/cal)
#   ./run_pipeline_40m.sh merge        # 7  calibrated merged 30 h predictions for the case study
#   ./run_pipeline_40m.sh case         # 8  case studies (truth mask and hull mask)
#   ./run_pipeline_40m.sh compare      # 9  side-by-side tables and figure
#   ./run_pipeline_40m.sh reptax       # 10 Rep_Taxonomy representation analyses (export, configs, analyze, summary;
#                                      #    each also runs alone as reptax_<step>; RT_DIR=/path/to/Rep_Taxonomy)
#   ./run_pipeline_40m.sh all          #    everything after 'configs' in sequence
#   ./run_pipeline_40m.sh resume       #    recal -> uq -> merge -> case -> compare -> reptax, redoing only
#                                      #    what is missing or older than its inputs (e.g. after adding files)
# recal / merge / case skip finished work (existing calibration.json, up-to-date merged files and
# case studies); uq reuses its prediction caches. FORCE=1 redoes everything.
# For the sweep on a cluster use the SLURM array in RUNBOOK_SWEEP_DEPLOY.md instead of 'sweep'.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIMO="$ROOT/emulator/mimo"
TAGS_SINGLE=(n705 n60 n456 n261 nadir 261 456 60 705)
TAGS_ALL=("${TAGS_SINGLE[@]}" multiangle)
cfg_sweep() { echo "config/sweep_40m/config_$1_mimo_sweep.yaml"; }
cfg_deployed() { echo "config/deployed_40m/config_$1_mimo_evaluate.yaml"; }

stage_configs() { (cd "$MIMO" && python make_sweep_configs.py --out config/sweep_40m); }

stage_prep() {
  for t in "${TAGS_ALL[@]}"; do
    (cd "$MIMO" && python pyct/train_cloud.py -y "$(cfg_sweep "$t")" --prepare-data)
  done
}

stage_sweep() {
  local n="${1:-40}"
  for t in "${TAGS_ALL[@]}"; do
    for ((i = 0; i < n; i++)); do
      (cd "$MIMO" && python pyct/train_cloud.py -y "$(cfg_sweep "$t")" --sweep local --trial-index "$i")
    done
  done
}

stage_deploy() { (cd "$MIMO" && python select_and_deploy.py -y config/deploy_40m.yaml); }
FORCE=${FORCE:-0}
skipflag() { [ "$FORCE" = 1 ] || echo "$1"; }
stage_recal() { (cd "$MIMO" && python recalibrate.py -y config/recalibration_40m.yaml $(skipflag --skip-existing)); }

stage_uq() {
  for v in single_raw single_cal multi_raw multi_cal; do
    (cd "$MIMO" && python run_uq_analysis.py -y "config/uq_eval_40m_${v}.yaml" $([ "$FORCE" = 1 ] && echo --overwrite))
  done
}

stage_merge() {
  local single=()
  for t in "${TAGS_SINGLE[@]}"; do single+=("$(cfg_deployed "$t")"); done
  (cd "$MIMO" && python evaluate_and_merge_mimo.py --time 30.0 --resolution 40 \
      --configs "${single[@]}" --apply-calibration --tag _MIMO_single_40m_toa_cal $(skipflag --skip-complete))
  (cd "$MIMO" && python evaluate_and_merge_mimo.py --time 30.0 --resolution 40 \
      --configs "$(cfg_deployed multiangle)" --apply-calibration --tag _MIMO_multi_40m_toa_cal $(skipflag --skip-complete))
}

stage_case() {
  for net in single multi; do
    for mk in truthmask hullmask; do
      (cd "$ROOT/reconstruction" && python cloud_case_study_uq.py -y "config/case_study_40m_${net}_cal_${mk}.yaml" $(skipflag --skip-done))
    done
  done
}

stage_compare() { (cd "$MIMO" && python compare_uq_runs.py -y config/compare_40m.yaml); }

RT_DIR=${RT_DIR:-$HOME/Rep_Taxonomy}
stage_reptax_export() { (cd "$MIMO" && python export_representations.py -y config/reptax_40m.yaml); }
stage_reptax_configs() { (cd "$MIMO" && python make_reptax_configs.py -y config/reptax_40m.yaml --rt-dir "$RT_DIR"); }
stage_reptax_analyze() {
  if [ ! -d "$RT_DIR/reptaxonomy" ]; then
    echo "[WARN] skip reptax analyze: no Rep_Taxonomy checkout at RT_DIR=$RT_DIR"; return 0
  fi
  local sh="$MIMO/output/reptax_40m/analysis/configs/run_reptax.sh"
  if [ ! -x "$sh" ]; then echo "[WARN] skip reptax analyze: $sh not found (run reptax_configs)"; return 0; fi
  RT_DIR="$RT_DIR" "$sh" || echo "[WARN] some Rep_Taxonomy analyses failed (see log); finished ones are kept"
}
stage_reptax_summary() { (cd "$MIMO" && python summarize_reptax.py -y config/reptax_40m.yaml); }
stage_reptax() { stage_reptax_export; stage_reptax_configs; stage_reptax_analyze; stage_reptax_summary; }

case "${1:-}" in
  configs) stage_configs ;;
  prep) stage_prep ;;
  sweep) stage_sweep "${2:-40}" ;;
  deploy) stage_deploy ;;
  recal) stage_recal ;;
  uq) stage_uq ;;
  merge) stage_merge ;;
  case) stage_case ;;
  compare) stage_compare ;;
  reptax) stage_reptax ;;
  reptax_export) stage_reptax_export ;;
  reptax_configs) stage_reptax_configs ;;
  reptax_analyze) stage_reptax_analyze ;;
  reptax_summary) stage_reptax_summary ;;
  resume) stage_recal; stage_uq; stage_merge; stage_case; stage_compare; stage_reptax ;;
  all) stage_prep; stage_sweep "${2:-40}"; stage_deploy; stage_recal; stage_uq; stage_merge; stage_case; stage_compare ;;
  *) sed -n '2,16p' "$0"; exit 1 ;;
esac
