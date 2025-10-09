#!/usr/bin/env bash
set -euo pipefail

# Batch runner for compute and plot over multiple subsets.
#
# Usage examples:
#   bash scripts/run_bootstrap_batches.sh both
#   bash scripts/run_bootstrap_batches.sh compute
#   bash scripts/run_bootstrap_batches.sh plot
#   bash scripts/run_bootstrap_batches.sh list       # list planned subsets
#   bash scripts/run_bootstrap_batches.sh dry-run    # print commands only
#
# Optional environment overrides:
#   TR=500 N_BOOT=2000 JOBS=8 CHUNK=256 TAU_INDEX=0 \
#   OUTDIR="" APPEND_SUBSET=1 LOAD_CACHE=1 \
#   bash scripts/run_bootstrap_batches.sh both

export PATHS_ENV="CLUSTER_FS"
export DATASET_NAME="julien_caillette"


ACTION="${1:-both}"  # compute | plot | both | list | dry-run

# Defaults (override via env vars as needed)
TR="${TR:-500}"
TAU_INDEX="${TAU_INDEX:-0}"
N_BOOT="${N_BOOT:-2000}"
JOBS="${JOBS:-8}"
CHUNK="${CHUNK:-256}"
POOL_THRESH="${POOL_THRESH:-median}"
POOL_ALL="${POOL_ALL:-1}"
REUSE_GROUP_BOOTS="${REUSE_GROUP_BOOTS:-1}"
BOOTS_FLOAT32="${BOOTS_FLOAT32:-1}"
LOAD_CACHE="${LOAD_CACHE:-1}"
BYWIN_GRID_COLS="${BYWIN_GRID_COLS:-2}"
GROUP_COLS="${GROUP_COLS:-genotype,treatment}"
POOL_COLS="${POOL_COLS:-genotype}"
POOL_EXCLUDE_SELF="${POOL_EXCLUDE_SELF:-0}"
N_ANIMALS="${N_ANIMALS:-48}"
CORRELATE_NOR="${CORRELATE_NOR:-0}"
CORRELATE_NOR_BY_GROUPS="${CORRELATE_NOR_BY_GROUPS:-0}"
PLOT_FORMAT="${PLOT_FORMAT:-png}"
PLOT_PROGRESS="${PLOT_PROGRESS:-0}"
PLOT_POOLTEST="${PLOT_POOLTEST:-1}"

# Optional output directory control (empty means default to --subset)
OUTDIR="${OUTDIR:-}"

outdir_flag=()
if [[ -n "${OUTDIR}" ]]; then
  outdir_flag=(--outdir "${OUTDIR}")
fi

cache_flag=()
if [[ "${LOAD_CACHE}" == "1" ]]; then
  cache_flag=(--load-cache)
fi

pool_flags=(--pool-threshold "${POOL_THRESH}")
if [[ "${POOL_ALL}" == "1" ]]; then
  pool_flags+=(--pool-all)
fi

pooltest_flags=()
if [[ -n "${POOL_COLS}" ]]; then
  pooltest_flags+=(--bootstrap-pool-cols "${POOL_COLS}")
fi
if [[ "${POOL_EXCLUDE_SELF}" == "1" ]]; then
  pooltest_flags+=(--pool-exclude-self)
fi

reuse_flag=()
if [[ "${REUSE_GROUP_BOOTS}" == "1" ]]; then
  reuse_flag=(--reuse-group-boots)
fi

boots32_flag=()
if [[ "${BOOTS_FLOAT32}" == "1" ]]; then
  boots32_flag=(--boots-float32)
fi

correlate_flags=()
if [[ "${CORRELATE_NOR}" == "1" ]]; then
  correlate_flags+=(--correlate-nor)
  if [[ "${CORRELATE_NOR_BY_GROUPS}" == "1" ]]; then
    correlate_flags+=(--correlate-nor-by-groups)
  fi
fi

# Subsets to process
# Subsets to process
declare -a SUBSETS
SUBSETS+=("regions500")
for flag in within touching; do
  for name in dmn mem sal lat 1st 2nd 3rd 4th; do
    SUBSETS+=("${name}_${flag}")
  done
done

if [[ "${ACTION}" == "list" ]]; then
  echo "Planned subsets (TR=${TR}):"
  for s in "${SUBSETS[@]}"; do echo "  - ${s}"; done
  exit 0
fi

run_compute() {
  local subset="$1"
  echo "[compute] subset=${subset}"
  if [[ "${ACTION}" == "dry-run" ]]; then
    echo python scripts/compute_speed_bootstrap.py \
      --tr "${TR}" --subset "${subset}" "${outdir_flag[@]}" \
      --tau-index "${TAU_INDEX}" "${pool_flags[@]}" \
      --group-cols "${GROUP_COLS}" "${pooltest_flags[@]}" \
      --n-boot "${N_BOOT}" --jobs "${JOBS}" --chunk "${CHUNK}" \
      --n-animals "${N_ANIMALS}" "${correlate_flags[@]}" \
      "${reuse_flag[@]}" "${boots32_flag[@]}" "${cache_flag[@]}"
    return 0
  fi
  python scripts/compute_speed_bootstrap.py \
    --tr "${TR}" --subset "${subset}" "${outdir_flag[@]}" \
    --tau-index "${TAU_INDEX}" "${pool_flags[@]}" \
    --group-cols "${GROUP_COLS}" "${pooltest_flags[@]}" \
    --n-boot "${N_BOOT}" --jobs "${JOBS}" --chunk "${CHUNK}" \
    --n-animals "${N_ANIMALS}" "${correlate_flags[@]}" \
    "${reuse_flag[@]}" "${boots32_flag[@]}" "${cache_flag[@]}"
}

run_plot() {
  local subset="$1"
  echo "[plot] subset=${subset}"
  if [[ "${ACTION}" == "dry-run" ]]; then
    echo python scripts/plot_speed_bootstrap.py \
      --tr "${TR}" --subset "${subset}" "${outdir_flag[@]}" \
      --plot-format "${PLOT_FORMAT}" \
      --plot-diffs-by-win --plot-diffs-bywin-grid --bywin-grid-cols "${BYWIN_GRID_COLS}" \
      --plot-pooled-diffs --plot-pooled-quantiles \
      "${cache_flag[@]}"
    echo python scripts/plot_speed_correlations.py \
      --tr "${TR}" --subset "${subset}" "${outdir_flag[@]}" \
      --metric both --plot-by-win --plot-pooled --bywin-group-grid --grid-cols 3 $([[ "${PLOT_PROGRESS}" == "1" ]] && echo --progress)
    if [[ "${PLOT_POOLTEST}" == "1" ]]; then
      echo python scripts/plot_speed_pooltest.py \
        --tr "${TR}" --subset "${subset}" "${outdir_flag[@]}" \
        --plot-format "${PLOT_FORMAT}" --bywin --pooled $([[ "${PLOT_PROGRESS}" == "1" ]] && echo --progress)
    fi
    return 0
  fi
  python scripts/plot_speed_bootstrap.py \
    --tr "${TR}" --subset "${subset}" "${outdir_flag[@]}" \
    --plot-format "${PLOT_FORMAT}" \
    --plot-diffs-by-win --plot-diffs-bywin-grid --bywin-grid-cols "${BYWIN_GRID_COLS}" \
    --plot-pooled-diffs --plot-pooled-quantiles \
    "${cache_flag[@]}"
  # Correlation plots (by-window, pooled, and grouped grid)
  python scripts/plot_speed_correlations.py \
    --tr "${TR}" --subset "${subset}" "${outdir_flag[@]}" \
    --metric both --plot-by-win --plot-pooled --bywin-group-grid --grid-cols 3 $([[ "${PLOT_PROGRESS}" == "1" ]] && echo --progress)
  # Pool-test plots (by-window and pooled)
  if [[ "${PLOT_POOLTEST}" == "1" ]]; then
    python scripts/plot_speed_pooltest.py \
      --tr "${TR}" --subset "${subset}" "${outdir_flag[@]}" \
      --plot-format "${PLOT_FORMAT}" --bywin --pooled $([[ "${PLOT_PROGRESS}" == "1" ]] && echo --progress)
  fi
}

for subset in "${SUBSETS[@]}"; do
  case "${ACTION}" in
    compute)
      run_compute "${subset}"
      ;;
    plot)
      run_plot "${subset}"
      ;;
    both)
      run_compute "${subset}"
      run_plot "${subset}"
      ;;
    dry-run)
      # Emit both compute and plot commands for preview
      run_compute "${subset}"
      run_plot "${subset}"
      ;;
    *)
      echo "Unknown action: ${ACTION}. Use compute|plot|both|list|dry-run" >&2
      exit 2
      ;;
  esac
done

echo "Done (${ACTION})."
