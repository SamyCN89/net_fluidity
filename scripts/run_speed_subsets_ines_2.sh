#!/usr/bin/env bash
set -euo pipefail

# --- Load environment variables from parent .env file ---
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a; source "${ENV_FILE}"; set +a
  echo "[info] Loaded environment from ${ENV_FILE}"
else
  echo "[warn] No .env file found at ${ENV_FILE}"
fi

# --- Load centralized dataset config early so DATASET_NAME is always defined ---
CONFIG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.dataset_config"
if [[ -f "$CONFIG_FILE" ]]; then
  set -a; source "$CONFIG_FILE"; set +a
  echo "[info] Loaded dataset config from $CONFIG_FILE"
else
  echo "[warn] No dataset config file found; using defaults."
  DATASET_NAME="ines_abdallah"
  TR="${TR:-450}"
  PROC="${PROC:-50}"
fi

# Fallbacks if .dataset_config missed them
DATASET_NAME="${DATASET_NAME:-ines_abdallah}"
TR="${TR:-450}"
PROC="${PROC:-50}"

# --- Auto-select PATHS_ROOT based on host and environment ---
HOSTNAME_LOWER=$(hostname | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
echo "[debug] Hostname detected: ${HOSTNAME_LOWER}"
echo "[debug] PATHS_ENV before detection: ${PATHS_ENV:-<unset>}"

unset PATHS_ENV 2>/dev/null || true

if [[ "${PATHS_ENV:-AUTO}" == "AUTO" ]]; then
  if [[ "$HOSTNAME_LOWER" == *"funsymania"* ]]; then
    PATHS_ENV="CLUSTER_NATIVE"
    PATHS_ROOT="${PROJECT_ROOT_CLUSTER:-/mnt/sdc/samy}"
  elif [[ "$HOSTNAME_LOWER" == *"funsy"* ]] || [[ "$HOSTNAME_LOWER" == *"stra"* ]]; then
    PATHS_ENV="CLUSTER_SSHFS"
    PATHS_ROOT="${PROJECT_ROOT_CLUSTER_SSHFS:-${HOME}/mnt/funsymania_sdc/samy}"
  else
    PATHS_ENV="LOCAL"
    PATHS_ROOT="${PROJECT_ROOT_LOCAL:-/tmp/net_fluidity_root}"
  fi
fi

PATHS_ROOT="${PATHS_ROOT:-/tmp/net_fluidity_root}"

echo "[info] Environment mode: ${PATHS_ENV}"
echo "[info] Using PATHS_ROOT=${PATHS_ROOT}"
export PATHS_ENV PATHS_ROOT
echo "[env] DATASET=${DATASET_NAME}"

# --- End load environment ---

ACTION="${1:-run}"   # run | dry-run | list

# ----------------- Defaults -----------------
WINDOW_MIN="${WINDOW_MIN:-5}"
WINDOW_MAX="${WINDOW_MAX:-100}"
WINDOW_STEP="${WINDOW_STEP:-1}"
LAG="${LAG:-1}"
TAU_RANGE="${TAU_RANGE:-0,1,2,3,4}"     # 0..4
TIME_OFFSET="${TIME_OFFSET:-}"          # optional
SPEED_CLI_BASE="python scripts/speed/dfc_speed_compute.py --dataset-name ${DATASET_NAME:?} --subset-name"

RUN_GLOBAL="${RUN_GLOBAL:-1}"
RUN_PER_REGION="${RUN_PER_REGION:-1}"
RUN_WITHIN="${RUN_WITHIN:-1}"       # DMN/memory within-network edges
RUN_TOUCHING="${RUN_TOUCHING:-1}"   # DMN/memory touching-network edges
PREPROCESS="${PREPROCESS:-0}"

# Respect BLAS thread caps
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# ----------------- Helpers -----------------
run_cmd() {
  if [[ "${ACTION}" == "dry-run" ]]; then
    echo "$@"
  else
    eval "$@"
  fi
}

maybe_time_offset() {
  if [[ -n "${TIME_OFFSET}" ]]; then
    echo "--time-offset ${TIME_OFFSET}"
  fi
}

COMMON_SPEED_FLAGS="\
  --window-min ${WINDOW_MIN} --window-max ${WINDOW_MAX} --window-step ${WINDOW_STEP} \
  --lag ${LAG} --tau-range ${TAU_RANGE} $(maybe_time_offset) --jobs ${PROC}"

# Check preprocessed assets; optionally run preprocess
ensure_preprocessed() {
  local tr="$1"
  local base="${PATHS_ROOT}/results/${DATASET_NAME:?}/preprocessed_data"
  local meta=("${base}/grouping_data_new.pkl")
  local tsnpz=("${base}/ts_and_meta_2m4m.npz")
  if ls "${meta[@]}" >/dev/null 2>&1 && ls "${tsnpz[@]}" >/dev/null 2>&1; then
    return 0
  fi
  if [[ "${PREPROCESS}" == "1" ]]; then
    echo "[info] Preprocessed files not found for tr=${tr}. (Hook for preprocess call.)"
    # put your preprocess command here if you want it auto-run
    return 0
  fi
  echo "[error] Preprocessed files missing for tr=${tr} under ${base}" >&2
  echo "        Or set PREPROCESS=1 to allow this script to run preprocessing." >&2
  exit 2
}

# ----------------- DMN / Memory definitions -----------------
# Labels as in your ines figure (blue = DMN, red = Memory)
DMN_LABELS="PL ILA,PFC,ACA,RSP"
MEMORY_LABELS="d HIP,v HIP,d DG,v DG,PERI,ENT,SUB,ReRh,THAL memory"
OTHER_LABELS="TEa,CLA,THAL sensory,THAL motor,Habenula,Hypo medial,Hypo SNC,Hypo PVN-ARC,Hypo lat,Septum,Insula,AUD,PIR,Motor,Somato,VIS,PTLp,ACB,CP,Pallidum,Amygdala,VTA,MBmot PAG,MBmot,MBbeh,MBsen,SN,Pons"

declare -a WITHIN_SETS=(
  "dmn_within;${DMN_LABELS}"
  "mem_within;${MEMORY_LABELS}"
  "other_within;${OTHER_LABELS}"
)

declare -a TOUCHING_SETS=(
  "dmn_touching;${DMN_LABELS}"
  "mem_touching;${MEMORY_LABELS}"
  "other_touching;${OTHER_LABELS}"
)

# ----------------- List planned subsets -----------------
list_subsets() {
  if [[ "${RUN_GLOBAL}" == "1" ]]; then
    echo "- all (global)"
  fi
  if [[ "${RUN_PER_REGION}" == "1" ]]; then
    echo "- regions500 (per-region)"
  fi
  if [[ "${RUN_WITHIN}" == "1" ]]; then
    for s in "${WITHIN_SETS[@]}"; do echo "- ${s%%;*}"; done
  fi
  if [[ "${RUN_TOUCHING}" == "1" ]]; then
    for s in "${TOUCHING_SETS[@]}"; do echo "- ${s%%;*}"; done
  fi
}

if [[ "${ACTION}" == "list" ]]; then
  echo "Planned subsets (TR=${TR}, PROC=${PROC}):"
  list_subsets
  exit 0
fi

# ----------------- Main execution -----------------
ensure_preprocessed "${TR}"

# Global (all edges)
if [[ "${RUN_GLOBAL}" == "1" ]]; then
  run_cmd "${SPEED_CLI_BASE} all ${COMMON_SPEED_FLAGS}"
fi

# Per-region (regions500 subset, touching edges)
if [[ "${RUN_PER_REGION}" == "1" ]]; then
  run_cmd "${SPEED_CLI_BASE} regions500 \
    --per-region --per-region-mode touching \
    ${COMMON_SPEED_FLAGS}"
fi

run_with_sets() {
  local mode="$1"; shift
  local -n arr="$1"
  for item in "${arr[@]}"; do
    local name="${item%%;*}"
    local labels="${item#*;}"
    run_cmd "${SPEED_CLI_BASE} ${name} \
      --region-labels '${labels}' --region-mode ${mode} \
      ${COMMON_SPEED_FLAGS}"
  done
}

if [[ "${RUN_WITHIN}" == "1" ]]; then
  run_with_sets within WITHIN_SETS
fi
if [[ "${RUN_TOUCHING}" == "1" ]]; then
  run_with_sets touching TOUCHING_SETS
fi

echo "Done (action=${ACTION})."
