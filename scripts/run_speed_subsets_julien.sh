#!/usr/bin/env bash
set -euo pipefail

# --- Load environment variables from parent .env file ---
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -a; source "$ENV_FILE"; set +a
  echo "[info] Loaded environment from $ENV_FILE"
else
  echo "[warn] No .env file found at $ENV_FILE"
fi

# --- Load centralized dataset config before using DATASET_NAME ---
CONFIG_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.dataset_config"
if [[ -f "$CONFIG_FILE" ]]; then
  set -a; source "$CONFIG_FILE"; set +a
  echo "[info] Loaded dataset config from $CONFIG_FILE"
else
  echo "[warn] No dataset config file found; defaulting."
  DATASET_NAME="julien_caillette"
fi

# --- Auto-detect environment and PATHS_ROOT ---
# --- Auto-detect environment and PATHS_ROOT ---
unset PATHS_ENV 2>/dev/null || true
HOSTNAME_LOWER=$(hostname | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')

if [[ -z "${PATHS_ENV:-}" || "${PATHS_ENV}" == "AUTO" ]]; then
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

# Defaults
WINDOW_MIN="${WINDOW_MIN:-5}"
WINDOW_MAX="${WINDOW_MAX:-100}"
WINDOW_STEP="${WINDOW_STEP:-1}"
LAG="${LAG:-1}"
TAU_RANGE="${TAU_RANGE:-0,4}"     # means 0..4
TIME_OFFSET="${TIME_OFFSET:-}"     # optional
DATASET_NAME="${DATASET_NAME:-julien_caillette}"
SPEED_CLI_BASE="python scripts/speed/dfc_speed_compute.py --dataset-name ${DATASET_NAME:?} --subset-name"

PROC="${PROC:-50}"
TR="${TR:-500}"
RUN_GLOBAL="${RUN_GLOBAL:-1}"
RUN_PER_REGION="${RUN_PER_REGION:-1}"
RUN_WITHIN="${RUN_WITHIN:-1}"
RUN_TOUCHING="${RUN_TOUCHING:-1}"
PREPROCESS="${PREPROCESS:-0}"

# Respect BLAS thread caps to avoid oversubscription
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Helper to run/echo
run_cmd() {
  if [[ "${ACTION}" == "dry-run" ]]; then
    echo "$@"
  else
    eval "$@"
  fi
}

# Helper to conditionally add --time-offset only if set
maybe_time_offset() {
  if [[ -n "${TIME_OFFSET}" ]]; then
    echo "--time-offset ${TIME_OFFSET}"
  fi
}

# (Optional) Common flag bundle to keep call sites short
COMMON_SPEED_FLAGS="\
  --window-min ${WINDOW_MIN} --window-max ${WINDOW_MAX} --window-step ${WINDOW_STEP} \
  --lag ${LAG} --tau-range ${TAU_RANGE} $(maybe_time_offset) --jobs ${PROC}"

# Check preprocessed assets; optionally run preprocess
ensure_preprocessed() {
  local tr="$1"
  local base="${PATHS_ROOT}/results/${DATASET_NAME:?}/preprocessed_data"

  shopt -s nullglob  # prevent literal globs when no match

  local meta=( "${base}"/metadata_animals_*_tr_${tr}.pkl )
  local tsnpz=( "${base}"/ts_filtered_animals_*_tr_${tr}.npz )

  shopt -u nullglob  # restore default globbing

  if (( ${#meta[@]} )) && (( ${#tsnpz[@]} )); then
    return 0
  fi

  if [[ "${PREPROCESS}" == "1" ]]; then
    echo "[info] Preprocessed files not found for tr=${tr}. Running preprocess..."
    run_cmd "python julien_data/src/preprocess.py --filter-mode exclude_shortest"
    return 0
  fi

  echo "[error] Preprocessed files missing for tr=${tr} under ${base}" >&2
  echo "        Or set PREPROCESS=1 to let this script run it." >&2
  exit 2
}


# Define label sets
declare -a WITHIN_SETS=(
  "dmn_within;ORB,PL,ILA,ACA,RSP,TEa"
  "mem_within;PERI,ENT,ECT,dhc,vhc,SUB,LSX"
  "sal_within;AI,ACB,AAA_CEA_MEA,ACA,GU_VISC,CP,VTA"
  "lat_within;MOp,MOs,SSp,SSs,MBsen,MBmot,MBsta,VTA,PO_PF,VPL_VPM"
  "1st_within;AI,LSX,PL,CLA,SUB,ENT,MBsta,HY,vhc"
  "2nd_within;PTLp,ORB,ILA,PL,MOs,ENT,PALv,AAA_CEA_MEA,HY,MBmot,MBsen"
  "3rd_within;MBmot,MBsen,AI,PL,ACA,RSP,PTLp,SSs,AUD,GU_VISC,ENT,dhc,vhc,SUB,CLA,PIR,PALd,ACB,CP,HY,VTA"
  "4th_within;PTLp,ORB,ILA,AAA_CEA_MEA,HY"
)

declare -a TOUCHING_SETS=(
  "dmn_touching;ORB,PL,ILA,ACA,RSP,TEa"
  "mem_touching;PERI,ENT,ECT,dhc,vhc,SUB,LSX"
  "sal_touching;AI,ACB,AAA_CEA_MEA,ACA,GU_VISC,CP,VTA"
  "lat_touching;MOp,MOs,SSp,SSs,MBsen,MBmot,MBsta,VTA,PO_PF,VPL_VPM"
  "1st_touching;AI,LSX,PL,CLA,SUB,ENT,MBsta,HY,vhc"
  "2nd_touching;PTLp,ORB,ILA,PL,MOs,ENT,PALv,AAA_CEA_MEA,HY,MBmot,MBsen"
  "3rd_touching;MBmot,MBsen,AI,PL,ACA,RSP,PTLp,SSs,AUD,GU_VISC,ENT,dhc,vhc,SUB,CLA,PIR,PALd,ACB,CP,HY,VTA"
  "4th_touching;PTLp,ORB,ILA,AAA_CEA_MEA,HY"
)

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


# Ensure preprocessed data exists (or preprocess if enabled)
ensure_preprocessed "${TR}"

# Run global and per-region first
if [[ "${RUN_GLOBAL}" == "1" ]]; then
  run_cmd "${SPEED_CLI_BASE} all ${COMMON_SPEED_FLAGS}"
fi
if [[ "${RUN_PER_REGION}" == "1" ]]; then
  run_cmd "${SPEED_CLI_BASE} regions500 \
  --per-region --per-region-mode touching \
  ${COMMON_SPEED_FLAGS}"
fi

run_with_sets() {
  local mode="$1"; shift
  local -n arr=$1
  for item in "${arr[@]}"; do
    local name="${item%%;*}"; local labels="${item#*;}"
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

