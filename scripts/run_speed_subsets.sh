#!/usr/bin/env bash
set -euo pipefail

# Batch runner for dFC speed computation using julien_data/src/speed_compute.py
#
# Usage:
#   bash scripts/run_speed_subsets.sh           # run all
#   bash scripts/run_speed_subsets.sh dry-run   # print commands only
#   bash scripts/run_speed_subsets.sh list      # list subsets to be run
#
# Environment overrides:
#   PROC=32 TR=500 PATHS_ROOT=/abs/root DATASET_NAME=julien_caillette \
#   RUN_ALL=1 RUN_WITHIN=1 RUN_TOUCHING=1 RUN_PER_REGION=1 RUN_GLOBAL=1 \
#   bash scripts/run_speed_subsets.sh
#
# Tips:
#   - Ensure PATHS_ROOT and DATASET_NAME resolve to valid paths; run
#       python scripts/paths_doctor.py --show
#   - If preprocessed files are missing, set PREPROCESS=1 to run preprocess.

ACTION="${1:-run}"   # run | dry-run | list

# Defaults
PROC="${PROC:-32}"
TR="${TR:-500}"
RUN_GLOBAL="${RUN_GLOBAL:-1}"
RUN_PER_REGION="${RUN_PER_REGION:-1}"
RUN_WITHIN="${RUN_WITHIN:-1}"
RUN_TOUCHING="${RUN_TOUCHING:-1}"
RUN_ALL="${RUN_ALL:-1}"
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

# Check preprocessed assets; optionally run preprocess
ensure_preprocessed() {
  local tr="$1"
  # Expect these under results/<dataset>/preprocessed_data
  local base="${PATHS_ROOT:?}/results/${DATASET_NAME:?}/preprocessed_data"
  local meta=("${base}/metadata_animals_*_tr_${tr}.pkl")
  local tsnpz=("${base}/ts_filtered_animals_*_tr_${tr}.npz")
  if ls ${meta[@]} >/dev/null 2>&1 && ls ${tsnpz[@]} >/dev/null 2>&1; then
    return 0
  fi
  if [[ "${PREPROCESS}" == "1" ]]; then
    echo "[info] Preprocessed files not found for tr=${tr}. Running preprocess..."
    run_cmd "python julien_data/src/preprocess.py --filter-mode exclude_shortest"
    return 0
  fi
  echo "[error] Preprocessed files missing for tr=${tr} under ${base}" >&2
  echo "        Run: python julien_data/src/preprocess.py --filter-mode exclude_shortest" >&2
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
  run_cmd "python julien_data/src/speed_compute.py --processors ${PROC} --tr ${TR} --subset-name all"
fi
if [[ "${RUN_PER_REGION}" == "1" ]]; then
  run_cmd "python julien_data/src/speed_compute.py --processors ${PROC} --tr ${TR} --subset-name regions500 --per-region"
fi

run_with_sets() {
  local mode="$1"; shift
  local -n arr=$1
  for item in "${arr[@]}"; do
    local name="${item%%;*}"; local labels="${item#*;}"
    run_cmd "python julien_data/src/speed_compute.py --processors ${PROC} --tr ${TR} --subset-name ${name} --selected-region-labels '${labels}' --region-mode ${mode}"
  done
}

if [[ "${RUN_WITHIN}" == "1" ]]; then
  run_with_sets within WITHIN_SETS
fi
if [[ "${RUN_TOUCHING}" == "1" ]]; then
  run_with_sets touching TOUCHING_SETS
fi

echo "Done (action=${ACTION})."

