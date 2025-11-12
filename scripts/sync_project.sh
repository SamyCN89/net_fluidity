#!/usr/bin/env bash
set -euo pipefail

# Load environment
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [[ -f "$ENV_FILE" ]]; then
  export $(grep -v '^#' "$ENV_FILE" | xargs)
  echo "[sync] Loaded environment from $ENV_FILE"
else
  echo "[error] Missing .env file."
  exit 1
fi

# Cluster configuration
REMOTE_HOST=funsystra
REMOTE_ROOT=${PROJECT_ROOT_CLUSTER}
LOCAL_ROOT=${PROJECT_ROOT_LOCAL}

# Define sync paths
LOCAL_DATA_DIR=${LOCAL_ROOT}/dataset/${DATASET_NAME}
LOCAL_RESULTS_DIR=${LOCAL_ROOT}/results/${DATASET_NAME}
REMOTE_DATA_DIR=${REMOTE_ROOT}/dataset/${DATASET_NAME}
REMOTE_RESULTS_DIR=${REMOTE_ROOT}/results/${DATASET_NAME}

# Exclusions
RSYNC_EXCLUDES="*.npz,*.npy,*.tmp,__pycache__/,*.log,*.bak"
IFS=',' read -ra EXCLUDE_PATTERNS <<< "$RSYNC_EXCLUDES"
RSYNC_EXCLUDE_FLAGS=()
for pattern in "${EXCLUDE_PATTERNS[@]}"; do
  RSYNC_EXCLUDE_FLAGS+=(--exclude "$pattern")
done

# Function to run rsync
run_rsync() {
  local src="$1"
  local dst="$2"
  echo "[sync] Syncing: $src → $dst"
  rsync -avz --progress "${RSYNC_EXCLUDE_FLAGS[@]}" "$src" "$dst"
}

# Determine action
MODE="${1:-}"
case "$MODE" in
  push)
    echo "[sync] Pushing local data/results to cluster..."
    run_rsync "${LOCAL_DATA_DIR}/" "${REMOTE_HOST}:${REMOTE_DATA_DIR}/"
    run_rsync "${LOCAL_RESULTS_DIR}/" "${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/"
    ;;
  pull)
    echo "[sync] Pulling results/data from cluster to local..."
    run_rsync "${REMOTE_HOST}:${REMOTE_DATA_DIR}/" "${LOCAL_DATA_DIR}/"
    run_rsync "${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/" "${LOCAL_RESULTS_DIR}/"
    ;;
  dry-run)
    echo "[sync] Preview of files to be transferred:"
    rsync -avzn "${RSYNC_EXCLUDE_FLAGS[@]}" "${LOCAL_RESULTS_DIR}/" "${REMOTE_HOST}:${REMOTE_RESULTS_DIR}/"
    ;;
  *)
    echo "Usage: bash scripts/sync_project.sh [push|pull|dry-run]"
    ;;
esac
echo "[sync] Done."
