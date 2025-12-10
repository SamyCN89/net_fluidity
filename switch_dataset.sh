#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE=".dataset_config"
DATASET=${1:-}

if [[ -z "$DATASET" ]]; then
  echo "Usage: bash switch_dataset.sh <julien|ines>"
  exit 1
fi

case "$DATASET" in
  julien)
    sed -i 's/^DATASET_NAME=.*/DATASET_NAME=julien_caillette/' "$CONFIG_FILE"
    ;;
  ines)
    sed -i 's/^DATASET_NAME=.*/DATASET_NAME=ines_abdallah/' "$CONFIG_FILE"
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    exit 1
    ;;
esac

echo "Switched dataset to $DATASET"
