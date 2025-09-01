#!/usr/bin/env bash
set -euo pipefail

# Simple helper to run style checks and generate reports.
# Usage:
#   bash scripts/run_checks.sh            # same as 'report'
#   bash scripts/run_checks.sh report     # write reports/style_check.md and reports/lint_ruff.md
#   bash scripts/run_checks.sh check      # run checks in terminal
#   bash scripts/run_checks.sh format     # apply Black and Ruff fixes where safe

DIRS=(shared_code/shared_code metaconnectivity allegiance/src julien_data)

cmd=${1:-report}

case "$cmd" in
  report)
    mkdir -p reports
    echo "[Black] writing reports/style_check.md"
    black --check "${DIRS[@]}" > reports/style_check.md 2>&1 || true
    if command -v ruff >/dev/null 2>&1; then
      echo "[Ruff] writing reports/lint_ruff.md"
      ruff check "${DIRS[@]}" > reports/lint_ruff.md 2>&1 || true
    else
      echo "Ruff not found. Install via conda/pip and re-run." > reports/lint_ruff.md
    fi
    ;;
  check)
    black --check "${DIRS[@]}"
    if command -v ruff >/dev/null 2>&1; then
      ruff check "${DIRS[@]}"
    else
      echo "[skip] Ruff not found. Install via conda/pip."
    fi
    ;;
  format|fix)
    black "${DIRS[@]}"
    if command -v ruff >/dev/null 2>&1; then
      ruff check --fix "${DIRS[@]}" || true
    else
      echo "[skip] Ruff not found. Install via conda/pip."
    fi
    ;;
  *)
    echo "Usage: $0 [report|check|format]" >&2
    exit 2
    ;;
esac

