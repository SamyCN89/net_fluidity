#!/usr/bin/env python3
"""
Paths Doctor: inspect and validate dataset/results paths configuration.

- Prints which root/profile is active, which dataset name is used, and all derived
  paths (dataset, results, figures).
- Optionally attempts to create missing results/figures directories and checks
  write access.
 - Never creates raw dataset files; only directories without a suffix are created
   when --create is set.

Examples
  # Read-only check (default):
  python scripts/paths_doctor.py

  # Select CLUSTER profile and show details:
  PATHS_ENV=CLUSTER_FS python scripts/paths_doctor.py --show

  # Force a root override and create missing results/fig dirs:
  PATHS_ROOT=/scratch/$USER/laura_harsan python scripts/paths_doctor.py --create --check-write
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict

try:
    from shared_code.shared_code.fun_paths import (
        get_paths,
        get_root_path,
        check_write_permissions,
    )
except ModuleNotFoundError:
    # Try package import fallback if installed as 'shared_code'
    from shared_code.fun_paths import (
        get_paths,
        get_root_path,
        check_write_permissions,
    )


KEY_GROUPS = {
    "dataset": ["root", "timeseries", "cog_data", "labels"],
    "results": [
        "results",
        "sorted",
        "preprocessed",
        "mc",
        "dfc",
        "speed",
        "mc_mod",
        "allegiance",
        "trimers",
        "cohesion",
    ],
    "figures": [
        "figures",
        "fmodularity",
        "f_mod",
        "f_cog",
        "f_speed",
        "f_dfc",
        "f_cohesion",
        "f_allegiance",
        "f_trimers",
    ],
}


def readable(p: Path) -> bool:
    try:
        return p.exists() and os.access(p, os.R_OK)
    except Exception:
        return False


def dir_writable(p: Path) -> bool:
    try:
        if not p.exists():
            return False
        test = p / ".write_test"
        with open(test, "w") as f:
            f.write("ok")
        test.unlink()
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect and validate net_fluidity paths configuration")
    ap.add_argument("--env", type=str, default=None, help="Profile label (e.g., LOCAL, CLUSTER_FS). Defaults to PATHS_ENV or LOCAL.")
    ap.add_argument("--dataset-name", type=str, default=None, help="Dataset name (defaults to env DATASET_NAME or code default)")
    ap.add_argument("--create", action="store_true", help="Create missing result/fig directories (no file creation)")
    ap.add_argument("--check-write", action="store_true", help="Check write access to result/fig directories")
    ap.add_argument("--show", action="store_true", help="Print all resolved paths grouped by category")
    args = ap.parse_args()

    # Resolve profile and root
    env_label = args.env or os.getenv("PATHS_ENV", "LOCAL")
    root_override = os.getenv("PATHS_ROOT")
    dataset_name = args.dataset_name or os.getenv("DATASET_NAME") or "julien_caillette"

    print("Paths Doctor")
    print("- Env label   :", env_label)
    print("- Dataset     :", dataset_name)
    print("- PATHS_ROOT  :", root_override or "<unset>")
    # get_root_path handles PATHS_ROOT override and PATHS_ENV fallback
    try:
        root = get_root_path(env_label if root_override is None else env_label)
        print("- Resolved root:", root)
    except Exception as e:
        print("! Could not resolve root:", e)
        print("  Suggestions:")
        print("  - export PATHS_ROOT=/abs/path/to/project/root")
        print("  - or export PATHS_ENV=CLUSTER_FS and set PROJECT_ROOT_CLUSTER_FS=/abs/path")
        return 2

    # Build paths (optionally creating directories)
    try:
        paths = get_paths(
            dataset_name=dataset_name,
            create=bool(args.create),
            check_write=False,
            env=env_label,
        )
    except Exception as e:
        print("! get_paths failed:", e)
        return 2

    # Show grouped paths
    if args.show:
        def fmt(p: Path) -> str:
            return str(p)

        for grp, keys in KEY_GROUPS.items():
            print(f"[{grp}]")
            for k in keys:
                if k in paths:
                    print(f"  {k:>12}: {fmt(paths[k])}")

    # Existence and readability
    print("\nChecks")
    # Dataset files/dirs
    ds_missing = []
    for k in KEY_GROUPS["dataset"]:
        p = paths.get(k)
        if p is None:
            continue
        if not readable(p):
            ds_missing.append((k, str(p)))
    if ds_missing:
        print("- Missing or unreadable dataset entries:")
        for k, p in ds_missing:
            print(f"    {k}: {p}")
        print("  Fix: mount/copy data or point PATHS_ROOT/PATHS_ENV to a valid dataset root.")
    else:
        print("- Dataset entries are present and readable (root, timeseries, cog_data, labels)")

    # Results/figures write checks
    rw_issues = []
    res_dirs = [*KEY_GROUPS["results"], *KEY_GROUPS["figures"]]
    for k in res_dirs:
        p = paths.get(k)
        if p is None:
            continue
        if p.suffix:
            continue
        if not p.exists():
            # Might be created with --create, otherwise note
            rw_issues.append((k, str(p), "does not exist"))
        else:
            if not dir_writable(p):
                rw_issues.append((k, str(p), "not writable"))

    if rw_issues:
        print("- Results/figures directory issues:")
        for k, p, msg in rw_issues:
            print(f"    {k}: {p} -> {msg}")
        print("  Suggestions (choose what fits your environment):")
        print("  - Use a writable root: export PATHS_ROOT=/scratch/$USER/laura_harsan (or set PATHS_ENV+PROJECT_ROOT_<ENV>)")
        print("  - Create and set permissions (example):")
        print("      mkdir -p <dir> && chmod -R u+rwX,g+rwX <dir>")
        print("      # or set ownership if needed: sudo chown -R $USER:$USER <dir>")
        print("  - To let the app create directories, rerun with --create")
    else:
        print("- Results/figures directories exist and are writable")

    # Optional strong write check via helper (after creation)
    if args.check_write:
        try:
            check_write_permissions(paths)
            print("- check_write_permissions: OK")
        except Exception as e:
            print("- check_write_permissions: FAILED ->", e)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

