#!/usr/bin/env python3
"""
Minimal, notebook-style tutorial (script) for shared_code.shared_code.fun_bootstrap.

Run:
  python scripts/fun_bootstrap_tutorial.py
"""
from __future__ import annotations

import numpy as np
from shared_code.shared_code.fun_bootstrap import (
    bootstrap_percentiles,
    bootstrap_diff_percentiles,
    bootstrap_groups_percentiles,
    bootstrap_groups_boots,
    ci_from_boots,
    pool_per_animal,
)


def demo_basic_percentiles():
    print("\n== Basic percentiles (single sample) ==")
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.0, scale=1.0, size=500)
    q = [5, 50, 95]
    point, lo, hi = bootstrap_percentiles(x, q=q, n_boot=2000, ci=95.0, seed=0, chunk=256)
    for qi, pt, l, h in zip(q, point, lo, hi):
        print(f"q={qi:>2}: point={pt: .3f}, 95% CI=({l: .3f}, {h: .3f})")


def demo_diff_percentiles():
    print("\n== Percentile differences between two samples ==")
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1.0, size=600)
    y = rng.normal(0.0, 1.0, size=600)
    q = [5, 50, 95]
    res = bootstrap_diff_percentiles(x, y, q=q, n_boot=2000, ci=95.0, seed=1, chunk=256)
    for qi, pt, l, h, sig in zip(res['q'], res['point'], res['lo'], res['hi'], res['sig']):
        print(f"q={int(qi):>2}: diff={pt: .3f}, 95% CI=({l: .3f}, {h: .3f}), inside_CI={bool(sig)}")


def demo_groups():
    print("\n== Per-group percentiles and reuse for pairwise diffs ==")
    rng = np.random.default_rng(2)
    # Build per-animal arrays: 6 animals; uneven lengths
    per_animal = [rng.normal(0.0, 1.0, size=s) for s in [100, 120, 80, 150, 90, 110]]
    groups = {('WT','VEH'): [0, 1, 2], ('WT','DRUG'): [3, 4, 5]}
    q = [5, 50, 95]
    qa = bootstrap_groups_percentiles(per_animal, groups, q=q, n_boot=1000, ci=95.0, seed=2, chunk=128)
    for gk, res in qa.items():
        print(f"Group {gk}: q={res['q'].tolist()} med={res['point'][1]: .3f}")
    # Reuse pattern: per-group boots => pairwise diffs and CI via ci_from_boots
    boots_map = bootstrap_groups_boots(per_animal, groups, q=q, n_boot=1000, seed=3, chunk=128)
    q_arr = boots_map['__q__']
    A, B = ('WT','VEH'), ('WT','DRUG')
    boots_A = boots_map[A]; boots_B = boots_map[B]
    diff_boots = boots_A - boots_B
    lo, hi = ci_from_boots(diff_boots, ci=95.0)
    # Point diff via pooled values
    xA = pool_per_animal(per_animal, groups[A]); xB = pool_per_animal(per_animal, groups[B])
    point = np.percentile(xA, q_arr) - np.percentile(xB, q_arr)
    print("Pairwise diff CI (reuse):")
    for qi, pt, l, h in zip(q_arr, point, lo, hi):
        print(f"  q={int(qi)}: diff={pt: .3f}, CI=({l: .3f}, {h: .3f})")


if __name__ == "__main__":
    demo_basic_percentiles()
    demo_diff_percentiles()
    demo_groups()

