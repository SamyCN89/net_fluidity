#!/usr/bin/env python3
"""
Compare shared_code vs metaconnectivity implementations for correctness and speed.

Outputs:
- reports/compat_benchmark.csv
- reports/compat_benchmark.md

Benchmarks kept tiny to avoid heavy runtime and external data.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import sys
import time

import numpy as np

# Import modules
from shared_code.shared_code import (
    fun_dfcspeed as S_dfc,
    fun_metaconnectivity as S_mc,
    fun_optimization as S_opt,
)

sys.path.append("metaconnectivity")
from metaconnectivity import (
    fun_dfcspeed as M_dfc,  # type: ignore
    fun_metaconnectivity as M_mc,  # type: ignore
    fun_optimization as M_opt,  # type: ignore
)


@dataclass
class Result:
    category: str
    name: str
    shape_equal: bool
    allclose: bool
    max_abs_diff: float | None
    mean_abs_diff: float | None
    shared_ms: float | None
    meta_ms: float | None
    speedup_shared_over_meta: float | None
    note: str = ""


def _cmp_arrays(a, b, rtol=1e-6, atol=1e-6):
    if a is None or b is None:
        return False, None, None
    if isinstance(a, tuple):
        a = np.array(a)
    if isinstance(b, tuple):
        b = np.array(b)
    if a.shape != b.shape:
        # Try flatten compare
        try:
            a_f = np.ravel(a)
            b_f = np.ravel(b)
            if a_f.shape != b_f.shape:
                return False, None, None
            ok = np.allclose(a_f, b_f, rtol=rtol, atol=atol, equal_nan=True)
            diff = np.abs(a_f - b_f)
            return False, float(np.nanmax(diff)), float(np.nanmean(diff))
        except Exception:
            return False, None, None
    ok = np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    diff = np.abs(a - b)
    return True, float(np.nanmax(diff)), float(np.nanmean(diff))


def _timeit(fn, runs=5):
    t0 = time.perf_counter()
    out = None
    for _ in range(runs):
        out = fn()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0 / runs, out


def benchmark():
    rng = np.random.default_rng(0)
    results: list[Result] = []

    # Synthetic data
    T, N = 120, 10
    ts = rng.standard_normal((T, N)).astype(np.float32)
    A = 1
    ts3 = rng.standard_normal((A, T, N)).astype(np.float32)

    # 1) Optimization: fast_corrcoef
    def s_fast():
        return S_opt.fast_corrcoef(ts)

    def m_fast():
        return M_opt.fast_corrcoef(ts)

    s_ms, s_out = _timeit(s_fast)
    m_ms, m_out = _timeit(m_fast)
    shape_eq, maxdiff, meandiff = _cmp_arrays(s_out, m_out)
    results.append(
        Result(
            category="optimization",
            name="fast_corrcoef",
            shape_equal=shape_eq,
            allclose=(maxdiff is not None and maxdiff <= 1e-6),
            max_abs_diff=maxdiff,
            mean_abs_diff=meandiff,
            shared_ms=s_ms,
            meta_ms=m_ms,
            speedup_shared_over_meta=(m_ms / s_ms if s_ms and m_ms else None),
        )
    )

    # 2) DFC stream
    def s_dfc_stream():
        return S_dfc.ts2dfc_stream(ts, window_size=30, lag=5, format_data="2D")

    def m_dfc_stream():
        return M_dfc.ts2dfc_stream(ts, window_size=30, lag=5, format_data="2D")

    s_ms, s_out = _timeit(s_dfc_stream)
    m_ms, m_out = _timeit(m_dfc_stream)
    shape_eq, maxdiff, meandiff = _cmp_arrays(s_out, m_out)
    results.append(
        Result(
            category="dfc",
            name="ts2dfc_stream(2D)",
            shape_equal=shape_eq,
            allclose=(maxdiff is not None and maxdiff <= 1e-5),
            max_abs_diff=maxdiff,
            mean_abs_diff=meandiff,
            shared_ms=s_ms,
            meta_ms=m_ms,
            speedup_shared_over_meta=(m_ms / s_ms if s_ms and m_ms else None),
        )
    )

    # 3) dFC speed (2D)
    dfc2d = s_out  # from ts2dfc_stream shared

    def s_speed():
        med, spd = S_dfc.dfc_speed(dfc2d, vstep=1)
        return np.array([med], dtype=np.float32), spd

    def m_speed():
        med, spd = M_dfc.dfc_speed(dfc2d, vstep=1)
        return np.array([med], dtype=np.float32), spd

    s_ms, s_out = _timeit(s_speed)
    m_ms, m_out = _timeit(m_speed)

    # compare median and speeds concatenated
    s_spd = np.ravel(s_out[1])
    m_spd = np.ravel(m_out[1])
    # Align by trimming to min length for a fair per-value compare
    min_len = min(s_spd.size, m_spd.size)
    s_cat = np.concatenate([s_out[0].ravel(), s_spd[:min_len]])
    m_cat = np.concatenate([m_out[0].ravel(), m_spd[:min_len]])
    shape_eq, maxdiff, meandiff = _cmp_arrays(s_cat, m_cat)
    results.append(
        Result(
            category="dfc",
            name="dfc_speed(vstep=1)",
            shape_equal=(s_spd.shape == m_spd.shape),
            allclose=(maxdiff is not None and maxdiff <= 1e-5),
            max_abs_diff=maxdiff,
            mean_abs_diff=meandiff,
            shared_ms=s_ms,
            meta_ms=m_ms,
            speedup_shared_over_meta=(m_ms / s_ms if s_ms and m_ms else None),
            note=(
                "length_mismatch:" + str((s_spd.size, m_spd.size))
                if s_spd.size != m_spd.size
                else ""
            ),
        )
    )

    # 3b) dFC speed (3D)
    dfc3d = S_dfc.ts2dfc_stream(ts, window_size=30, lag=5, format_data="3D")

    def s_speed_3d():
        med, spd = S_dfc.dfc_speed(dfc3d, vstep=1)
        return np.array([med], dtype=np.float32), spd

    def m_speed_3d():
        med, spd = M_dfc.dfc_speed(dfc3d, vstep=1)
        return np.array([med], dtype=np.float32), spd

    s_ms, s_out = _timeit(s_speed_3d)
    m_ms, m_out = _timeit(m_speed_3d)
    s_spd = np.ravel(s_out[1])
    m_spd = np.ravel(m_out[1])
    min_len = min(s_spd.size, m_spd.size)
    s_cat = np.concatenate([s_out[0].ravel(), s_spd[:min_len]])
    m_cat = np.concatenate([m_out[0].ravel(), m_spd[:min_len]])
    shape_eq, maxdiff, meandiff = _cmp_arrays(s_cat, m_cat)
    results.append(
        Result(
            category="dfc",
            name="dfc_speed(3D, vstep=1)",
            shape_equal=(s_spd.shape == m_spd.shape),
            allclose=(maxdiff is not None and maxdiff <= 1e-5),
            max_abs_diff=maxdiff,
            mean_abs_diff=meandiff,
            shared_ms=s_ms,
            meta_ms=m_ms,
            speedup_shared_over_meta=(m_ms / s_ms if s_ms and m_ms else None),
            note=(
                "length_mismatch:" + str((s_spd.size, m_spd.size))
                if s_spd.size != m_spd.size
                else ""
            ),
        )
    )

    # 4) Meta-connectivity (very small)
    def s_mc_fn():
        return S_mc.compute_metaconnectivity(ts3, window_size=10, lag=5, save_path=None, n_jobs=1)

    def m_mc_fn():
        return M_mc.compute_metaconnectivity(ts3, window_size=10, lag=5, return_dfc=False, save_path=None, n_jobs=1)

    try:
        s_ms, s_out = _timeit(s_mc_fn, runs=1)
        m_ms, m_out = _timeit(m_mc_fn, runs=1)
        shape_eq, maxdiff, meandiff = _cmp_arrays(s_out, m_out, rtol=1e-4, atol=1e-4)
        results.append(
            Result(
                category="metaconnectivity",
                name="compute_metaconnectivity",
                shape_equal=shape_eq,
                allclose=(maxdiff is not None and maxdiff <= 5e-4),
                max_abs_diff=maxdiff,
                mean_abs_diff=meandiff,
                shared_ms=s_ms,
                meta_ms=m_ms,
                speedup_shared_over_meta=(m_ms / s_ms if s_ms and m_ms else None),
            )
        )
    except Exception as e:
        results.append(
            Result(
                category="metaconnectivity",
                name="compute_metaconnectivity",
                shape_equal=False,
                allclose=False,
                max_abs_diff=math.nan,
                mean_abs_diff=math.nan,
                shared_ms=None,
                meta_ms=None,
                speedup_shared_over_meta=None,
                note=f"skipped: {e}",
            )
        )

    return results


def write_reports(results: list[Result], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSV
    csv_path = out_dir / "compat_benchmark.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(asdict(results[0]).keys()),
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Markdown
    md_path = out_dir / "compat_benchmark.md"
    lines = [
        "# Compatibility and Performance Benchmark\n",
        "\n",
        "This compares selected functions between `shared_code` and `metaconnectivity` on small synthetic inputs.\n",
        "\n",
        "| Category | Name | Shape Equal | Allclose | Max | Mean | Shared ms | Meta ms | Speedup (meta/shared) | Note |\n",
        "|---|---|:--:|:--:|---:|---:|---:|---:|---:|---|\n",
    ]
    for r in results:
        lines.append(
            f"| {r.category} | {r.name} | {int(r.shape_equal)} | {int(r.allclose)} | {r.max_abs_diff} | {r.mean_abs_diff} | {r.shared_ms} | {r.meta_ms} | {r.speedup_shared_over_meta} | {r.note} |\n"
        )
    # Add a short diagnostic note about dfc_speed indexing differences
    lines += [
        "\n## Notes on dFC Speed Differences\n",
        "- metaconnectivity.dfc_speed(2D/3D) computes speeds for each t vs t+vstep, yielding n_frames - vstep values.\n",
        "- shared_code.fun_dfcspeed.dfc_speed uses an internal index stride that currently produces n_frames - vstep - 1 values for vstep=1 on 2D input; reports align after trimming to the shorter length.\n",
        "- For 3D input, shared_code extracts lower-triangular FC values (excluding diagonal), while some legacy implementations may include diagonals or duplicates when reshaping.\n",
    ]
    md_path.write_text("".join(lines), encoding="utf-8")
    return csv_path, md_path


def main():
    results = benchmark()
    csv_p, md_p = write_reports(results, Path("reports"))
    print(f"Wrote {csv_p} and {md_p}")


if __name__ == "__main__":
    main()
