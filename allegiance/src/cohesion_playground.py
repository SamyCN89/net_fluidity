#!/usr/bin/env python3
"""
Jupyter-friendly playground for cohesion_compute.py outputs.

Loads NPZ summaries and Parquet events and provides small helpers to
inspect links, list top-N by metrics, and make quick plots.

Usage in a notebook:

    from allegiance.src.cohesion_playground import *
    paths = get_paths(timecourse_folder="Timecourses_updated_03052024")
    data = load_summaries(paths, window_size=9, lag=1, tau=3, scope="all")
    events = load_events(paths, window_size=9, lag=1, tau=3, scope="all")
    links = link_table(data)
    links.head()
    top = top_links_by(data["burstiness"], links, k=10)
    plot_top_time_ratio_bars(data["time_ratio"], top)
    plot_duration_hist(events, link_idx=int(top.iloc[0].link), bins=30)

This module avoids heavy CLI and focuses on reusable helpers.
"""
#%%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared_code.fun_paths import get_paths
#%%

@dataclass
class CohesionSummaries:
    time_ratio: np.ndarray  # (A, L)
    mean_duration: np.ndarray  # (A, L)
    std_duration: np.ndarray  # (A, L)
    burstiness: np.ndarray  # (A, L)
    pair_labels: np.ndarray  # (L, 2)
    anat_labels_sorted: np.ndarray  # (N,)
    n_animals: int
    n_windows: int
    tag: str


def _npz_path(paths: dict, ws: int, lag: int, tau: int, scope: str) -> Path:
    return (paths["allegiance"] / "cohesion_data" / f"cohesion_data_w{ws}_lag{lag}_tau{tau}_{scope}.npz").expanduser()


def _events_path(paths: dict, ws: int, lag: int, tau: int, scope: str) -> tuple[Path | None, Path | None]:
    base = f"events_w{ws}_lag{lag}_tau{tau}_{scope}"
    pq = (paths["allegiance"] / "cohesion_data" / f"{base}.parquet").expanduser()
    csv = (paths["allegiance"] / "cohesion_data" / f"{base}.csv").expanduser()
    return (pq if pq.exists() else None, csv if csv.exists() else None)


def load_summaries(paths: dict, *, window_size: int, lag: int, tau: int, scope: str = "all") -> CohesionSummaries:
    npz = np.load(_npz_path(paths, window_size, lag, tau, scope), allow_pickle=True)
    return CohesionSummaries(
        time_ratio=np.asarray(npz["time_ratio"]).astype(float),
        mean_duration=np.asarray(npz["mean_duration"]).astype(float),
        std_duration=np.asarray(npz["std_duration"]).astype(float),
        burstiness=np.asarray(npz["burstiness"]).astype(float),
        pair_labels=np.asarray(npz["pair_labels"]),
        anat_labels_sorted=np.asarray(npz["anat_labels_sorted"]),
        n_animals=int(npz["n_animals"]),
        n_windows=int(npz["n_windows"]),
        tag=f"w{window_size}_lag{lag}_tau{tau}_{scope}",
    )


def load_events(paths: dict, *, window_size: int, lag: int, tau: int, scope: str = "all") -> pd.DataFrame:
    pq, csv = _events_path(paths, window_size, lag, tau, scope)
    if pq is not None:
        return pd.read_parquet(pq)
    if csv is not None:
        return pd.read_csv(csv)
    raise FileNotFoundError("Neither Parquet nor CSV events found for the given tag.")


def link_table(data: CohesionSummaries) -> pd.DataFrame:
    """Return a DataFrame mapping link index ↔ pair labels."""
    pairs = [f"{a}–{b}" for a, b in data.pair_labels]
    return pd.DataFrame({"link": np.arange(len(pairs), dtype=int), "pair": pairs})


def top_links_by(metric_AL: np.ndarray, links_df: pd.DataFrame, k: int = 10, *, agg: str = "mean") -> pd.DataFrame:
    """Return top-k links sorted by aggregate across animals (mean or max)."""
    if agg == "mean":
        scores = metric_AL.mean(axis=0)
    elif agg == "max":
        scores = metric_AL.max(axis=0)
    else:
        raise ValueError("agg must be 'mean' or 'max'")
    out = links_df.copy()
    out[agg] = scores
    return out.sort_values(agg, ascending=False).head(k).reset_index(drop=True)


def plot_top_time_ratio_bars(time_ratio: np.ndarray, top_links: pd.DataFrame, *, title: str | None = None) -> None:
    """Bar chart of mean time_ratio for the provided top links."""
    idx = top_links["link"].to_numpy(dtype=int)
    vals = time_ratio.mean(axis=0)[idx]
    labels = top_links["pair"].tolist()
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(idx)), vals, color="#4477AA")
    plt.xticks(np.arange(len(idx)), labels, rotation=60, ha="right")
    plt.ylabel("mean time_ratio")
    plt.ylim(0, 1)
    plt.title(title or "Top links by mean time_ratio")
    plt.tight_layout()
    plt.show()


def plot_duration_hist(events: pd.DataFrame, *, link_idx: int, bins: int = 30, title: str | None = None) -> None:
    """Histogram of event durations for a specific link (all animals)."""
    durations = events.loc[events["link"] == int(link_idx), "duration"].to_numpy()
    plt.figure(figsize=(6, 4))
    plt.hist(durations, bins=bins, color="#66CCEE", alpha=0.9)
    plt.xlabel("duration [windows]")
    plt.ylabel("count")
    plt.title(title or f"Durations for link {link_idx}")
    plt.tight_layout()
    plt.show()


def scatter_mean_vs_burstiness(data: CohesionSummaries, *, by: str = "mean_duration") -> None:
    """Scatter per animal×link of a duration stat vs burstiness."""
    if by not in {"mean_duration", "std_duration"}:
        raise ValueError("by must be 'mean_duration' or 'std_duration'")
    X = getattr(data, by).ravel()
    Y = data.burstiness.ravel()
    plt.figure(figsize=(5, 5))
    plt.scatter(X, Y, s=4, alpha=0.3)
    plt.xlabel(by.replace("_", " "))
    plt.ylabel("burstiness")
    plt.title("Per animal×link")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def scatter_burstiness_vs_std_duration(data: CohesionSummaries, *, alpha: float = 0.3) -> None:
    """Scatter of std_duration vs burstiness across all animal×link points.

    Useful to see whether higher variability of event durations (std) aligns with
    higher burstiness. Consider filtering to links with at least 2 events when
    interpreting this plot (std=0 can reflect too few samples).
    """
    import matplotlib.pyplot as plt  # local import for notebook environments

    X = data.std_duration.ravel()
    Y = data.burstiness.ravel()
    plt.figure(figsize=(5, 5))
    plt.scatter(X, Y, s=4, alpha=alpha)
    plt.xlabel("std duration")
    plt.ylabel("burstiness")
    plt.title("std duration vs burstiness (per animal×link)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def scatter_mean_std_colored(
    data: CohesionSummaries,
    *,
    alpha: float = 0.9,
    s: int = 6,
    cmap: str = "coolwarm",
    vmin: float = -.2,
    vmax: float = .2,
) -> None:
    """Scatter mean_duration vs std_duration, colored by burstiness.

    Notes:
    - Points are animal×link; consider filtering to links with ≥2 events for
      more stable std estimates (std=0 can reflect single-event cases).
    """
    import matplotlib.pyplot as plt  # local import for notebook environments

    X = data.mean_duration.ravel()
    Y = data.std_duration.ravel()
    C = data.burstiness.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(X, Y, c=C, cmap=cmap, s=s, alpha=alpha, vmin=vmin, vmax=vmax)
    ax.set_xlabel("mean duration")
    ax.set_ylabel("std duration")
    ax.set_title("mean vs std duration (colored by burstiness)")
    ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("burstiness [-1, 1]")
    fig.tight_layout()
    plt.show()


def example(paths: dict, *, window_size: int = 9, lag: int = 1, tau: int = 3, scope: str = "all") -> dict[str, Any]:
    """Load, list top links, and draw a couple of basic plots."""
    data = load_summaries(paths, window_size=window_size, lag=lag, tau=tau, scope=scope)
    events = load_events(paths, window_size=window_size, lag=lag, tau=tau, scope=scope)
    links = link_table(data)
    top = top_links_by(data.burstiness, links, k=10, agg="mean")
    plot_top_time_ratio_bars(data.time_ratio, top, title="Top 10 links by mean burstiness (time_ratio shown)")
    plot_duration_hist(events, link_idx=int(top.iloc[0].link), bins=25)
    scatter_mean_vs_burstiness(data, by="mean_duration")
    scatter_burstiness_vs_std_duration(data)
    scatter_mean_std_colored(data)
    return {"data": data, "events": events, "links": links, "top": top}


if __name__ == "__main__":
    # Minimal non-interactive example
    p = get_paths(timecourse_folder="Timecourses_updated_03052024")
    try:
        example(p, window_size=9, lag=1, tau=3, scope="dmn")
        # example(p, window_size=9, lag=1, tau=3, scope="all")
    except FileNotFoundError as e:
        print("Cohesion summaries/events not found. Run cohesion_compute.py first.")
        print(e)


# %%
