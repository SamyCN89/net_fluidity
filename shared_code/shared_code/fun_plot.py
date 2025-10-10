"""
Lightweight matplotlib/stat helpers shared across plotting scripts.

Public functions provide type hints and concise docstrings per repo standards.
"""

from __future__ import annotations

import re
from typing import Iterable

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.transforms import blended_transform_factory


def add_sig_bar_axes(
    ax: Axes,
    x1: float,
    x2: float,
    y_axes: float,
    text: str,
    *,
    h_axes: float = 0.02,
    lw: float = 1.0,
    fontsize: int = 8,
) -> None:
    """Draw a significance bar with text using axes y-coordinates.

    - x is in data coordinates; y is in axes coordinates (0..1).
    - If ``h_axes`` is negative, ticks point downward and the label is placed above.
    - Colors by p-value parsed from ``text``:
      p>0.1 → gray; 0.1≥p≥0.05 → pale red; p<0.05 → pale green.

    Parameters
    - ax: target axes
    - x1, x2: data x-positions to connect
    - y_axes: baseline y (axes coords)
    - text: label to draw between ticks (e.g., "p=0.012")
    - h_axes: tick height in axes coords (default 0.02)
    - lw: line width
    - fontsize: label font size
    """
    trans = blended_transform_factory(ax.transData, ax.transAxes)

    # determine bar/text color based on p-value in label
    match = re.search(r"p\s*=\s*([0-9]*\.?[0-9]+(?:e-?\d+)?)", text, re.I)
    p_value = float(match.group(1)) if match else None
    if p_value is None:
        color = "black"
    elif p_value > 0.1:
        color = "gray"
    elif p_value >= 0.05:
        color = "lightcoral"  # pale red
    else:
        color = "palegreen"

    y0, y1 = y_axes, y_axes + h_axes
    ax.plot([x1, x1, x2, x2], [y0, y1, y1, y0], transform=trans, color=color, lw=lw, clip_on=False)

    va = "bottom" if h_axes >= 0 else "top"
    ax.text((x1 + x2) * 0.5, y1, text, transform=trans, ha="center", va=va, fontsize=fontsize, color=color, clip_on=False)


def errbar(values: np.ndarray, mode: str = "sd") -> float:
    """Return dispersion for ``values``: 'sd', 'sem', or 'var'.

    Returns NaN for empty input and 0.0 when a single sample is provided.
    """
    n = int(values.size)
    if n == 0:
        return float("nan")
    if n == 1:
        return 0.0
    if mode == "sd":
        return float(np.std(values, ddof=1))
    if mode == "sem":
        return float(np.std(values, ddof=1) / np.sqrt(n))
    return float(np.var(values, ddof=1))


def annotate_inset(ax: Axes, lines: Iterable[str]) -> None:
    """Draw a small text box with ``lines`` in the lower-right corner."""
    txt = "\n".join([line for line in lines if line])
    if not txt:
        return
    ax.text(
        0.98,
        0.02,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
    )


def compute_pvalue(v_left: np.ndarray, v_right: np.ndarray, *, paired: bool) -> float:
    """Return a two-sided p-value (Wilcoxon if paired, else Mann–Whitney).

    Returns NaN if inputs are empty or if the test fails.
    """
    from scipy.stats import mannwhitneyu  # local import to keep dependency optional

    if v_left.size == 0 or v_right.size == 0:
        return float("nan")
    try:
        if paired:
            from scipy.stats import wilcoxon

            return float(wilcoxon(v_left, v_right, zero_method="wilcox", alternative="two-sided").pvalue)
        return float(mannwhitneyu(v_left, v_right, alternative="two-sided").pvalue)
    except Exception:
        return float("nan")
