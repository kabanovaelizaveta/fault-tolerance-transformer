"""Figures: vulnerability maps, degradation grids, pseudo-observations, tau curves."""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, path: pathlib.Path) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def vulnerability_map(single_head_df: pd.DataFrame, title: str,
                      path: pathlib.Path, value_col: str = "criticality"):
    """Layer x head heatmap of single-head criticality (Figure 2 analogue)."""
    pivot = single_head_df.pivot(index="layer", columns="head", values=value_col)
    fig, ax = plt.subplots(figsize=(1.0 + 0.7 * pivot.shape[1], 1.2 + 0.6 * pivot.shape[0]))
    im = ax.imshow(pivot.to_numpy(), cmap="Blues", aspect="auto", vmin=0.0)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("head")
    ax.set_ylabel("encoder layer")
    ax.set_title(title, fontsize=10)
    vmax = float(np.nanmax(pivot.to_numpy()))
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            ax.text(j, i, "%.3f" % val, ha="center", va="center", fontsize=6,
                    color="white" if val > 0.55 * vmax else "black")
    fig.colorbar(im, ax=ax, label=value_col)
    return _save(fig, path)


def degradation_grid_plot(grid: pd.DataFrame, title: str, path: pathlib.Path,
                          value_label: str):
    """Mean degradation by fault fraction x criticality quartile (Fig 3/4)."""
    # Match the source-paper layout: quartiles on the vertical axis and failed
    # fraction bins on the horizontal axis.
    plot_grid = grid.T
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    data = plot_grid.to_numpy(dtype=float)
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0.0)
    ax.set_xticks(range(plot_grid.shape[1]))
    ax.set_xticklabels(plot_grid.columns, rotation=40, ha="right")
    ax.set_yticks(range(plot_grid.shape[0]))
    ax.set_yticklabels(plot_grid.index)
    ax.set_xlabel("failed-head fraction")
    ax.set_ylabel("mean failed-head criticality")
    ax.set_title(title, fontsize=10)
    finite = data[np.isfinite(data)]
    vmax = float(finite.max()) if finite.size else 1.0
    for i in range(plot_grid.shape[0]):
        for j in range(plot_grid.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, "%.3f" % val, ha="center", va="center", fontsize=7,
                        color="white" if val > 0.55 * vmax else "black")
    fig.colorbar(im, ax=ax, label=value_label)
    return _save(fig, path)


def pseudo_observation_panel(pairs: list[tuple[str, np.ndarray, np.ndarray]],
                             title: str, path: pathlib.Path):
    """Pseudo-observation scatter for each severity/degradation pair (Fig 5-8)."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.4 * len(pairs), 3.4))
    if len(pairs) == 1:
        axes = [axes]
    for ax, (label, u, v) in zip(axes, pairs):
        ax.scatter(u, v, s=1.5, alpha=0.25, color="#1f4e79", linewidths=0)
        ax.set_xlabel("u  (%s)" % label.split(" vs ")[0])
        ax.set_ylabel("v  (%s)" % label.split(" vs ")[-1])
        ax.set_title(label, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
    fig.suptitle(title, fontsize=10)
    return _save(fig, path)


def required_tau_plot(tau_df: pd.DataFrame, title: str, path: pathlib.Path,
                      ylabel: str):
    """tau_required as the admissible failure boundary widens (Figure 9)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    if tau_df.empty:
        ax.text(0.5, 0.5, "no boundary reached the minimum configuration count",
                ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return _save(fig, path)
    for c, sub in tau_df.groupby("n_crit_fail_max"):
        sub = sub.sort_values("n_fail_max")
        ax.plot(sub["n_fail_max"], sub["required_tau"], marker="o", markersize=3,
                label="N_crit_fail <= %d" % int(c))
    ax.set_xlabel("maximum total failed heads (n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    return _save(fig, path)


def criticality_profile(single_head_df: pd.DataFrame, title: str,
                        path: pathlib.Path, ylabel: str):
    """Sorted criticality profile: how concentrated is head importance."""
    vals = np.sort(single_head_df["criticality"].to_numpy())[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.bar(range(len(vals)), vals, color="#2f75b5")
    ax.set_xlabel("heads, ranked by criticality")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, path)


def degradation_curves(curves: dict, title: str, path: pathlib.Path,
                       ylabel: str, normalise: bool = False,
                       xlabel: str = "number of failed attention heads"):
    """Mean degradation vs number of failed heads, one line per dataset.

    curves: {label: (n_fail array, degradation array)}
    normalise: divide each curve by its own maximum, so datasets measured on
    different scales (probability damage vs delta MAE) can share an axis.
    """
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for label, (x, y) in curves.items():
        grouped = pd.DataFrame({"n": np.asarray(x), "y": np.asarray(y)}) \
            .groupby("n")["y"].mean()
        vals = grouped.to_numpy(dtype=float)
        if normalise:
            m = float(np.nanmax(np.abs(vals))) if vals.size else 0.0
            if m > 0:
                vals = vals / m
        ax.plot(grouped.index.to_numpy(), vals, marker="o", markersize=2.5,
                linewidth=1.2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, path)


def criticality_comparison(profiles: dict, title: str, path: pathlib.Path,
                           ylabel: str = "criticality / max criticality"):
    """Sorted, self-normalised criticality profiles for several models."""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for label, vals in profiles.items():
        v = np.sort(np.asarray(vals, dtype=float))[::-1]
        m = float(v.max()) if v.size else 0.0
        ax.plot(range(len(v)), v / m if m > 0 else v, marker="o", markersize=2.5,
                linewidth=1.2, label=label)
    ax.set_xlabel("heads, ranked by criticality")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    return _save(fig, path)
