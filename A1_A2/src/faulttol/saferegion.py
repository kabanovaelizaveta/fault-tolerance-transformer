"""Empirical probabilistic fault-tolerance regions (Paper B, Section 3.6).

A boundary (n, c) admits every fault configuration with at most n failed heads
in total and at most c failed critical heads.  It is 95%-safe for threshold tau
when

    P(Y <= tau | N_fail <= n, N_crit_fail <= c) >= 0.95

estimated from the random multi-head trials, and evaluated only when the
boundary contains at least MIN_CONFIGS_PER_BOUNDARY configurations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


GRID_COLUMNS = ["n_fail_max", "n_crit_fail_max", "n_configs", "coverage_percent",
                "p_safe", "is_safe", "tau", "required_tau"]

TAU_COLUMNS = ["n_fail_max", "n_crit_fail_max", "n_configs", "required_tau",
               "median_y", "max_y"]


def _boundary_mask(df: pd.DataFrame, n: int, c: int) -> np.ndarray:
    return (df["N_fail"].to_numpy() <= n) & (df["N_crit_fail"].to_numpy() <= c)


def safe_region_grid(df: pd.DataFrame, degradation_col: str, tau: float,
                     n_heads: int, n_critical: int, min_configs: int = 100,
                     safe_prob: float = 0.95) -> pd.DataFrame:
    """P_safe for every candidate boundary at one threshold."""
    y = df[degradation_col].to_numpy(dtype=float)
    total = len(df)
    rows = []
    for n in range(0, n_heads + 1):
        for c in range(0, n_critical + 1):
            mask = _boundary_mask(df, n, c)
            count = int(mask.sum())
            if count < min_configs:
                continue
            p_safe = float(np.mean(y[mask] <= tau))
            rows.append(
                {
                    "n_fail_max": n,
                    "n_crit_fail_max": c,
                    "n_configs": count,
                    "coverage_percent": 100.0 * count / total,
                    "p_safe": p_safe,
                    "is_safe": bool(p_safe >= safe_prob),
                    "tau": tau,
                    "required_tau": float(np.quantile(y[mask], safe_prob)),
                }
            )
    return pd.DataFrame(rows, columns=GRID_COLUMNS)


def widest_safe_boundary(grid: pd.DataFrame) -> dict:
    """The safe boundary admitting the most configurations."""
    if grid.empty:
        return {
            "tau": float("nan"), "safe_boundary": None, "n_fail_max": None,
            "n_crit_fail_max": None, "max_p_safe": float("nan"),
            "coverage_percent": 0.0, "smallest_required_tau": float("nan"),
            "n_boundaries_evaluated": 0,
        }
    safe = grid[grid["is_safe"].astype(bool)]
    if safe.empty:
        return {
            "tau": float(grid["tau"].iloc[0]) if len(grid) else float("nan"),
            "safe_boundary": None,
            "n_fail_max": None,
            "n_crit_fail_max": None,
            "max_p_safe": float(grid["p_safe"].max()) if len(grid) else float("nan"),
            "coverage_percent": 0.0,
            "smallest_required_tau": float(grid["required_tau"].min()),
            "n_boundaries_evaluated": int(len(grid)),
        }
    best = safe.loc[safe["coverage_percent"].idxmax()]
    return {
        "tau": float(best["tau"]),
        "safe_boundary": "N_fail<=%d, N_crit_fail<=%d"
                         % (int(best["n_fail_max"]), int(best["n_crit_fail_max"])),
        "n_fail_max": int(best["n_fail_max"]),
        "n_crit_fail_max": int(best["n_crit_fail_max"]),
        "max_p_safe": float(best["p_safe"]),
        "coverage_percent": float(best["coverage_percent"]),
        "smallest_required_tau": float(grid["required_tau"].min()),
        "n_boundaries_evaluated": int(len(grid)),
    }


def required_tau_table(df: pd.DataFrame, degradation_col: str, n_heads: int,
                       n_critical: int, min_configs: int = 100,
                       safe_prob: float = 0.95) -> pd.DataFrame:
    """tau_required(n, c) = 95th percentile of Y inside the boundary."""
    y = df[degradation_col].to_numpy(dtype=float)
    rows = []
    for n in range(0, n_heads + 1):
        for c in range(0, n_critical + 1):
            mask = _boundary_mask(df, n, c)
            if int(mask.sum()) < min_configs:
                continue
            rows.append(
                {
                    "n_fail_max": n,
                    "n_crit_fail_max": c,
                    "n_configs": int(mask.sum()),
                    "required_tau": float(np.quantile(y[mask], safe_prob)),
                    "median_y": float(np.median(y[mask])),
                    "max_y": float(np.max(y[mask])),
                }
            )
    return pd.DataFrame(rows, columns=TAU_COLUMNS)


def run_safe_region_analysis(df: pd.DataFrame, degradation_col: str, taus,
                             n_heads: int, n_critical: int,
                             min_configs: int = 100,
                             safe_prob: float = 0.95,
                             tau_labels=None):
    """Full analysis over several thresholds -> (grids, summary)."""
    grids, summary = [], []
    labels = list(tau_labels) if tau_labels is not None else [None] * len(list(taus))
    for tau, label in zip(taus, labels):
        grid = safe_region_grid(df, degradation_col, float(tau), n_heads,
                                n_critical, min_configs, safe_prob)
        if label is not None:
            grid["threshold_name"] = label
        grids.append(grid)
        row = widest_safe_boundary(grid)
        if label is not None:
            row["threshold_name"] = label
        summary.append(row)
    grids_df = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()
    return grids_df, pd.DataFrame(summary)
