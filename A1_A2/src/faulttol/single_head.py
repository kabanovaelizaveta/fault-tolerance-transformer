"""Single-head fault injection: criticality, redundancy, layer fragility.

Paper B, Section 3.3.1 / 3.4.  For every head the model is evaluated with that
head crashed and nothing else changed, then

    C_{l,h} = mean over samples of max(damage_i, 0)          (criticality)
    R_{l,h} = 1 - C_{l,h} / (max_{j,k} C_{j,k} + eps)        (redundancy)
    W_{l,h} = C_{l,h} / (mean over heads of C + eps)         (severity weight)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .injection import (
    disable_single_head,
    get_all_heads,
    head_name,
    reset_head_masks,
)

EPS = 1e-12


def run_single_head_analysis(model, evaluate, damage_fn, criticality_col: str,
                             model_name: str, verbose: bool = True) -> pd.DataFrame:
    """Crash every head in turn and tabulate the damage it causes.

    evaluate() -> outputs dict for the current mask state.
    damage_fn(clean_outputs, fault_outputs) -> dict of scalar degradation metrics.
    """
    reset_head_masks(model)
    clean = evaluate()

    rows = []
    for layer_id, head_id in get_all_heads(model):
        disable_single_head(model, layer_id, head_id)
        fault = evaluate()
        row = {
            "model": model_name,
            "layer": layer_id,
            "head": head_id,
            "head_name": head_name(layer_id, head_id),
        }
        row.update(damage_fn(clean, fault))
        rows.append(row)
        if verbose:
            print(
                "  %-8s %s = %.6f"
                % (row["head_name"], criticality_col, row[criticality_col]),
                flush=True,
            )
    reset_head_masks(model)

    df = pd.DataFrame(rows)
    crit = df[criticality_col].to_numpy(dtype=float)
    crit_pos = np.maximum(crit, 0.0)
    df["criticality"] = crit_pos
    df["redundancy"] = 1.0 - crit_pos / (crit_pos.max() + EPS)
    df["severity_weight"] = crit_pos / (crit_pos.mean() + EPS)
    return df


def build_layer_summary(single_head_df: pd.DataFrame) -> pd.DataFrame:
    grouped = single_head_df.groupby("layer")
    summary = grouped.agg(
        mean_layer_criticality=("criticality", "mean"),
        max_head_criticality=("criticality", "max"),
        min_head_criticality=("criticality", "min"),
        mean_redundancy=("redundancy", "mean"),
        n_heads=("head", "count"),
    ).reset_index()
    summary["most_critical_head"] = [
        single_head_df.loc[single_head_df["layer"] == layer, "criticality"].idxmax()
        for layer in summary["layer"]
    ]
    summary["most_critical_head"] = single_head_df.loc[
        summary["most_critical_head"], "head_name"
    ].to_numpy()
    return summary


def build_model_summary(single_head_df: pd.DataFrame, layer_summary: pd.DataFrame,
                        model_name: str, clean_metrics: dict) -> dict:
    top = single_head_df.loc[single_head_df["criticality"].idxmax()]
    # "most fragile layer" is reported two ways because the DOCX and the EXP_B
    # notebook output disagree: by layer mean, and by the strongest single head.
    by_mean = layer_summary.loc[layer_summary["mean_layer_criticality"].idxmax()]
    by_max = layer_summary.loc[layer_summary["max_head_criticality"].idxmax()]
    return {
        "model": model_name,
        "n_heads": int(len(single_head_df)),
        "n_layers": int(single_head_df["layer"].nunique()),
        "most_critical_head": top["head_name"],
        "max_criticality": float(top["criticality"]),
        "mean_criticality": float(single_head_df["criticality"].mean()),
        "median_criticality": float(single_head_df["criticality"].median()),
        "min_criticality": float(single_head_df["criticality"].min()),
        "mean_redundancy": float(single_head_df["redundancy"].mean()),
        "n_heads_above_half_max": int(
            (single_head_df["criticality"] > 0.5 * top["criticality"]).sum()
        ),
        "most_fragile_layer_by_mean": int(by_mean["layer"]),
        "most_fragile_layer_by_mean_value": float(by_mean["mean_layer_criticality"]),
        "most_fragile_layer_by_max": int(by_max["layer"]),
        "most_fragile_layer_by_max_value": float(by_max["max_head_criticality"]),
        **{"clean_" + k: v for k, v in clean_metrics.items()},
    }


def build_head_weight_lookup(single_head_df: pd.DataFrame) -> dict:
    """(layer, head) -> normalised severity weight W_{l,h}."""
    return {
        (int(r.layer), int(r.head)): float(r.severity_weight)
        for r in single_head_df.itertuples()
    }


def critical_head_set(single_head_df: pd.DataFrame,
                      top_fraction: float = 0.10) -> list[tuple[int, int]]:
    """Top-fraction of heads ranked by criticality (Paper B, Section 3.6)."""
    # The source methodology defines the top fraction with ceil.  For the
    # Base model this is ceil(0.10 * 32) = 4 critical heads.
    n = max(1, int(math.ceil(top_fraction * len(single_head_df))))
    top = single_head_df.nlargest(n, "criticality")
    return [(int(r.layer), int(r.head)) for r in top.itertuples()]
