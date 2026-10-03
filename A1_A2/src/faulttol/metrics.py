"""Clean-performance and degradation metrics.

Classification metrics reproduce compute_damage_metrics / metrics_from_outputs
of transformers_EXP_A (probability damage on the true class).  Regression
metrics reproduce the EXP_B error-change family (delta MAE / MAPE and mean
harmful absolute-error change).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------
# forward passes
# --------------------------------------------------------------------------
@torch.inference_mode()
def classification_outputs(model, X: torch.Tensor, y: np.ndarray, device,
                           chunk: int = 512) -> dict:
    """Probabilities assigned to the true class, plus hard predictions."""
    model.eval()
    p_true, preds, p_all = [], [], []
    y_t = torch.as_tensor(y, dtype=torch.long)
    for i in range(0, X.size(0), chunk):
        xb = X[i : i + chunk].to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits.float(), dim=-1)
        yb = y_t[i : i + chunk].to(probs.device)
        p_true.append(probs.gather(1, yb.view(-1, 1)).squeeze(1).cpu().numpy())
        preds.append(probs.argmax(dim=-1).cpu().numpy())
        p_all.append(probs.cpu().numpy())
    return {
        "y_true": np.asarray(y),
        "p_true": np.concatenate(p_true),
        "y_pred": np.concatenate(preds),
        "p_all": np.concatenate(p_all, axis=0),
    }


@torch.inference_mode()
def regression_outputs(model, X: torch.Tensor, y: np.ndarray, device,
                       chunk: int = 512) -> dict:
    model.eval()
    yhat = []
    for i in range(0, X.size(0), chunk):
        xb = X[i : i + chunk].to(device, non_blocking=True)
        yhat.append(model(xb).float().cpu().numpy())
    return {"y_true": np.asarray(y, dtype=np.float64),
            "y_pred": np.concatenate(yhat).astype(np.float64)}


# --------------------------------------------------------------------------
# clean performance
# --------------------------------------------------------------------------
def classification_clean_metrics(out: dict) -> dict:
    y, pred, p_all = out["y_true"], out["y_pred"], out["p_all"]
    n_classes = p_all.shape[1]
    avg = "binary" if n_classes == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y, pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y, pred, average=avg, zero_division=0)),
    }
    try:
        if n_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(y, p_all[:, 1]))
        else:
            metrics["roc_auc"] = float(
                roc_auc_score(y, p_all, multi_class="ovr", average="macro")
            )
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def regression_clean_metrics(out: dict,
                             denorm: Callable[[np.ndarray], np.ndarray] | None = None) -> dict:
    y, yhat = out["y_true"], out["y_pred"]
    err = y - yhat
    metrics = {
        "mse": float(np.mean(err ** 2)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
    }
    if denorm is not None:
        yo, yho = denorm(y), denorm(yhat)
        metrics["mape"] = float(100.0 * np.mean(np.abs((yo - yho) / yo)))
    return metrics


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------
def classification_damage(clean: dict, fault: dict) -> dict:
    """Probability damage D_i = p_clean(true) - p_fault(true) and summaries."""
    if not np.array_equal(clean["y_true"], fault["y_true"]):
        raise ValueError("Clean and fault outputs use different samples or order.")

    damage = clean["p_true"] - fault["p_true"]
    positive = np.maximum(damage, 0.0)
    negative = np.minimum(damage, 0.0)
    absolute = np.abs(damage)

    clean_m = classification_clean_metrics(clean)
    fault_m = classification_clean_metrics(fault)

    return {
        "D_mean": float(np.mean(damage)),
        "D_pos_mean": float(np.mean(positive)),
        "D_neg_mean": float(np.mean(negative)),
        "D_abs_mean": float(np.mean(absolute)),
        "D_q90": float(np.quantile(damage, 0.90)),
        "D_q95": float(np.quantile(damage, 0.95)),
        "D_q99": float(np.quantile(damage, 0.99)),
        "D_pos_q90": float(np.quantile(positive, 0.90)),
        "D_pos_q95": float(np.quantile(positive, 0.95)),
        "D_pos_q99": float(np.quantile(positive, 0.99)),
        "D_abs_q95": float(np.quantile(absolute, 0.95)),
        "harmful_rate_001": float(np.mean(damage > 0.01)),
        "harmful_rate_005": float(np.mean(damage > 0.05)),
        "harmful_rate_010": float(np.mean(damage > 0.10)),
        "helpful_rate_001": float(np.mean(damage < -0.01)),
        "helpful_rate_005": float(np.mean(damage < -0.05)),
        "helpful_rate_010": float(np.mean(damage < -0.10)),
        "f1_clean": clean_m["f1"],
        "f1_fault": fault_m["f1"],
        "delta_f1": float(clean_m["f1"] - fault_m["f1"]),
        "acc_clean": clean_m["accuracy"],
        "acc_fault": fault_m["accuracy"],
        "delta_accuracy": float(clean_m["accuracy"] - fault_m["accuracy"]),
    }


def regression_damage(clean: dict, fault: dict,
                      denorm: Callable[[np.ndarray], np.ndarray] | None = None) -> dict:
    """Sample-level error change E_i and the delta MAE / MAPE summaries."""
    if not np.array_equal(clean["y_true"], fault["y_true"]):
        raise ValueError("Clean and fault outputs use different samples or order.")

    y = clean["y_true"]
    abs_clean = np.abs(y - clean["y_pred"])
    abs_fault = np.abs(y - fault["y_pred"])
    error_change = abs_fault - abs_clean          # > 0 means degradation
    harmful = np.maximum(error_change, 0.0)

    out = {
        "mae_clean": float(np.mean(abs_clean)),
        "mae_fault": float(np.mean(abs_fault)),
        "delta_mae": float(np.mean(abs_fault) - np.mean(abs_clean)),
        "mse_clean": float(np.mean((y - clean["y_pred"]) ** 2)),
        "mse_fault": float(np.mean((y - fault["y_pred"]) ** 2)),
        "delta_mse": float(np.mean((y - fault["y_pred"]) ** 2)
                           - np.mean((y - clean["y_pred"]) ** 2)),
        "mean_error_change": float(np.mean(error_change)),
        "mean_harmful_error_change": float(np.mean(harmful)),
        "q95_harmful_error_change": float(np.quantile(harmful, 0.95)),
        "q99_harmful_error_change": float(np.quantile(harmful, 0.99)),
        "harmful_rate": float(np.mean(error_change > 0.0)),
    }
    if denorm is not None:
        yo = denorm(y)
        mape_clean = 100.0 * np.mean(np.abs((yo - denorm(clean["y_pred"])) / yo))
        mape_fault = 100.0 * np.mean(np.abs((yo - denorm(fault["y_pred"])) / yo))
        out["mape_clean"] = float(mape_clean)
        out["mape_fault"] = float(mape_fault)
        out["delta_mape"] = float(mape_fault - mape_clean)
    return out


# Column used as the criticality signal and as the copula degradation variable.
CLS_CRITICALITY_COL = "D_pos_mean"
CLS_DEGRADATION_COL = "D_pos_mean"
REG_CRITICALITY_COL = "mean_harmful_error_change"
REG_DEGRADATION_COL = "delta_mae"
