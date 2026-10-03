"""Training loops and checkpointing (ports of the notebook helpers)."""
from __future__ import annotations

import copy
import os
import pathlib
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import (
    classification_clean_metrics,
    classification_outputs,
    regression_clean_metrics,
    regression_outputs,
)


class EarlyStopping:
    """Port of the notebook EarlyStopping."""

    def __init__(self, patience: int = 6, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.stop = False

    def step(self, score: float) -> bool:
        if self.best is None:
            self.best = score
            return False
        improved = score > self.best if self.mode == "max" else score < self.best
        if improved:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True
        return self.stop


def save_checkpoint(path: pathlib.Path, model, optimizer, epoch: int,
                    best_score: float, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "config": config,
        }
    )
    _atomic_torch_save(payload, path)


def _atomic_torch_save(payload: dict, path: pathlib.Path) -> None:
    """Write a checkpoint without risking a half-written final file."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def load_checkpoint(path: pathlib.Path, model) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return ckpt


def make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int,
                shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False)


def train_classifier(model, X_train, y_train, X_val, y_val, device, *,
                     lr: float, weight_decay: float, epochs: int, batch_size: int,
                     patience: int, ckpt_path: pathlib.Path, config: dict,
                     selection_metric: str = "roc_auc", grad_clip: float | None = None,
                     cosine: bool = False, label_smoothing: float = 0.0,
                     augment=None, resume_path: pathlib.Path | None = None,
                     resume: bool = False) -> dict:
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        if cosine else None
    )
    stopper = EarlyStopping(patience=patience, mode="max")
    best_state = None

    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    loader = make_loader(X_train, y_train_t, batch_size, shuffle=True)

    history, best, best_epoch = [], -np.inf, None
    start_epoch = 1

    if resume and resume_path is not None and pathlib.Path(resume_path).is_file():
        try:
            saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise RuntimeError(
                "The resume checkpoint could not be read: %s. The file was not "
                "silently ignored, because that would restart training and waste "
                "compute. Check the .tmp file or restore the checkpoint." % resume_path
            ) from exc

        if saved.get("config") != config:
            raise ValueError(
                "The saved training checkpoint belongs to a different configuration. "
                "Set FORCE_RESTART_TRAINING=True only if you intentionally want a new run."
            )

        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        if scheduler is not None and saved.get("scheduler_state") is not None:
            scheduler.load_state_dict(saved["scheduler_state"])
        history = list(saved.get("history", []))
        best = float(saved.get("best_score", -np.inf))
        best_epoch = saved.get("best_epoch")
        start_epoch = int(saved["epoch"]) + 1
        stopper.best = saved.get("early_stopping_best", best)
        stopper.counter = int(saved.get("early_stopping_counter", 0))
        stopper.stop = bool(saved.get("early_stopping_stop", False))
        _restore_rng_state(saved.get("rng_state"))
        print(
            "    RESUME: completed epoch %d; continuing at epoch %d (best %.6f)"
            % (start_epoch - 1, start_epoch, best),
            flush=True,
        )

    if stopper.stop:
        print("    training was already completed by early stopping; no retraining", flush=True)
    elif start_epoch > epochs:
        print("    all %d epochs were already completed; no retraining" % epochs, flush=True)

    for epoch in range(start_epoch, epochs + 1):
        if stopper.stop:
            break
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for xb, yb in loader:
            if augment is not None:
                xb = augment(xb)
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        if scheduler is not None:
            scheduler.step()

        val = classification_clean_metrics(
            classification_outputs(model, X_val, y_val, device)
        )
        score = val[selection_metric]
        history.append({"epoch": epoch, "train_loss": total / max(n, 1),
                        "secs": time.time() - t0, **{"val_" + k: v for k, v in val.items()}})
        print(
            "    epoch %2d  loss %.4f  val_acc %.4f  val_f1 %.4f  val_auc %.4f  (%.0fs)"
            % (epoch, total / max(n, 1), val["accuracy"], val["f1"], val["roc_auc"],
               time.time() - t0),
            flush=True,
        )

        if score > best:
            best = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(ckpt_path, model, optimizer, epoch, best, config)
        should_stop = stopper.step(score)

        # Save the LAST completed epoch independently of the BEST model.  This
        # includes optimizer, scheduler, early-stopping, history and RNG state,
        # so a new Colab runtime continues with the next epoch.
        if resume_path is not None:
            _atomic_torch_save(
                {
                    "checkpoint_type": "last_completed_epoch",
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch,
                    "best_score": best,
                    "best_epoch": best_epoch,
                    "history": history,
                    "early_stopping_best": stopper.best,
                    "early_stopping_counter": stopper.counter,
                    "early_stopping_stop": stopper.stop,
                    "rng_state": _rng_state(),
                    "config": config,
                },
                pathlib.Path(resume_path),
            )
            print("    resume checkpoint saved:", resume_path, flush=True)

        if should_stop:
            print("    early stopping at epoch %d" % epoch, flush=True)
            break

    # Restore from memory, not from disk: another process running the same
    # notebook can overwrite the checkpoint file between save and reload.
    if best_state is not None:
        model.load_state_dict(best_state)
    elif pathlib.Path(ckpt_path).is_file():
        # Required when the current process resumed after the best epoch was
        # reached in an earlier Colab runtime.
        load_checkpoint(pathlib.Path(ckpt_path), model)
    return {
        "history": history,
        "best_score": float(best),
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "resumed_from_epoch": start_epoch - 1,
    }


def train_regressor(model, X_train, y_train, X_val, y_val, device, *,
                    lr: float, weight_decay: float, epochs: int, batch_size: int,
                    patience: int, ckpt_path: pathlib.Path, config: dict,
                    grad_clip: float | None = 1.0) -> dict:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopping(patience=patience, mode="min")
    best_state = None

    y_train_t = torch.as_tensor(y_train, dtype=torch.float32)
    loader = make_loader(X_train, y_train_t, batch_size, shuffle=True)

    history, best = [], np.inf
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        val = regression_clean_metrics(regression_outputs(model, X_val, y_val, device))
        score = val["mae"]
        history.append({"epoch": epoch, "train_mse": total / max(n, 1),
                        "secs": time.time() - t0, **{"val_" + k: v for k, v in val.items()}})
        print(
            "    epoch %2d  train_mse %.6f  val_mae %.6f  val_rmse %.6f  (%.0fs)"
            % (epoch, total / max(n, 1), val["mae"], val["rmse"], time.time() - t0),
            flush=True,
        )

        if score < best:
            best = score
            best_state = copy.deepcopy(model.state_dict())
            save_checkpoint(ckpt_path, model, optimizer, epoch, best, config)
        if stopper.step(score):
            print("    early stopping at epoch %d" % epoch, flush=True)
            break

    # Restore from memory, not from disk (see train_classifier).
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_score": float(best), "selection_metric": "val_mae"}
