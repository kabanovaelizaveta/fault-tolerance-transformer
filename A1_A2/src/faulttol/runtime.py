"""Runtime knobs: run device and thread count."""
from __future__ import annotations

import os

import torch


def configure_threads(default: int | None = None) -> int:
    """Set the torch thread count from FT_THREADS (0/unset = leave as is)."""
    try:
        n = int(os.environ.get("FT_THREADS", default or 0))
    except ValueError:
        n = 0
    if n > 0:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1) if torch.get_num_interop_threads() > 1 else None
    return torch.get_num_threads()


def select_device(prefer: str | None = None) -> torch.device:
    """Pick the run device: CUDA when available, CPU otherwise.

    FT_DEVICE overrides the choice - `FT_DEVICE=cpu` reproduces the original
    CPU-only runs, `FT_DEVICE=cuda:1` pins a specific GPU.  An explicit CUDA
    request raises instead of falling back to CPU: a silent fallback here means
    a multi-hour run that looks like it is on the GPU and is not.

    TF32 is left OFF on CUDA so that matmuls and the CIFAR patch convolution
    stay full fp32 and the damage numbers remain comparable with the CPU
    results; set FT_TF32=1 to trade that comparability for speed.
    """
    name = (os.environ.get("FT_DEVICE") or prefer or "").strip()
    explicit = bool(name)
    if not name:
        name = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested (%s) but torch.cuda.is_available() is False - "
                "this is a CPU-only torch build or the NVIDIA driver is not "
                "visible. Install a CUDA build of torch, or set FT_DEVICE=cpu."
                % ("FT_DEVICE=%s" % name if explicit else name)
            )
        if device.index is not None:
            if device.index >= torch.cuda.device_count():
                raise RuntimeError("cuda:%d requested but only %d GPU(s) visible"
                                   % (device.index, torch.cuda.device_count()))
            # plain "cuda" already means the current device, and set_device
            # rejects a device without an index
            torch.cuda.set_device(device)
        tf32 = os.environ.get("FT_TF32", "0") == "1"
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    return device


def device_report(device) -> str:
    """One-line description of the device actually in use."""
    device = torch.device(device)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        p = torch.cuda.get_device_properties(idx)
        return ("cuda:%d %s | %.1f GB | sm_%d%d | torch cuda %s | tf32 %s"
                % (idx, p.name, p.total_memory / 1024 ** 3, p.major, p.minor,
                   torch.version.cuda,
                   "on" if torch.backends.cuda.matmul.allow_tf32 else "off"))
    return "cpu | %d threads" % torch.get_num_threads()
