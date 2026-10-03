"""Attention-head fault injection helpers.

Direct port of the injection utilities in transformers_EXP_A / EXP_B:
reset_head_masks, disable_single_head, disable_multiple_heads, get_all_heads,
get_layer_fault_counts, get_number_of_affected_layers, format_failed_heads.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch

from .model import MultiHeadAttention


def attention_modules(model) -> list[MultiHeadAttention]:
    """Attention modules of the encoder stack, ordered by layer index."""
    return [layer.attn for layer in model.layers]


def reset_head_masks(model) -> None:
    """Restore every head mask to 1 (fault-free model)."""
    for attn in attention_modules(model):
        attn.head_mask.fill_(1.0)


def set_store_maps(model, flag: bool) -> None:
    for attn in attention_modules(model):
        attn.store_maps = flag


def get_all_heads(model) -> list[tuple[int, int]]:
    """All (layer, head) pairs in the model."""
    heads: list[tuple[int, int]] = []
    for layer_id, attn in enumerate(attention_modules(model)):
        heads.extend((layer_id, h) for h in range(attn.num_heads))
    return heads


def disable_single_head(model, layer_id: int, head_id: int) -> None:
    """Crash exactly one head; every other head stays alive."""
    reset_head_masks(model)
    attn = attention_modules(model)[layer_id]
    if not 0 <= head_id < attn.num_heads:
        raise ValueError("head_id %d out of range for layer %d" % (head_id, layer_id))
    attn.head_mask[head_id] = 0.0


def disable_multiple_heads(model, failed_heads: Iterable[tuple[int, int]]) -> None:
    """Crash all listed heads simultaneously."""
    reset_head_masks(model)
    mods = attention_modules(model)
    for layer_id, head_id in failed_heads:
        attn = mods[layer_id]
        if not 0 <= head_id < attn.num_heads:
            raise ValueError("head_id %d out of range for layer %d" % (head_id, layer_id))
        attn.head_mask[head_id] = 0.0


def format_failed_heads(failed_heads: Sequence[tuple[int, int]]) -> str:
    return ";".join("L%d_H%d" % (l, h) for l, h in sorted(failed_heads))


def head_name(layer_id: int, head_id: int) -> str:
    return "L%d_H%d" % (layer_id, head_id)


def get_layer_fault_counts(failed_heads: Iterable[tuple[int, int]],
                           num_layers: int) -> list[int]:
    counts = [0] * num_layers
    for layer_id, _ in failed_heads:
        counts[layer_id] += 1
    return counts


def get_number_of_affected_layers(layer_fault_counts: Sequence[int]) -> int:
    return int(sum(1 for c in layer_fault_counts if c > 0))


@torch.inference_mode()
def collect_attention_stats_per_head(model, loader, device) -> list[dict]:
    """Mean/max cumulative attention received per head on the clean model.

    Mirrors collect_attention_stats_per_head in EXP_A: for each head it
    summarises the column sums of the attention map (total attention each key
    receives from all queries), which is the quantity Paper A calls the
    cumulative attention contribution.
    """
    model.eval()
    reset_head_masks(model)
    set_store_maps(model, True)

    mods = attention_modules(model)
    sums = [torch.zeros(m.num_heads, dtype=torch.float64) for m in mods]
    maxima = [torch.zeros(m.num_heads, dtype=torch.float64) for m in mods]
    n_batches = 0

    for batch in loader:
        x = batch[0].to(device)
        model(x)
        for li, attn_mod in enumerate(mods):
            a = attn_mod.last_attn                       # (B, H, L, L)
            cumulative = a.sum(dim=2)                    # (B, H, L) per key
            sums[li] += cumulative.mean(dim=(0, 2)).double().cpu()
            maxima[li] = torch.maximum(
                maxima[li], cumulative.amax(dim=(0, 2)).double().cpu()
            )
        n_batches += 1

    set_store_maps(model, False)

    rows = []
    for li, m in enumerate(mods):
        for h in range(m.num_heads):
            rows.append(
                {
                    "layer": li,
                    "head": h,
                    "head_name": head_name(li, h),
                    "mean_cumulative_attention": float(sums[li][h] / max(n_batches, 1)),
                    "max_cumulative_attention": float(maxima[li][h]),
                }
            )
    return rows
