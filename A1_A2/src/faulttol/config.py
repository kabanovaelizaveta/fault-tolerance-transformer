"""Shared configuration for the attention-head fault-tolerance experiments.

The architecture is deliberately held FIXED across all datasets (the "Base"
configuration of Paper B / transformers_EXP_A / transformers_EXP_B) so that
layer-position findings are comparable across modalities.  Only the input
embedding, the task head and the activation follow the source notebook for the
corresponding task family.
"""
from __future__ import annotations

import os


def _env_int(name, default):
    """Allow smoke runs / reduced budgets without editing the file."""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default

# Base architecture: 4 encoder layers x 8 heads = 32 attention heads.
BASE_PARAM = {
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.05,
}

MODEL_NAME = "base_transformer"

# Second architecture, used by the experiments/tiny_vit family: the Tiny ViT of
# "Dropout and the Outliers" (Hezbri et al.), which is 6 blocks x 6 heads =
# 36 attention heads at width 384 with an MLP expansion factor of 4.  Relative
# to BASE_PARAM this changes depth, width, head count, head width (d_k 64 not
# 32), normalisation placement, positional encoding and the q/k/v biases, so it
# is a bundle of choices rather than a single-factor contrast - see
# experiments/tiny_vit/README.md.
TINY_VIT_PARAM = {
    "d_model": 384,
    "num_heads": 6,
    "num_layers": 6,
    "d_ff": 1536,
    "dropout": 0.05,
    "norm_first": True,     # ViT places the norm inside the residual branch
    "qkv_bias": False,      # ViT: biases everywhere except q, k and v
    "learned_pe": True,     # ViT: learnable absolute positional embedding
}

# Third architecture, used by the experiments/tiny_gpt2 family: the Tiny GPT-2
# of the same paper (Table 4), which has the SAME 6 blocks x 6 heads at width
# 384 with MLP expansion 4 as the Tiny ViT above - the paper deliberately gives
# all three of its models one shape.  So relative to TINY_VIT_PARAM only two
# things change, and the first is nearly free in parameter terms:
#   bias=False    GPT-2 has no biases at all, where ViT keeps them everywhere
#                 except q/k/v.  This removes 18,432 of 10,639,872 stack
#                 parameters (0.17%).
#   causal=True   GPT-2 is decoder-only: every query attends to its own
#                 position and to the left only.  This is the substantive
#                 contrast - bidirectional versus autoregressive attention -
#                 and it is what makes this family worth running.
# cls_last follows from causal: see _EncoderBackbone in model.py.
TINY_GPT2_PARAM = {
    "d_model": 384,
    "num_heads": 6,
    "num_layers": 6,
    "d_ff": 1536,
    "dropout": 0.05,
    "norm_first": True,     # GPT-2: pre-layer norms
    "qkv_bias": False,      # GPT-2: no biases anywhere ...
    "bias": False,          # ... including W_o, the FFN and the LayerNorms
    "learned_pe": True,     # GPT-2: learnable absolute positional embedding
    "causal": True,         # GPT-2: decoder-only, left-to-right attention
    "cls_last": True,       # read-out token must sit where it can see the input
}

# Seeds, copied from transformers_EXP_A / transformers_EXP_B.
DATA_SEED = 42
SPLIT_SEED = 42
TRAIN_SEED = 42
FAULT_SPLIT_SEED = 2026
MULTIHEAD_FAULT_SEED = 1024
FAULT_ANALYSIS_SEED = 999
BOOTSTRAP_SEED = 2026

# Fault-injection protocol (Paper B, Section 3.3.2).
K_TRIALS = _env_int("FT_K_TRIALS", 10_000)   # random multi-head fault configurations per model
N_FAULT = _env_int("FT_N_FAULT", 1_000)      # fixed evaluation subset size used for every trial
BATCH_SIZE = 128

# Copula analysis.
COPULA_FAMILIES = ("clayton", "gumbel", "frank", "survival_clayton", "survival_gumbel")
CVM_BOOTSTRAP_REPS = _env_int("FT_BOOTSTRAP", 500)

# Safe-region analysis (Paper B, Section 3.6).
CRITICAL_TOP_FRACTION = 0.10
SAFE_PROB = 0.95
MIN_CONFIGS_PER_BOUNDARY = 100
CLS_TAUS = (0.010, 0.025, 0.050)          # absolute D_pos_mean thresholds
REG_ALPHAS = (0.10, 0.25, 0.50)           # fractions of clean MAPE
