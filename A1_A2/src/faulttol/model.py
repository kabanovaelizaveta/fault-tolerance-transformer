"""Transformer encoders (and a decoder-only variant) with head fault injection.

Faithful port of the MultiHeadAttention / EncoderLayer classes in
transformers_EXP_A.ipynb and transformers_EXP_B (1).ipynb.  The crash-fault
model of Definition 1 is implemented as a per-head binary mask applied to the
head output tensor AFTER (attn @ V) and BEFORE concatenation and the output
projection W_o.

The AG News text model also accepts a key-padding mask.  PAD token embeddings
are zero, but positional encodings make padded positions non-zero again; the
mask therefore has to be applied to the attention logits before softmax.

store_maps avoids unconditional caching of last_attn / last_head_output on
every forward pass.  Caching is off by default and switched on only when
attention statistics are actually collected.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (identical in both notebooks)."""

    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class LearnedPositionalEmbedding(nn.Module):
    """Learnable absolute positional embedding, as used by ViT, BERT and GPT-2."""

    def __init__(self, d_model: int, max_seq_length: int):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        nn.init.trunc_normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with a crash mask per head.

    head_mask[h] == 0 means head h has crashed: its output tensor is replaced
    by zeros after the attention weights multiply V and before the heads are
    concatenated and projected by W_o.

    causal=True masks every key strictly to the right of the query, which is
    the decoder-only (GPT-2) arrangement.  The diagonal is always kept, so no
    softmax row is fully masked.
    """

    def __init__(self, d_model: int, num_heads: int, qkv_bias: bool = True,
                 bias: bool = True, causal: bool = False,
                 max_seq_length: int | None = None):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if causal and max_seq_length is None:
            raise ValueError("causal attention needs max_seq_length")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.causal = causal

        # qkv_bias=False reproduces the ViT convention of biases everywhere
        # except on the key, query and value projections; bias=False drops the
        # remaining ones too, which is the GPT-2 convention.
        self.W_q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        if causal:
            # persistent=False: the mask is a constant, so it stays out of the
            # state dict and every existing checkpoint keeps loading.
            self.register_buffer(
                "causal_mask",
                torch.ones(max_seq_length, max_seq_length, dtype=torch.bool).tril(),
                persistent=False,
            )

        # fault injection: crashed head contributes zeros
        self.register_buffer("head_mask", torch.ones(num_heads))
        self.store_maps = False
        self.last_attn = None
        self.last_head_output = None

    def set_head_mask(self, mask) -> None:
        mask = torch.as_tensor(mask, dtype=self.head_mask.dtype,
                               device=self.head_mask.device)
        if mask.shape != self.head_mask.shape:
            raise ValueError(
                "Expected mask shape %s, got %s"
                % (tuple(self.head_mask.shape), tuple(mask.shape))
            )
        self.head_mask.copy_(mask)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape

        Q = self.W_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.causal:
            if L > self.causal_mask.size(0):
                raise ValueError(
                    "sequence length %d exceeds the causal mask size %d"
                    % (L, self.causal_mask.size(0))
                )
            scores = scores.masked_fill(~self.causal_mask[:L, :L], float("-inf"))
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, L):
                raise ValueError(
                    "Expected key_padding_mask shape %s, got %s"
                    % ((B, L), tuple(key_padding_mask.shape))
                )
            # True means that the position is padding and cannot be used as a
            # key/value.  The CLS position is always False, so every softmax
            # row retains at least one finite score.
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :].to(torch.bool),
                float("-inf"),
            )
        attn = torch.softmax(scores, dim=-1)
        head_output = attn @ V

        if self.store_maps:
            self.last_attn = attn.detach()
            self.last_head_output = head_output.detach()

        mask = self.head_mask.view(1, -1, 1, 1)
        head_output = head_output * mask
        head_output = head_output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.W_o(head_output)


class EncoderLayer(nn.Module):
    """Encoder layer, post-norm by default (as in both source notebooks).

    norm_first=True switches to the pre-normalisation arrangement used by ViT
    and GPT-2, where the normalisation sits inside the residual branch.
    """

    _ACTS = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout: float = 0.05, activation: str = "relu",
                 norm_first: bool = False, qkv_bias: bool = True,
                 bias: bool = True, causal: bool = False,
                 max_seq_length: int | None = None):
        super().__init__()
        if activation not in self._ACTS:
            raise ValueError("Unknown activation: %s" % activation)

        self.norm_first = norm_first
        self.attn = MultiHeadAttention(d_model, num_heads, qkv_bias=qkv_bias,
                                       bias=bias, causal=causal,
                                       max_seq_length=max_seq_length)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            self._ACTS[activation](),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )
        self.norm1 = nn.LayerNorm(d_model, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            x = x + self.dropout(
                self.attn(self.norm1(x), key_padding_mask=key_padding_mask)
            )
            x = x + self.dropout(self.ff(self.norm2(x)))
        else:
            x = self.norm1(
                x + self.dropout(self.attn(x, key_padding_mask=key_padding_mask))
            )
            x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class _EncoderBackbone(nn.Module):
    """CLS-token encoder stack shared by every task head."""

    def __init__(self, config: dict, max_len: int, activation: str):
        super().__init__()
        self.d_model = config["d_model"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        # Optional keys keep every existing config and checkpoint working: the
        # defaults are exactly the original post-norm, sinusoidal-PE model.
        norm_first = bool(config.get("norm_first", False))
        qkv_bias = bool(config.get("qkv_bias", True))
        bias = bool(config.get("bias", True))
        self.causal = bool(config.get("causal", False))
        # Under a causal mask a CLS token at position 0 could attend only to
        # itself, so the decoder-only configuration appends it instead and the
        # read-out moves to the last position.  Default stays position 0.
        self.cls_last = bool(config.get("cls_last", self.causal))
        if self.causal and not self.cls_last:
            raise ValueError(
                "causal=True with cls_last=False would leave the CLS token "
                "attending only to itself"
            )
        self.pos = (
            LearnedPositionalEmbedding(config["d_model"], max_len + 1)
            if config.get("learned_pe", False)
            else PositionalEncoding(config["d_model"], max_len + 1)
        )
        self.cls = nn.Parameter(torch.randn(1, 1, config["d_model"]))
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=config["d_model"],
                    num_heads=config["num_heads"],
                    d_ff=config["d_ff"],
                    dropout=config["dropout"],
                    activation=activation,
                    norm_first=norm_first,
                    qkv_bias=qkv_bias,
                    bias=bias,
                    causal=self.causal,
                    max_seq_length=max_len + 1,
                )
                for _ in range(config["num_layers"])
            ]
        )
        # A pre-norm stack leaves its output un-normalised, so ViT and GPT-2
        # both apply a final normalisation before the task head.  Post-norm
        # already ends in a LayerNorm, hence Identity there: no new parameters
        # and no change to existing state dicts.
        self.norm_out = (
            nn.LayerNorm(config["d_model"], bias=bias) if norm_first else nn.Identity()
        )
        self.dropout = nn.Dropout(config["dropout"])

    def encode(self, tokens: torch.Tensor,
               key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (B, L, d_model) -> CLS representation (B, d_model)."""
        B = tokens.size(0)
        cls = self.cls.expand(B, 1, -1)
        x = torch.cat([tokens, cls], dim=1) if self.cls_last \
            else torch.cat([cls, tokens], dim=1)
        if key_padding_mask is not None:
            if key_padding_mask.shape != tokens.shape[:2]:
                raise ValueError(
                    "Expected token padding mask shape %s, got %s"
                    % (tuple(tokens.shape[:2]), tuple(key_padding_mask.shape))
                )
            cls_mask = torch.zeros(
                (B, 1), dtype=torch.bool, device=key_padding_mask.device
            )
            key_padding_mask = (
                torch.cat([key_padding_mask, cls_mask], dim=1)
                if self.cls_last
                else torch.cat([cls_mask, key_padding_mask], dim=1)
            )
        x = self.pos(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        x = self.norm_out(x)
        return x[:, -1] if self.cls_last else x[:, 0]


class TextClassifier(_EncoderBackbone):
    """Token-embedding encoder classifier (AG News); mirrors EXP_A."""

    def __init__(self, vocab_size: int, config: dict, max_len: int,
                 num_classes: int, activation: str = "relu"):
        super().__init__(config, max_len, activation)
        self.embedding = nn.Embedding(vocab_size, config["d_model"], padding_idx=0)
        self.classifier = nn.Linear(config["d_model"], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # padding_idx=0 keeps the token embedding at zero, while this explicit
        # mask also removes PAD keys/values after positional encoding is added.
        key_padding_mask = x.eq(0)
        return self.classifier(
            self.encode(self.embedding(x), key_padding_mask=key_padding_mask)
        )


class ImageClassifier(_EncoderBackbone):
    """ViT-style patch-embedding encoder classifier (CIFAR-10)."""

    def __init__(self, config: dict, image_size: int, patch_size: int,
                 in_channels: int, num_classes: int, activation: str = "relu"):
        n_patches = (image_size // patch_size) ** 2
        super().__init__(config, n_patches, activation)
        self.patch = nn.Conv2d(in_channels, config["d_model"],
                               kernel_size=patch_size, stride=patch_size)
        self.n_patches = n_patches
        self.classifier = nn.Linear(config["d_model"], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.patch(x).flatten(2).transpose(1, 2)  # (B, n_patches, d_model)
        return self.classifier(self.encode(t))


class SeriesRegressor(_EncoderBackbone):
    """Scalar-projection encoder regressor (PJM load); mirrors EXP_B."""

    def __init__(self, config: dict, max_len: int, activation: str = "silu"):
        super().__init__(config, max_len, activation)
        self.input_proj = nn.Linear(1, config["d_model"])
        self.regressor = nn.Linear(config["d_model"], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.input_proj(x.unsqueeze(-1))
        return self.regressor(self.encode(t)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
