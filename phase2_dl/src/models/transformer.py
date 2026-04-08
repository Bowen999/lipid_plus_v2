"""
transformer.py — Multi-task Transformer for lipid structure prediction.

Architecture:
  Input: top-100 peaks as (norm_mz, norm_NL, sqrt_intensity) tokens
         + scalar precursor_mz (Da)
  Linear(3→128) token projection
  Learnable CLS token  +  precursor_mz register token (Linear(1→128))
  Sequence: [CLS, PMZ_REG, peak_1 … peak_100]  (length 102)
  6 × TransformerEncoderLayer(d_model=128, nheads=4, dim_ff=512, dropout=0.1)
  CLS token output (index 0) → 14 per-target classification heads

The precursor_mz register token gives the model explicit global mass context
that is otherwise only implicitly available via the neutral-loss channel.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import TARGETS, TRANS_TOP_K, TRANS_D_TOKEN

_D_MODEL  = 128
_N_HEADS  = 4
_DIM_FF   = 512
_N_LAYERS = 6
_DROPOUT  = 0.1


class LipidTransformer(nn.Module):
    """
    Transformer-based multi-task model with precursor_mz register token.

    All heads receive the CLS token representation.

    Parameters
    ----------
    n_classes : {target_name: number_of_classes}
    """

    INPUT_KEYS = ["tokens", "precursor_mz"]

    def __init__(self, n_classes: dict[str, int]) -> None:
        super().__init__()

        # Project 3-dim peak token to d_model
        self.token_proj = nn.Linear(TRANS_D_TOKEN, _D_MODEL)

        # Precursor m/z register: scalar (in Da) → d_model embedding
        self.pmz_proj = nn.Linear(1, _D_MODEL)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, _D_MODEL))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = _D_MODEL,
            nhead           = _N_HEADS,
            dim_feedforward = _DIM_FF,
            dropout         = _DROPOUT,
            batch_first     = True,
            norm_first      = True,    # pre-norm (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=_N_LAYERS)

        # Per-target classification heads
        self.heads = nn.ModuleDict({
            t: nn.Linear(_D_MODEL, n_classes[t]) for t in TARGETS
        })

        # Weight initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.token_proj.weight)
        nn.init.xavier_uniform_(self.pmz_proj.weight)
        nn.init.zeros_(self.pmz_proj.bias)
        for head in self.heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        precursor_mz: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        tokens       : (B, TRANS_TOP_K, 3) float32
                       Zero-padded rows (all-zero token) indicate absent peaks.
        precursor_mz : (B,) float32  — original precursor m/z in Da

        Returns
        -------
        {target: (B, K) logits}
        """
        B = tokens.size(0)

        # Padding mask for peak tokens: zero token → padded
        peak_pad = (tokens.sum(dim=-1) == 0)                          # (B, top_k)

        # Project peak tokens
        x = self.token_proj(tokens)                                    # (B, top_k, 128)

        # Precursor m/z register token: normalise to [0, 1] then project
        pmz_norm  = (precursor_mz / 2000.0).view(B, 1)                # (B, 1)
        pmz_token = self.pmz_proj(pmz_norm).unsqueeze(1)              # (B, 1, 128)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)                        # (B, 1, 128)

        # [CLS, PMZ_REG, peak_1 … peak_top_k]
        x = torch.cat([cls, pmz_token, x], dim=1)                     # (B, 2+top_k, 128)

        # Padding mask: CLS and PMZ_REG are never padded
        cls_pmz_pad  = peak_pad.new_zeros(B, 2)                       # (B, 2) all False
        padding_mask = torch.cat([cls_pmz_pad, peak_pad], dim=1)      # (B, 2+top_k)

        # Transformer encoder
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # CLS token output
        cls_out = x[:, 0]                                              # (B, 128)

        return {t: head(cls_out) for t, head in self.heads.items()}
