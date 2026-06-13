"""
EmotionalDialogueEncoder — replaces ConversationLSTM.

Input:  GoE history window [B, N, 28] — last N real GoE distributions (oldest first)
Output: prior [B, 28] Softmax, phase_logits [B, 6]

prior is stored at trajectory:{conv_id}:prior and consumed by meta_learner.py
as feature vector slot [79:107].  FEATURE_DIM=107 and the slot width are unchanged.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

PHASES   = ["opening", "escalation", "peak", "turning_point", "resolution", "sustained"]
N_PHASES = len(PHASES)
PHASE_IDX = {p: i for i, p in enumerate(PHASES)}


class EmotionalDialogueEncoder(nn.Module):
    def __init__(
        self,
        n_emotions: int   = 28,
        n_phases:   int   = N_PHASES,
        d_model:    int   = 64,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        max_window: int   = 12,
        dropout:    float = 0.15,
    ):
        super().__init__()
        self.n_emotions = n_emotions
        self.n_phases   = n_phases
        self.max_window = max_window
        self.d_model    = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(n_emotions, d_model),
            nn.LayerNorm(d_model),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Embedding(max_window + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        self.prior_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_emotions),
            nn.Softmax(dim=-1),
        )

        self.phase_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_phases),
        )

    def forward(
        self,
        history:      torch.Tensor,                    # [B, N, 28]
        padding_mask: Optional[torch.Tensor] = None,   # [B, N] True = padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = history.shape

        x   = self.input_proj(history)                       # [B, N, d]
        cls = self.cls_token.expand(B, -1, -1)               # [B, 1, d]
        x   = torch.cat([cls, x], dim=1)                     # [B, N+1, d]

        positions = torch.arange(N + 1, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            mask = torch.cat([cls_mask, padding_mask], dim=1)  # [B, N+1]
        else:
            mask = None

        h       = self.transformer(x, src_key_padding_mask=mask)
        cls_out = h[:, 0]                                    # [B, d]

        prior = self.prior_head(cls_out)   # [B, 28] — Softmax
        phase = self.phase_head(cls_out)   # [B,  6] — raw logits

        return prior, phase
