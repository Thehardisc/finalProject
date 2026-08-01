"""EmotionalDialogueEncoder — replaces ConversationLSTM."""

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
        history:      torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = history.shape

        x   = self.input_proj(history)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)

        positions = torch.arange(N + 1, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=padding_mask.device)
            mask = torch.cat([cls_mask, padding_mask], dim=1)
        else:
            mask = None

        h       = self.transformer(x, src_key_padding_mask=mask)
        cls_out = h[:, 0]

        prior = self.prior_head(cls_out)
        phase = self.phase_head(cls_out)

        return prior, phase
