"""ConversationLSTM — learns emotional trajectory dynamics from conversation sequences."""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional

from features.schema import MSG_DIM, N_EMOTIONS


class ConversationLSTM(nn.Module):
    def __init__(
        self,
        input_dim:  int = MSG_DIM,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = N_EMOTIONS,
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout   = nn.Dropout(dropout)
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Args:"""
        B, T, _ = x.shape

        x_proj = self.input_proj(x)

        if lengths is not None:
            packed = pack_padded_sequence(x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, (h_n, c_n) = self.lstm(packed, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=T)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x_proj, hidden)

        lstm_out    = self.dropout(lstm_out)
        lstm_out    = self.lstm_norm(lstm_out)
        predictions = self.output_head(lstm_out)

        return predictions, (h_n, c_n)

    def get_trajectory_embedding(self, h_n: torch.Tensor) -> torch.Tensor:
        """Collapse the multi-layer hidden state into a single trajectory vector."""
        return h_n[-1]

    def predict_next(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Single-step inference — feed one message, get prediction + updated state."""
        with torch.no_grad():
            preds, (h_n, c_n) = self.forward(x, hidden=hidden)
        next_emotions     = preds[0, 0]
        trajectory_vector = self.get_trajectory_embedding(h_n)[0]
        return next_emotions, trajectory_vector, (h_n, c_n)
