"""
ConversationLSTM — learns emotional trajectory dynamics from conversation sequences.

Input  at each timestep: 67-dim feature vector (go_emotions + bert + vader + emojinet)
Output at each timestep: 28-dim predicted emotion distribution for the NEXT message

The LSTM hidden state h_t IS the "conversation state" — a continuous learned
representation of where the conversation has been and where it's heading.
After training, h_t is extracted and injected into the main pipeline as context.

Architecture:
  Linear(67 → input_dim)  — projection + normalisation
  LSTM(input_dim, hidden_dim, num_layers)
  Dropout
  Linear(hidden_dim → 28)
  Sigmoid  — output is a probability-like distribution over 28 emotions
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional

from features.schema import MSG_DIM, N_EMOTIONS


class ConversationLSTM(nn.Module):
    def __init__(
        self,
        input_dim:  int = MSG_DIM,     # 67
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = N_EMOTIONS,  # 28
        dropout:    float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection — maps raw 67-dim vector to model's internal dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Core LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Output head — predict next-message emotion distribution
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),   # outputs in [0, 1] — emotion scores
        )

    def forward(
        self,
        x: torch.Tensor,                          # [B, T, 67]
        lengths: Optional[torch.Tensor] = None,   # [B] actual sequence lengths
        hidden: Optional[Tuple] = None,           # (h_0, c_0) for stateful inference
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x:       [B, T, 67]  padded input sequences
            lengths: [B]         actual lengths (for packing); None = all equal
            hidden:  (h, c)      initial hidden state; None = zeros

        Returns:
            predictions: [B, T, 28]  next-step emotion distribution at each timestep
            (h_n, c_n):              final hidden state (the "conversation state")
        """
        B, T, _ = x.shape

        # Project input
        x_proj = self.input_proj(x)   # [B, T, hidden_dim]

        # Pack for variable-length efficiency
        if lengths is not None:
            packed = pack_padded_sequence(x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, (h_n, c_n) = self.lstm(packed, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=T)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x_proj, hidden)

        lstm_out = self.dropout(lstm_out)         # [B, T, hidden_dim]
        predictions = self.output_head(lstm_out)  # [B, T, 28]

        return predictions, (h_n, c_n)

    def get_trajectory_embedding(self, h_n: torch.Tensor) -> torch.Tensor:
        """
        Collapse the multi-layer hidden state into a single trajectory vector.
        Uses the last layer's hidden state.
        Returns: [B, hidden_dim]
        """
        return h_n[-1]   # last layer: [B, hidden_dim]

    def predict_next(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Single-step inference — feed one message, get prediction + updated state.

        Args:
            x:      [1, 1, 67]  single message features (batch=1, seq_len=1)
            hidden: previous (h, c) or None

        Returns:
            next_emotions:      [28]   predicted emotion distribution for NEXT message
            trajectory_vector:  [hidden_dim]  current conversation state embedding
            (h_n, c_n):         updated hidden state
        """
        with torch.no_grad():
            preds, (h_n, c_n) = self.forward(x, hidden=hidden)
        next_emotions     = preds[0, 0]                        # [28]
        trajectory_vector = self.get_trajectory_embedding(h_n)[0]  # [hidden_dim]
        return next_emotions, trajectory_vector, (h_n, c_n)
