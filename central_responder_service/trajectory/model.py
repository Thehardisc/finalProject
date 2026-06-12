"""
Self-contained ConversationLSTM definition for inference inside central_responder.
Mirrors conversation_state_learner/models/lstm.py exactly — kept in sync manually.

Input  per timestep : 79-dim  (go_emotions 28 + bert 7 + vader 4 + cdm_context 40)
Output per timestep : 28-dim  predicted next-message emotion distribution
Hidden state h_t    : [num_layers, 1, hidden_dim] — the learned conversation state

input_dim is read from trajectory_config.json at load time (default 79).
See inference.py:load_trajectory_model and conversation_state_learner/features/schema.py.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional

MSG_DIM    = 79  # GoEmotions(28) + BERT(7) + VADER(4) + CDM context(40)
N_EMOTIONS = 28


class ConversationLSTM(nn.Module):
    def __init__(
        self,
        input_dim:  int   = MSG_DIM,
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        output_dim: int   = N_EMOTIONS,
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

        self.dropout    = nn.Dropout(dropout)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x, lengths=None, hidden=None):
        x_proj = self.input_proj(x)
        if lengths is not None:
            packed    = pack_padded_sequence(x_proj, lengths.cpu(), batch_first=True, enforce_sorted=False)
            lstm_out, (h_n, c_n) = self.lstm(packed, hidden)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=x.shape[1])
        else:
            lstm_out, (h_n, c_n) = self.lstm(x_proj, hidden)

        lstm_out    = self.dropout(lstm_out)
        predictions = self.output_head(lstm_out)
        return predictions, (h_n, c_n)

    def predict_next(
        self,
        x:      torch.Tensor,
        hidden: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Single-step inference.
        x: [1, 1, 79]
        Returns:
            next_emotions:     [28]        predicted next-message distribution
            trajectory_vector: [hidden_dim] conversation state embedding
            (h_n, c_n):        updated hidden state
        """
        with torch.no_grad():
            preds, (h_n, c_n) = self.forward(x, hidden=hidden)
        next_emotions     = preds[0, 0]       # [28]
        trajectory_vector = h_n[-1, 0]        # [hidden_dim]  last-layer hidden
        return next_emotions, trajectory_vector, (h_n, c_n)
