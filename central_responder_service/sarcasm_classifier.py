import json
import os
from typing import List, Optional

import torch
import torch.nn as nn

from shared.utils.logger import get_logger

logger = get_logger("sarcasm_classifier")

_TOKENIZER = None


def _get_tokenizer(base_model: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import DistilBertTokenizerFast
        _TOKENIZER = DistilBertTokenizerFast.from_pretrained(base_model)
    return _TOKENIZER


class _SarcasmNet(nn.Module):
    def __init__(self, base_model: str, dropout: float):
        super().__init__()
        from transformers import DistilBertModel
        self.bert = DistilBertModel.from_pretrained(base_model)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(self.drop(cls)).squeeze(-1)


class SarcasmInference:
    def __init__(
        self,
        model:      _SarcasmNet,
        tokenizer,
        threshold:  float,
        max_length: int,
    ) -> None:
        self.model      = model
        self.tokenizer  = tokenizer
        self.threshold  = threshold
        self.max_length = max_length

    @staticmethod
    def _build_text(text: str, context: List[str]) -> str:
        parts = [c.strip() for c in context if c.strip()]
        parts.append(text.strip())
        return " [SEP] ".join(parts)

    def predict(self, text: str, context: Optional[List[str]] = None) -> float:
        try:
            input_text = self._build_text(text, context or [])
            enc = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logit = self.model(enc["input_ids"], enc["attention_mask"])
            return float(torch.sigmoid(logit).item())
        except Exception as e:
            logger.warning(f"Sarcasm inference error: {e}")
            return 0.0


def load_sarcasm_model(model_path: str, config_path: str) -> Optional[SarcasmInference]:
    if not os.path.exists(model_path):
        logger.warning(
            f"Sarcasm model not found at {model_path} — sarcasm_score will be 0.0 "
            "until the model is trained and placed in models/."
        )
        return None

    try:
        config: dict = {}
        if os.path.exists(config_path):
            with open(config_path) as fh:
                config = json.load(fh)

        base_model = config.get("base_model", "distilbert-base-uncased")
        dropout    = float(config.get("dropout",    0.3))
        threshold  = float(config.get("threshold",  0.5))
        max_length = int(config.get("max_length",   256))

        net = _SarcasmNet(base_model=base_model, dropout=dropout)

        ckpt = torch.load(model_path, map_location="cpu")
        net.load_state_dict(ckpt["model_state"])
        net.eval()

        tokenizer = _get_tokenizer(base_model)

        logger.info(
            f"Sarcasm classifier loaded: base={base_model} "
            f"threshold={threshold:.2f} "
            f"val_f1={config.get('val_f1', '?')} "
            f"val_auc={config.get('val_auc', '?')}"
        )
        return SarcasmInference(
            model=net,
            tokenizer=tokenizer,
            threshold=threshold,
            max_length=max_length,
        )

    except Exception as e:
        logger.error(f"Failed to load sarcasm model from {model_path}: {e}")
        return None
