"""
trainer — Periodic retraining package for central_responder_service.

Exposes start_trainer_thread() at the package level so that
`from trainer import start_trainer_thread` continues to work in main.py.
"""
from trainer.logging_utils import _setup_hf_logging

# Route HuggingFace logging at import time, just as the original trainer.py did
_setup_hf_logging()

from trainer.cycle import start_trainer_thread

__all__ = ["start_trainer_thread"]
