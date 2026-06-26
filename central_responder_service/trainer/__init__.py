from trainer.logging_utils import _setup_hf_logging

_setup_hf_logging()

from trainer.cycle import start_trainer_thread

__all__ = ["start_trainer_thread"]
