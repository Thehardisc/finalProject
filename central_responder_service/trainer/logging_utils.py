"""
trainer/logging_utils.py — HuggingFace logging capture utilities.
"""

import re
import sys
import logging
from shared.utils.logger import get_logger

logger = get_logger("trainer")


def _setup_hf_logging():
    """Route HuggingFace datasets + transformers logging into our trainer logger."""
    _ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

    class _HFHandler(logging.Handler):
        def emit(self, record):
            msg = _ANSI.sub('', self.format(record)).strip()
            if msg:
                logger.info(f"[HF] {msg}")

    handler = _HFHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    for name in ("datasets", "datasets.utils", "datasets.builder",
                 "datasets.download", "fsspec", "transformers"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)
        lg.propagate = False


class _StderrToLogger:
    """Context manager: redirect stderr lines to logger.info during downloads."""
    _ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')

    def __init__(self):
        self._real = None
        self._buf  = ""

    def write(self, data):
        self._buf += data
        # tqdm uses \r to overwrite lines — treat both \r and \n as line ends
        for chunk in re.split(r'[\r\n]', self._buf):
            pass  # iterate to last segment
        lines = re.split(r'[\r\n]', self._buf)
        self._buf = lines[-1]  # keep incomplete last segment
        for line in lines[:-1]:
            line = self._ANSI.sub('', line).strip()
            if line:
                logger.info(f"[HF] {line}")

    def flush(self):
        pass

    def fileno(self):
        return self._real.fileno()

    def __enter__(self):
        self._real = sys.stderr
        sys.stderr  = self
        return self

    def __exit__(self, *_):
        sys.stderr = self._real
