import re
import sys
import logging
from shared.utils.logger import get_logger

logger = get_logger("trainer")


def _setup_hf_logging():
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
    _ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]|\r')

    def __init__(self):
        self._real = None
        self._buf  = ""

    def write(self, data):
        self._buf += data
        for chunk in re.split(r'[\r\n]', self._buf):
            pass
        lines = re.split(r'[\r\n]', self._buf)
        self._buf = lines[-1]
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
