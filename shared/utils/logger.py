"""
shared/utils/logger.py — Central logger factory used by every service.

What this module gives you:
  - get_logger(name)              → a CustomLogger writing to stdout (and optionally a file)
  - CustomLogger.bind(**fields)   → a child logger carrying structured fields
                                    merged into every log call (visible in
                                    TEXT mode as "[k=v ...]" suffix and as
                                    top-level keys in JSON mode)
  - sanitize_email(email)         → "s***@gmail.com" — for audit logs
  - log_exception / log_stats     → unchanged from the prior API

Env vars:
  LOG_LEVEL  = DEBUG | INFO | WARNING | ERROR | CRITICAL   (default INFO)
  LOG_FORMAT = TEXT | JSON                                  (default TEXT)
  LOG_DIR    = /path/to/log/dir   (optional — enables rotating file logs)
"""
import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime

_VALID_FORMATS = ("TEXT", "JSON")

# Ensure stdout is line-buffered even in non-TTY environments (e.g. Docker).
# This prevents logs from being silently lost on crashes due to buffering.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _resolve_log_format() -> str:
    """Return a validated LOG_FORMAT value. Falls back to TEXT on bad input."""
    raw = os.environ.get("LOG_FORMAT", "TEXT")
    fmt = (raw or "TEXT").upper()
    if fmt not in _VALID_FORMATS:
        sys.stderr.write(
            f"[logger] LOG_FORMAT={raw!r} is not one of {_VALID_FORMATS}; "
            f"falling back to TEXT\n"
        )
        fmt = "TEXT"
    return fmt


def sanitize_email(email: str) -> str:
    """
    Redact an email for audit logs: 'steven.furman@example.com' -> 's***@example.com'.
    Returns the input unchanged if it doesn't look like an email.
    """
    if not isinstance(email, str) or "@" not in email:
        return "<invalid-email>"
    local, _, domain = email.partition("@")
    if not local:
        return f"***@{domain}"
    return f"{local[0]}***@{domain}"


class _BoundFieldsFormatter(logging.Formatter):
    """TEXT formatter that surfaces bound `extra_data` fields as a [k=v ...] suffix."""

    def __init__(self):
        super().__init__(
            "[%(asctime)s] [%(levelname)-8s] [%(name)-24s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record):
        base = super().format(record)
        extra = getattr(record, "extra_data", None)
        if not extra:
            return base
        suffix = " ".join(f"{k}={_short_repr(v)}" for k, v in extra.items())
        return f"{base} [{suffix}]"


class JsonFormatter(logging.Formatter):
    """Structured JSON output. Bound fields become top-level keys."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level":     record.levelname,
            "service":   record.name,
            "message":   record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_data", None)
        if extra:
            log_entry.update(extra)
        return json.dumps(log_entry, default=str)


def _short_repr(v) -> str:
    """Concise printable form for TEXT-mode bound fields. Quotes strings with spaces."""
    if isinstance(v, str):
        return v if (v and " " not in v and "=" not in v) else json.dumps(v)
    return json.dumps(v, default=str)


class _OurHandler(logging.StreamHandler):
    """Marker class so we can detect our own handler in a pre-populated root logger."""


class CustomLogger:
    """
    Thin wrapper around `logging.Logger` that adds:
      - bind(**fields)  → returns a child wrapper carrying merged fields
      - log_stats / log_exception (unchanged helpers)

    Bound fields are forwarded via `extra={"extra_data": {...}}` so both the
    TEXT formatter (suffix) and the JSON formatter (top-level keys) can render them.
    """

    __slots__ = ("logger", "_bound")

    def __init__(self, logger: logging.Logger, bound: dict = None):
        self.logger = logger
        self._bound = bound or {}

    # ── public ───────────────────────────────────────────────────────────────

    def bind(self, **fields) -> "CustomLogger":
        """Return a new wrapper that carries these fields on every log call."""
        merged = {**self._bound, **fields}
        return CustomLogger(self.logger, merged)

    def info(self, msg, *args, **kwargs):     self._emit(logging.INFO,     msg, args, kwargs)
    def debug(self, msg, *args, **kwargs):    self._emit(logging.DEBUG,    msg, args, kwargs)
    def warning(self, msg, *args, **kwargs):  self._emit(logging.WARNING,  msg, args, kwargs)
    def error(self, msg, *args, **kwargs):    self._emit(logging.ERROR,    msg, args, kwargs)
    def critical(self, msg, *args, **kwargs): self._emit(logging.CRITICAL, msg, args, kwargs)

    def log_exception(self, title: str, e: Exception):
        """Log an unexpected error with full stack trace + bound fields."""
        self._emit(logging.ERROR, f"{title}: {e}", (), {"exc_info": True})

    def log_stats(self, title: str, stats: dict, level: int = logging.INFO):
        """Boxed table in TEXT mode, structured event in JSON mode."""
        if _resolve_log_format() == "JSON":
            payload = {**self._bound, **stats}
            self.logger.log(level, f"STATS: {title}", extra={"extra_data": payload})
            return

        if not self.logger.isEnabledFor(level):
            return

        width = 60
        sep    = f"+{'-' * (width - 2)}+"
        header = f"|{f' {title} ':-^{width - 2}}|"
        lines = [sep, header, sep]
        for key, value in stats.items():
            val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            if len(val_str) > 27:
                val_str = val_str[:24] + "..."
            lines.append(f"|  {key:<25} : {val_str:<27}|")
        lines.append(sep)
        for line in lines:
            self.logger.log(level, line)

    # ── internal ─────────────────────────────────────────────────────────────

    def _emit(self, level, msg, args, kwargs):
        """Merge bound fields + caller's `extra` then forward to stdlib logger."""
        if self._bound or "extra" in kwargs:
            user_extra = kwargs.pop("extra", None) or {}
            user_fields = user_extra.get("extra_data", user_extra)
            merged = {**self._bound, **user_fields}
            kwargs["extra"] = {"extra_data": merged}
        self.logger.log(level, msg, *args, **kwargs)


def _configure_root_logger(log_level: int, log_format: str) -> None:
    """
    Install our handler on the root logger exactly once per process.

    We use a marker class (_OurHandler) to detect if we already configured it,
    so we never double-install — even when uvicorn or another framework has
    already added its own handlers before we run.
    """
    root = logging.getLogger()

    # If our handler is already present, just ensure the level is right.
    if any(isinstance(h, _OurHandler) for h in root.handlers):
        root.setLevel(log_level)
        return

    formatter = JsonFormatter() if log_format == "JSON" else _BoundFieldsFormatter()

    # Remove any pre-existing StreamHandlers that uvicorn/fastapi may have added
    # so we don't get duplicate lines in two different formats.
    for h in root.handlers[:]:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            root.removeHandler(h)

    stdout_handler = _OurHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)

    # Optional rotating file log — enabled when LOG_DIR is set.
    log_dir = os.environ.get("LOG_DIR", "").strip()
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        service_name = os.environ.get("SERVICE_NAME", "service")
        log_path = os.path.join(log_dir, f"{service_name}.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(log_level)

    # Tame chatty libraries — keep them at INFO even when we're at DEBUG.
    for lib in ("uvicorn", "uvicorn.access", "gunicorn", "transformers",
                "urllib3", "filelock", "datasets", "fsspec"):
        lib_logger = logging.getLogger(lib)
        lib_logger.setLevel(logging.INFO)
        lib_logger.propagate = True


def get_logger(name: str) -> CustomLogger:
    """
    Return a CustomLogger writing to stdout. Idempotent — safe to call repeatedly.
    Configures the root logger once on first call so external libraries inherit
    the same handler/format.
    """
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_format = _resolve_log_format()

    _configure_root_logger(log_level, log_format)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = True
    return CustomLogger(logger)
