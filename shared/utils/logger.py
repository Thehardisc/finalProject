"""
shared/utils/logger.py — Central logger factory used by every service.

What this module gives you:
  - get_logger(name)              → a CustomLogger writing to stdout
  - CustomLogger.bind(**fields)   → a child logger carrying structured fields
                                    merged into every log call (visible in
                                    TEXT mode as "[k=v ...]" suffix and as
                                    top-level keys in JSON mode)
  - sanitize_email(email)         → "s***@gmail.com" — for audit logs
  - log_exception / log_stats     → unchanged from the prior API

Env vars:
  LOG_LEVEL  = DEBUG | INFO | WARNING | ERROR | CRITICAL   (default INFO)
  LOG_FORMAT = TEXT | JSON                                  (default TEXT)
"""
import logging
import os
import sys
import json
from datetime import datetime

_VALID_FORMATS = ("TEXT", "JSON")


def _resolve_log_format() -> str:
    """Return a validated LOG_FORMAT value. Falls back to TEXT on bad input."""
    raw = os.environ.get("LOG_FORMAT", "TEXT")
    fmt = (raw or "TEXT").upper()
    if fmt not in _VALID_FORMATS:
        # Use stderr directly — the logging system isn't fully configured yet.
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
        # Compact "k=v k=v" form. repr() ensures str values with spaces stay parseable.
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
            # Merge bound fields so the stats event also carries correlation.
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
            # Caller may pass {"extra_data": {...}} OR a plain dict — accept both.
            user_fields = user_extra.get("extra_data", user_extra)
            merged = {**self._bound, **user_fields}
            kwargs["extra"] = {"extra_data": merged}
        self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str) -> CustomLogger:
    """
    Return a CustomLogger writing to stdout. Idempotent — safe to call repeatedly.
    Configures the root logger once on first call so external libraries inherit
    the same handler/format.
    """
    log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    log_format = _resolve_log_format()

    root_logger = logging.getLogger()
    if not getattr(root_logger, "_is_global_configured", False):
        root_logger.setLevel(log_level)

        if not root_logger.handlers:
            rh = logging.StreamHandler(sys.stdout)
            rh.setLevel(log_level)
            rh.setFormatter(JsonFormatter() if log_format == "JSON" else _BoundFieldsFormatter())
            root_logger.addHandler(rh)

        # Tame chatty libraries — force them to INFO even when we're at DEBUG.
        for lib in ("uvicorn", "uvicorn.access", "gunicorn", "transformers",
                    "urllib3", "filelock", "datasets", "fsspec"):
            lib_logger = logging.getLogger(lib)
            lib_logger.setLevel(logging.INFO)
            lib_logger.propagate = True

        setattr(root_logger, "_is_global_configured", True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = True
    return CustomLogger(logger)
