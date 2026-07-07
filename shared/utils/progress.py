"""Shared training-progress reporting.

Every training loop in the repo reports through TrainingProgress so progress
lines look identical everywhere (meta-learner trainer, GoE fine-tune, context
pretrain, conversation LSTM, sarcasm classifier):

    [meta_gating] epoch  12/80  15% | loss=0.8123 val_F1=0.4312 | 1.9s/epoch elapsed=23.1s ETA=2m 09s

In JSON log mode (LOG_FORMAT=JSON) the same values are emitted as structured
fields (event=train_progress / train_done) instead of a formatted string,
matching the repo-wide logging contract.

Creating a TrainingProgress also silences third-party tqdm bars (the
sentence-transformers "Batches: 100%|…" lines, HF download bars). Those bars
track a single encode()/download call's internal batches — they hit 100%
immediately and say nothing about overall training progress, so the shared
progress lines are the only progress signal.
"""

import logging
import os
import time

from shared.utils.logger import _resolve_log_format

_bars_silenced = False


def silence_library_bars():
    """Disable per-call tqdm bars from sentence-transformers / HF libraries."""
    global _bars_silenced
    if _bars_silenced:
        return
    _bars_silenced = True
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # sentence-transformers shows its "Batches" bar whenever its logger is at
    # INFO/DEBUG (the stack default) — raise it to WARNING.
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        import datasets
        datasets.disable_progress_bars()
    except Exception:
        pass


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int(seconds % 3600 // 60):02d}m"
    if seconds >= 60:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{seconds:.1f}s"


def _fmt_metric(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}" if (v == 0.0 or 1e-3 <= abs(v) < 1e4) else f"{v:.2e}"
    return str(v)


class TrainingProgress:
    """Uniform progress reporter for training loops.

    Usage:
        prog = TrainingProgress(logger, task="meta_gating", total=n_epochs)
        for epoch in range(n_epochs):
            ...
            prog.step(metrics={"loss": avg_loss, "val_F1": val_f1})
        prog.done(metrics={"best_val_F1": best_f1})
    """

    def __init__(self, logger, task: str, total: int, unit: str = "epoch",
                 log_every: int = 1):
        silence_library_bars()
        self.logger    = logger
        self.task      = task
        self.total     = max(1, int(total))
        self.unit      = unit
        self.log_every = max(1, int(log_every))
        self.count     = 0
        self._t0       = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self._t0

    def step(self, n: int = 1, metrics: dict = None, note: str = "",
             force: bool = False):
        """Advance by ``n`` units and log if due (every ``log_every``, at the
        end, or when ``force`` is set — e.g. on a new-best epoch)."""
        self.count += n
        if not (force or self.count % self.log_every == 0 or self.count >= self.total):
            return
        elapsed  = self.elapsed
        rate     = elapsed / self.count if self.count else 0.0
        eta      = rate * max(0, self.total - self.count)
        pct      = 100.0 * self.count / self.total
        metrics  = metrics or {}

        if _resolve_log_format() == "JSON":
            self.logger.info("train_progress", extra={
                "event": "train_progress", "task": self.task, "unit": self.unit,
                "current": self.count, "total": self.total, "pct": round(pct, 1),
                "sec_per_unit": round(rate, 3), "elapsed_sec": round(elapsed, 1),
                "eta_sec": round(eta, 1), **({"note": note} if note else {}),
                **metrics,
            })
            return

        metric_str = " ".join(f"{k}={_fmt_metric(v)}" for k, v in metrics.items())
        pad = len(str(self.total))
        self.logger.info(
            f"[{self.task}] {self.unit} {self.count:>{pad}}/{self.total} {pct:3.0f}%"
            + (f" | {metric_str}" if metric_str else "")
            + f" | {rate:.2f}s/{self.unit} elapsed={fmt_duration(elapsed)} ETA={fmt_duration(eta)}"
            + (f" {note}" if note else "")
        )

    def done(self, metrics: dict = None, note: str = ""):
        elapsed = self.elapsed
        rate    = elapsed / self.count if self.count else 0.0
        metrics = metrics or {}

        if _resolve_log_format() == "JSON":
            self.logger.info("train_done", extra={
                "event": "train_done", "task": self.task, "unit": self.unit,
                "completed": self.count, "total": self.total,
                "sec_per_unit": round(rate, 3), "elapsed_sec": round(elapsed, 1),
                **({"note": note} if note else {}), **metrics,
            })
            return

        metric_str = " ".join(f"{k}={_fmt_metric(v)}" for k, v in metrics.items())
        self.logger.info(
            f"[{self.task}] done — {self.count}/{self.total} {self.unit}s "
            f"in {fmt_duration(elapsed)} ({rate:.2f}s/{self.unit})"
            + (f" | {metric_str}" if metric_str else "")
            + (f" {note}" if note else "")
        )
