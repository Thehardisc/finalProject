"""
trainer/reporter.py — Logging utilities for training cycle reports.
"""
from shared.utils.logger import get_logger

logger = get_logger("trainer")


def _bar(value: float, width: int = 25) -> str:
    """Renders a unicode progress bar for logging."""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)


def print_report(prev_meta: dict, new_acc: float, new_f1: float,
                 n_train: int, n_filtered: int, deployed: bool,
                 duration_s: float = None) -> None:
    """Log a structured summary after each training cycle."""
    prev_acc = prev_meta.get("test_accuracy")
    delta    = (new_acc - prev_acc) if prev_acc is not None else None

    stats = {
        "Previous Accuracy":  f"{prev_acc:.4f}" if prev_acc is not None else "N/A",
        "New Accuracy (test)": f"{new_acc:.4f}  {_bar(new_acc)}",
        "New F1 (macro)":      f"{new_f1:.4f}  {_bar(new_f1)}",
        "Samples Trained":     n_train,
        "Samples Filtered":    n_filtered,
        "Deployment":          "✅ DEPLOYED" if deployed else "❌ REJECTED (accuracy/regression)"
    }

    if delta is not None:
        direction    = "↑" if delta >= 0 else "↓"
        stats["Delta"] = f"{direction} {delta*100:+.2f}%"

    if duration_s is not None:
        stats["Cycle Duration"] = f"{duration_s:.1f}s"

    logger.log_stats("Retraining Report", stats)
