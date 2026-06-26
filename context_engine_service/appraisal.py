import numpy as np


def compute_appraisal(ctx):
    novelty = float(np.clip(ctx.get("entry_abruptness", 0.0), 0.0, 1.0))
    goal_congruence = float(np.clip(
        ctx.get("current_valence", 0.0) * ctx.get("topic_resonance", 0.5) * 2.0,
        -1.0, 1.0,
    ))
    coping = float(np.clip(
        (1.0 - ctx.get("volatility", 0.5)) * 0.6
        + (1.0 - ctx.get("speaker_divergence", 0.5)) * 0.4,
        0.0, 1.0,
    ))
    return {"novelty": novelty, "goal_congruence": goal_congruence, "coping": coping}
