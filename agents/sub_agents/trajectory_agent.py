from .base_agent import BaseAgent


class TrajectoryAgent(BaseAgent):
    name = "trajectory"
    description = (
        "EmotionalDialogueEncoder (EDE) LSTM: consumes GoEmotions history window [B,N,28] "
        "(N=12, oldest-first) and outputs predicted_next [B,28] softmax + phase_logits [B,6]. "
        "Phases: opening/escalation/peak/turning_point/resolution/sustained. "
        "After each step writes predicted_next to Redis trajectory:{conv_id}:prior (28-dim JSON). "
        "Feature slot [82:110] in the 116-dim vector. "
        "Owns: central_responder_service/trajectory/model.py, inference.py."
    )
    key_files = [
        "central_responder_service/trajectory/model.py",
        "central_responder_service/trajectory/inference.py",
        "central_responder_service/main.py",
        "shared/constants.py",
    ]
