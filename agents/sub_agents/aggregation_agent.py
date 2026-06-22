from .base_agent import BaseAgent


class AggregationAgent(BaseAgent):
    name = "aggregation"
    description = (
        "Conversation state tracking: reads emotion_stream, maintains per-conversation "
        "valence/mood trajectory, writes conv:{cid}:spk:{uid}:valence_seq (JSON list, "
        "oldest-first) and updates conversation:{cid} hash. "
        "Owns: aggregation_service/main.py, state/conversation_state.py, rules/dynamic_rules.py. "
        "Also manages mood_trajectory, emotional_inertia, and contagion calculations "
        "(Dynamics block [111:113] in the 116-dim feature vector)."
    )
    key_files = [
        "aggregation_service/main.py",
        "aggregation_service/state/conversation_state.py",
        "aggregation_service/rules/dynamic_rules.py",
        "shared/constants.py",
    ]
