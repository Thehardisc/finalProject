from .base_agent import BaseAgent


class ContextEngineAgent(BaseAgent):
    name = "context_engine"
    description = (
        "CDM (Conversation Dynamics Machine) HMM with 15 intent states × 28 observations. "
        "Produces a 40-dim context vector: CDM one-hot[0:15], scalars[15:39], intent_stability[39]. "
        "Episodic memory via Qdrant. Per-speaker Redis keys: "
        "conv:{cid}:spk:{uid}:cdm_state|hmm_alpha|state_hist|intent_stab|last_embed. "
        "Owns: context_engine_service/main.py, cdm.py, train_cdm_hmm.py."
    )
    key_files = [
        "context_engine_service/main.py",
        "context_engine_service/cdm.py",
        "context_engine_service/train_cdm_hmm.py",
        "shared/constants.py",
    ]
