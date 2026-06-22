from .base_agent import BaseAgent


class MetaLearnerAgent(BaseAgent):
    name = "meta_learner"
    description = (
        "GatingEnsembleNet meta-learner: fuses VADER+BERT+GoE+VAD+CTX into a 28-class "
        "emotion prediction. Feature vector is exactly 116 dims "
        "(VADER[0:4] BERT[4:11] GoE[11:39] VAD[39:42] CDM[42:82] Traj[82:110] "
        "Sarcasm[110] Dynamics[111:113] Appraisal[113:116]). "
        "Owns: central_responder_service/meta_learner.py, trainer/, sarcasm_classifier.py, "
        "implicit_emotion.py, metrics.py. "
        "Key invariant: trainer/cycle.py feature building MUST match build_feature_vector() exactly."
    )
    key_files = [
        "central_responder_service/meta_learner.py",
        "central_responder_service/trainer/cycle.py",
        "central_responder_service/trainer/models.py",
        "central_responder_service/sarcasm_classifier.py",
        "central_responder_service/implicit_emotion.py",
        "shared/constants.py",
    ]
