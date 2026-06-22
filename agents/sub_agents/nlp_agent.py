from .base_agent import BaseAgent


class NLPAgent(BaseAgent):
    name = "nlp"
    description = (
        "NLP inference services: VADER (4 sentiment scores), BERT Ekman 7-class "
        "(j-hartmann/emotion-english-distilroberta-base), GoEmotions 28-class "
        "(bhadresh-savani/bert-base-go-emotion) + emoji scoring. "
        "Owns: vader_service, bert_service, goemotions_service. "
        "Outputs feed partial_analysis_stream with model_name tag."
    )
    key_files = [
        "vader_service/main.py",
        "bert_service/main.py",
        "goemotions_service/main.py",
        "shared/constants.py",
    ]
