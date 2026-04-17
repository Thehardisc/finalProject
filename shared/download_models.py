import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# List of models to pre-cache in the Docker image
MODELS = [
    "bhadresh-savani/bert-base-go-emotion",         # Used by goemotions_service
    "j-hartmann/emotion-english-distilroberta-base" # Used by bert_service
]

def download_models():
    """
    Downloads and caches models and tokenizers to the default HF cache directory.
    This should be run during the Docker build phase.
    """
    cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Starting model pre-download to {cache_dir}...")
    
    for model_id in MODELS:
        print(f"Downloading {model_id}...")
        # Download tokenizer
        AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        # Download model
        AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir)
        
    print("All models successfully cached.")

if __name__ == "__main__":
    download_models()
