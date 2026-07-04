import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = [
    "bhadresh-savani/bert-base-go-emotion",
    "j-hartmann/emotion-english-distilroberta-base"
]

def download_models():
    """Downloads and caches models and tokenizers to the default HF cache directory."""
    cache_dir = os.getenv("TRANSFORMERS_CACHE", "/app/model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Starting model pre-download to {cache_dir}...")
    
    for model_id in MODELS:
        print(f"Downloading {model_id}...")
        AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=cache_dir)
        
    print("All models successfully cached.")

if __name__ == "__main__":
    download_models()
