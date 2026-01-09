from transformers import pipeline
import torch

class SentimentAnalyzer:
    def __init__(self):
        print("Loading Sentiment Analysis (SST-2) model...")
        
        # Determine device
        device = -1
        if torch.cuda.is_available():
            device = 0
            print("Using CUDA GPU for Sentiment")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Using MPS GPU for Sentiment")
            
        self.classifier = pipeline(
            "text-classification", 
            model="distilbert-base-uncased-finetuned-sst-2-english", 
            top_k=None,
            device=device
        )
        print("Sentiment Analysis model loaded.")
    
    def analyze(self, text: str) -> dict:
        # returns {'POSITIVE': 0.99, 'NEGATIVE': 0.01}
        if len(text) > 512:
            text = text[:512]
            
        results = self.classifier(text)[0]
        scores = {res['label']: res['score'] for res in results}
        
        # Namespace keys to avoid collision
        return {f"sentiment_{k.lower()}": v for k, v in scores.items()}
