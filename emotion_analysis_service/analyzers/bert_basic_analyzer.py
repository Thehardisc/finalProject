import torch
from transformers import pipeline

class BasicBertAnalyzer:
    def __init__(self):
        print("Loading Basic (Ekman) BERT model...")
        
        # Determine device: CUDA (Nvidia), MPS (Mac), or CPU
        device = -1
        if torch.cuda.is_available():
            device = 0 # CUDA device 0
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Using Apple Metal (MPS) GPU")
        else:
            print("Using CPU")

        # Using a model fine-tuned on 7 basic emotions (Ekman)
        # joy, anger, sadness, fear, surprise, disgust, neutral
        self.classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base", 
            top_k=None, 
            device=device
        )
        print("Basic BERT model loaded.")
    
    def analyze(self, text: str) -> dict:
        # returns dict with emotion scores
        
        # Max length for this model is 512, truncate if needed
        if len(text) > 512:
            text = text[:512]
            
        results = self.classifier(text)[0]
        
        # Convert to simple dict: {'joy': 0.9, 'anger': 0.01, ...}
        # Normalize labels to lowercase just in case
        emotions = {res['label'].lower(): res['score'] for res in results}
        
        return emotions
