import torch
from transformers import pipeline

class GoEmotionsAnalyzer:
    def __init__(self):
        print("Loading GoEmotions BERT model...")
        
        # Determine device: CUDA (Nvidia), MPS (Mac), or CPU
        device = -1
        if torch.cuda.is_available():
            device = 0 # CUDA device 0
            print("Using CUDA GPU")
        elif torch.backends.mps.is_available():
            # MPS backend for Metal (Mac M1/M2/M3)
            # Transformers pipeline usually accepts "mps" string or torch.device object
            # BUT: In some versions pipeline integer index only works for CUDA. 
            # Passing device="mps" is safer for Mac.
            device = "mps"
            print("Using Apple Metal (MPS) GPU")
        else:
            print("Using CPU")

        # Using a model fine-tuned on GoEmotions (28 labels)
        self.classifier = pipeline(
            "text-classification", 
            model="bhadresh-savani/bert-base-go-emotion", 
            top_k=None, # Replaces return_all_scores=True
            device=device
        )
        print("GoEmotions BERT model loaded.")
    
    def analyze(self, text: str) -> dict:
        # returns dict with emotion scores
        
        # Max length for this model is 512, truncate if needed
        if len(text) > 512:
            text = text[:512]
            
        results = self.classifier(text)[0]
        
        # Convert to simple dict: {'admiration': 0.9, 'joy': 0.01, ...}
        emotions = {res['label']: res['score'] for res in results}
        
        return emotions
