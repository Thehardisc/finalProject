from transformers import pipeline
import torch

class DistilBertAnalyzer:
    def __init__(self):
        print("Loading DistilBERT Emotion model...")
        # Using a popular fine-tuned model for emotion detection
        self.classifier = pipeline(
            "text-classification", 
            model="bhadresh-savani/distilbert-base-uncased-emotion", 
            return_all_scores=True,
            device=-1 # CPU
        )
        print("DistilBERT Emotion model loaded.")
    
    def analyze(self, text: str) -> dict:
        # returns dict with emotion scores
        # Output format from pipeline: [[{'label': 'sadness', 'score': 0.1}, ...]]
        
        # Max length for this model is 512, truncate if needed
        if len(text) > 512:
            text = text[:512]
            
        results = self.classifier(text)[0]
        
        # Convert to simple dict: {'joy': 0.9, 'anger': 0.01, ...}
        # The model labels are usually: sadness, joy, love, anger, fear, surprise
        emotions = {res['label']: res['score'] for res in results}
        
        return emotions
