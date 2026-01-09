from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class VaderAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
    
    def analyze(self, text: str) -> dict:
        # returns dict with 'neg', 'neu', 'pos', 'compound'
        scores = self.analyzer.polarity_scores(text)
        
        # We can map compund score to a primary emotion for simplicity if needed,
        # but for now we return the raw VADER scores as "emotions"
        return {
            "vader_neg": scores["neg"],
            "vader_neu": scores["neu"],
            "vader_pos": scores["pos"],
            "vader_compound": scores["compound"]
        }
