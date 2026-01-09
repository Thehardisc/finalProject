import re

class RuleBasedAnalyzer:
    def __init__(self):
        pass
    
    def analyze(self, text: str, original_text: str = "") -> dict:
        """
        Analyzes for structural emphasis and specific patterns.
        Note: We use original_text if available because preprocessing removes caps.
        """
        text_to_check = original_text if original_text else text
        
        scores = {
            "emphasis_caps": 0.0,
            "emphasis_exclamation": 0.0,
            "emphasis_question": 0.0
        }
        
        if not text_to_check:
            return scores

        # Check for ALL CAPS (if length > 3 to avoid acronyms)
        # Ratio of caps to total alpha characters
        alpha_chars = [c for c in text_to_check if c.isalpha()]
        if len(alpha_chars) > 3:
            caps_count = sum(1 for c in alpha_chars if c.isupper())
            caps_ratio = caps_count / len(alpha_chars)
            if caps_ratio > 0.7:
                scores["emphasis_caps"] = 1.0
            elif caps_ratio > 0.4:
                scores["emphasis_caps"] = 0.5

        # Check for multiple exclamation marks
        if "!!" in text_to_check:
            scores["emphasis_exclamation"] = 1.0
        elif "!" in text_to_check:
            scores["emphasis_exclamation"] = 0.5

        # Check for repeated question marks
        if "??" in text_to_check:
             scores["emphasis_question"] = 1.0
             
        return scores
