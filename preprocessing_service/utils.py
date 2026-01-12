import re
try:
    import emoji
except ImportError:
    emoji = None

def clean_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def demojize_text(text: str) -> str:
    # Convert emojis to text, e.g. 👍 -> :thumbs_up:
    if emoji:
        return emoji.demojize(text)
    return text

def preprocess_message(text: str) -> str:
    text = clean_text(text)
    # text = demojize_text(text) # Disabled to preserve raw emojis for VADER/BERT
    return text
