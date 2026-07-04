import re
try:
    import emoji
except ImportError:
    emoji = None

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def demojize_text(text: str) -> str:
    if emoji:
        d = emoji.demojize(text)
        return d.replace(':', ' ').replace('_', ' ').strip()
    return text

def preprocess_message(text: str) -> str:
    text = clean_text(text)
    return text
