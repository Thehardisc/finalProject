import re
try:
    import emoji
except ImportError:
    emoji = None

# High-signal emotion emoji → the emotion the sender means, not the picture.
# Generic demojize renders "❤️" as "red heart", which BERT/GoEmotions read as
# sadness-adjacent noise; these override before the generic pass. VADER still
# receives the raw text, so its own emoji handling is unaffected.
EMOJI_EMOTION_MAP = {
    "❤️": " love ", "❤": " love ", "🧡": " love ", "💛": " love ", "💚": " love ",
    "💙": " love ", "💜": " love ", "🖤": " love ", "🤍": " love ", "🤎": " love ",
    "💕": " love ", "💞": " love ", "💓": " love ", "💗": " love ", "💖": " love ",
    "💘": " love ", "💝": " love ", "😍": " adoring love ", "🥰": " adoring love ",
    "💔": " heartbroken sad ",
    "😂": " haha hilarious ", "🤣": " haha hilarious ", "😆": " haha funny ",
    "😊": " happy ", "☺️": " happy ", "😁": " happy ", "😄": " happy ",
    "😃": " happy ", "🙂": " content ",
    "🎉": " celebration hooray ", "🎊": " celebration hooray ", "🥳": " celebrating excited ",
    "🙄": " ugh annoyed eye roll ", "😒": " unimpressed annoyed ",
    "😢": " sad ", "😞": " sad ", "😔": " sad ", "☹️": " sad ", "🙁": " sad ",
    "😭": " crying ",
    "😡": " angry ", "🤬": " furious ", "😠": " angry ",
    "😱": " shocked terrified ", "😨": " afraid ", "😰": " anxious ",
    "😅": " nervous relieved laugh ", "😬": " awkward nervous ",
    "🙏": " please hoping grateful ",
    "😏": " smirking ", "🤔": " thinking hmm ",
    "😴": " tired sleepy ", "😩": " weary longing ", "😫": " exhausted frustrated ",
    "🤗": " warm hug ", "😳": " embarrassed flustered ",
    "🥺": " pleading touched ", "😤": " determined huffing ",
    "👍": " approval ", "👎": " disapproval ", "💀": " haha dying funny ",
}

_EMOJI_MAP_RE = re.compile("|".join(re.escape(e) for e in
                                    sorted(EMOJI_EMOTION_MAP, key=len, reverse=True)))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def demojize_text(text: str) -> str:
    text = _EMOJI_MAP_RE.sub(lambda m: EMOJI_EMOTION_MAP[m.group(0)], text)
    if emoji:
        text = emoji.demojize(text)
        text = text.replace(':', ' ').replace('_', ' ')
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_message(text: str) -> str:
    text = clean_text(text)
    return text
