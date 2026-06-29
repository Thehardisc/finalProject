import random                                                  # seeded RNG for deterministic generation
import unicodedata                                             # NFC normalisation for accent fuzzing

FAMILY = {                                                     # emotion -> set of labels counted as a family hit
    "joy":     {"joy", "amusement", "excitement", "admiration", "approval", "optimism",
                "love", "gratitude", "pride", "relief", "caring", "desire"},
    "anger":   {"anger", "annoyance", "disapproval", "disgust"},
    "sadness": {"sadness", "grief", "disappointment", "remorse", "embarrassment"},
    "fear":    {"fear", "nervousness"},
    "surprise":{"surprise", "realization", "confusion", "curiosity"},
    "neutral": {"neutral", "realization"},
}
LABEL_FAMILY = {                                               # each of the 28 labels -> its family key
    "admiration": "joy", "amusement": "joy", "approval": "joy", "caring": "joy",
    "desire": "joy", "excitement": "joy", "gratitude": "joy", "joy": "joy",
    "love": "joy", "optimism": "joy", "pride": "joy", "relief": "joy",
    "anger": "anger", "annoyance": "anger", "disapproval": "anger", "disgust": "anger",
    "disappointment": "sadness", "grief": "sadness", "remorse": "sadness",
    "sadness": "sadness", "embarrassment": "sadness",
    "fear": "fear", "nervousness": "fear",
    "confusion": "surprise", "curiosity": "surprise", "realization": "surprise",
    "surprise": "surprise",
    "neutral": "neutral",
}

EMOTION_TRIGGERS = {                                           # ~8 strongly-cued sentences per label (~224 cases)
    "admiration": ["I really admire how you handled that", "You're incredibly talented at this",
                   "What an impressive piece of work", "I look up to her so much",
                   "That was a brilliant performance", "He is such a remarkable leader",
                   "I have so much respect for you", "Truly an admirable achievement"],
    "amusement": ["That joke had me laughing out loud", "This is absolutely hilarious",
                  "Haha that's so funny", "I can't stop giggling at this",
                  "What a ridiculous and funny situation", "That meme is comedy gold",
                  "You always crack me up", "This is the funniest thing all week"],
    "anger": ["I am absolutely furious about this", "This makes me so angry",
              "I'm enraged by how they treated us", "I'm seething with rage right now",
              "How dare they do that to me", "I'm livid and want answers",
              "This injustice infuriates me", "I'm boiling with anger"],
    "annoyance": ["This is so annoying", "Ugh, stop interrupting me constantly",
                  "I'm irritated by the noise", "That's really getting on my nerves",
                  "How irritating this whole thing is", "I'm fed up with these delays",
                  "It bugs me when this happens", "Such a frustrating little glitch"],
    "approval": ["Yes, I completely agree with this plan", "That sounds like a great idea",
                 "I approve of this decision", "This is exactly the right call",
                 "Good job, that works perfectly", "I'm on board with that",
                 "That's a solid and sensible choice", "Absolutely, let's go with it"],
    "caring": ["I really care about how you're doing", "Please take good care of yourself",
               "I'm here for you whenever you need", "Let me help you through this",
               "I worry about you and want you safe", "Sending you warmth and support",
               "You matter so much to me", "I'll always look after you"],
    "confusion": ["I'm so confused about what happened", "Wait, I don't understand this at all",
                  "This makes no sense to me", "I'm completely puzzled right now",
                  "Huh, what does that even mean", "I'm lost and can't follow",
                  "None of this is clear to me", "I'm baffled by these instructions"],
    "curiosity": ["I'm really curious how this works", "Tell me more, I want to know",
                  "What happens if we try this", "I wonder what's behind that door",
                  "That's intriguing, how does it function", "I'd love to learn more about it",
                  "What an interesting question to explore", "I'm eager to find out the answer"],
    "desire": ["I really want this so badly", "I'm craving a slice of pizza",
               "I wish I could have that", "I long to travel the world",
               "I desperately want to win this", "I yearn for a quiet vacation",
               "I'd give anything to be there", "I really hope to get that job"],
    "disappointment": ["I'm so disappointed by the result", "This let me down completely",
                       "What a letdown that turned out to be", "I expected more and got nothing",
                       "Sadly it didn't live up to the hype", "I'm gutted it fell through",
                       "That was a real disappointment", "I had high hopes and they were dashed"],
    "disapproval": ["I strongly disapprove of this", "That was the wrong thing to do",
                    "I do not condone this behavior", "This is unacceptable and I object",
                    "I can't support a decision like that", "That's not okay at all",
                    "I disagree with this entirely", "This shouldn't have been allowed"],
    "disgust": ["That is absolutely disgusting", "This makes me sick to my stomach",
                "How revolting and gross", "I'm repulsed by what I saw",
                "That smell is utterly nauseating", "Ew, that's vile and repugnant",
                "I find this thoroughly disgusting", "What a sickening sight"],
    "embarrassment": ["I'm so embarrassed right now", "That was mortifying in front of everyone",
                      "I want to hide, this is humiliating", "How awkward and shameful that was",
                      "I blushed with embarrassment", "I felt so ashamed of myself",
                      "That was a cringeworthy moment", "I'm red-faced and flustered"],
    "excitement": ["I'm so excited I can hardly wait", "This is thrilling and amazing",
                   "I can't wait, this is exhilarating", "So pumped for the big day",
                   "What an exciting adventure ahead", "I'm buzzing with anticipation",
                   "This is going to be incredible", "I'm electrified with excitement"],
    "fear": ["I'm terrified of what comes next", "This is genuinely frightening",
             "I'm scared something bad will happen", "I'm afraid of the dark hallway",
             "That noise made me freeze in fear", "I dread walking in there alone",
             "I'm petrified and shaking", "A wave of fear washed over me"],
    "gratitude": ["Thank you so much for your help", "I'm truly grateful for everything",
                  "I really appreciate your kindness", "Thanks a million, you saved me",
                  "I can't thank you enough", "So thankful to have you here",
                  "I owe you my deepest gratitude", "Bless you for being so generous"],
    "grief": ["I'm devastated by the loss", "My heart is broken with grief",
              "I'm mourning someone I loved deeply", "The sorrow is overwhelming me",
              "I can't stop crying over this loss", "Grief has swallowed me whole",
              "I miss them more than words can say", "The pain of losing them is unbearable"],
    "joy": ["I'm so happy today", "This fills me with pure joy",
            "What a joyful and wonderful day", "I'm overjoyed beyond belief",
            "My heart is bursting with happiness", "I feel so delighted right now",
            "Everything is bright and cheerful", "I'm beaming with joy"],
    "love": ["I love you with all my heart", "You mean everything to me",
             "I'm deeply in love with this", "My affection for you is endless",
             "I adore you completely", "I cherish every moment with you",
             "You have my whole heart", "I'm devoted to you forever"],
    "nervousness": ["I'm so nervous about the interview", "My stomach is in knots with worry",
                    "I'm anxious and can't sit still", "I feel uneasy about tomorrow",
                    "I'm on edge waiting for the results", "Butterflies are fluttering nervously",
                    "I'm jittery and apprehensive", "I keep fretting about what might go wrong"],
    "optimism": ["I'm hopeful things will get better", "The future looks bright to me",
                 "I'm optimistic this will work out", "Tomorrow holds great promise",
                 "I believe good things are coming", "Stay positive, it'll be fine",
                 "I have high hopes for us", "Brighter days are surely ahead"],
    "pride": ["I'm so proud of what we built", "I take great pride in this work",
              "We accomplished something to be proud of", "I'm proud of how far I've come",
              "What an achievement to be proud of", "I beam with pride at the result",
              "I'm proud to stand by this", "This success makes me proud"],
    "realization": ["Oh, I finally understand it now", "It just dawned on me what happened",
                    "I suddenly realize what went wrong", "Now it all makes sense to me",
                    "Ah, so that's how it works", "I just figured out the trick",
                    "It hit me that I was mistaken", "I came to understand the truth"],
    "relief": ["What a relief, it's finally over", "I'm so relieved that worked out",
               "Phew, that was a close one", "A weight lifted off my shoulders",
               "Thank goodness everyone is safe", "I can finally relax and breathe",
               "So glad the worst has passed", "The relief is overwhelming"],
    "remorse": ["I'm so sorry for what I did", "I deeply regret my mistake",
                "I feel terrible and full of remorse", "I wish I could take it back",
                "Forgive me, I was wrong", "The guilt is eating me alive",
                "I regret hurting you", "I'm ashamed and remorseful about it"],
    "sadness": ["I feel so sad and empty", "I'm heartbroken and down today",
                "Tears keep rolling down my face", "Everything feels gloomy and bleak",
                "I'm miserable and can't shake it", "A deep sadness has settled in",
                "I feel so low and blue", "My heart aches with sorrow"],
    "surprise": ["Wow, I did not see that coming", "What a shocking surprise",
                 "I'm stunned by this news", "That caught me completely off guard",
                 "Whoa, that's totally unexpected", "I can't believe this just happened",
                 "What an astonishing turn of events", "I'm astounded right now"],
    "neutral": ["The meeting is at three o'clock", "Please send me the report by Friday",
                "The store opens at nine in the morning", "Here is the address you asked for",
                "The bus arrives every fifteen minutes", "I'll be there in about ten minutes",
                "The file is saved in that folder", "We have a call scheduled for tomorrow"],
}


def labeled_cases():                                           # flatten the per-emotion corpus
    out = []                                                   # accumulator
    for label, texts in EMOTION_TRIGGERS.items():              # each emotion and its sentences
        fam = FAMILY[LABEL_FAMILY[label]]                      # acceptable family set for that label
        for t in texts:                                        # each sentence
            out.append((t, label, fam))                        # (text, target label, family set)
    return out                                                 # list of labeled tuples


_BASE = [                                                      # clean seed sentences the fuzzers mutate
    "I'm so happy about this", "This makes me really angry", "I feel sad and tired",
    "Thank you so much for everything", "I'm terrified of the storm", "What a hilarious joke",
    "I can't believe you did that", "Please send the report tomorrow", "I love this so much",
    "I'm a bit nervous about the test", "That is absolutely disgusting", "I'm so proud of you",
    "What time is the meeting", "I'm confused about the plan", "We won the championship",
    "My package never arrived", "The weather is nice today", "I'm grateful for your help",
    "This is the worst day ever", "I'm excited for the trip", "Let's grab lunch later",
    "I regret saying that", "She is incredibly talented", "Stop bothering me please",
    "I hope things get better soon", "The food was amazing", "I miss my old friends",
    "Can you help me with this", "That was so embarrassing", "I'm relieved it's over",
    "Wow that is shocking news", "I'm curious how it works", "He always cracks me up",
    "I'm fed up with the delays", "Everything will be okay", "I really want that job",
]
_EMOJI = list("😀😂😍🥰😢😭😠😡😱😨🙏🔥💔🎉👍👎😅😬🤔💯✨😤😩🥳😏")  # emoji pool for emoji-spam fuzzing
_SLANG = {"you": "u", "are": "r", "your": "ur", "for": "4", "to": "2", "be": "b",  # word -> slang substitutions
          "great": "gr8", "later": "l8r", "please": "plz", "thanks": "thx",
          "really": "rlly", "tomorrow": "tmrw", "because": "bc", "people": "ppl"}


def _typo(s, rng):                                             # inject character-level typos
    if len(s) < 2:                                             # too short to mutate
        return s
    chars = list(s)                                            # mutable char list
    for _ in range(max(1, len(chars) // 12)):                  # a few edits proportional to length
        i = rng.randrange(len(chars))                          # random position
        op = rng.choice(("swap", "drop", "dup", "sub"))        # pick an edit op
        if op == "swap" and i + 1 < len(chars):                # swap adjacent chars
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "drop":                                     # delete a char
            chars[i] = ""
        elif op == "dup":                                      # repeat a char 2-4x
            chars[i] = chars[i] * rng.randint(2, 4)
        else:                                                  # substitute a random letter
            chars[i] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)                                      # reassemble


def _leet(s, rng):                                             # leetspeak substitution
    table = str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"})  # letter->digit map
    return s.translate(table)                                  # apply map


def _case_scramble(s, rng):                                    # randomise letter case
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in s)  # per-char coin flip


def _emoji_spam(s, rng):                                       # surround text with emoji bursts
    burst = "".join(rng.choice(_EMOJI) for _ in range(rng.randint(1, 6)))  # 1-6 random emoji
    return f"{burst} {s} {burst}" if rng.random() < 0.5 else f"{s} {burst}"  # wrap or append


def _punct_spam(s, rng):                                       # append punctuation noise
    return s + "".join(rng.choice("!?.,;~*") for _ in range(rng.randint(2, 10)))  # 2-10 marks


def _repeat_chars(s, rng):                                     # elongate random letters
    out = []                                                   # accumulator
    for c in s:                                                # each char
        out.append(c * rng.randint(2, 5) if c.isalpha() and rng.random() < 0.18 else c)  # sometimes repeat
    return "".join(out)                                        # reassemble


def _spacing(s, rng):                                          # mangle whitespace
    if rng.random() < 0.5:                                     # widen spaces
        return s.replace(" ", "   ")
    return "  ".join(s.split())                                # or double-space tokens


def _slangify(s, rng):                                         # apply slang substitutions
    for k, v in _SLANG.items():                                # each word/slang pair
        if rng.random() < 0.6:                                 # apply most of the time
            s = s.replace(k, v)
    return s                                                   # slangified text


def _accentify(s, rng):                                        # add combining accent marks
    marks = ["́", "̀", "̈"]                                       # combining accent code points
    out = []                                                   # accumulator
    for c in s:                                                # each char
        out.append(c + rng.choice(marks) if c.isalpha() and rng.random() < 0.15 else c)  # sometimes accent
    return unicodedata.normalize("NFC", "".join(out))          # normalise to composed form


def _truncate(s, rng):                                         # cut the string short
    if len(s) < 4:                                             # too short to cut
        return s
    return s[: rng.randint(1, len(s) - 1)]                     # random prefix


def _concat(s, rng):                                           # glue on another base sentence
    return s + " " + rng.choice(_BASE)


_MUTATORS = [_typo, _leet, _case_scramble, _emoji_spam, _punct_spam,  # all mutators chained at random
             _repeat_chars, _spacing, _slangify, _accentify, _truncate, _concat]


def generate_fuzz(n=950, seed=1234):                           # n deterministic noisy strings
    rng = random.Random(seed)                                  # seeded RNG
    out = set()                                                # unique results
    guard = 0                                                  # loop safety counter
    while len(out) < n and guard < n * 80:                     # until we have n (bounded)
        guard += 1                                             # tick guard
        s = rng.choice(_BASE)                                  # pick a clean base
        for _ in range(rng.randint(1, 3)):                     # apply 1-3 mutators
            s = rng.choice(_MUTATORS)(s, rng)
        s = s[:600].strip()                                    # cap length, trim
        if s:                                                  # skip empties
            out.add(s)
    out.update({"a", "?", "!!!", "🔥🔥🔥", "ok", "...", "x" * 500})  # explicit hard edge cases
    return sorted(out)                                         # stable ordering


ARC_TEMPLATES = {                                              # arc -> (per-turn emotions, expected valence_trend)
    "escalation":     (["neutral", "neutral", "annoyance", "annoyance", "anger"],        "down"),
    "argument":       (["neutral", "annoyance", "anger", "sadness", "sadness"],          "down"),
    "meltdown":       (["neutral", "disappointment", "anger", "grief"],                  "down"),
    "celebration":    (["neutral", "curiosity", "excitement", "joy", "excitement"],      "up"),
    "good_news":      (["neutral", "surprise", "joy", "excitement"],                     "up"),
    "grief_support":  (["grief", "sadness", "caring", "caring", "gratitude"],            "up"),
    "anxiety_relief": (["nervousness", "fear", "neutral", "relief", "joy"],              "up"),
    "reconciliation": (["anger", "annoyance", "sadness", "caring", "gratitude"],         "up"),
    "reversal":       (["excitement", "joy", "disappointment", "sadness"],               "up_then_down"),
    "gratitude":      (["gratitude", "caring", "admiration", "gratitude"],               "positive"),
    "disgust":        (["neutral", "disgust", "disgust"],                                "negative"),
    "logistics":      (["neutral", "neutral", "neutral", "neutral"],                     "flat"),
}
_ARCS = sorted(ARC_TEMPLATES)                                  # arc names, stable order for cycling


def generate_conversations(n=3000, seed=99):                   # n deterministic full conversations
    rng = random.Random(seed)                                  # seeded RNG for sentence picks
    convs = []                                                 # accumulator
    for i in range(n):                                         # build n conversations
        arc = _ARCS[i % len(_ARCS)]                            # cycle through arcs evenly
        emotions, trend = ARC_TEMPLATES[arc]                   # emotion sequence + expected trend
        turns = [{"speaker": "A" if t % 2 == 0 else "B",       # alternate speakers
                  "text": rng.choice(EMOTION_TRIGGERS[emo]),   # a sentence cueing that emotion
                  "emotion": emo}                              # the per-turn target emotion
                 for t, emo in enumerate(emotions)]            # one turn per emotion slot
        convs.append({"id": f"{arc}_{i:04d}", "arc": arc,      # unique id + arc
                      "valence_trend": trend, "turns": turns}) # + expected trend + turns
    return convs                                               # list of conversation dicts
