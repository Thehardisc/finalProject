"""
situation_encoder.py — Stage 2: COMET-ATOMIC-2020-inspired situation encoder.

NOTE: single-word keywords use word-boundary matching (\bword\b) to avoid
false positives (e.g. "died" inside "studied"). Multi-word phrases use
substring matching (no boundary issue for phrases ≥ 2 words).

Encodes commonsense causal knowledge ("PersonX did Y → PersonX feels Z") as
structured Python rules. Classifies text into situation archetypes, detects
contextual modifiers, and returns a GoEmotions-aligned emotion distribution.

Three components:
  1. Situation archetype detection — ~20 archetypes (MEDICAL, ACHIEVEMENT, etc.)
  2. Modifier detection — uncertainty / completion / reversal / temporal cues
  3. Situation-emotion priors — (archetype × modifiers) → emotion distribution

No external ML dependencies — pure Python, deterministic, fast.
"""

import re
from typing import Optional


def _kw_in(kw: str, text_lower: str) -> bool:
    """Match keyword with word-boundary awareness for single words."""
    if ' ' in kw:
        return kw in text_lower
    return bool(re.search(r'\b' + re.escape(kw) + r'\b', text_lower))

# ── Situation archetypes ───────────────────────────────────────────────────────
# Each archetype is defined by keyword clusters. A text matches the archetype
# if at least one keyword from ANY cluster is present.

_ARCHETYPES: list[tuple[str, list[list[str]]]] = [

    ("MEDICAL_UNCERTAIN", [
        ["doctor said", "physician said", "hospital said", "specialist said",
         "run more tests", "more tests", "additional tests", "further tests",
         "waiting for results", "waiting to hear back", "test results",
         "scan results", "mri", "ct scan", "biopsy", "blood test",
         "found something", "found a lump", "found a mass", "suspicious area",
         "could be serious", "might be serious", "needs investigating",
         "wouldn't tell me why", "didn't say why", "they want to admit"],
    ]),

    ("MEDICAL_RESOLVED", [
        ["all clear", "came back negative", "came back normal", "benign",
         "nothing serious", "nothing to worry about", "clean bill",
         "surgery went well", "operation was a success", "pulled through",
         "made it through surgery", "out of intensive care",
         "test came back fine", "results were fine", "results were good"],
    ]),

    ("ACHIEVEMENT_SUCCESS", [
        ["got the job", "got in", "got accepted", "got promoted",
         "got the scholarship", "got the offer", "got the grant",
         "passed the exam", "passed the test", "aced the interview",
         "made the team", "made the cut", "won the competition",
         "they picked me", "they chose me", "i was selected",
         "crossed the finish line", "finished the race", "completed the marathon",
         "graduated", "certified", "licensed", "awarded the",
         "presented with the", "published", "launched", "shipped"],
        ["first in my family", "first in the family", "youngest ever",
         "first woman to", "first man to", "record-breaking",
         "nobody else had", "never been done before", "making history"],
    ]),

    ("ACHIEVEMENT_FAIL", [
        ["didn't get the job", "didn't get in", "didn't get accepted",
         "rejection letter", "they said no", "got rejected",
         "didn't make the team", "didn't make the cut",
         "failed the exam", "failed the test", "didn't pass",
         "missed the deadline", "missed the cutoff",
         "application was rejected", "proposal was rejected"],
        ["studied for", "practiced for", "prepared for", "trained for",
         "worked hard for", "put everything into", "gave it everything"],
    ]),

    ("LOSS_DEATH", [
        ["passed away", "died", "passed this morning", "passed last night",
         "didn't make it", "gone now", "they're gone", "she's gone",
         "he's gone", "no longer with us", "lost them",
         "funeral", "memorial", "burial", "cremation",
         "said my last goodbye", "final goodbye", "one last time"],
    ]),

    ("LOSS_ABANDONMENT", [
        ["left me", "walked out", "walked away", "moved out",
         "packed up and left", "never came back", "never showed up",
         "didn't show up", "stood me up", "ghosted me",
         "stopped answering", "stopped texting", "went silent",
         "no one came", "nobody came", "table is still set",
         "chair is empty", "place is set but"],
    ]),

    ("LOSS_CONNECTION", [
        ["we used to", "used to go there", "used to see them",
         "used to talk every day", "used to be close",
         "the house is so quiet", "place feels empty",
         "nobody left", "no one left here", "everyone has moved on",
         "last time i saw", "last time we spoke",
         "don't talk anymore", "grew apart"],
    ]),

    ("PRE_PERFORMANCE", [
        ["tomorrow is the", "presentation is tomorrow", "interview is tomorrow",
         "exam is tomorrow", "pitch is tomorrow", "match is tomorrow",
         "speech is tomorrow", "audition is tomorrow", "first day is tomorrow",
         "goes live tomorrow", "launches tomorrow",
         "presentation is in", "interview is in", "exam is in",
         "in two days", "in three days", "this weekend",
         "the night before", "eve of the", "about to go on",
         "about to present", "about to perform", "about to speak"],
        ["hands are shaking", "heart is pounding", "heart is racing",
         "palms are sweaty", "can't sleep", "couldn't sleep",
         "stomach is in knots", "dry mouth", "going over my lines",
         "rehearsed a hundred times", "practised in the mirror",
         "keep rehearsing", "going over it in my head"],
    ]),

    ("PERSONAL_REGRET", [
        ["last thing i said", "last words i said", "the last time i spoke",
         "never got to say", "never got to tell", "never got to see",
         "never told them", "never said", "wish i had said",
         "wish i had told", "should have said", "should have been there",
         "should have called", "took it for granted", "didn't appreciate",
         "was too busy", "too caught up", "i pushed them away",
         "i ignored them", "wasn't there for them", "let them down"],
    ]),

    ("SOCIAL_EMBARRASSMENT", [
        # Only phrases where the SPEAKER is the one embarrassed
        ["i froze up completely", "i went completely blank",
         "i forgot the words", "i forgot my lines", "i tripped on stage",
         "i fell on stage", "i autocorrected", "i sent to the wrong person",
         "i replied all by accident", "i replied to all",
         "caught me", "walked in on me", "barged in on me",
         "everyone heard me", "everyone stared at me", "everyone laughed at me",
         "i was caught", "i got caught",
         "room went silent", "everyone went quiet"],
        ["called the teacher mom", "called my teacher mom", "called my boss mom",
         "called him dad", "called her mom", "called the prof"],
    ]),

    ("BETRAYAL_DECEPTION", [
        ["lied to me", "lied to my face", "looked me in the eye and lied",
         "told me a lie", "deceived me", "manipulated me",
         "behind my back", "without telling me", "kept it secret",
         "found out they had been", "turns out they were",
         "it was all fake", "none of it was real", "it was a lie",
         "they were using me", "used me", "took advantage of me",
         "took credit for my work", "stole my idea", "claimed credit"],
    ]),

    ("CONFLICT_INJUSTICE", [
        ["completely ignored", "they ignored my", "no one listened",
         "they dismissed me", "talked over me", "shut me out",
         "kept me out of", "excluded me from",
         "unfair treatment", "treated me unfairly",
         "that's not fair", "that's not right",
         "they always do this", "they keep doing this",
         "for the hundredth time", "every single time",
         "not for the first time", "third time this week"],
    ]),

    ("UNEXPECTED_POSITIVE", [
        ["completely out of the blue", "totally unexpected",
         "never saw it coming", "last thing i expected",
         "came out of nowhere", "surprised me completely",
         "walked in to find", "opened the door and saw",
         "turned around and", "was already there when"],
    ]),

    ("WORRY_UNKNOWN", [
        ["haven't heard from", "can't reach", "no one's heard from",
         "been missing since", "phone goes straight to voicemail",
         "phone keeps going to voicemail", "no answer",
         "not responding", "something's wrong", "i have a bad feeling",
         "something feels off", "can't shake the feeling",
         "something is not right", "i don't know what's wrong"],
    ]),

    ("GRATITUDE_ACT", [
        ["you didn't have to", "didn't need to do that",
         "went out of your way", "drove all the way here",
         "came all this way", "dropped everything to help",
         "stayed up all night", "spent hours on", "took time out",
         "made such a difference", "without you i wouldn't",
         "i owe you so much", "means so much to me",
         "means the world to me", "so thoughtful",
         "covered for me", "took my shift", "took the blame"],
    ]),

    ("ADMIRATION_ACT", [
        ["raised all those kids", "raised three kids", "raised two kids",
         "raised them alone", "all by herself", "all by himself",
         "single-handedly", "built it from nothing", "started from scratch",
         "worked three jobs", "works two jobs", "never complained",
         "never asked for anything", "gave everything",
         "inspiring the whole", "set the example", "led by example",
         "breakthrough", "pioneered", "revolutionised", "revolutionized"],
    ]),

    ("LONGING_ROMANTIC", [
        ["been thinking about you", "can't stop thinking about you",
         "thinking of you all day", "you're always on my mind",
         "miss you so much", "counting the days",
         "every day without you", "can't imagine without you",
         "told everyone about you", "keep looking at your photo",
         "saved all your messages", "smile every time i see your name",
         "face lights up when"],
    ]),

    ("ANTICIPATION_POSITIVE", [
        # "looking forward to" intentionally excluded — covered by optimism rules
        ["can't wait for", "can't wait until", "can't wait —", "can't wait,",
         "so excited for", "counting down to", "only a few more days",
         "finally happening", "it's actually happening",
         "we're finally taking", "finally taking the trip",
         "planned for two years", "planned for years",
         "been planning this for", "been saving up for",
         "been waiting for this moment", "dreamed of this for years",
         "the day has finally come"],
    ]),

    ("REALIZATION_INSIGHT", [
        ["just realized", "just realised", "only now realised",
         "it clicked", "it dawned on me", "it hit me",
         "makes sense now", "all makes sense now",
         "the pattern was there", "the signs were there",
         "i can see it now", "looking back i can see",
         "the whole time", "all along", "been blind to",
         "didn't see it", "couldn't see it before"],
    ]),

    ("PRIDE_PERSONAL", [
        ["first in my family", "first in the family to",
         "youngest ever to", "oldest ever to",
         "first woman to", "first man to",
         "graduated today", "just graduated",
         "finally finished", "completed the",
         "certification", "licensed now", "official now",
         "years in the making", "worked years for this"],
    ]),
]

# ── Archetype → emotion priors ─────────────────────────────────────────────────
# Maps archetype name to a dict of {emotion_label: base_confidence}
# These priors are boosted / reduced by modifier detection below.

_PRIORS: dict[str, dict[str, float]] = {
    "MEDICAL_UNCERTAIN":     {"fear": 0.72, "nervousness": 0.65, "confusion": 0.45},
    "MEDICAL_RESOLVED":      {"relief": 0.82, "joy": 0.45, "gratitude": 0.40},
    "ACHIEVEMENT_SUCCESS":   {"joy": 0.75, "excitement": 0.68, "pride": 0.65, "gratitude": 0.40},
    "ACHIEVEMENT_FAIL":      {"disappointment": 0.78, "sadness": 0.60, "remorse": 0.40},
    "LOSS_DEATH":            {"grief": 0.88, "sadness": 0.72, "remorse": 0.50},
    "LOSS_ABANDONMENT":      {"sadness": 0.78, "disappointment": 0.60, "loneliness": 0.55},
    "LOSS_CONNECTION":       {"sadness": 0.68, "grief": 0.55, "remorse": 0.45},
    "PRE_PERFORMANCE":       {"nervousness": 0.80, "fear": 0.55, "excitement": 0.40},
    "PERSONAL_REGRET":       {"remorse": 0.82, "sadness": 0.65, "grief": 0.50},
    "SOCIAL_EMBARRASSMENT":  {"embarrassment": 0.85, "amusement": 0.30},
    "BETRAYAL_DECEPTION":    {"anger": 0.72, "disgust": 0.68, "disappointment": 0.60, "sadness": 0.45},
    "CONFLICT_INJUSTICE":    {"annoyance": 0.70, "anger": 0.65, "disapproval": 0.60},
    "UNEXPECTED_POSITIVE":   {"surprise": 0.72, "joy": 0.50},
    "WORRY_UNKNOWN":         {"fear": 0.70, "nervousness": 0.65, "sadness": 0.45},
    "GRATITUDE_ACT":         {"gratitude": 0.84, "love": 0.40, "joy": 0.35},
    "ADMIRATION_ACT":        {"admiration": 0.80, "pride": 0.45, "approval": 0.40},
    "LONGING_ROMANTIC":      {"love": 0.82, "sadness": 0.40},
    "ANTICIPATION_POSITIVE": {"excitement": 0.72, "joy": 0.55, "optimism": 0.60},
    "REALIZATION_INSIGHT":   {"realization": 0.75, "surprise": 0.45},
    "PRIDE_PERSONAL":        {"pride": 0.80, "joy": 0.55, "excitement": 0.50},
}

# ── Contextual modifiers ───────────────────────────────────────────────────────
# Each modifier: (phrases, {emotion: delta})
# Positive delta = boost; negative delta = suppress (relative to priors).

_MODIFIERS: list[tuple[list[str], dict[str, float]]] = [

    # Uncertainty language → amplifies fear, suppresses relief
    (["don't know", "not sure", "might be", "could be", "possibly",
      "maybe", "uncertain", "unclear", "waiting to find out"],
     {"fear": +0.08, "nervousness": +0.06, "relief": -0.15}),

    # Good outcome confirmed → amplifies joy/relief, suppresses fear
    (["all clear", "came back fine", "came back normal", "negative result",
      "benign", "nothing serious", "worked out perfectly",
      "made it", "pulled through", "survived", "they're okay"],
     {"relief": +0.10, "joy": +0.08, "fear": -0.20, "nervousness": -0.20}),

    # Personal uniqueness / milestone → amplifies pride
    (["first in my family", "first time", "never done before",
      "youngest ever", "record", "milestone", "breakthrough",
      "generations", "family history"],
     {"pride": +0.12, "joy": +0.05}),

    # Temporal proximity → amplifies nervousness (performance context)
    (["tomorrow", "tonight", "in an hour", "in two hours",
      "this afternoon", "this morning", "in a few hours",
      "next hour", "right now", "any minute"],
     {"nervousness": +0.10, "excitement": +0.05}),

    # Physical anxiety signals → amplifies nervousness over fear
    (["hands are shaking", "heart is pounding", "can't sleep",
      "stomach in knots", "dry mouth", "sweating", "palms sweaty",
      "voice is shaking", "legs are shaking"],
     {"nervousness": +0.15, "fear": -0.05}),

    # Past tense / irreversible → amplifies remorse/grief
    (["never got to", "never said", "never told", "too late",
      "last time", "final", "goodbye", "already gone",
      "it's over", "can't take it back", "no going back"],
     {"remorse": +0.12, "grief": +0.08, "sadness": +0.05}),

    # Presence of "I loved" / "I love" in regret context → suppress love, boost remorse
    (["last thing i said", "last words", "never told them i loved",
      "never got to say i love", "wish i had said i love"],
     {"remorse": +0.15, "love": -0.25}),

    # Betrayal language → amplifies anger/disgust
    (["lied", "deceived", "manipulated", "fake", "pretend",
      "behind my back", "using me", "took credit"],
     {"anger": +0.08, "disgust": +0.08, "disappointment": +0.05}),

    # Achievement with effort → amplifies pride
    (["worked for", "years of", "months of", "trained for",
      "sacrificed", "gave up", "hard work", "dedication"],
     {"pride": +0.10, "joy": +0.05}),

    # Completion / finality → amplifies relief
    (["finally", "at last", "after all that", "it's over",
      "behind me now", "done now", "finished now"],
     {"relief": +0.08, "joy": +0.05}),

    # Isolation / alone → amplifies sadness/fear
    (["alone", "by myself", "no one", "nobody",
      "completely alone", "all alone"],
     {"sadness": +0.06, "fear": +0.06}),

    # Surprise / disbelief → amplifies surprise
    (["can't believe", "cannot believe", "unbelievable",
      "never expected", "totally unexpected",
      "came out of nowhere", "out of the blue"],
     {"surprise": +0.10}),
]


def _detect_archetypes(text_lower: str) -> list[str]:
    """Return list of matching archetype names."""
    matched = []
    for name, clusters in _ARCHETYPES:
        for cluster in clusters:
            if any(_kw_in(kw, text_lower) for kw in cluster):
                matched.append(name)
                break  # one cluster match is enough per archetype
    return matched


def _apply_modifiers(
    text_lower: str, scores: dict[str, float]
) -> dict[str, float]:
    """Apply modifier deltas to the current score dict."""
    result = dict(scores)
    for phrases, deltas in _MODIFIERS:
        if not any(_kw_in(p, text_lower) for p in phrases):
            continue
        for emo, delta in deltas.items():
            if emo in result:
                result[emo] = max(0.0, min(0.95, result[emo] + delta))
            elif delta > 0:
                result[emo] = delta
    return result


def encode(text: str) -> Optional[dict[str, float]]:
    """
    Run the situation encoder on a text.

    Returns a dict of {emotion: confidence} for the detected archetype(s),
    or None if no archetype matched.
    """
    text_lower = text.lower()
    archetypes  = _detect_archetypes(text_lower)

    if not archetypes:
        return None

    # Merge priors from all matched archetypes (take the max per emotion)
    merged: dict[str, float] = {}
    for arch in archetypes:
        for emo, conf in _PRIORS.get(arch, {}).items():
            merged[emo] = max(merged.get(emo, 0.0), conf)

    # Apply contextual modifiers
    merged = _apply_modifiers(text_lower, merged)

    # Remove emotions with negligible scores
    return {e: round(c, 3) for e, c in merged.items() if c >= 0.40}


def top_emotion(text: str) -> Optional[tuple[str, float]]:
    """
    Return (emotion, confidence) for the highest-scoring archetype emotion,
    or None if no archetype matched or top score < 0.52.
    """
    scores = encode(text)
    if not scores:
        return None
    top = max(scores.items(), key=lambda x: x[1])
    return top if top[1] >= 0.52 else None
