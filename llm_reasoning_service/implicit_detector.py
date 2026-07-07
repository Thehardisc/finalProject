import os
from typing import Optional
import situation_encoder as _SIT

MIN_CONFIDENCE     = 0.52
MULTI_MATCH_BOOST  = 0.07
CONTEXT_WEIGHT     = 0.75
NARRATIVE_BOOST    = 0.06
MAX_NARRATIVE      = 0.12
NEGATION_PENALTY   = 0.30

_NARRATIVE_MARKERS = [
    "ended up", "turned out", "only to", "only to find", "only to discover",
    "found out", "just found out", "just heard", "just got", "just learned",
    "after all that", "after everything", "after three", "after two", "after a week",
    "after months", "after years", "last minute", "at the last minute",
    "out of nowhere", "completely out of", "for the first time",
    "and then", "but then", "so then", "and just",
    "tried so hard", "worked so hard", "gave everything",
    "wasn't supposed to", "wasn't meant to", "never expected",
    "can't believe", "cannot believe", "i couldn't believe",
]

_NEGATION_TOKENS = [
    "not ", "didn't ", "did not ", "isn't ", "is not ", "wasn't ",
    "was not ", "don't ", "do not ", "can't ", "cannot ",
    "never ", "hardly ", "barely ",
]

_EMO_KEYWORDS: dict[str, list[str]] = {
    "joy":             ["happy", "joy", "excited", "thrilled", "delighted"],
    "sadness":         ["sad", "upset", "down", "depressed", "heartbroken", "crying"],
    "anger":           ["angry", "furious", "mad", "rage", "livid"],
    "fear":            ["scared", "afraid", "terrified", "anxious", "panicking"],
    "embarrassment":   ["embarrassed", "mortified", "humiliated", "ashamed"],
    "disappointment":  ["disappointed", "let down", "gutted"],
    "gratitude":       ["grateful", "thankful", "appreciative"],
    "amusement":       ["amused", "laughing", "funny", "hilarious"],
    "love":            ["love", "adore", "cherish", "devoted"],
    "surprise":        ["surprised", "shocked", "stunned", "astonished"],
    "curiosity":       ["curious", "interested", "fascinated", "intrigued"],
    "annoyance":       ["annoyed", "irritated", "frustrated", "bothered"],
    "optimism":        ["hopeful", "optimistic", "looking forward", "positive"],
    "pride":           ["proud", "accomplished", "achieved"],
    "remorse":         ["guilty", "regret", "sorry", "remorseful"],
    "grief":           ["lost", "bereaved", "mourning", "grieving"],
    "nervousness":     ["nervous", "worried", "anxious", "tense"],
    "disgust":         ["disgusted", "revolted", "sickened", "grossed out"],
    "relief":          ["relieved", "glad it's over", "finally over"],
    "excitement":      ["excited", "pumped", "stoked", "hyped"],
    "caring":          ["caring", "supportive", "looking after", "protective"],
    "disapproval":     ["disapprove", "disagree", "wrong of", "shouldn't have"],
    "admiration":      ["admire", "look up to", "inspired by", "in awe"],
    "realization":     ["realized", "just realized", "dawned on me", "clicked for me"],
    "confusion":       ["confused", "lost", "don't understand", "can't figure out"],
    "approval":        ["approve", "agreed", "makes sense", "right call"],
    "desire":          ["want so badly", "wish i had", "dying for"],
}

_RULES: list[tuple[list[str], str, float]] = [

    (["they picked me", "i got in", "i got accepted", "i got the job",
      "i start monday", "i start next week", "i got the call",
      "they said yes", "i passed", "we won", "it worked out",
      "i made it", "i got through", "officially accepted",
      "got the offer", "got promoted", "got the scholarship",
      "made the team", "made the cut", "got the grant",
      "they chose me", "i was selected", "we're finally",
      "worked out perfectly", "it all worked out"],
     "joy", 0.73),

    (["can't wait until", "can't wait —", "can't wait,",
      "we're finally taking", "finally taking the trip",
      "planned for two years", "planned for years",
      "counting down the days", "only two days",
      "leaves tomorrow", "trip is tomorrow", "flight tomorrow",
      "first time ever", "biggest day of my life", "this is it",
      "the moment i've been", "been waiting for this",
      "finally happening", "it's actually happening"],
     "excitement", 0.70),

    (["never showed up", "didn't show up", "never came", "never arrived",
      "table is still set", "chair is still empty", "place is still set",
      "still waiting for", "she left me", "he left me", "they left",
      "no one came", "nobody came", "stood me up",
      "packed up and left", "moved out without",
      "they ghosted me", "he ghosted me", "she ghosted me",
      "never called back", "stopped answering", "stopped texting",
      "said goodbye and", "last thing he ever", "last thing she ever",
      "last time i saw",
      "we used to", "i used to go there", "i used to see them",
      "the house is so quiet", "the place feels empty",
      "nobody left", "there is no one"],
     "sadness", 0.73),

    (["passed away", "passed this morning", "passed last night",
      "didn't make it", "they're gone", "she's gone now", "he's gone now",
      "the funeral is", "we said goodbye", "final goodbye",
      "one last time", "last time together",
      "keep expecting to see", "keep looking for",
      "still can't believe they're gone", "still feels unreal"],
     "grief", 0.80),

    (["studied for weeks", "studied every night", "worked so hard for",
      "practiced for months", "practiced for years", "prepared for months", "got cancelled", "was cancelled",
      "got canceled", "was canceled", "all for nothing", "wasted my time",
      "didn't even get", "never happened", "didn't make it through",
      "got rejected", "rejection letter", "they said no",
      "didn't get the job", "didn't get in", "didn't get the part",
      "missed the cutoff", "missed the deadline",
      "after all that work", "after everything i put in",
      "the deal fell through", "fell through at the last",
      "cancelled last minute", "called it off",
      "never even replied", "still no response"],
     "disappointment", 0.73),

    (["wish i had", "wish i hadn't", "should have said",
      "should have told", "should have been there",
      "the last thing i said", "last words i said",
      "never got to say", "never got to tell",
      "i took it for granted", "i didn't appreciate",
      "i ignored her", "i ignored him", "i pushed them away",
      "i was too busy", "too busy to notice",
      "i let them down", "i wasn't there for"],
     "remorse", 0.72),

    (["called the teacher mom", "called my teacher mom", "called her mom",
      "called him dad", "called the prof", "in front of the whole class",
      "in front of everyone", "everyone heard me", "she heard me",
      "he heard me", "the whole room heard", "the whole office heard",
      "room went silent", "everyone went quiet", "everyone stared",
      "everyone laughed at", "they were all looking at me",
      "walked in on", "barged in on", "caught me",
      "autocorrected to", "sent to the wrong person",
      "replied all by accident", "replied to all",
      "flew off during", "wig came off", "tripped on stage",
      "completely froze up", "went completely blank",
      "forgot the words", "forgot my lines"],
     "embarrassment", 0.73),

    (["walked straight into the glass", "walked into the glass door",
      "walked into a glass", "walked into the wall",
      "fell off the chair", "tripped in front of",
      "knocked over the entire", "knocked everything over",
      "in front of the entire office", "in front of the whole team",
      "everyone burst out", "everyone cracked up",
      "dog ate my", "cat knocked over",
      "sleep-talked", "sleepwalked",
      "auto-replied with", "accidentally sent",
      "spilled all over", "dropped everything",
      "ran into a lamp post", "ran into a pole"],
     "amusement", 0.71),

    (["you didn't have to", "you really didn't have to", "didn't need to do that",
      "means so much to me", "means the world to me",
      "that was so kind", "you have no idea what",
      "you have no idea how much", "i really appreciate",
      "so thoughtful of you", "went out of your way",
      "drove all the way", "came all the way",
      "stayed up all night to help",
      "dropped everything to",
      "without you i wouldn't have",
      "made such a difference",
      "i owe you so much", "i owe you one"],
     "gratitude", 0.74),

    (["run more tests", "more tests", "wouldn't tell me why", "didn't tell me why",
      "waiting for results", "waiting to hear back", "could be serious",
      "something might be wrong", "what if it's", "i don't know what's wrong",
      "they found something", "found a lump", "found something suspicious",
      "biopsy results", "scan results", "test results came back",
      "the mri showed", "the ct showed",
      "could be cancer", "might be serious",
      "they want to admit me", "they're keeping me",
      "alarm went off", "someone tried the door",
      "heard footsteps", "heard a noise downstairs",
      "i'm alone and", "nobody knows i'm here",
      "engine light came on", "brakes aren't working",
      "can't find them anywhere", "been missing since",
      "no one's heard from", "phone keeps going to voicemail"],
     "fear", 0.71),

    (["my hands are shaking", "hands won't stop shaking",
      "heart is pounding", "heart was racing",
      "dry mouth", "couldn't sleep last night",
      "haven't slept properly", "keep rehearsing",
      "going over it in my head", "can't stop thinking about",
      "what will they think", "what if i mess up",
      "pitch is tomorrow", "interview is tomorrow",
      "presentation is in", "speech is in",
      "first day is tomorrow", "first shift is",
      "going live in", "going on stage"],
     "nervousness", 0.70),

    (["they just changed", "raised the price again", "increased the price",
      "without any warning", "without notice", "no notice",
      "behind my back", "without asking me",
      "took credit for my", "stole my idea",
      "lied to my face", "looked me in the eye and lied",
      "they promised and then", "they swore and then",
      "they keep doing this", "they always do this",
      "once again they", "yet again they",
      "after i specifically asked", "after i told them not to",
      "cut me off in", "slammed the door",
      "left me on read", "ignored me again"],
     "anger", 0.70),

    (["raised the price again", "increased the price again",
      "did it again without", "no warning whatsoever",
      "no explanation whatsoever", "zero notice", "no notice at all",
      "didn't even tell us", "without asking",
      "how could they do this", "keeps happening",
      "every single time", "not for the first time",
      "the third time this week", "for the hundredth time",
      "still hasn't fixed", "still hasn't replied",
      "been waiting for days", "been waiting for hours",
      "already told them twice", "already told them three times",
      "right in the middle of", "keeps interrupting"],
     "annoyance", 0.66),

    (["found out they had been", "turns out they were",
      "it was all fake", "it was a scam", "none of it was real",
      "they never cared", "they were just using",
      "the whole thing was staged", "planted evidence",
      "hidden camera", "secret recording",
      "used me for", "took advantage of",
      "the conditions were terrible", "inhumane conditions",
      "left in filth", "left in the dark",
      "rotted in the back", "expired and still",
      "found a cockroach", "found a bug in",
      "couldn't even look"],
     "disgust", 0.68),

    (["they shouldn't have", "that was wrong", "that was not okay",
      "that's not how you", "that's not how anyone should",
      "the way they handled", "how they dealt with",
      "it was completely avoidable", "entirely their fault",
      "should have known better", "should have done better",
      "that's unacceptable", "that's inexcusable"],
     "disapproval", 0.66),

    (["was already there", "already arrived", "left at the same time",
      "got there before me", "arrived before", "we literally left together",
      "wait — already", "wait, already",
      "how is that possible", "how did you get here",
      "i can't believe how fast",
      "completely out of the blue", "came out of nowhere",
      "last person i expected", "never saw that coming",
      "walked in to find", "opened the door and",
      "turned around and",
      "all of a sudden", "just like that"],
     "surprise", 0.66),

    (["been thinking about you all day", "been thinking about you",
      "can't stop thinking about you", "thinking of you",
      "you mean everything to me", "you mean the world to me",
      "miss you so much", "counting the days",
      "keep looking at your photo", "saved all your messages",
      "smile every time i see", "my face lights up",
      "told everyone about you", "couldn't stop talking about you",
      "can't imagine without you", "the day feels different without"],
     "love", 0.74),

    (["couldn't have done it without", "she did it alone",
      "he figured it out on his own", "without any help",
      "raised all those kids by herself", "raised them alone",
      "built it from nothing", "started from zero",
      "worked three jobs", "works two jobs",
      "never once complained", "never asked for anything",
      "gave everything she had", "gave everything he had",
      "inspiring the whole team", "set the standard"],
     "admiration", 0.70),

    (["first in my family", "first in the family",
      "nobody else managed to", "never been done before",
      "youngest ever to", "first woman to", "first man to",
      "they finally recognised", "they finally noticed",
      "awarded the", "presented with the",
      "stood on that stage", "walked across that stage",
      "crossed the finish line", "finished the race",
      "completed the program", "certified now",
      "graduated today", "just graduated"],
     "pride", 0.71),

    (["how does that even work", "how does that work", "how does it work",
      "can you explain", "explain it again", "i don't understand how",
      "what does that mean exactly", "wait, explain",
      "tell me more about", "i need to know more",
      "never thought about that before", "never considered that",
      "made me think about", "got me wondering",
      "how did they come up with", "who came up with",
      "what's the story behind", "where does that come from"],
     "curiosity", 0.70),

    (["don't understand what happened", "don't know what happened",
      "makes no sense", "doesn't add up",
      "what just happened", "one minute", "first it was",
      "then it changed to", "then it became",
      "completely contradicted", "said one thing then",
      "no explanation given", "no reason given",
      "can't figure out why", "can't work out why",
      "everyone else seems to get it", "am i the only one"],
     "confusion", 0.67),

    (["finally over", "it's finally done", "finally behind me",
      "test came back clear", "results came back normal",
      "all clear", "nothing serious", "benign",
      "she pulled through", "he pulled through", "they made it",
      "surgery went well", "operation was a success",
      "landed safely", "got home safely",
      "just in time", "arrived just before",
      "found it", "showed up eventually",
      "it wasn't as bad as", "much better than i feared",
      "turned out to be nothing"],
     "relief", 0.72),

    (["can't wait to try", "really want to try", "excited to start",
      "looking forward to", "hope it goes well", "fingers crossed",
      "hopefully this time", "this time will be different",
      "fresh start", "new beginning", "turning a corner",
      "things are looking up", "finally a light at the end",
      "first step toward", "working toward",
      "getting there", "making progress"],
     "optimism", 0.64),

    (["only just realised", "just realised", "just realized", "only now realised",
      "clicked for me", "it hit me", "dawned on me",
      "look back and see", "looking back i can see",
      "the whole time it was", "it was staring me in the face",
      "the pattern was there", "the signs were there",
      "i finally understand", "now i get it",
      "all those times", "all those years",
      "it makes sense now", "it all makes sense now"],
     "realization", 0.68),

    (["totally the right call", "right decision",
      "exactly what needed to happen", "handled it well",
      "couldn't have done better", "impressed by how",
      "came through when it mattered", "rose to the occasion",
      "made the hard choice", "did the right thing",
      "earned that", "well deserved"],
     "approval", 0.65),

    (["can't stop thinking about", "keep imagining what it would be like",
      "dreamed about this", "dreamed of this",
      "all i want right now", "all i want is",
      "would do anything for", "would give anything for",
      "just want one chance", "one more chance",
      "if only i had", "if only there was"],
     "desire", 0.65),

    (["check in on them", "check on her", "check on him",
      "make sure they're okay", "making sure she's alright",
      "stayed with her all night", "stayed with him all night",
      "dropped everything to be with", "flew out to be with",
      "brought them food", "left food at the door",
      "sat with them while", "held their hand",
      "covered for them", "took their shift"],
     "caring", 0.68),

    (["stomach is in knots", "can't eat", "too nervous to eat",
      "palms are sweaty", "going over my lines",
      "rehearsed a hundred times", "practiced in the mirror",
      "big presentation", "final exam tomorrow",
      "jury is tomorrow", "defence is tomorrow",
      "board exam", "licensing exam"],
     "nervousness", 0.68),
]


def _apply_negation(text_lower: str, scores: dict[str, float]) -> dict[str, float]:
    """Penalise emotions whose keywords appear right after a negation word."""
    penalised = {}
    for emo, conf in scores.items():
        neg_hit = False
        for kw in _EMO_KEYWORDS.get(emo, []):
            for neg in _NEGATION_TOKENS:
                if (neg + kw) in text_lower:
                    neg_hit = True
                    break
            if neg_hit:
                break
        penalised[emo] = conf * (1.0 - NEGATION_PENALTY) if neg_hit else conf
    return penalised


def _narrative_boost(text_lower: str, top_emotion: str, score: float) -> float:
    """Add a boost when narrative/situational markers are present."""
    hits = sum(1 for m in _NARRATIVE_MARKERS if m in text_lower)
    boost = min(MAX_NARRATIVE, NARRATIVE_BOOST * hits)
    return min(0.95, score + boost)


def _score_rules(text: str) -> dict[str, float]:
    """Return per-emotion confidence scores for a text."""
    text_lower = text.lower()
    scores: dict[str, float] = {}

    for phrases, emotion, base_conf in _RULES:
        hits = sum(1 for p in phrases if p in text_lower)
        if hits == 0:
            continue
        boost = MULTI_MATCH_BOOST * (hits - 1) * (0.8 ** max(0, hits - 2))
        conf  = min(0.95, base_conf + boost)
        if emotion not in scores or conf > scores[emotion]:
            scores[emotion] = conf

    return scores


def _fuse(rule_scores: dict[str, float], sit_scores: dict[str, float]) -> tuple[dict[str, float], str]:
    """Fuse phrase-rule scores (Layer 1) with situation-encoder scores (Layer 2)."""
    if not sit_scores:
        return rule_scores, "rule-based-v2"

    if not rule_scores:
        return sit_scores, "situation-encoder"

    top_rule = max(rule_scores.values()) if rule_scores else 0.0
    top_sit  = max(sit_scores.values())  if sit_scores  else 0.0

    merged: dict[str, float] = {}
    all_emotions = set(rule_scores) | set(sit_scores)

    for emo in all_emotions:
        r = rule_scores.get(emo, 0.0)
        s = sit_scores.get(emo, 0.0)

        if r > 0 and s > 0:
            merged[emo] = round(0.45 * r + 0.55 * s, 3)
        elif r > 0:
            merged[emo] = round(r * 0.85, 3)
        else:
            merged[emo] = round(s, 3)

    method = "situation+rules" if (top_rule >= MIN_CONFIDENCE and top_sit >= MIN_CONFIDENCE) else (
             "situation-encoder" if top_sit > top_rule else "rule-based-v2")
    return merged, method


def detect(text: str, history: list) -> Optional[dict]:
    """Stage 2 implicit emotion detection."""
    text_lower = text.lower()

    rule_scores = _score_rules(text)
    sit_scores  = _SIT.encode(text) or {}

    # History may only REINFORCE emotions the current message itself evidences —
    # never author them, or the previous message's emotion bleeds into this one.
    if history:
        context = " ".join(h for h in history[-2:] if h != text)
        ctx_scores = _score_rules(context)
        for emo, conf in ctx_scores.items():
            if rule_scores.get(emo, 0.0) > 0 or sit_scores.get(emo, 0.0) > 0:
                rule_scores[emo] = max(rule_scores.get(emo, 0.0), conf * CONTEXT_WEIGHT)

    merged, method = _fuse(rule_scores, sit_scores)

    if not merged:
        return None

    merged = _apply_negation(text_lower, merged)

    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)
    if not ranked or ranked[0][1] < MIN_CONFIDENCE:
        return None

    top_emotion_label, top_conf = ranked[0]
    top_conf = _narrative_boost(text_lower, top_emotion_label, top_conf)
    ranked[0] = (top_emotion_label, top_conf)
    ranked.sort(key=lambda x: x[1], reverse=True)

    if ranked[0][1] < MIN_CONFIDENCE:
        return None

    top_emotion_label, top_conf = ranked[0]

    # Keyword/prototype matches carry hand-tuned confidences (0.68–0.80) that were
    # regularly beating the meta-learner's calibrated posteriors. Scale them down so
    # a phrase match reads as a hint, not a verdict.
    conf_scale = float(os.environ.get("IMPLICIT_DETECTOR_CONF_SCALE", "0.85"))
    top_conf   = top_conf * conf_scale

    scores: dict[str, float] = {"neutral": round(max(0.0, 1.0 - top_conf), 3)}
    for emo, conf in ranked[:5]:
        scores[emo] = round(conf * conf_scale, 3)

    return {
        "emotion":    top_emotion_label,
        "confidence": round(top_conf, 3),
        "scores":     scores,
        "method":     method,
    }
