"""
Emotional arc templates for demo conversation generation.

Each trajectory defines:
  - type:        machine-readable identifier (used as JSONL key)
  - description: what this arc represents emotionally
  - arc:         ordered list of dominant emotions per phase
  - length:      target message count (8-14)
  - prompt:      instruction injected into the Claude generation prompt
  - count:       how many distinct conversations to generate per trajectory

Designed to cover the full GoEmotions-28 label space across trajectories.
"""

TRAJECTORIES = [
    {
        "type": "escalating_conflict",
        "description": "Starts as a friendly chat, a misunderstanding appears and tension builds into an argument.",
        "arc": ["neutral", "curiosity", "annoyance", "disapproval", "anger"],
        "length": 10,
        "count": 15,
        "prompt": (
            "Write a 10-message conversation (single person, first-person perspective) "
            "that starts as a relaxed chat, then a frustrating misunderstanding appears, "
            "and the emotional tone steadily escalates from mild annoyance to real anger by the end. "
            "Keep it realistic and natural — no sudden tone jumps."
        ),
    },
    {
        "type": "de_escalation",
        "description": "Starts angry or upset, gradually calms down through reflection or reassurance.",
        "arc": ["anger", "annoyance", "disapproval", "realization", "neutral", "relief"],
        "length": 10,
        "count": 15,
        "prompt": (
            "Write a 10-message conversation (single person, first-person perspective) "
            "where the person starts very angry or upset about something, then step by step "
            "processes their feelings, reconsiders the situation, and ends in a calmer, "
            "more neutral or relieved emotional state. Realistic internal emotional arc."
        ),
    },
    {
        "type": "conflict_resolution",
        "description": "Argument → negotiation → apology → forgiveness → relief.",
        "arc": ["anger", "disapproval", "realization", "remorse", "approval", "gratitude", "relief"],
        "length": 12,
        "count": 15,
        "prompt": (
            "Write a 12-message conversation (single person) where they start frustrated in an argument, "
            "gradually see the other person's point of view, feel remorse for their own behavior, "
            "apologize, receive forgiveness, and end with relief and gratitude. "
            "The emotional arc should feel earned and authentic."
        ),
    },
    {
        "type": "growing_excitement",
        "description": "Neutral start, gradually building interest, curiosity, excitement, joy.",
        "arc": ["neutral", "curiosity", "approval", "excitement", "joy", "admiration"],
        "length": 10,
        "count": 15,
        "prompt": (
            "Write a 10-message conversation (single person) that starts with mild, everyday tone, "
            "then good news or an interesting development is introduced, "
            "and excitement and joy build naturally through each message. "
            "The last messages should feel genuinely happy and energized."
        ),
    },
    {
        "type": "disappointment_spiral",
        "description": "Hope and optimism → reality doesn't match → sadness → acceptance.",
        "arc": ["optimism", "approval", "anticipation", "confusion", "disappointment", "sadness"],
        "length": 10,
        "count": 15,
        "prompt": (
            "Write a 10-message conversation (single person) that starts with hopeful, optimistic feelings "
            "about something they were looking forward to, then things don't go as planned, "
            "the mood shifts through confusion and disappointment, "
            "and ends in quiet sadness or resigned acceptance."
        ),
    },
    {
        "type": "emotional_support",
        "description": "Person shares pain/vulnerability, receives care, ends in warmth and gratitude.",
        "arc": ["sadness", "fear", "grief", "caring", "love", "gratitude", "relief"],
        "length": 12,
        "count": 15,
        "prompt": (
            "Write a 12-message conversation (single person) where they open up about something painful "
            "they are going through — grief, fear, or sadness. As they express themselves, "
            "they feel heard and supported, the tone shifts to warmth and gratitude. "
            "Emotional authenticity is key — vulnerability and care should feel real."
        ),
    },
    {
        "type": "celebration",
        "description": "Good news arrives → joy builds → pride, gratitude, love.",
        "arc": ["surprise", "joy", "excitement", "pride", "admiration", "gratitude", "love"],
        "length": 10,
        "count": 15,
        "prompt": (
            "Write a 10-message conversation (single person) around a piece of genuinely great news — "
            "a promotion, a win, an achievement, or a meaningful moment. "
            "The emotional arc should go from surprise to growing joy, pride, and deep gratitude. "
            "Feel celebratory and warm throughout."
        ),
    },
    {
        "type": "anxiety_resolution",
        "description": "Nervous / fearful → vulnerability → reassurance builds → calm relief.",
        "arc": ["nervousness", "fear", "confusion", "caring", "realization", "relief"],
        "length": 10,
        "count": 15,
        "prompt": (
            "Write a 10-message conversation (single person) where they are anxious or nervous "
            "about something coming up — a presentation, a difficult conversation, a medical result. "
            "Through their messages they process the fear, receive internal or external reassurance, "
            "and gradually arrive at a calmer, more grounded place."
        ),
    },
    {
        "type": "passive_aggression",
        "description": "Surface neutral / polite but building annoyance underneath, ending in a clear emotional break.",
        "arc": ["neutral", "annoyance", "disapproval", "disgust", "anger"],
        "length": 10,
        "count": 15,
        "prompt": (
            "Write a 10-message conversation (single person) where they are clearly bothered by something "
            "but initially try to be polite about it. Their frustration leaks out through tone and word choice "
            "even when the content seems neutral. By the end, the mask comes off and the real emotion shows. "
            "This is the passive-aggression arc: surface civility, underlying anger."
        ),
    },
    {
        "type": "nostalgia_bittersweet",
        "description": "Warm memories bring joy, but the awareness of loss or change pulls into sadness.",
        "arc": ["joy", "admiration", "love", "realization", "sadness", "grief", "acceptance"],
        "length": 12,
        "count": 15,
        "prompt": (
            "Write a 12-message conversation (single person) where they are reminiscing about a happy time "
            "— a relationship, a phase of life, a person no longer around. Start with warm nostalgia and joy, "
            "then the awareness of what has been lost pulls the mood toward sadness and grief. "
            "End in quiet, bittersweet acceptance."
        ),
    },
    {
        "type": "volatile_mix",
        "description": "Rapid emotional swings — joy, anger, fear, amusement in quick succession.",
        "arc": ["joy", "surprise", "anger", "amusement", "fear", "relief", "confusion"],
        "length": 10,
        "count": 15,
        "prompt": (
            "Write a 10-message conversation (single person) with rapid emotional shifts — "
            "they might be happy, then suddenly annoyed, then laugh it off, then scared, then relieved. "
            "The key is that the emotional tone changes every 2-3 messages in a believable way, "
            "like a chaotic but real day of events."
        ),
    },
    {
        "type": "gradual_sadness",
        "description": "Starts neutral or slightly positive, slowly and quietly slides into deep sadness.",
        "arc": ["neutral", "realization", "disappointment", "sadness", "grief"],
        "length": 10,
        "count": 15,
        "prompt": (
            "Write a 10-message conversation (single person) where they start in an ordinary mood, "
            "then something quiet and heavy starts pulling them down — a realization, a memory, "
            "a slow understanding of loss. The decline should be gradual and subtle, "
            "ending in deep sadness or grief. No dramatic events — just quiet emotional weight."
        ),
    },
]

TRAJECTORY_TYPES = [t["type"] for t in TRAJECTORIES]
TOTAL_CONVERSATIONS = sum(t["count"] for t in TRAJECTORIES)
