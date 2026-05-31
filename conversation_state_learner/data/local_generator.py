"""
local_generator.py — Generates realistic conversation data WITHOUT any API call.

Uses parameterized templates with random variation across:
  - Subjects (work, relationship, friendship, family, personal growth)
  - Intensities (mild → severe)
  - Writing styles (casual, emotional, formal)

Produces the same JSON format as generator.py so collect.py --phase run works unchanged.

Usage:
    python3 -m data.local_generator              # generate all trajectories
    python3 -m data.local_generator --count 5    # 5 per trajectory
    python3 -m data.local_generator --type escalating_conflict
"""

import json
import random
import argparse
from pathlib import Path
from typing import List

from data.trajectories import TRAJECTORIES

RAW_DIR = Path(__file__).parent.parent / "raw_conversations"
random.seed()   # new seed each run for variety


# ── Shared vocabulary pools ───────────────────────────────────────────────────

SUBJECTS = [
    ("a project at work",     "my manager",       "the deadline"),
    ("our friendship",        "my best friend",   "what happened last night"),
    ("a family dinner",       "my sister",        "what she said"),
    ("a relationship issue",  "my partner",       "the argument we had"),
    ("a university exam",     "my professor",     "the grade I got"),
    ("moving out",            "my roommate",      "the lease situation"),
    ("a job interview",       "the recruiter",    "the offer"),
    ("a creative project",    "my collaborator",  "the feedback I got"),
    ("a health scare",        "my doctor",        "the test results"),
    ("a financial stress",    "my bank",          "the unexpected bill"),
    ("a wedding plan",        "my fiancé",        "the venue decision"),
    ("a sports competition",  "my coach",         "the training session"),
]

OPENERS_NEUTRAL = [
    "Hey, can we talk about {subject}?",
    "So I've been thinking about {subject} a lot lately.",
    "Something came up with {subject} and I need to process it.",
    "Quick thing — {subject} is on my mind today.",
    "I wanted to check in about {subject}.",
]

INTENSIFIERS = ["really", "honestly", "seriously", "genuinely", "actually"]
SOFTENERS    = ["I think", "I feel like", "it seems", "I guess", "maybe"]


# ── Trajectory-specific message pools ────────────────────────────────────────

TRAJECTORY_POOLS = {

"escalating_conflict": [
    [
        "Hey, can we talk? Something's been bothering me.",
        "It's about {subject} — {person} made a decision without telling me and I'm a bit annoyed.",
        "I asked them about it and they acted like it wasn't a big deal. But it {intensifier} is.",
        "This is the second time they've done something like this. I'm starting to lose patience.",
        "I told them clearly how I feel and they just brushed it off. That's not okay.",
        "Do they think my feelings don't matter? Because it's starting to feel that way.",
        "I'm not going to keep letting this slide. I need to address it properly.",
        "I sent them a message laying everything out. Now I'm waiting and I'm furious.",
        "Still no reply. The silence is making it so much worse.",
        "That's it. I'm done being the patient one here.",
    ],
    [
        "Something happened with {subject} and I need to vent.",
        "So {person} just completely ignored what we agreed on. I'm {intensifier} irritated.",
        "They knew this was important to me. There's no excuse.",
        "When I brought it up they got defensive right away. Classic.",
        "They turned it around and started blaming me. I can't believe this.",
        "My hands are shaking I'm so frustrated right now.",
        "I said things I probably shouldn't have but I meant every word.",
        "They hung up on me. HUNG. UP.",
        "I am absolutely done tolerating this disrespect.",
        "We are NOT okay right now. Not even close.",
    ],
    [
        "Okay so {thing} has officially become a problem.",
        "{person} keeps doing this over and over and I keep letting it go.",
        "Not this time. I called them out and they didn't like it.",
        "They tried to make it seem like I was overreacting. I wasn't.",
        "The argument escalated fast. We said some real things to each other.",
        "I'm shaking. I've never been this angry with them before.",
        "They actually accused me of being the problem here. ME.",
        "I walked away before I said something I couldn't take back.",
        "Now I'm just sitting here, furious, trying to breathe.",
        "I don't know where we go from here but I'm done pretending this is fine.",
    ],
],

"de_escalation": [
    [
        "I'm so angry right now. {subject} completely blew up today.",
        "{person} said something that cut really deep and I snapped back.",
        "I've been pacing around my room just trying to cool down.",
        "Why do I always say things I regret when I'm this upset?",
        "Okay. Okay. I need to actually think about this instead of just reacting.",
        "Maybe I was also partly responsible for how things escalated.",
        "I sent them an apology for the things I said — not for how I felt, but for how I expressed it.",
        "They actually responded kindly. That helped.",
        "I think we can work through this if we're both calm.",
        "I'm still hurt but I'm not angry anymore. That's something.",
    ],
    [
        "I came home and just started crying. {subject} was awful today.",
        "I said things to {person} I can't take back and it's eating me alive.",
        "I've been sitting here for an hour just replaying it.",
        "The anger is fading and now there's just this heavy guilt.",
        "I need to reach out. I need to make this right.",
        "I wrote out what I wanted to say and deleted it four times.",
        "Finally sent something honest. Just that I'm sorry and I want to understand.",
        "They said they appreciated me reaching out. I felt some of the weight lift.",
        "We're not fully okay yet but we're talking. That's better than silence.",
        "I'm exhausted but I feel calmer. Tomorrow we'll figure the rest out.",
    ],
],

"conflict_resolution": [
    [
        "I don't even know how to start. {person} and I had a really bad fight.",
        "It was about {subject} and we both said things that weren't fair.",
        "I've been angry for two days but today I just feel tired.",
        "I keep thinking about their perspective and honestly… they had a point.",
        "I picked up the phone three times and put it down.",
        "Finally called. They picked up. There was this long silence.",
        "We both started talking at the same time. Then we both laughed a little.",
        "I told them I was sorry. A real sorry. Not a defensive one.",
        "They said they were sorry too. I could hear it in their voice.",
        "I feel like we came out of this actually understanding each other better.",
        "Something shifted. I think this made us stronger, weirdly enough.",
        "I'm genuinely grateful we worked through it instead of walking away.",
    ],
    [
        "The fight about {thing} has been weighing on me all week.",
        "I was so sure I was right. I went over the argument in my head a hundred times.",
        "But the more I thought about it the more I realized I wasn't listening.",
        "I asked {person} if we could meet up and talk properly.",
        "They agreed. We sat down without phones, without distractions.",
        "I let them speak first and I actually heard them — maybe for the first time.",
        "Then I shared my side and they listened too. Really listened.",
        "We found the misunderstanding at the center of all of it.",
        "Silly, almost. Such a big fight over a misread message.",
        "We hugged before we left. A real one.",
        "Relief is the only word. Pure relief.",
        "Things feel lighter now than they have in weeks.",
    ],
],

"growing_excitement": [
    [
        "Okay so something is happening with {subject} and I can feel it building.",
        "I got a message from {person} this morning that has me curious.",
        "They said they have news but didn't say what. Now I can't focus on anything.",
        "Just got a follow-up message — it's something good. Definitely something good.",
        "My stomach is doing flips waiting for the full story.",
        "Okay they called me. They're excited. Now I'm EXCITED.",
        "THIS IS HAPPENING. It's actually happening.",
        "I cannot believe this is real. I keep rereading the confirmation.",
        "Screaming internally. This is everything I've been working toward.",
        "I have to tell everyone. Starting with you. WE DID IT.",
    ],
    [
        "Something small happened today with {subject} that's making me hopeful.",
        "It might be nothing but it feels like the beginning of something.",
        "{person} mentioned an opportunity and my brain has not stopped since.",
        "I looked into it and the more I read the more excited I get.",
        "I applied. I applied and now I'm terrified and thrilled at the same time.",
        "Just got an acknowledgment email. This is real. It's actually real.",
        "They want to interview me. They actually want to interview me.",
        "Interview went so well. I left feeling like I was floating.",
        "Got the call. I got it. I got it!!",
        "I've wanted this for so long and now it's here and I don't know what to do with myself.",
    ],
],

"disappointment_spiral": [
    [
        "I had such high hopes for {subject} this time.",
        "I spent weeks preparing and telling myself this would finally work out.",
        "The results just came in and… no. It didn't work.",
        "I can't believe it. I honestly thought I had done everything right.",
        "I keep going over what I could have done differently.",
        "{person} tried to console me but the words aren't landing right now.",
        "I feel embarrassed. I had told everyone this was going to happen.",
        "Now I have to explain it didn't. I dread those conversations.",
        "I'm sitting in the quiet and it's heavy in a way I can't describe.",
        "Maybe I'm just not meant for this. That thought is new and it scares me.",
    ],
    [
        "I was so sure about {subject}. Told myself this time was different.",
        "Checked the results. Failed again.",
        "I don't understand. I really don't understand what I'm missing.",
        "Called {person} to tell them. The silence on their end said everything.",
        "I'm not crying. I just feel hollow.",
        "The energy I put into this… where does it go when things don't work out?",
        "I turned my phone off. I don't want to talk to anyone right now.",
        "Sitting here in the dark just letting it sit.",
        "I don't want to give up but I don't know how to keep going either.",
        "This is the lowest I've felt in a while.",
    ],
],

"emotional_support": [
    [
        "I need to talk to someone. Something happened and I'm really struggling.",
        "{subject} has been harder than I expected and I've been holding it together by a thread.",
        "I lost someone close to me last month and I don't think I've really grieved yet.",
        "I just keep pushing through the days but at night it hits me.",
        "The grief comes in waves. Right now it's a wave.",
        "I'm scared of how heavy it feels. Like I can't breathe sometimes.",
        "Just saying all this out loud helps, somehow.",
        "Someone sent me a message today that really moved me. Reminded me I'm not alone.",
        "I cried for the first time since it happened. It felt like releasing something.",
        "I still miss them. I will for a long time. But I think I'll be okay.",
        "Thank you for being here. It matters more than I can say.",
        "I feel a little lighter tonight. That's enough for now.",
    ],
    [
        "I've been avoiding talking about {subject} because I didn't know how.",
        "But I think I need to say it out loud: I'm not okay.",
        "I've been pretending everything is fine and it's exhausting.",
        "{person} noticed and gently asked. That broke something open in me.",
        "I told them the truth for the first time. All of it.",
        "They didn't try to fix it. They just listened. That's what I needed.",
        "I cried so hard I couldn't speak for a minute.",
        "When I finally could speak I said 'I'm so tired of carrying this alone'.",
        "They said 'you don't have to anymore'.",
        "I don't know how to describe what that felt like. Safe, I think.",
        "Still processing. Still sad. But less alone.",
        "Sometimes being heard is the only thing that helps.",
    ],
],

"celebration": [
    [
        "I literally cannot stop smiling. {subject} just changed everything.",
        "The news came in this morning and I had to read it three times.",
        "{person} called and we were both just shouting with excitement.",
        "I've been working toward this for so long and it's finally here.",
        "My hands were shaking when I accepted. Happy shaking.",
        "I called my family. My mom started crying. I started crying.",
        "This is one of those days you remember forever.",
        "I feel proud. Like genuinely, deeply proud of myself.",
        "I want to thank everyone who believed in me when I didn't.",
        "Life is so good right now. I want to hold this feeling forever.",
    ],
    [
        "Okay I have to tell someone — something incredible just happened.",
        "{thing} came through. The thing I've been hoping for for two years.",
        "I got the email at 9am and I sat completely still for a full minute.",
        "Then I screamed into a pillow so I wouldn't wake my neighbors.",
        "Sent a voice note to {person} who was with me through the whole journey.",
        "They sent one back and they were crying. Happy crying.",
        "I took myself out to lunch to celebrate. Treated myself right.",
        "Walking home I just felt… grateful. Deeply, overwhelmingly grateful.",
        "I'm going to remember this day every time I doubt myself.",
        "This is what it feels like when you don't give up. Remember this.",
    ],
],

"anxiety_resolution": [
    [
        "I have a big thing tomorrow and I can't calm down.",
        "{subject} has been on my mind for weeks and tomorrow is the day.",
        "I keep running worst-case scenarios in my head.",
        "What if I freeze? What if everything goes wrong?",
        "I've been pacing for an hour. My palms won't stop sweating.",
        "I called {person} and just talking helped a little.",
        "They reminded me how much I've prepared. They're right. I have.",
        "I can't control the outcome but I can control how I show up.",
        "Going to do my breathing exercises and get some sleep.",
        "Whatever happens, I'll handle it. I always do.",
    ],
    [
        "The anxiety about {subject} is at a ten today.",
        "It's the kind of nervous that's in your chest, not your head.",
        "I know I'm ready. But knowing and feeling are different things.",
        "I wrote down everything that could go wrong. Then everything that could go right.",
        "The right list was longer. That helped.",
        "Got a message from {person}: 'You've got this.'",
        "Something released in me when I read that.",
        "I'm still nervous but it's the productive kind now.",
        "About to head in. Taking three deep breaths.",
        "Okay. Whatever happens, I showed up. That's what matters.",
    ],
],

"passive_aggression": [
    [
        "Fine. Everything is {intensifier} fine.",
        "{person} did the thing again with {subject} and I'm totally not bothered.",
        "I said nothing because what's the point, right? It never changes anyway.",
        "I'm being helpful and pleasant and absolutely not seething inside.",
        "They asked if I was okay. I said 'yep!' with a little too much energy.",
        "I reorganized the entire situation around them without saying a word.",
        "Every time they speak I feel the pressure building.",
        "I'm smiling so hard my face hurts.",
        "One more comment. One more dismissive comment and I will say something.",
        "I said something. Not politely.",
    ],
    [
        "Oh it's totally fine that {person} took credit for {subject}. Totally.",
        "I'm not going to make a scene. I'm just going to remember it.",
        "And remember it. And remember it.",
        "When they asked for my help today I helped. Obviously. I'm professional.",
        "But there was an edge in my voice that even I could hear.",
        "They didn't notice. Or pretended not to.",
        "I started keeping a log. Just for clarity. Just in case.",
        "Someone asked why I seemed off. I said 'tired'. I wasn't tired.",
        "Something finally slipped out of me in the meeting. Not much. But enough.",
        "The room went quiet. Good. I'm tired of being quiet.",
    ],
],

"nostalgia_bittersweet": [
    [
        "Found some old photos today. {subject} from years ago.",
        "I forgot how happy we were. I forgot how easy things felt.",
        "There's {person} in the background, laughing. I miss that laugh.",
        "We were so young. So sure that everything good would just keep going.",
        "I wonder if we knew at the time how precious it was.",
        "I don't think we did. We never do.",
        "Scrolling through these slowly, trying to hold each one.",
        "I can feel how far away that time is now.",
        "Things changed. We changed. Life does that.",
        "I'm grateful it happened even if it's over.",
        "But tonight I'll sit with missing it for a while.",
        "Some sadness is just love that doesn't know where to go.",
    ],
    [
        "I drove past the old place today. {subject}.",
        "Everything looked the same and completely different.",
        "I thought of {person} immediately. Of course I did.",
        "We made so many memories there. I can still hear the sounds of it.",
        "It's strange how a place can hold a whole chapter of your life.",
        "I sat in my car for a while just looking.",
        "There was joy in the memory and underneath it a kind of ache.",
        "I miss that version of us. Of me.",
        "I don't regret anything. I just feel the distance.",
        "Some things end not because they fail but because time moves.",
        "I think I needed to feel this today.",
        "Goodbye, again, to something I loved.",
    ],
],

"volatile_mix": [
    [
        "TODAY. What a day. Where to even start.",
        "Got great news about {subject} this morning. Actually jumped up and down.",
        "Then {person} said something that completely ruined my mood.",
        "I was furious for about twenty minutes and then something funny happened.",
        "I laughed so hard I forgot I was angry.",
        "Then the funny thing led to an awkward thing and now I'm kind of embarrassed.",
        "Oh and then I got a confusing email about {thing} that made my heart race.",
        "Turned out to be nothing. Relief flooded through me.",
        "Now I'm tired. Emotionally whiplashed.",
        "I don't even know what I'm feeling anymore. Just all of it at once.",
    ],
    [
        "Everything happened at the same time today.",
        "Started with excitement about {subject} — legitimate excitement.",
        "Then a misunderstanding with {person} that had me spiraling.",
        "A full emotional free-fall for like thirty minutes.",
        "Then they clarified and it was all fine. The relief was almost funny.",
        "Then I got some news that scared me.",
        "But then more context made it less scary.",
        "I laughed at my own catastrophizing.",
        "Now I'm exhausted and kind of giddy.",
        "My emotional range today deserves an award honestly.",
    ],
],

"gradual_sadness": [
    [
        "It was a normal day. Nothing dramatic happened.",
        "{subject} went fine. {person} was around. The usual.",
        "But somewhere around mid-afternoon something felt off.",
        "Hard to name. Like a slow dimming.",
        "I found myself staring out the window without meaning to.",
        "Thinking about things I usually keep tucked away.",
        "A feeling I haven't felt in a while came back. Quiet and heavy.",
        "I ate dinner alone and didn't mind. That itself felt like a sign.",
        "I don't want to talk to anyone. I just want to sit with it.",
        "The sadness isn't sharp. It's more like weather. Low clouds.",
    ],
    [
        "Nothing happened today. That might be the problem.",
        "I went through the motions. Work, food, screen, sleep.",
        "But there was this undertone I couldn't shake.",
        "Like something is missing and I can't remember what it is.",
        "I texted {person} but didn't send it.",
        "The afternoon dragged. I noticed the light changing and felt something tighten.",
        "I'm thinking about {subject} more than I should.",
        "Not obsessively. Just… persistently.",
        "I'll probably feel better tomorrow. I usually do.",
        "But tonight I'm letting myself be sad without explaining it.",
    ],
],

}


# ── Generator ─────────────────────────────────────────────────────────────────

def _pick(pool: list):
    return random.choice(pool)


def _fill(template: str) -> str:
    """Replace {placeholders} with random values."""
    s, p, t = _pick(SUBJECTS)
    return (template
            .replace("{subject}", s)
            .replace("{person}", p)
            .replace("{thing}", t)
            .replace("{intensifier}", _pick(INTENSIFIERS))
            .replace("{softener}", _pick(SOFTENERS)))


def generate_local(trajectory_type: str, count: int) -> List[List[str]]:
    """
    Generate `count` conversations for the given trajectory type.
    Each conversation is a list of message strings.
    """
    pool = TRAJECTORY_POOLS.get(trajectory_type)
    if not pool:
        raise ValueError(f"No local templates for trajectory '{trajectory_type}'")

    results = []
    for _ in range(count):
        template_msgs = _pick(pool)
        filled = [_fill(m) for m in template_msgs]
        results.append(filled)
    return results


def generate_all(count_per_trajectory: int = 15) -> dict:
    """Generate conversations for all trajectory types that have local templates."""
    results = {}
    for t_type, pool in TRAJECTORY_POOLS.items():
        convs = generate_local(t_type, count_per_trajectory)
        results[t_type] = convs
        print(f"  {t_type:<28} {len(convs)} conversations generated")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate conversations locally (no API key needed)")
    parser.add_argument("--count",  type=int, default=15, help="Conversations per trajectory")
    parser.add_argument("--type",   default=None, help="Single trajectory type to generate")
    parser.add_argument("--output", default=str(RAW_DIR), help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    print(f"Generating conversations locally → {out_dir}/")

    if args.type:
        targets = {args.type: generate_local(args.type, args.count)}
    else:
        targets = generate_all(args.count)

    total = 0
    for t_type, convs in targets.items():
        # Figure out starting index (don't overwrite existing files)
        existing = sorted(out_dir.glob(f"{t_type}_*.json"))
        start    = len(existing)
        for i, messages in enumerate(convs):
            path = out_dir / f"{t_type}_{start + i:03d}.json"
            path.write_text(json.dumps({"trajectory_type": t_type, "messages": messages}, indent=2))
            total += 1

    print(f"\nDone — {total} conversation files written to {out_dir}/")


if __name__ == "__main__":
    main()
