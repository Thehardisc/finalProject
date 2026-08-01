#!/usr/bin/env python3
"""Replay the audited miss list through the live stack and diff verdicts.

Usage: python3 replay_misses.py <tag>
The <tag> namespaces conversation IDs so runs don't share context
(e.g. "rulebased-baseline", "retrained-v3").
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else f"run{int(time.time())}"
API = os.environ.get("INGESTION_URL", "http://localhost:8000") + "/messages"
KEY = subprocess.run(
    ["grep", "^INTERNAL_API_KEY", "/Users/stevenfurman/Documents/Projects-Final/.env"],
    capture_output=True, text=True).stdout.strip().split("=", 1)[1]

CASES = [
    ("It's everything at once. Work, family... I feel overwhelmed.", "neutral", {"sadness", "nervousness", "fear", "grief"}),
    ("I keep telling myself to be stronger but I'm just tired.", "neutral", {"sadness", "disappointment", "grief"}),
    ("Are we even talking, You never freaking listen :(", "neutral", {"anger", "annoyance", "disappointment", "sadness"}),
    ("That sounds exhausting. You don't have to carry it alone.", "neutral", {"caring", "sadness"}),
    ("Sometimes you don't need a reason. Let it out.", "neutral", {"caring", "neutral", "approval"}),
    ("Anytime. I'm not going anywhere.", "disapproval", {"caring", "love", "neutral", "approval"}),
    ("❤️", "sadness", {"love"}),
    ("Hey, how's the project going?", "surprise", {"curiosity", "neutral"}),
    ("oh great, exactly what I needed today \U0001f644", "surprise", {"annoyance", "disappointment", "disapproval", "anger", "sadness"}),
    ("Same! I really missed hanging out like this.", "sadness", {"joy", "love", "excitement", "caring"}),
    ("I sent a message! Check your phone.", "grief", {"neutral", "annoyance", "surprise"}),
    ("Let's take a breath.", "grief", {"neutral", "caring", "relief"}),
    ("i love you motherfucker", "anger", {"love", "amusement", "joy"}),
    ("yo i just checked and the weather's perfect this weekend, we HAVE to do that road trip we talked about", "nervousness", {"excitement", "joy", "optimism", "desire"}),
    ("YO. I just booked flights to Paris for us, I'm literally shaking rn \U0001f1eb\U0001f1f7✈️ tell me you're free those dates", "fear", {"excitement", "joy", "surprise"}),
    ("prepare to be personally victimized by how good this place looks, not even exaggerating \U0001f3e1✨", "fear", {"excitement", "amusement", "admiration", "joy"}),
    ("YESSS I love you SO much rn \U0001f62d okay we're going to Paris, this is happening", "disapproval", {"love", "excitement", "joy", "gratitude"}),
    ("honestly can't argue with that, food's the only part of vacation hype that never disappoints", "disappointment", {"approval", "joy", "optimism", "admiration", "amusement"}),
    ("OMG IT'S FINALLY HAPPENING, Colombia tomorrow!! \U0001f1e8\U0001f1f4✈️ I literally cannot sleep I'm so hyped, are you packed", "surprise", {"excitement", "joy", "surprise"}),
    ("Noice now its working", "neutral", {"joy", "approval", "relief", "admiration", "excitement"}),
    ("Okay then its going not bad at all", "surprise", {"approval", "optimism", "joy", "relief", "neutral"}),
    ("Honestly I'm furious about how this was handled.", "anger", {"anger", "annoyance"}),
    ("I am honestly terrified about the surgery tomorrow", "fear", {"fear", "nervousness"}),
    ("Thank you for saying that. I really needed to hear it.", "gratitude", {"gratitude"}),
    ("i love you", "love", {"love"}),
    ("This is amazing", "admiration", {"admiration", "excitement", "joy"}),
    ("holy shit", "surprise", {"surprise"}),
    ("I'm here. What's going on?", "curiosity", {"curiosity", "caring", "neutral"}),
    ("I am happy!", "joy", {"joy", "excitement"}),
]

ids = {}
for i, (text, _, _) in enumerate(CASES):
    payload = json.dumps({
        "message_id": "x", "conversation_id": f"replay-{TAG}-{i}",
        "user_id": "qa-replay", "text": text, "timestamp": time.time(),
    }).encode()
    req = urllib.request.Request(API, data=payload, headers={
        "Content-Type": "application/json", "X-API-Key": KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        ids[i] = json.load(r)["message_id"]
    time.sleep(0.4)

print(f"sent {len(ids)} messages, waiting for analysis...", file=sys.stderr)

def fetch(msg_ids):
    q = ",".join(f"'{m}'" for m in msg_ids)
    sql = ("SELECT m.message_id, a.pipeline_log_json::json->>'dominant_selected', "
           "a.pipeline_log_json::json->>'meta_confidence', "
           "a.pipeline_log_json::json->>'decision_mode' "
           f"FROM messages m JOIN emotion_analysis a ON m.message_id=a.message_id WHERE m.message_id IN ({q})")
    out = subprocess.run(["docker", "exec", "projects-final-db-1", "psql", "-U", "user",
                          "-d", "emotion_db", "-tAc", sql], capture_output=True, text=True).stdout
    res = {}
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) >= 4:
            res[parts[0]] = (parts[1], float(parts[2] or 0), parts[3])
    return res

results = {}
for _ in range(45):
    results = fetch(list(ids.values()))
    if len(results) >= len(ids):
        break
    time.sleep(2)

flipped = kept_ok = still_wrong = regressed = 0
rows = []
for i, (text, old, ok) in enumerate(CASES):
    got = results.get(ids[i])
    new, conf, mode = got if got else ("<no result>", 0.0, "?")
    was_ok = old in ok
    now_ok = new in ok
    if was_ok and now_ok: verdict = "OK (kept)";      kept_ok += 1
    elif was_ok:          verdict = "REGRESSED";      regressed += 1
    elif now_ok:          verdict = "FIXED";          flipped += 1
    else:                 verdict = "still wrong";    still_wrong += 1
    rows.append((verdict, old, new, conf, mode, text[:58]))

for v, old, new, conf, mode, t in rows:
    print(f"{v:12s} {old:14s} -> {new:14s} {conf:.2f} [{mode[:4]}] | {t}")
print(f"\nTAG={TAG}  fixed={flipped}  still_wrong={still_wrong}  controls_kept={kept_ok}  REGRESSED={regressed}")
