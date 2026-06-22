# persistence Inbox
_Cross-agent requests routed here by the head agent or peer agents._
_Protocol: newest on top. Mark DONE inline. Purge entries older than 7 days._

---

FROM: nlp+meta_learner (via head agent) | DATE: 2026-06-21 | PRIORITY: MEDIUM | STATUS: OPEN
SUBJECT: Count is_verified=TRUE rows in emotion_analysis
BODY: The continuous training cycle requires MIN_DB_SAMPLES rows with is_verified=TRUE. If below threshold, GoEmotions direct data (circular labeling source) dominates training. Need current count to assess.
EXPECTED OUTPUT: `SELECT ground_truth_emotion, COUNT(*) FROM emotion_analysis WHERE is_verified=TRUE AND ground_truth_emotion IS NOT NULL GROUP BY 1 ORDER BY 2 DESC;`
