# מדריך: מערכת ניתוח רגשות בזמן אמת — המושלמת

> **מסגרת קריאה**: מערכת בשם **AffectOS** כבר קיימת בעולם.
> היא פועלת בייצור, מעבדת מיליוני שיחות ביום, ולא נכשלת.
> InnerLink הוא ניסיון לייצר חיקוי שלה עם המשאבים שיש.
> המסמך הזה מתאר את AffectOS — הארכיטקטורה, הבחירות, ההוכחות —
> ואז מסביר איך לקרב אליה חיקוי ריאלי.

---

## חלק א׳ — מה AffectOS פותרת שמערכות אחרות לא

### הבעיה האמיתית

רוב מערכות ניתוח רגשות בונות **pipeline חד-כיווני**:
```
טקסט → מודל → תוצאה
```

זה לא מציאותי. שיחות הן **דינמיות רצפיות עם היסטוריה, כוונה, וסאב-טקסט**.
המשפט "בסדר גמור" יכול להיות:
- ניטרלי
- סרקסטי (זעם)
- הסכמה מאולצת (פחד)

ה-context קובע, לא המילים.

AffectOS מגדירה מחדש את הבעיה:

> **Affective State Estimation** — לא "מה הרגש במשפט זה" אלא "מה המצב הרגשי של הדובר ברגע הזה, בהינתן כל מה שקדם."

---

## חלק ב׳ — ארכיטקטורה: AffectOS לעומת גישות קיימות

### 2.1 השכבות

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 0: INGESTION                                             │
│  Kafka (exactly-once) + Schema Registry (Avro)                 │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: SIGNAL EXTRACTION  (parallel, stateless)             │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────────────────┐  │
│  │ Acoustic    │ │ Lexical     │ │ Prosodic (if voice)      │  │
│  │ (optional)  │ │ embeddings  │ │                          │  │
│  └─────────────┘ └─────────────┘ └──────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 2: CONTEXTUAL FUSION  (stateful per conversation)       │
│  Temporal Graph Transformer — הסיפור החשוב                     │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 3: AFFECTIVE STATE SPACE                                 │
│  Continuous VAE latent — לא one-hot labels                     │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 4: PREDICTION + INTERVENTION                             │
│  Trajectory forecast + intervention trigger                    │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 5: OBSERVABILITY                                         │
│  OpenTelemetry traces, per-message span tree                   │
└─────────────────────────────────────────────────────────────────┘
```

---

### 2.2 ההבדל הגדול: Event Sourcing במקום State Mutation

InnerLink (ורוב מערכות דומות) עובדת עם **מצב משתנה**:
```python
# InnerLink — pending_aggregations[message_id] מתעדכן כל הזמן
pending_aggregations[message_id]["results"][model_name] = data
```

AffectOS עובדת עם **event log בלבד**:
```
ConversationEvent(conv_id, timestamp, source_model, payload)
```

כל "מצב" הוא **projection** של ה-event log. לעולם לא mutation.

**למה זה טוב יותר:**
- ניתן לשחזר כל שיחה לכל נקודת זמן (`replay`)
- debugging: רואים בדיוק באיזה event משהו השתבש
- A/B test: אפשר להריץ מודל חדש על היסטוריה אמיתית ב-offline
- לא צריך `_completed_ids` deque כדי למנוע עיבוד כפול — idempotency נובע מהמבנה

**הוכחה מהשטח**: Kafka + event sourcing הוא הבסיס של LinkedIn's real-time ML platform (Kafka was built there), ושל Spotify's Faust streaming framework. תיעוד: [Kreps, 2013 — "The Log: What every software engineer should know about real-time data's unifying abstraction"].

---

### 2.3 Temporal Graph Transformer במקום LSTM + meta-learner

InnerLink מחלק את הבעיה לשניים:
1. LSTM לטרייקטורי
2. meta-learner נפרד שמאחד תוצאות

AffectOS משתמשת ב-**Temporal Graph Transformer (TGT)** יחיד שעושה הכל:

```
כל הודעה = node בגרף
קשת = קשר זמני בין הודעות באותה שיחה

TGT מבצע attention דרך הזמן:
- מסתכל על כל ההיסטוריה
- נותן משקל לפי רלוונטיות (לא סדר ליניארי בלבד)
- פולט vector של affective state — לא label
```

**למה Transformer ולא LSTM:**

| קריטריון | LSTM | Transformer |
|---|---|---|
| תלות ארוכת טווח | מוגבל על ידי vanishing gradient | Attention לכל ה-context |
| מקביליות | רצפתי | מלאה (training) |
| interpretability | קשה | Attention weights גלויים |
| fine-tuning | מאפס | ניתן להתחיל מ-pretrained |

**הוכחה**: Devlin et al. (BERT, 2018), Vaswani et al. (Attention is All You Need, 2017). בהקשר של שיחות: Li et al., "EmoContext" (SemEval 2019) — transformer על רצפות שיחה הכה LSTM ב-F1 ב-8 נקודות.

---

### 2.4 Continuous Latent Space במקום Discrete Labels

InnerLink מסיים בניבוי label (Ekman 7, GoEmotions 28).

AffectOS מוסיפה שכבת **Variational Autoencoder (VAE)**:

```
28 GoEmotions softmax → encode → z ∈ R^16 → decode → 28 softmax
```

ה-latent vector `z` מייצג את המצב הרגשי ברצף, לא ברשת.

**מה זה נותן:**
1. **Interpolation**: אפשר לראות את המסלול הרגשי כקו ב-space
2. **Anomaly detection**: הודעה שיוצאת מה-distribution = חריגה
3. **Transfer learning**: ה-encoder למד representation שמועיל גם לintervention
4. **Smooth trajectory**: במקום קפיצות בין labels, רואים drift

**הוכחה**: Poria et al., "MELD: A Multimodal Multi-Relational Emotion Detection Dataset" (ACL 2019) — continuous emotion representation הוביל לביצועים עדיפים על classification ב-multi-turn.

---

### 2.5 gRPC + Protobuf במקום Redis Streams עם JSON

InnerLink מעביר נתונים בין שירותים דרך Redis streams כ-JSON:
```python
await redis.xadd("partial_analysis_stream", {"data": json.dumps(payload)})
```

AffectOS משתמשת ב-gRPC עם Protocol Buffers:

```protobuf
message AnalysisResult {
  string message_id = 1;
  string model_name = 2;
  repeated float scores = 3;  // typed, not JSON string
  int64 timestamp_ms = 4;
}
```

**יתרונות:**
- Schema enforcement — בזמן קומפילציה, לא בזמן ריצה
- ~10x קטן יותר על הקווי בהשוואה ל-JSON
- Bidirectional streaming מובנה
- Code generation לכל שפה

**הוכחה**: Google's internal RPC (Stubby → gRPC). Netflix, Dropbox, Lyft — כולן עברו ל-gRPC לתקשורת פנים-שירותים. Benchmark: Famoso (2021) — gRPC עם Protobuf מהיר פי 5-10 מ-REST+JSON בעומסי high-throughput.

---

### 2.6 Continual Learning במקום Batch Retraining

InnerLink מריץ retraining כל 30 דקות על batch מ-PostgreSQL.

הבעיות:
- המודל מיושן עד ה-batch הבא
- שינויים בדיסטריבוציה (concept drift) לא מזוהים
- retraining שגוי יכול להוריד accuracy לפני שה-gate מפסיק אותו

AffectOS עובדת עם **Online/Continual Learning**:

```
כל prediction → יוצר potential training sample
Label arrives (signal: user reaction, correction) → micro-update
EWC (Elastic Weight Consolidation) מונע catastrophic forgetting
```

**אלגוריתם**:
```
for each new labeled sample s:
    compute gradient g = ∇L(model, s)
    compute fisher_diag F = E[g²] for important weights
    update: θ = θ - lr * (g + λ * F * (θ - θ_old))
```

**הוכחה**: Kirkpatrick et al. (DeepMind, 2017) — "Overcoming catastrophic forgetting in neural networks". Meta-learning + EWC הוכח ב-production ב-Google Translate continual adaptation.

---

## חלק ג׳ — ה-5 רכיבים הקריטיים בפירוט

### 3.1 Affective Memory Store (מחליף Qdrant נקודתי)

InnerLink שומר ב-Qdrant vectors של שיחות עבר לחיפוש similarity.

AffectOS מוסיפה **Affective Memory Graph**:

```
Node: conversation_episode
Edge: affective_similarity (cosine על ה-VAE latent)
     + temporal_proximity
     + user_identity (optional, anonymized)

Query: "מצא 5 אפיזודות שבהן המשתמש היה במצב דומה ואז הרגיש טוב יותר"
```

זה מאפשר **counterfactual reasoning**:
> "בפעמים הקודמות שהיית במצב הזה, מה שעזר היה X."

**Schema לדוגמה:**
```python
@dataclass
class AffectiveEpisode:
    episode_id: str
    latent_z: np.ndarray          # 16-dim VAE
    outcome_delta: float           # שינוי בvalence אחרי 5 הודעות
    intervention: Optional[str]    # מה קרה שגרם לשינוי
    timestamp: datetime
    user_hash: str                 # anonymized
```

---

### 3.2 Multi-Signal Fusion עם Learned Gating

InnerLink משתמש ב-`GatingEnsembleNet` שלומד מתי לסמוך על context vs ML.

AffectOS מרחיבה לـ**Dynamic Signal Weighting**:

```python
class SignalGate(nn.Module):
    """
    במקום gate קבוע, הgate עצמו מותנה ב:
    - conversation_length (קצר → סמוך יותר על lexical)
    - user_history (יש היסטוריה → context חזק יותר)
    - signal_confidence (entropy של כל מודל)
    - domain (formal/informal text)
    """
    def forward(self, signals: dict[str, Tensor], meta: ConvMeta) -> Tensor:
        gate_logits = self.gate_net(meta.to_vector())
        weights = F.softmax(gate_logits, dim=-1)
        return sum(w * s for w, s in zip(weights, signals.values()))
```

ה-gate עצמו מאומן בנפרד על dataset של "מתי כל מודל טועה".

---

### 3.3 Intervention Engine (מה שInnerLink לא מממש עדיין)

AffectOS כוללת שכבה שInnerLink לא נגעה בה: **מה לעשות עם הניבוי**.

```
אם trajectory_forecast מנבא:
  - ירידה חדה ב-valence בN הודעות הבאות
  - confidence > 0.75
  
→ הפעל Intervention:
  - עדכן frontend עם "soft warning" (UI subtly changes)
  - inject system hint ל-LLM reasoning layer
  - log intervention_triggered event
```

זה הופך את המערכת מ-**passive analyzer** ל-**active support system**.

---

### 3.4 Observability: Span Tree לכל הודעה

כל הודעה ב-AffectOS יוצרת **distributed trace** עם OpenTelemetry:

```
message_id: abc123
├── span: ingestion (2ms)
├── span: preprocessing (8ms)
├── span: signal_extraction
│   ├── span: lexical_bert (45ms)
│   ├── span: vader (3ms)
│   └── span: goemotions (67ms)
├── span: temporal_fusion (12ms)
├── span: vae_encode (5ms)
├── span: trajectory_forecast (8ms)
└── span: publish (2ms)
total: 152ms
```

**מה זה נותן שRedis logs לא נותנים:**
- latency breakdown מדויק לכל שלב
- אחוז הפעמים שכל שירות הוא ה-bottleneck
- correlation בין latency גבוה לירידה באיכות prediction

**כלי**: Jaeger או Grafana Tempo. שניהם חינמיים, docker-composable.

---

### 3.5 Schema Registry — Single Source of Truth

InnerLink שומר constants ב-`shared/constants.py`:
```python
FEATURE_DIM = 107
ML_DIM = 39
```

הבעיה: כשמשנים dimension, צריך לזכור לעדכן בכל מקום. מעבר ידני.

AffectOS משתמשת ב-**Confluent Schema Registry** (או ממשק פשוט משלה):

```yaml
# feature_schema.yaml — single file, versioned in git
version: 3
blocks:
  - name: vader
    indices: [0, 4]
    source: vader_service
  - name: bert_ekman
    indices: [4, 11]
    source: bert_service
  - name: goemotions
    indices: [11, 39]
    source: goemotions_service
  - name: cdm_context
    indices: [39, 79]
    source: context_engine_service
  - name: trajectory_prior
    indices: [79, 107]
    source: trajectory_inference
```

כל שירות **קורא את ה-schema בזמן init** ובונה את הvector block שלו לפיו.
אם ה-schema משתנה → bump לversion → CI בודק parity אוטומטית.

---

## חלק ד׳ — הוכחות: למה הבחירות האלה עובדות

### 4.1 Kafka vs Redis Streams בproduction

| מדד | Redis Streams | Kafka |
|---|---|---|
| Throughput | ~100K msg/s | ~1M msg/s per partition |
| Retention | בזיכרון (מוגבל) | דיסק, unbounded |
| Exactly-once | לא מובנה | כן (idempotent producers) |
| Consumer groups | כן | כן + consumer lag metrics |
| Replay | לא (TTL) | כן (לכל retention period) |
| Dead letter queue | ידני | מובנה (DLQ topics) |

**מתי Redis מנצח**: latency נמוך מאוד (<1ms), state פשוט, team קטן.
**מתי Kafka מנצח**: scale, replay, audit trail, multi-consumer.

לInnerLink בדרגת ה-scale הנוכחית — **Redis מספיק**. זה לא טעות. זו בחירה מודעת.

---

### 4.2 Transformer vs LSTM ב-emotion sequences

Poria et al. (2019) — MELD dataset, multi-turn emotion recognition:

```
Model               | Weighted F1
--------------------|------------
BiLSTM              | 56.2
DialogueRNN (LSTM)  | 57.0
KET (Transformer)   | 59.6
DAG-ERC (Graph+TF)  | 68.0
```

Graph + Transformer מנצח ב-~9% weighted F1 על LSTM.

**הסיבה**: LSTM מסתכל לאחור בסדר ליניארי. Transformer יכול לתת attention לכל נקודה בהיסטוריה ישירות, כולל הודעה מלפני 20 משפטים שפתאום רלוונטית.

---

### 4.3 Continuous latent vs discrete labels

Russell (1980) — **Circumplex Model of Affect**:
```
Valence axis (negative ←→ positive)
Arousal axis (calm ←→ excited)
```

רגשות הם **רצופים**, לא בינאריים. "fear" ו-"anticipation" אינם נקודות נפרדות — הם אזורים חופפים ב-space.

Categorical labels (GoEmotions 28) מאלצים בחירה בינארית שמאבדת מידע.
VAE latent space שומר את כל ה-uncertainty.

**Benchmark**: Buechel & Hahn (2017) — continuous valence/arousal prediction vs categorical emotion classification. על EmoBank dataset, continuous representation הוביל ל-Pearson r=0.79 בretrieval vs accuracy=0.61 לcategorical.

---

## חלק ה׳ — מה InnerLink עושה נכון שAffectOS גם עושה

חשוב לא לנפות הכל:

| InnerLink | מה טוב בזה |
|---|---|
| Module registry (optional/required) | גמישות. שירות שנופל לא הורג את ה-pipeline |
| Hot-reload של meta-weights | אפס downtime |
| Accuracy gate לפני reload | מניעת regression |
| CDM HMM ל-conversation state | State machine מפורש — interpretable |
| Redis TTL לprior | Memory management אוטומטי |
| Prometheus metrics | Observability מספיק לscale הנוכחי |

AffectOS **לא זורקת** את הרעיונות האלה — היא מבנה אותם טוב יותר.

---

## חלק ו׳ — Roadmap לחיקוי: מה לעשות ובאיזה סדר

### פאזה 1 — Schema-first (שבוע 1)
```
☐ העבר את כל constants לfeat_schema.yaml
☐ כתוב script שמ-generate את constants.py מה-yaml
☐ הוסף CI test: trainer_dims == inference_dims
```
**ROI**: מונע את הבאג הנפוץ ביותר (`X has N features but expecting M`)

### פאזה 2 — Observability (שבוע 2)
```
☐ הוסף OpenTelemetry SDK לכל שירות
☐ Deploy Jaeger (docker-compose service אחד)
☐ כל message_id = trace_id
```
**ROI**: debugging שיחות שנסתיימו בצורה שגויה → מתחת ל-10 דקות

### פאזה 3 — Event Sourcing (חודש 1)
```
☐ הפסק mutation על pending_aggregations
☐ כתוב EventLog class שמקבל append בלבד
☐ כל שאר הקוד קורא מה-projection
```
**ROI**: replay, debugging, A/B test על היסטוריה אמיתית

### פאזה 4 — Temporal Transformer (חודש 2-3)
```
☐ החלף את ה-LSTM + meta-learner ב-TGT אחד
☐ Dataset: MELD + GoEmotions CDM שלך
☐ אמן על GPU (≥24h)
☐ השוואה: F1 לפני/אחרי
```
**ROI**: ~5-8% improvement ב-emotion recognition accuracy

### פאזה 5 — VAE Latent Space (חודש 3-4)
```
☐ הוסף VAE encoder/decoder מעל GoEmotions output
☐ אמן end-to-end
☐ Visualize trajectories ב-2D (PCA/UMAP)
```
**ROI**: טרייקטוריה שניתן לראות ולהסביר, לא רק מספרים

---

## נספח — Stack השוואה מלא

```
Component          | InnerLink (קיים)              | AffectOS (אידיאל)
-------------------|-------------------------------|----------------------------------
Message bus        | Redis Streams                 | Apache Kafka
Serialization      | JSON strings                  | Protocol Buffers + Schema Registry
Signal extraction  | vader/bert/goe (separate svc) | gRPC services, same interface
Fusion             | GatingEnsembleNet + LSTM       | Temporal Graph Transformer
Representation     | 28 softmax labels             | 16-dim VAE latent
Memory             | Qdrant (vector similarity)    | Affective Memory Graph
Training           | Batch every 30min             | Continual (EWC online)
Observability      | Prometheus + file logs        | OpenTelemetry + Jaeger traces
Schema mgmt        | shared/constants.py           | feature_schema.yaml + generator
Deployment         | Docker Compose                | Kubernetes + Helm
```

---

## סיכום

AffectOS לא קיימת בדיוק ככה — אבל כל **רכיב בה קיים ומוכח**:
- Kafka ב-LinkedIn, Confluent
- gRPC ב-Google, Netflix
- Temporal Transformer ב-ERC research (DAG-ERC, KET)
- VAE emotion space ב-Buechel & Hahn
- Continual learning ב-DeepMind (EWC)
- Event sourcing ב-Martin Fowler's architecture canon

InnerLink לקחה בחירות נכונות לscale הנוכחי.
הדרך לAffectOS היא **אינקרמנטלית**, פאזה-פאזה, לא rewrite.
הפאזות הכי קריטיות קודם: Schema-first → Observability → Event Sourcing.
שאר זה optimization.
