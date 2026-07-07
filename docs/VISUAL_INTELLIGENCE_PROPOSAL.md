# InnerLink — Visual Emotional Intelligence Layer
## Product Development Proposal v2.0 — Synthesized
### Engineering Initiative | June 2026
### Revised after Academic Peer Review: Cognitive Load Theory, Calm Technology, Emotion Regulation

---

## מבוא: למה הגרסה הראשונה הייתה חלקית

הגרסה הראשונה של ההצעה הכילה פיצ'רים נכונים לצד פיצ'רים שנבנו על הנחת יסוד שגויה: שמשתמש בזמן שיחה רגשית מסוגל לקרוא מדדים, גרפים ומספרים. המחקר הקוגניטיבי מפריך זאת.

**שלושה כשלים שנפסלו:**
- **Cognitive Load** (Sweller 1988): Gauges, Scatter Plots ו-Health Scores שואבים Working Memory בדיוק ברגע שהמשתמש זקוק לו לוויסות רגשי
- **Observer Effect** (Boehner et al. 2007): ציון "22 At Risk" הורס את מה שהוא מנסה לשמור — המדידה יוצרת את הבעיה
- **Reactance** (Brehm 1966 + Weiser 1995): Pop-up שאומר "קח הפסקה" בזמן ריב מגביר תוקפנות, לא מפחית אותה

**ועוד שניים שנוספו בסינתזה:**
- **Morphing Typography**: שינוי משקל פונט לפי emotional state מפרש מחדש את מילות המשתמש — בעיה אתית ובעיית נגישות (dyslexia)
- **Spatial Shrink**: בועות מתכווצות של הצד הנסוג עלולות להתפרש כ"ניצחון" של התוקף, הפוך מהכוונה

**המסקנה:** הפרדיגמה הנכונה היא **Contextual UI** — לא "Zero-UI" ולא "Dashboard-UI". סמוי לגמרי בזמן ריב, גלוי לגמרי כשהמשתמש מבקש עומק ויש לו את ה-bandwidth לעבד אותו.

---

## 1. החזון — לא השתנה

> "People don't have a superpower to understand what the other person is really feeling.
> We can build that."

כל שיחה מכילה שתי שכבות — מה שנאמר ומה שמורגש. בני אדם מנהלים את השיחה דרך השכבה הראשונה, אבל הנזק והריפוי מתרחשים בשנייה. InnerLink גושרת בין השתיים — לא אחרי השיחה, אלא בזמן שהיא מתרחשת. ה-HOW השתנה; ה-WHY נשאר.

---

## 2. הבעיה שאנחנו פותרים — לא השתנה

**חמישה כשלי תקשורת ללא פתרון קיים:**

1. **עיוורון דינמי** — אנשים לא רואים שהם עצמם יצרו את ה-tension
2. **פער Surface/Implicit** — "בסדר" = WITHDRAWAL + frustration. הצד השני מאמין לטקסט
3. **אין עצירה בנקודה הנכונה** — שיחות ממשיכות אחרי ה-peak כי אין תמונת מאקרו
4. **contagion עיוור** — טון מבחוץ נכנס לשיחה ואף אחד לא רואה אותו
5. **אסימטריה לא מזוהה** — אחד ב-arousal גבוה, שני ב-withdrawal, שניהם מדברים בקצבים שונים

---

## 3. הנכס הטכנולוגי — לא השתנה

```
┌─────────────────────────────────────────────────────────────────────┐
│  VADER sentiment        [0:4]    — 4 dims  — lexical baseline       │
│  BERT Ekman emotions    [4:11]   — 7 dims  — basic emotion set      │
│  GoEmotions            [11:39]  — 28 dims  — fine-grained taxonomy  │
│  VAD lexicon           [39:42]  — 3 dims   — valence/arousal/domin. │
│  CDM context vector    [42:82]  — 40 dims  — conversation dynamics  │
│  Trajectory prior      [82:110] — 28 dims  — predicted next state   │
│  Sarcasm score         [110]    — 1 dim    — implicit/explicit gap  │
│  Emotional dynamics    [111:113]— 2 dims   — inertia + contagion    │
│  Appraisal             [113:116]— 3 dims   — novelty/goals/coping   │
└─────────────────────────────────────────────────────────────────────┘
```

כל הנתונים כבר יורדים ב-WebSocket payload. אין צורך בשינוי backend לשכבות 1-2.

---

## 4. ארכיטקטורת השכבות — הלב של הגרסה החדשה

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Real-time Ambient          (ALWAYS ON, zero attention)   │
│  Living Aura + Semantic Breathing                                   │
│  Edge Accent per message                                            │
│  Inertia Trail                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — Real-time Friction         (INVISIBLE, triggered)        │
│  Micro-delay on Send (200-300ms, covert)                            │
│  No text, no popup, no explanation                                  │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 3 — Post-Mortem Dashboard      (ON DEMAND, after conv ends)  │
│  Constellation Map                                                  │
│  Narrative Arc replay                                               │
│  Full numeric breakdown                                             │
│  Conversation Health Score                                          │
└─────────────────────────────────────────────────────────────────────┘
```

**הכלל:** מה שדורש קריאה ✗ בזמן אמת. מה שנקלט בראייה היקפית ✓ בזמן אמת.

---

## 5. Layer 1 — Real-time Ambient Features

---

### פיצ'ר 1: Living Aura + Semantic Breathing

**מה זה:**
הרקע של חלון השיחה משנה צבע וקצב פעימה לפי CDM state + valence. שתי ממדים — לא אחד.

**ממד א' — צבע (CDM state):**
```javascript
const AURA_COLORS = {
  NEUTRAL:        'rgba(100,100,120, 0.04)',
  WARMTH:         'rgba(251,191,36,  0.06)',
  EMPATHY:        'rgba(52,211,153,  0.05)',
  HUMOR:          'rgba(251,191,36,  0.08)',
  CURIOSITY:      'rgba(96,165,250,  0.05)',
  AGREEMENT:      'rgba(52,211,153,  0.07)',
  TENSION:        'rgba(239,68,68,   0.07)',
  FRUSTRATION:    'rgba(239,68,68,   0.08)',
  CONFLICT:       'rgba(220,38,38,   0.10)',
  ARGUMENT:       'rgba(220,38,38,   0.12)',
  WITHDRAWAL:     'desaturate(40%)',           // CSS filter, not color
  RECONCILIATION: 'rgba(139,92,246,  0.06)',
};
```

**ממד ב' — קצב פעימה (Semantic Breathing):**
```javascript
// arousal → breathing rate → animation duration
// מסתנכרן עם מערכת העצבים הפאראסימפתטית
const breathingDuration = (arousal) => {
  if (arousal > 0.7) return '1.2s';   // distress — נשימה מהירה
  if (arousal > 0.4) return '2.0s';   // elevated
  return '4.0s';                       // calm — 15 נשימות/דקה, כמו במנוחה
};
```

**contagion → מהירות מעבר צבע:**
```javascript
const transitionSpeed = (contagion) =>
  contagion > 0.6 ? '400ms' : contagion < 0.2 ? '2000ms' : '900ms';
```

**מה המשתמש חווה:** הגוף קולט לפני השכל. שיחה ב-CONFLICT מרגישה שונה פיזית משיחה ב-WARMTH — בלי לקרוא מילה.

**מורכבות:** נמוכה. 2-3 ימים.

---

### פיצ'ר 2: Edge Accent — חתימת הודעה

**מה זה:**
במקום Weather Badge, כל בועת הודעה מקבלת פס צבע עדין (3-4px) בשולי הבועה. הצבע = dominant emotion מה-EmotionPalette. אין אייקון, אין טקסט.

```jsx
<div
  className="message-bubble"
  style={{
    borderLeft: `3px solid rgba(${EmotionPalette[dom]}, ${confidence * 0.6})`,
    // opacity מדרגת לפי confidence — אי-ודאות = עדין יותר
  }}
/>
```

**סרקזם > 0.5:** הפס הופך ל-dashed. ללא הסבר — המשתמש שמעיין ישאל, זה יפתח שיחה.

**מה זה לא:** לא Weather Badge, לא אמוג'י, לא טקסט. רק צבע בשוליים.

**מורכבות:** יום אחד.

---

### פיצ'ר 3: Inertia Trail — עקבות רגשיות

**מה זה:**
כש-inertia גבוהה (> 0.5), הודעות קודמות בצבע הרגש הנוכחי מקבלות glow עמום שדועך אחורה. המשתמש רואה ויזואלית שהרגש "תקוע".

```javascript
const TRAIL_LOOKBACK = Math.round(inertia * 10); // inertia 0.8 → 8 הודעות
const trailOpacity = (i) => Math.max(0, (inertia - i * 0.1) * 0.15);
```

**מה זה לא:** מספר, ציון, מילה. רק צבע שדועך.

**מורכבות:** נמוכה. 3-4 ימים.

---

### פיצ'ר 4: Narrative Phase Ambient

**מה זה (גרסה מעודכנת של Narrative Arc):**
לא פס אופקי עם "אתה ב-Escalation" — זה דורש קריאה. במקום, שלב הנרטיב מתבטא ברמת ה-opacity של ה-Living Aura.

```javascript
const PHASE_INTENSITY = {
  opening:       0.04,   // עדין מאוד
  escalation:    0.08,   // מתחזק
  peak:          0.14,   // חזק
  turning_point: 0.10,   // מתרכך
  resolution:    0.06,
  sustained:     0.05,
};
// opacity של הצבע הרלוונטי גדל לפי השלב
```

הרקע עצמו "מספר" היכן השיחה נמצאת — בלי טקסט.

**מורכבות:** שעות. זה תוספת לפיצ'ר 1.

---

## 6. Layer 2 — Real-time Friction (Invisible)

---

### פיצ'ר 5: Covert Micro-Delay on Send

**מה זה:**
כשהמערכת מזהה הודעה הרסנית פוטנציאלית, כפתור השליחה מגיב — אבל בהשהייה שאינה גלויה למשתמש. לא Long Press (שנייה שלמה), לא animation מפורש, לא הסבר.

```javascript
const FRICTION_DELAY_MS = 280; // בלתי מורגש, אבל קוטע amygdala hijack

const shouldApplyFriction = (data) => (
  data.sarcasm_score > 0.75 &&
  ['TENSION','CONFLICT','ARGUMENT','FRUSTRATION'].includes(data.cdm_state)
) || (
  data.dynamics?.inertia > 0.80 &&
  data.vad?.valence < -0.5
);

// בפרונטאנד:
const handleSend = async () => {
  if (shouldApplyFriction(currentAnalysis?.data)) {
    await new Promise(r => setTimeout(r, FRICTION_DELAY_MS));
  }
  onSend();
};
```

**הבסיס המדעי:** 280ms מספיק להפעלת System 2 (Kahneman) בלי שהמשתמש מרגיש שמשהו מונע ממנו. זה לא Long Press — זה latency שנראה כרגיל.

**מה זה לא:** Pop-up, alert, הסבר, הנחייה. שום דבר לא נאמר.

**מורכבות:** שעתיים. שורות בודדות.

---

## 7. Layer 3 — Post-Mortem Dashboard (On Demand)

כל הפיצ'רים שנפסלו מזמן אמת — חיים כאן. המשתמש שחוזר על שיחה יום לאחר מכן, עם רוחב פס מנטלי מלא, יכול לראות הכל.

---

### פיצ'ר 6: Emotional Constellation Map

**בזמן אמת:** לא זמין.
**Post-Mortem:** scatter plot מלא — Valence × Arousal, כל הודעה כנקודה, קו נסיעה, hover להודעה המקורית.

```jsx
// מסך "Conversation Review" — נפרד לגמרי מהצ'אט
<ConstellationMap messages={completedConversation.messages} />
```

**מה המשתמש רואה:** "הנסיעה" הרגשית של השיחה כולה כנרטיב ויזואלי. "ראיתי שנפלתי ל-High Arousal Negative ב-הודעה 7."

**מורכבות:** בינונית. Recharts ScatterChart. 7 ימים.

---

### פיצ'ר 7: Conversation Health Score

**בזמן אמת:** לא זמין — Observer Effect.
**Post-Mortem:** ציון 0-100 עם breakdown מלא.

```python
def conversation_health_retrospective(messages):
    valence_avg      = mean([m.vad.valence for m in messages])
    coping_avg       = mean([m.appraisal.coping for m in messages])
    goal_align_avg   = mean([m.appraisal.goal_congruence for m in messages])
    conflict_ratio   = sum(1 for m in messages
                          if m.cdm_state in CONFLICT_STATES) / len(messages)
    inertia_negative = mean([max(0, -m.dynamics.inertia) for m in messages])

    health = (
        (valence_avg + 1) / 2 * 0.30 +
        coping_avg          * 0.25 +
        (goal_align_avg + 1) / 2 * 0.25 -
        conflict_ratio      * 0.15 -
        inertia_negative    * 0.05
    )
    return max(0, min(100, health * 100))
```

**Post-Mortem תצוגה:**
```
Conversation Health: 67/100

  Valence average:    +0.34  ✓
  Coping capacity:     0.61  ✓
  Goal alignment:     -0.22  ~
  Conflict ratio:      18%   ~
  Negative inertia:    0.31  ✗
```

---

### פיצ'ר 8: Narrative Arc Replay

**בזמן אמת:** ambient בלבד (פיצ'ר 4).
**Post-Mortem:** ציר זמן אינטראקטיבי עם שלבי השיחה, CDM state, ונקודות מפנה.

```
Opening ──── Escalation ──●── Peak ──── Turning Point ──── Resolution
                          ↑
                    "הודעה 7 — כניסה לקונפליקט"
                    [לחץ לראות ניתוח מלא]
```

---

### פיצ'ר 9: Pressure Dashboard (Post-Mortem בלבד)

**בזמן אמת:** לא זמין.
**Post-Mortem:** 4 gauges (Tension, Momentum, Spread, Coping) עבור ממוצעי השיחה כולה, לא per-message.

---

## 8. מה לא לבנות — ומדוע

| פיצ'ר | סיבת פסילה |
|--------|------------|
| Health Meter בזמן אמת (מספר) | Observer Effect — המדידה יוצרת את הבעיה |
| Pop-up Intervention Alerts | Reactance — "תרגע" מעצבן יותר ממה שמרגיע |
| Morphing Typography | מפרש מחדש את מילות המשתמש (אתיקה) + Dyslexia |
| Long Press (שנייה שלמה) | גלוי מדי — מגביר תסכול בשיחות לגיטימיות |
| Spatial Shrink (בועות מתכווצות) | עלול להתפרש כ"ניצחון" התוקף — הפוך מהכוונה |
| Tarot Symbols (זמן אמת) | Cultural noise — "The Moon" לא ברור ללא הקשר |
| Shared numeric map בזמן אמת | מספרי הצד השני יכולים לשמש כנשק |

---

## 9. Roadmap מעודכן

```
שבוע 1      │ Edge Accent + Covert Micro-Delay
             │ → שתי שורות קוד כל אחד. עובד מהיום הראשון.
             │
שבוע 1-2    │ Living Aura + Semantic Breathing
             │ → ambient layer מלא. המשתמש מרגיש את השיחה.
             │
שבוע 2-3    │ Inertia Trail + Narrative Phase Ambient
             │ → per-message + phase — עדיין אפס cognitive load.
             │
שבוע 3-5    │ Post-Mortem screen — Constellation Map
             │ → נרטיב השיחה אחרי סיומה.
             │
שבוע 5-7    │ Post-Mortem — Narrative Arc Replay + Health Score
             │ → מסך Review מלא עם כל הנתונים הגולמיים.
             │
שבוע 7-10   │ Post-Mortem — Pressure Dashboard
             │ → ממוצעי שיחה כollה, לא real-time.
```

---

## 10. מטריקת עדיפות מעודכנת

| פיצ'ר | Layer | Impact | Effort | עדיפות |
|--------|-------|--------|--------|--------|
| Edge Accent | Real-time | בינוני | נמוך מאוד | **1** |
| Covert Micro-Delay | Real-time | גבוה מאוד | נמוך מאוד | **1** |
| Living Aura + Breathing | Real-time | גבוה | נמוך | **2** |
| Inertia Trail | Real-time | בינוני | נמוך | **3** |
| Narrative Phase Ambient | Real-time | בינוני | נמוך מאוד | **3** |
| Constellation Map | Post-Mortem | גבוה | בינוני | **4** |
| Health Score + Breakdown | Post-Mortem | גבוה | נמוך | **4** |
| Narrative Arc Replay | Post-Mortem | גבוה | בינוני | **5** |
| Pressure Dashboard | Post-Mortem | בינוני | בינוני | **6** |

---

## 11. חוקות הממשק הסופיות

**חוק 1 — אפס מספרים בזמן אמת.**
אין מד, אין אחוז, אין ציון. הנתונים עוברים דרך הצבע, הקצב, הצבעוניות.

**חוק 2 — אין פקודות.**
המערכת לא אומרת למשתמש מה לעשות. היא משנה את המרחב כדי שהוא ירצה לעשות את הדבר הנכון.

**חוק 3 — חיכוך סמוי בלבד.**
ה-Micro-Delay (280ms) הוא כלי ההתערבות החזק ביותר במערכת. הוא קוטע Amygdala Hijack בלי שהמשתמש יודע שמשהו קרה.

**חוק 4 — Post-Mortem = full access.**
לאחר סיום השיחה, כל הנתונים הגולמיים זמינים. Power users, מטפלים, חוקרים — מקבלים את כל 116 הממדים. Agency מלאה.

**חוק 5 — טיפוגרפיה קדושה.**
לעולם לא לשנות משקל פונט, גודל או צורת טקסט לפי ניתוח רגשי. הטקסט שייך למשתמש — לא למערכת.

---

## 12. מדדי הצלחה

### Layer 1 — Ambient
- Living Aura transition frame rate > 55fps
- Zero jank — אנימציית breathing לא משפיעה על render time של messages

### Layer 2 — Friction
- מדידה: שיחות שהגיעו ל-CONFLICT עם Micro-Delay פעיל — האם rate of escalation יורד?
- A/B test: 50% users עם friction, 50% בלי. מדד: mean inertia negative לאחר peak

### Layer 3 — Post-Mortem
- Engagement: משתמשים פותחים Conversation Review > 30% מהשיחות המשמעותיות
- Dwell time: > 45 שניות בממוצע במסך Review

### איכותי (הכי חשוב)
- "הבנתי מה קרה לנו" — לא "ראיתי גרפים יפים"

---

## 13. עקרון המנחה — Contextual UI

> "Invisible when you need space to feel.
> Explicit when you're ready to understand."

זה לא Zero-UI ולא Dashboard-UI. זה מערכת שמכירה את ההקשר:
- **בזמן ריב** — כלום לא נאמר, הכל מורגש
- **ביום למחרת** — הכל גלוי, המשתמש בוחר מה לראות

זה הפרדיגמה שלא קיימת בשום כלי תקשורת היום.

---

*InnerLink Visual Intelligence Proposal | v2.0 — Post Academic Review*
*June 2026 | Synthesized from: Sweller 1988, Boehner et al. 2007, Weiser & Brown 1995,*
*Brehm 1966, Kahneman 2011, Kuppens 2010, Gross 1998*
