import React, { useState, useEffect, useRef } from 'react';

const DEMO_MESSAGES = [
  { text: "Sure, totally fine with that 🙄", emotion: 'annoyance',  tokens: [{ w: 'Sure',   role: 'hedge' }, { w: ', totally fine with that 🙄', role: 'default' }] },
  { text: "I feel so happy today!", emotion: 'joy',       tokens: [{ w: 'feel', role: 'emotion-pos' }, { w: ' so ', role: 'intensifier' }, { w: 'happy', role: 'emotion-pos' }] },
  { text: "This is absolutely amazing work!", emotion: 'admiration', tokens: [{ w: 'absolutely', role: 'absolute' }, { w: ' amazing', role: 'emotion-pos' }] },
  { text: "I'm not sure I agree with that…", emotion: 'disapproval', tokens: [{ w: "not", role: 'negation' }, { w: ' sure', role: 'hedge' }, { w: ' agree', role: 'agreement' }] },
  { text: "Can you help me with this? I'm really struggling.", emotion: 'nervousness', tokens: [{ w: 'Can you', role: 'request' }, { w: ' really', role: 'intensifier' }, { w: ' struggling', role: 'emotion-neg' }] },
  { text: "That was genuinely unexpected and kind of shocking.", emotion: 'surprise', tokens: [{ w: 'genuinely', role: 'absolute' }, { w: ' unexpected', role: 'emotion-neg' }, { w: ' kind of', role: 'hedge' }, { w: ' shocking', role: 'emotion-neg' }] },
];

const EMOTION_COLORS = {
  joy: '#facc15', admiration: '#f472b6', annoyance: '#fb923c',
  disapproval: '#f87171', nervousness: '#a78bfa', surprise: '#60a5fa',
  neutral: '#94a3b8',
};

const TOKEN_COLORS = {
  'emotion-pos': '#4ade80', 'emotion-neg': '#f87171', 'intensifier': '#fbbf24',
  'negation': '#f87171', 'hedge': '#94a3b8', 'request': '#60a5fa',
  'agreement': '#34d399', 'absolute': '#c084fc', 'default': 'rgba(255,255,255,0.82)',
};

// SVG icons for features — no emoji
const FEATURE_ICONS = {
  emotion: (c) => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={`rgb(${c})`} strokeWidth="1.6" strokeLinecap="round">
      <circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>
    </svg>
  ),
  trajectory: (c) => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={`rgb(${c})`} strokeWidth="1.6" strokeLinecap="round">
      <polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>
    </svg>
  ),
  cdm: (c) => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={`rgb(${c})`} strokeWidth="1.6" strokeLinecap="round">
      <circle cx="12" cy="5" r="2"/><circle cx="5" cy="19" r="2"/><circle cx="19" cy="19" r="2"/>
      <line x1="12" y1="7" x2="5" y2="17"/><line x1="12" y1="7" x2="19" y2="17"/><line x1="7" y1="19" x2="17" y2="19"/>
    </svg>
  ),
  pipeline: (c) => (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke={`rgb(${c})`} strokeWidth="1.6" strokeLinecap="round">
      <rect x="2" y="7" width="6" height="10" rx="1"/><rect x="9" y="9" width="6" height="6" rx="1"/>
      <rect x="16" y="5" width="6" height="14" rx="1"/>
    </svg>
  ),
};

const FEATURES = [
  {
    iconKey: 'emotion',
    title: 'Deep Emotion Intelligence',
    desc: '28-class GoEmotions model + 7-class Ekman BERT + VADER lexicon fused by a trained meta-learner into a single confidence score.',
    stat: '28 emotion classes',
    color: '167,139,250',
  },
  {
    iconKey: 'trajectory',
    title: 'Trajectory Prediction',
    desc: 'An LSTM Emotional Dialogue Encoder reads your conversation history and predicts the next emotional state before you even reply.',
    stat: '12-message window',
    color: '96,165,250',
  },
  {
    iconKey: 'cdm',
    title: 'Conversation Dynamics',
    desc: 'A 15-state CDM Hidden Markov Machine tracks Tension, Reconciliation, Empathy, Conflict, Humor and 10 more conversation phases in real time.',
    stat: '15 dynamic states',
    color: '74,222,128',
  },
  {
    iconKey: 'pipeline',
    title: 'Sub-second Pipeline',
    desc: 'VADER, BERT, GoEmotions and the Context Engine run in parallel on every message. Results arrive before you finish reading.',
    stat: '< 1s latency',
    color: '251,191,36',
  },
];

const STEPS = [
  { n: '01', title: 'Write naturally', desc: 'Send messages just like any chat. No special commands. No setup.' },
  { n: '02', title: 'AI pipeline activates', desc: 'Four ML models analyze in parallel: sentiment, emotion, context, sarcasm.' },
  { n: '03', title: 'See what words carry', desc: 'Every word gets highlighted by emotional role. Trajectory shows where you\'re headed.' },
];

const STATS = [
  { value: '116', label: 'feature dimensions' },
  { value: '28',  label: 'emotion classes' },
  { value: '15',  label: 'conversation states' },
  { value: '< 1s',label: 'analysis latency' },
];

const INJECTED_CSS = `
@keyframes lp-orb1 {
  0%,100% { transform: translate(0,0) scale(1); }
  33% { transform: translate(80px,-60px) scale(1.10); }
  66% { transform: translate(-50px,70px) scale(0.92); }
}
@keyframes lp-orb2 {
  0%,100% { transform: translate(0,0) scale(1); }
  40% { transform: translate(-70px,60px) scale(1.06); }
  75% { transform: translate(60px,-40px) scale(0.95); }
}
@keyframes lp-orb3 {
  0%,100% { transform: translate(0,0) scale(1); }
  50% { transform: translate(40px,50px) scale(1.08); }
}
@keyframes lp-fadeUp {
  from { opacity:0; transform:translateY(28px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes lp-msgIn {
  from { opacity:0; transform:translateX(-12px); }
  to   { opacity:1; transform:translateX(0); }
}
@keyframes lp-pulse {
  0%,100% { opacity:1; }
  50% { opacity:.45; }
}
@keyframes lp-float {
  0%,100% { transform:translateY(0); }
  50% { transform:translateY(-8px); }
}
@keyframes lp-spin { to { transform:rotate(360deg); } }
@keyframes lp-glow {
  0%,100% { box-shadow: 0 0 20px rgba(109,40,217,0.30); }
  50% { box-shadow: 0 0 44px rgba(109,40,217,0.65); }
}
@keyframes lp-scan {
  0% { transform: translateY(-100%); opacity:0; }
  10% { opacity:1; }
  90% { opacity:1; }
  100% { transform: translateY(100%); opacity:0; }
}
.lp-feature-card:hover {
  transform: translateY(-4px) !important;
  border-color: rgba(139,92,246,0.35) !important;
  box-shadow: 0 20px 60px rgba(0,0,0,0.45), 0 0 0 1px rgba(139,92,246,0.20) !important;
}
.lp-nav-cta:hover {
  background: rgba(139,92,246,0.25) !important;
  border-color: rgba(139,92,246,0.60) !important;
}
.lp-hero-btn:hover {
  box-shadow: 0 10px 50px rgba(109,40,217,0.75), inset 0 1px 0 rgba(255,255,255,0.12) !important;
  transform: translateY(-2px) !important;
}
.lp-ghost-btn:hover {
  background: rgba(255,255,255,0.08) !important;
  border-color: rgba(255,255,255,0.28) !important;
}
`;

function DemoChat() {
  const [visible, setVisible] = useState([]);
  const idx = useRef(0);

  useEffect(() => {
    const tick = () => {
      setVisible(prev => {
        const msg = DEMO_MESSAGES[idx.current % DEMO_MESSAGES.length];
        idx.current++;
        const next = [...prev, { ...msg, key: Date.now() }];
        return next.slice(-4);
      });
    };
    tick();
    const id = setInterval(tick, 2200);
    return () => clearInterval(id);
  }, []);

  return (
    <div style={{
      background: 'rgba(255,255,255,0.04)', backdropFilter: 'blur(20px)',
      WebkitBackdropFilter: 'blur(20px)',
      border: '1px solid rgba(255,255,255,0.09)',
      borderRadius: 20, padding: '20px 18px', width: '100%', maxWidth: 420,
      boxShadow: '0 24px 60px rgba(0,0,0,0.50)',
      animation: 'lp-float 6s ease-in-out infinite',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16, paddingBottom: 12, borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#4ade80', animation: 'lp-pulse 2s ease infinite' }} />
        <span style={{ fontSize: '0.70rem', fontWeight: 700, color: 'rgba(255,255,255,0.45)', letterSpacing: '.10em', textTransform: 'uppercase' }}>Live Analysis</span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 5 }}>
          {['rgba(239,68,68,.6)', 'rgba(251,191,36,.6)', 'rgba(74,222,128,.6)'].map((c, i) => (
            <div key={i} style={{ width: 9, height: 9, borderRadius: '50%', background: c }} />
          ))}
        </div>
      </div>

      {/* Messages */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minHeight: 140 }}>
        {visible.map((msg) => {
          const emoColor = EMOTION_COLORS[msg.emotion] || EMOTION_COLORS.neutral;
          return (
            <div key={msg.key} style={{ animation: 'lp-msgIn .35s ease both', display: 'flex', flexDirection: 'column', gap: 4 }}>
              <div style={{
                alignSelf: Math.random() > 0.5 ? 'flex-end' : 'flex-start',
                background: `rgba(${msg.emotion === 'joy' ? '250,204,21' : msg.emotion === 'admiration' ? '167,139,250' : '255,255,255'},.08)`,
                border: `1px solid ${emoColor}30`,
                borderRadius: 14, padding: '9px 13px',
                fontSize: '0.83rem', lineHeight: 1.5,
                maxWidth: '88%',
              }}>
                {msg.tokens.map((t, i) => (
                  <span key={i} style={{ color: TOKEN_COLORS[t.role] || 'rgba(255,255,255,0.82)', fontWeight: t.role !== 'default' ? 600 : 400 }}>
                    {i === 0 ? t.w : (t.w.startsWith(' ') ? t.w : ' ' + t.w)}
                  </span>
                ))}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 5, paddingLeft: 4 }}>
                <div style={{ width: 6, height: 6, borderRadius: '50%', background: emoColor }} />
                <span style={{ fontSize: '0.62rem', color: emoColor, fontWeight: 600, letterSpacing: '.04em' }}>{msg.emotion}</span>
                <span style={{ fontSize: '0.60rem', color: 'rgba(255,255,255,0.25)', marginLeft: 4 }}>real-time</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Mini stat strip */}
      <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.07)', display: 'flex', gap: 12 }}>
        {[['VADER', '#facc15'], ['BERT', '#60a5fa'], ['GoE', '#4ade80']].map(([label, color]) => (
          <div key={label} style={{ flex: 1, textAlign: 'center' }}>
            <div style={{ fontSize: '0.58rem', color: 'rgba(255,255,255,0.30)', letterSpacing: '.08em', textTransform: 'uppercase', marginBottom: 3 }}>{label}</div>
            <div style={{ height: 3, borderRadius: 2, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${40 + Math.random() * 55}%`, background: color, borderRadius: 2, transition: 'width .8s ease' }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function LandingPage({ onSignIn }) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    document.body.style.background = '#05060F';
    document.documentElement.style.background = '#05060F';
    const onScroll = () => setScrolled(window.scrollY > 40);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <div style={{ minHeight: '100vh', background: '#05060F', color: '#fff', fontFamily: '"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif', overflowX: 'hidden' }}>
      <style>{INJECTED_CSS}</style>

      {/* ── Background orbs ───────────────────────────────────────────────── */}
      <div style={{ position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 0 }}>
        <div style={{ position: 'absolute', width: 800, height: 800, borderRadius: '50%', top: -300, left: -250, background: 'radial-gradient(circle, rgba(79,40,210,0.42) 0%, transparent 70%)', filter: 'blur(100px)', animation: 'lp-orb1 22s ease-in-out infinite' }} />
        <div style={{ position: 'absolute', width: 700, height: 700, borderRadius: '50%', bottom: -200, right: -200, background: 'radial-gradient(circle, rgba(46,16,130,0.55) 0%, transparent 70%)', filter: 'blur(90px)', animation: 'lp-orb2 28s ease-in-out infinite' }} />
        <div style={{ position: 'absolute', width: 500, height: 500, borderRadius: '50%', top: '40%', right: '20%', background: 'radial-gradient(circle, rgba(109,40,217,0.18) 0%, transparent 70%)', filter: 'blur(80px)', animation: 'lp-orb3 18s ease-in-out infinite' }} />
      </div>

      {/* ── Navbar ────────────────────────────────────────────────────────── */}
      <nav style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
        padding: '0 48px',
        height: 64,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: scrolled ? 'rgba(5,6,15,0.85)' : 'transparent',
        backdropFilter: scrolled ? 'blur(20px)' : 'none',
        WebkitBackdropFilter: scrolled ? 'blur(20px)' : 'none',
        borderBottom: scrolled ? '1px solid rgba(255,255,255,0.07)' : 'none',
        transition: 'all 0.3s ease',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 30, height: 30, borderRadius: 9, background: 'linear-gradient(135deg,#7c3aed,#4338ca)', boxShadow: '0 4px 14px rgba(109,40,217,0.45)' }} />
          <span style={{ fontWeight: 800, fontSize: '1.05rem', letterSpacing: '-0.3px' }}>InnerLink</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            className="lp-nav-cta"
            onClick={onSignIn}
            style={{ padding: '8px 18px', background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.14)', borderRadius: 10, color: 'rgba(255,255,255,0.82)', fontSize: '0.85rem', fontWeight: 600, cursor: 'pointer', transition: 'all 0.2s', fontFamily: 'inherit' }}
          >
            Sign In
          </button>
          <button
            className="lp-hero-btn"
            onClick={onSignIn}
            style={{ padding: '8px 20px', background: 'linear-gradient(135deg,#4c1d95,#6d28d9)', border: '1px solid rgba(139,92,246,0.35)', borderRadius: 10, color: '#fff', fontSize: '0.85rem', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s', boxShadow: '0 4px 20px rgba(109,40,217,0.40)', fontFamily: 'inherit' }}
          >
            Get Started
          </button>
        </div>
      </nav>

      {/* ── Hero ──────────────────────────────────────────────────────────── */}
      <section style={{ position: 'relative', zIndex: 1, minHeight: '100vh', display: 'flex', alignItems: 'center', padding: '100px 48px 80px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', width: '100%', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 80, alignItems: 'center' }}>

          {/* Left: copy */}
          <div style={{ animation: 'lp-fadeUp .7s ease both' }}>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: 8, background: 'rgba(109,40,217,0.15)', border: '1px solid rgba(139,92,246,0.30)', borderRadius: 99, padding: '5px 14px', marginBottom: 28 }}>
              <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#a78bfa', animation: 'lp-pulse 2s ease infinite' }} />
              <span style={{ fontSize: '0.72rem', fontWeight: 700, color: '#a78bfa', letterSpacing: '.06em', textTransform: 'uppercase' }}>Real-time emotion AI</span>
            </div>

            <h1 style={{ margin: '0 0 22px', fontSize: 'clamp(2.4rem, 4vw, 3.4rem)', fontWeight: 800, lineHeight: 1.12, letterSpacing: '-0.03em' }}>
              Where AI reads<br />
              <span style={{ background: 'linear-gradient(135deg,#a78bfa,#60a5fa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>between the lines.</span>
            </h1>

            <p style={{ margin: '0 0 36px', fontSize: '1.05rem', color: 'rgba(255,255,255,0.52)', lineHeight: 1.65, maxWidth: 480 }}>
              InnerLink analyzes every message in real time — detecting 28 emotion classes, predicting conversation trajectory, and mapping dynamics through a 15-state machine.
            </p>

            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
              <button
                className="lp-hero-btn"
                onClick={onSignIn}
                style={{ padding: '14px 32px', background: 'linear-gradient(135deg,#4c1d95,#6d28d9)', border: '1px solid rgba(139,92,246,0.35)', borderRadius: 14, color: '#fff', fontSize: '1rem', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s', boxShadow: '0 6px 28px rgba(109,40,217,0.48)', fontFamily: 'inherit', animation: 'lp-glow 3s ease infinite' }}
              >
                Start analyzing →
              </button>
              <button
                className="lp-ghost-btn"
                onClick={onSignIn}
                style={{ padding: '14px 28px', background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.14)', borderRadius: 14, color: 'rgba(255,255,255,0.70)', fontSize: '1rem', fontWeight: 600, cursor: 'pointer', transition: 'all 0.2s', fontFamily: 'inherit' }}
              >
                Try demo account
              </button>
            </div>

            {/* Social proof strip */}
            <div style={{ marginTop: 44, display: 'flex', gap: 28, flexWrap: 'wrap' }}>
              {STATS.map(s => (
                <div key={s.label}>
                  <div style={{ fontSize: '1.5rem', fontWeight: 800, background: 'linear-gradient(135deg,#a78bfa,#60a5fa)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>{s.value}</div>
                  <div style={{ fontSize: '0.72rem', color: 'rgba(255,255,255,0.38)', marginTop: 2, letterSpacing: '.02em' }}>{s.label}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Right: live chat preview */}
          <div style={{ display: 'flex', justifyContent: 'center', animation: 'lp-fadeUp .7s .2s ease both' }}>
            <DemoChat />
          </div>
        </div>
      </section>

      {/* ── Features ──────────────────────────────────────────────────────── */}
      <section style={{ position: 'relative', zIndex: 1, padding: '80px 48px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 56 }}>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#a78bfa', letterSpacing: '.12em', textTransform: 'uppercase', marginBottom: 14 }}>What it does</div>
            <h2 style={{ margin: 0, fontSize: 'clamp(1.8rem, 3vw, 2.4rem)', fontWeight: 800, letterSpacing: '-0.02em' }}>Four layers of intelligence</h2>
            <p style={{ marginTop: 12, color: 'rgba(255,255,255,0.42)', fontSize: '1rem' }}>Stacked ML models that work together on every single message.</p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 20 }}>
            {FEATURES.map((f, i) => (
              <div
                key={i}
                className="lp-feature-card"
                style={{
                  background: 'rgba(255,255,255,0.04)',
                  border: '1px solid rgba(255,255,255,0.08)',
                  borderRadius: 20, padding: '28px 24px',
                  transition: 'all 0.3s ease',
                  cursor: 'default',
                  animation: `lp-fadeUp .6s ${i * 0.1}s ease both`,
                }}
              >
                <div style={{ width: 40, height: 40, borderRadius: 12, background: `rgba(${f.color},0.10)`, border: `1px solid rgba(${f.color},0.20)`, display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 16 }}>
                  {FEATURE_ICONS[f.iconKey]?.(f.color)}
                </div>
                <h3 style={{ margin: '0 0 10px', fontSize: '1.0rem', fontWeight: 700, color: '#fff' }}>{f.title}</h3>
                <p style={{ margin: '0 0 18px', fontSize: '0.83rem', color: 'rgba(255,255,255,0.45)', lineHeight: 1.6 }}>{f.desc}</p>
                <div style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  background: `rgba(${f.color},0.12)`,
                  border: `1px solid rgba(${f.color},0.25)`,
                  borderRadius: 99, padding: '4px 12px',
                }}>
                  <div style={{ width: 5, height: 5, borderRadius: '50%', background: `rgb(${f.color})` }} />
                  <span style={{ fontSize: '0.68rem', fontWeight: 700, color: `rgb(${f.color})`, letterSpacing: '.04em' }}>{f.stat}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How it works ──────────────────────────────────────────────────── */}
      <section style={{ position: 'relative', zIndex: 1, padding: '80px 48px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 56 }}>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#60a5fa', letterSpacing: '.12em', textTransform: 'uppercase', marginBottom: 14 }}>How it works</div>
            <h2 style={{ margin: 0, fontSize: 'clamp(1.8rem, 3vw, 2.4rem)', fontWeight: 800, letterSpacing: '-0.02em' }}>Three steps, zero friction</h2>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 2, position: 'relative' }}>
            {STEPS.map((step, i) => (
              <div key={i} style={{ position: 'relative', padding: '32px 28px', animation: `lp-fadeUp .6s ${i * 0.15}s ease both` }}>
                {/* Connector line */}
                {i < 2 && (
                  <div style={{ position: 'absolute', top: 48, right: -1, width: 2, height: 24, background: 'linear-gradient(180deg, rgba(109,40,217,0.6), transparent)', borderRadius: 2 }} />
                )}
                <div style={{ fontSize: '0.68rem', fontWeight: 800, color: '#a78bfa', letterSpacing: '.14em', marginBottom: 18 }}>{step.n}</div>
                <div style={{ width: 44, height: 44, borderRadius: 14, background: 'rgba(109,40,217,0.15)', border: '1px solid rgba(139,92,246,0.25)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 18 }}>
                  {i === 0
                    ? <svg width="18" height="18" fill="none" stroke="rgba(167,139,250,0.80)" strokeWidth="1.8" strokeLinecap="round" viewBox="0 0 24 24"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 013 3L7 19l-4 1 1-4L16.5 3.5z"/></svg>
                    : i === 1
                    ? <svg width="18" height="18" fill="none" stroke="rgba(167,139,250,0.80)" strokeWidth="1.8" strokeLinecap="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/></svg>
                    : <svg width="18" height="18" fill="none" stroke="rgba(167,139,250,0.80)" strokeWidth="1.8" strokeLinecap="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                  }
                </div>
                <h3 style={{ margin: '0 0 10px', fontSize: '1.05rem', fontWeight: 700 }}>{step.title}</h3>
                <p style={{ margin: 0, fontSize: '0.85rem', color: 'rgba(255,255,255,0.44)', lineHeight: 1.6 }}>{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pipeline visualization ─────────────────────────────────────────── */}
      <section style={{ position: 'relative', zIndex: 1, padding: '80px 48px' }}>
        <div style={{ maxWidth: 1200, margin: '0 auto' }}>
          <div style={{ textAlign: 'center', marginBottom: 48 }}>
            <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#4ade80', letterSpacing: '.12em', textTransform: 'uppercase', marginBottom: 14 }}>Under the hood</div>
            <h2 style={{ margin: 0, fontSize: 'clamp(1.8rem, 3vw, 2.4rem)', fontWeight: 800, letterSpacing: '-0.02em' }}>The ML pipeline</h2>
          </div>

          <div style={{
            background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.08)',
            borderRadius: 24, padding: '36px 40px',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 0, flexWrap: 'wrap', justifyContent: 'center' }}>
              {[
                { label: 'Message',       color: '255,255,255', abbr: 'MSG' },
                { label: 'VADER',         color: '250,204,21',  abbr: 'VAD' },
                { label: 'BERT',          color: '96,165,250',  abbr: 'BRT' },
                { label: 'GoEmotions',    color: '167,139,250', abbr: 'GoE' },
                { label: 'Context Engine',color: '74,222,128',  abbr: 'CTX' },
                { label: 'Meta-Learner',  color: '251,113,133', abbr: 'MTA' },
                { label: 'Result',        color: '255,255,255', abbr: '✓'   },
              ].map((node, i, arr) => (
                <React.Fragment key={i}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
                    <div style={{
                      width: 52, height: 52, borderRadius: 16,
                      background: `rgba(${node.color},0.10)`,
                      border: `1px solid rgba(${node.color},0.25)`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: '0.62rem', fontWeight: 800, color: `rgba(${node.color},0.85)`, letterSpacing: '.03em',
                    }}>
                      {node.abbr}
                    </div>
                    <span style={{ fontSize: '0.65rem', fontWeight: 600, color: `rgba(${node.color},0.80)`, textAlign: 'center', maxWidth: 70 }}>{node.label}</span>
                  </div>
                  {i < arr.length - 1 && (
                    <div style={{ width: 32, height: 1, background: 'rgba(255,255,255,0.12)', margin: '0 2px 24px', flexShrink: 0 }} />
                  )}
                </React.Fragment>
              ))}
            </div>
            <div style={{ marginTop: 28, paddingTop: 20, borderTop: '1px solid rgba(255,255,255,0.07)', display: 'flex', gap: 32, justifyContent: 'center', flexWrap: 'wrap' }}>
              {[
                ['116', 'feature dimensions', '#a78bfa'],
                ['4 models', 'run in parallel', '#60a5fa'],
                ['< 1s', 'per message', '#4ade80'],
                ['Real-time', 'WebSocket push', '#fbbf24'],
              ].map(([v, l, c]) => (
                <div key={l} style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '1.0rem', fontWeight: 800, color: c }}>{v}</div>
                  <div style={{ fontSize: '0.68rem', color: 'rgba(255,255,255,0.32)', marginTop: 2 }}>{l}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA ───────────────────────────────────────────────────────────── */}
      <section style={{ position: 'relative', zIndex: 1, padding: '80px 48px 120px' }}>
        <div style={{ maxWidth: 680, margin: '0 auto', textAlign: 'center' }}>
          <div style={{
            background: 'rgba(109,40,217,0.08)', border: '1px solid rgba(139,92,246,0.20)',
            borderRadius: 28, padding: '56px 48px',
          }}>
            <div style={{ width: 48, height: 48, borderRadius: 14, background: 'rgba(109,40,217,0.20)', border: '1px solid rgba(139,92,246,0.30)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 20px' }}>
            <svg width="22" height="22" fill="none" stroke="rgba(167,139,250,0.88)" strokeWidth="1.6" strokeLinecap="round" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/></svg>
          </div>
            <h2 style={{ margin: '0 0 14px', fontSize: '2rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
              Ready to understand your conversations?
            </h2>
            <p style={{ margin: '0 0 36px', color: 'rgba(255,255,255,0.48)', fontSize: '1rem', lineHeight: 1.6 }}>
              Sign in with a demo account in one click — no registration required. Real emotion analysis, live.
            </p>
            <button
              className="lp-hero-btn"
              onClick={onSignIn}
              style={{ padding: '16px 44px', background: 'linear-gradient(135deg,#4c1d95,#6d28d9)', border: '1px solid rgba(139,92,246,0.35)', borderRadius: 14, color: '#fff', fontSize: '1.05rem', fontWeight: 700, cursor: 'pointer', transition: 'all 0.2s', boxShadow: '0 6px 28px rgba(109,40,217,0.48)', fontFamily: 'inherit' }}
            >
              Start analyzing →
            </button>
          </div>
        </div>
      </section>

      {/* ── Footer ────────────────────────────────────────────────────────── */}
      <footer style={{ position: 'relative', zIndex: 1, padding: '24px 48px', borderTop: '1px solid rgba(255,255,255,0.07)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 22, height: 22, borderRadius: 7, background: 'linear-gradient(135deg,#7c3aed,#4338ca)' }} />
          <span style={{ fontWeight: 700, fontSize: '0.85rem', color: 'rgba(255,255,255,0.60)' }}>InnerLink</span>
        </div>
        <span style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.25)' }}>
          BERT · GoEmotions · VADER · LSTM · CDM-HMM — all running locally
        </span>
      </footer>
    </div>
  );
}
