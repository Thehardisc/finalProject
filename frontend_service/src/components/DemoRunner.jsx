import React, { useState, useRef } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001';

const SCENARIOS = {
  escalating_conflict: {
    label: 'Escalating Conflict',
    emoji: '🔥',
    color: '255,80,80',
    messages: [
      { from: 'bob',   text: "Did you move my stuff again? I've told you not to do that." },
      { from: 'alice', text: "I just tidied up a little, it's not a big deal." },
      { from: 'bob',   text: "It IS a big deal. I need things where I put them." },
      { from: 'alice', text: "You're being ridiculous. I was trying to help." },
      { from: 'bob',   text: "You never ask. You just do whatever you want." },
      { from: 'alice', text: "I can't believe you're this upset over nothing." },
      { from: 'bob',   text: "It's not nothing to me. Why don't you ever listen?" },
      { from: 'alice', text: "Fine. I won't touch anything ever again. Happy?" },
    ],
  },
  emotional_support: {
    label: 'Emotional Support',
    emoji: '💙',
    color: '0,150,255',
    messages: [
      { from: 'alice', text: "I've been really struggling lately. Just feeling lost." },
      { from: 'bob',   text: "I'm here. What's going on?" },
      { from: 'alice', text: "It's everything at once. Work, family... I feel overwhelmed." },
      { from: 'bob',   text: "That sounds exhausting. You don't have to carry it alone." },
      { from: 'alice', text: "I keep telling myself to be stronger but I'm just tired." },
      { from: 'bob',   text: "It's okay to be tired. You've been carrying so much." },
      { from: 'alice', text: "Thank you for saying that. I really needed to hear it." },
      { from: 'bob',   text: "Anytime. I'm not going anywhere." },
    ],
  },
  growing_excitement: {
    label: 'Growing Excitement',
    emoji: '🚀',
    color: '255,160,0',
    messages: [
      { from: 'alice', text: "Okay so you're not going to believe this." },
      { from: 'bob',   text: "What happened??" },
      { from: 'alice', text: "I just got the call. They want to offer me the position!" },
      { from: 'bob',   text: "NO WAY. The one in New York?!" },
      { from: 'alice', text: "YES! I've been shaking for the past hour!" },
      { from: 'bob',   text: "I knew you'd get it. I ALWAYS knew!" },
      { from: 'alice', text: "I can't stop smiling. This is everything I've worked for." },
      { from: 'bob',   text: "Let's celebrate tonight. This calls for something special!" },
    ],
  },
  de_escalation: {
    label: 'De-Escalation',
    emoji: '🕊️',
    color: '40,190,90',
    messages: [
      { from: 'alice', text: "I'm still upset about yesterday. You really hurt me." },
      { from: 'bob',   text: "I know. I've been thinking about it all day. I'm sorry." },
      { from: 'alice', text: "I just don't understand why you said that in front of everyone." },
      { from: 'bob',   text: "It was wrong. I wasn't thinking. That's on me." },
      { from: 'alice', text: "I just need you to understand how much it affected me." },
      { from: 'bob',   text: "I do. And I want to make it right, if you'll let me." },
      { from: 'alice', text: "I appreciate that you're taking responsibility." },
      { from: 'bob',   text: "I care about this too much to let it go unfixed." },
    ],
  },
  gradual_sadness: {
    label: 'Gradual Sadness',
    emoji: '🌧️',
    color: '120,120,200',
    messages: [
      { from: 'alice', text: "I've been thinking about how things used to be." },
      { from: 'bob',   text: "Yeah? What brought that on?" },
      { from: 'alice', text: "I don't know. I miss how simple everything felt." },
      { from: 'bob',   text: "Things do change. It's hard sometimes." },
      { from: 'alice', text: "I feel like I lost a version of myself I can't get back." },
      { from: 'bob',   text: "That kind of grief is real, even when it's hard to explain." },
      { from: 'alice', text: "I cried earlier and I don't even know exactly why." },
      { from: 'bob',   text: "Sometimes you don't need a reason. Let it out." },
    ],
  },
  volatile_mix: {
    label: 'Volatile Mix',
    emoji: '⚡',
    color: '160,0,255',
    messages: [
      { from: 'bob',   text: "I've been looking forward to this all week." },
      { from: 'alice', text: "Same! I really missed hanging out like this." },
      { from: 'bob',   text: "Wait... did you seriously not tell me about the change of plans?" },
      { from: 'alice', text: "I sent a message! Check your phone." },
      { from: 'bob',   text: "I never got it. I've wasted the whole afternoon." },
      { from: 'alice', text: "I'm sorry, I genuinely thought I sent it. I feel terrible." },
      { from: 'bob',   text: "It's fine... I just get frustrated when this happens." },
      { from: 'alice', text: "Let's just reset. I really do want to spend time with you." },
    ],
  },
};

const MSG_DELAY_MS = 3200;

export default function DemoRunner({ currentUser, socketRef, onDemoStart }) {
  const [open, setOpen]       = useState(false);
  const [selected, setSelected] = useState(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const abortRef = useRef(false);

  const delay = ms => new Promise(r => setTimeout(r, ms));

  const startDemo = async () => {
    const scenario = SCENARIOS[selected];
    if (!scenario || !currentUser || !socketRef?.current) return;

    setRunning(true);
    setOpen(false);
    abortRef.current = false;
    setProgress({ current: 0, total: scenario.messages.length });

    try {
      // Get Bob's user_id without touching Alice's session cookie
      const bobRes = await axios.post(
        `${API_BASE}/auth/demo-login/1`, {},
        { withCredentials: false }
      );
      const bobId = bobRes.data.user_id;

      // Create or reuse direct conversation between Alice and Bob
      const convRes = await axios.post(`${API_BASE}/conversations`, {
        user_id:        currentUser.user_id,
        target_user_id: bobId,
      });
      const convId = convRes.data.conversation_id;

      // Switch UI to this conversation and clear messages
      onDemoStart(convId);
      await delay(600);

      // Send each message through the live pipeline
      for (let i = 0; i < scenario.messages.length; i++) {
        if (abortRef.current) break;
        const { from, text } = scenario.messages[i];
        const senderId = from === 'alice' ? currentUser.user_id : bobId;

        socketRef.current.send(JSON.stringify({
          text,
          sender_id:       senderId,
          conversation_id: convId,
        }));

        setProgress({ current: i + 1, total: scenario.messages.length });
        await delay(MSG_DELAY_MS);
      }
    } catch (err) {
      console.error('[DemoRunner] error:', err);
    } finally {
      setRunning(false);
      setProgress({ current: 0, total: 0 });
    }
  };

  const stopDemo = () => {
    abortRef.current = true;
    setRunning(false);
    setProgress({ current: 0, total: 0 });
  };

  const sc = SCENARIOS[selected];

  return (
    <>
      {/* ── Trigger button ─────────────────────────────────────────────────── */}
      <button
        onClick={running ? stopDemo : () => setOpen(true)}
        title={running ? 'Stop demo' : 'Run a live demo conversation through the ML pipeline'}
        style={{
          display: 'flex', alignItems: 'center', gap: 5,
          padding: '5px 12px', borderRadius: 8, cursor: 'pointer',
          background: running ? 'rgba(255,80,80,0.10)' : 'rgba(0,190,90,0.10)',
          border: `1px solid ${running ? 'rgba(255,80,80,0.30)' : 'rgba(0,190,90,0.30)'}`,
          color: running ? '#e05050' : '#18a860',
          fontSize: '0.72rem', fontWeight: 600, transition: 'all 0.15s',
        }}
        onMouseOver={e => { if (!running) { e.currentTarget.style.background = 'rgba(0,190,90,0.18)'; }}}
        onMouseOut={e =>  { if (!running) { e.currentTarget.style.background = 'rgba(0,190,90,0.10)'; }}}
      >
        {running
          ? `⏹ ${progress.current}/${progress.total}`
          : '▶ Live Demo'}
      </button>

      {/* ── Scenario picker modal ───────────────────────────────────────────── */}
      {open && (
        <div
          onClick={e => e.target === e.currentTarget && setOpen(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 9999,
            background: 'rgba(10,10,20,0.60)', backdropFilter: 'blur(10px)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          <div style={{
            background: 'rgba(28,28,46,0.95)',
            border: '1.5px solid rgba(255,255,255,0.12)',
            borderRadius: 22, padding: '28px 28px 24px', width: 520, maxWidth: '92vw',
            boxShadow: '0 24px 60px rgba(0,0,0,0.5)',
          }}>
            <h2 style={{ margin: '0 0 4px', color: '#fff', fontSize: '1.1rem', fontWeight: 700 }}>
              Live Demo
            </h2>
            <p style={{ margin: '0 0 20px', color: 'rgba(255,255,255,0.45)', fontSize: '0.80rem' }}>
              Messages are sent through the real ML pipeline — analysis updates live.
            </p>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 22 }}>
              {Object.entries(SCENARIOS).map(([key, s]) => (
                <button
                  key={key}
                  onClick={() => setSelected(key)}
                  style={{
                    padding: '12px 14px', borderRadius: 12, textAlign: 'left', cursor: 'pointer',
                    background: selected === key ? `rgba(${s.color},0.18)` : 'rgba(255,255,255,0.04)',
                    border: `1.5px solid ${selected === key ? `rgba(${s.color},0.55)` : 'rgba(255,255,255,0.09)'}`,
                    transition: 'all 0.14s',
                  }}
                >
                  <div style={{ fontSize: '1.25rem', marginBottom: 3 }}>{s.emoji}</div>
                  <div style={{ color: '#fff', fontSize: '0.80rem', fontWeight: 600 }}>{s.label}</div>
                  <div style={{ color: 'rgba(255,255,255,0.38)', fontSize: '0.72rem', marginTop: 2 }}>
                    {s.messages.length} messages · ~{Math.round(s.messages.length * MSG_DELAY_MS / 1000)}s
                  </div>
                </button>
              ))}
            </div>

            <div style={{ display: 'flex', gap: 8 }}>
              <button
                onClick={() => setOpen(false)}
                style={{
                  flex: 1, padding: '10px 0', borderRadius: 10, cursor: 'pointer',
                  border: '1.5px solid rgba(255,255,255,0.12)',
                  background: 'transparent', color: 'rgba(255,255,255,0.55)', fontSize: '0.84rem',
                }}
              >
                Cancel
              </button>
              <button
                onClick={startDemo}
                disabled={!selected}
                style={{
                  flex: 2, padding: '10px 0', borderRadius: 10, cursor: selected ? 'pointer' : 'not-allowed',
                  border: 'none',
                  background: selected ? `rgba(${sc?.color},0.80)` : 'rgba(255,255,255,0.08)',
                  color: selected ? '#fff' : 'rgba(255,255,255,0.25)',
                  fontWeight: 700, fontSize: '0.88rem', transition: 'all 0.15s',
                }}
              >
                ▶ Start
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
