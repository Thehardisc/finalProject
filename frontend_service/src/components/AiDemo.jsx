import { useState, useRef, useEffect } from 'react';
import { adminAPI } from '../api/client';

// Admin-only launcher: two Claude agents improvise a conversation between two
// randomly generated users; messages run through the real emotion pipeline.
export default function AiDemo({ onDemoStart }) {
  const [open, setOpen]         = useState(false);
  const [topic, setTopic]       = useState('');
  const [count, setCount]       = useState(10);
  const [running, setRunning]   = useState(false);
  const [progress, setProgress] = useState({ current: 0, total: 0 });
  const [error, setError]       = useState('');
  const pollRef = useRef(null);

  useEffect(() => () => clearInterval(pollRef.current), []);

  const finish = () => {
    clearInterval(pollRef.current);
    pollRef.current = null;
    setRunning(false);
    setProgress({ current: 0, total: 0 });
  };

  const start = async () => {
    const t = topic.trim();
    if (!t || running) return;
    setError('');
    setRunning(true);
    setOpen(false);
    setProgress({ current: 0, total: count });
    try {
      const res = await adminAPI.startAiDemo(t, count);
      onDemoStart(res.data.conversation_id);
      pollRef.current = setInterval(async () => {
        try {
          const s = (await adminAPI.aiDemoStatus()).data;
          setProgress({ current: s.sent, total: s.total });
          if (s.status === 'error') setError(s.error || 'Run failed.');
          if (['done', 'error', 'stopped', 'idle'].includes(s.status)) finish();
        } catch { finish(); }
      }, 3000);
    } catch (err) {
      console.error('[AiDemo] start failed:', err);
      setError(err.apiMessage || 'Failed to start AI demo.');
      finish();
      setOpen(true);
    }
  };

  const stop = () => {
    adminAPI.stopAiDemo().catch(() => {});
    finish();
  };

  const canStart = topic.trim().length >= 2 && !running;

  return (
    <>
      <button
        type="button"
        onClick={running ? stop : () => setOpen(true)}
        title={running ? 'Stop AI demo' : 'Run an AI-generated demo conversation'}
        style={{
          display: 'flex', alignItems: 'center', gap: 5,
          padding: '5px 12px', borderRadius: 8, cursor: 'pointer',
          background: running ? 'rgba(239,68,68,0.08)' : 'rgba(109,40,217,0.08)',
          border: `1px solid ${running ? 'rgba(239,68,68,0.30)' : 'rgba(109,40,217,0.25)'}`,
          color: running ? '#ef4444' : '#7c3aed',
          fontSize: '0.72rem', fontWeight: 600, transition: 'all 0.15s',
        }}
      >
        {running ? `⏹ ${progress.current}/${progress.total}` : '✨ AI Demo'}
      </button>

      {open && (
        <div
          onClick={e => e.target === e.currentTarget && setOpen(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 300,
            background: 'rgba(0,0,0,0.60)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          <div style={{
            background: 'var(--ig-surf)', border: '1px solid var(--ig-bord)',
            borderRadius: 20, padding: '24px 24px 20px', width: 420, maxWidth: '92vw',
            boxShadow: '0 12px 40px rgba(0,0,0,.14)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <div style={{ width: 32, height: 32, borderRadius: 9, background: 'rgba(109,40,217,0.08)', border: '1px solid rgba(109,40,217,0.18)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.95rem' }}>✨</div>
              <span style={{ fontWeight: 700, fontSize: '1rem', color: 'var(--ig-txt)' }}>AI Demo</span>
            </div>
            <p style={{ margin: '0 0 16px', color: 'var(--ig-txt2)', fontSize: '0.78rem', lineHeight: 1.5 }}>
              Two Claude agents improvise a conversation between two randomly generated
              users. Every message runs through the live emotion pipeline.
            </p>

            <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 600, color: 'var(--ig-txt2)', marginBottom: 4 }}>Topic</label>
            <input
              value={topic}
              onChange={e => setTopic(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && canStart && start()}
              placeholder="e.g. planning a road trip"
              maxLength={80}
              autoFocus
              style={{
                width: '100%', boxSizing: 'border-box', padding: '9px 12px', marginBottom: 12,
                borderRadius: 10, border: '1px solid var(--ig-bord)',
                background: 'var(--ig-surf2)', color: 'var(--ig-txt)', fontSize: '0.84rem', outline: 'none',
              }}
            />

            <label style={{ display: 'block', fontSize: '0.72rem', fontWeight: 600, color: 'var(--ig-txt2)', marginBottom: 4 }}>
              Messages ({count})
            </label>
            <input
              type="range" min={4} max={30} value={count}
              onChange={e => setCount(+e.target.value)}
              style={{ width: '100%', marginBottom: 14, accentColor: '#7c3aed' }}
            />

            {error && (
              <div style={{ color: '#ef4444', fontSize: '0.74rem', marginBottom: 10 }}>{error}</div>
            )}

            <div style={{ display: 'flex', gap: 8 }}>
              <button
                type="button"
                onClick={() => setOpen(false)}
                style={{
                  flex: 1, padding: '10px 0', borderRadius: 10, cursor: 'pointer',
                  border: '1px solid var(--ig-bord)', background: 'transparent',
                  color: 'var(--ig-txt2)', fontSize: '0.84rem',
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={start}
                disabled={!canStart}
                style={{
                  flex: 2, padding: '10px 0', borderRadius: 10, border: 'none',
                  cursor: canStart ? 'pointer' : 'not-allowed',
                  background: canStart ? '#6d28d9' : 'var(--ig-surf2)',
                  color: canStart ? '#fff' : 'var(--ig-txt3)',
                  fontWeight: 700, fontSize: '0.86rem', transition: 'all 0.15s',
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
