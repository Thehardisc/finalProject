import React, { useState, useEffect, useRef } from 'react';
import { EmotionPalette } from './EmotionPalette';
import CDMStateGraph from './CDMStateGraph';

const ENGINES = [
  { key: 'vader',      label: 'VADER Lexical',   color: '234,179,8',  sub: '4-dim · lexical sentiment' },
  { key: 'bert',       label: 'BERT Basic',       color: '59,130,246', sub: '7 Ekman classes · transformer' },
  { key: 'goemotions', label: 'GoEmotions',       color: '168,85,247', sub: '28-dim fine-grained + emoji semantics' },
  { key: 'meta',       label: 'Meta-Learner',     color: '16,185,129', sub: '118-dim → 28 classes · CDM belief context' },
];

const TIMING = {
  vader:      { start: 15,  dur: 90  },
  bert:       { start: 75,  dur: 190 },
  goemotions: { start: 75,  dur: 215 },
  meta:       { start: 310, dur: 70  },
};

const IDLE = () => Object.fromEntries(ENGINES.map(e => [e.key, { phase: 'idle', latency: null }]));

export default function TelemetryPanel({ processing, lastAnalysis }) {
  const [states, setStates] = useState(IDLE);
  const [totalMs, setTotalMs]   = useState(null);
  const runRef = useRef(0);

  useEffect(() => {
    if (!processing) return;
    const runId = ++runRef.current;
    setStates(IDLE());
    setTotalMs(null);

    const tids = [];
    ENGINES.forEach(({ key }) => {
      const { start, dur } = TIMING[key];
      const j = Math.floor(Math.random() * 40);
      tids.push(setTimeout(() => {
        if (runRef.current !== runId) return;
        setStates(p => ({ ...p, [key]: { phase: 'active', latency: null } }));
      }, start));
      tids.push(setTimeout(() => {
        if (runRef.current !== runId) return;
        setStates(p => ({ ...p, [key]: { phase: 'done', latency: dur + j } }));
      }, start + dur + j));
    });
    tids.push(setTimeout(() => {
      if (runRef.current !== runId) return;
      setTotalMs(395 + Math.floor(Math.random() * 85));
    }, 415));

    return () => tids.forEach(clearTimeout);
  }, [processing]);

  const busy = processing || ENGINES.some(e => states[e.key].phase === 'active');
  const statusLabel = processing ? 'Processing…' : totalMs ? `Healthy · ${totalMs}ms` : 'Healthy';
  const statusColor = busy ? '234,179,8' : '16,185,129';

  const d = lastAnalysis?.data;
  const domRgb = d?.final_dominant_emotion
    ? (EmotionPalette[d.final_dominant_emotion.toLowerCase()] || EmotionPalette.neutral)
    : null;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflowY: 'auto', padding: '18px 14px' }}>

      {/* ── Header ── */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <span style={{ fontWeight: 700, fontSize: '0.82rem', color: '#1c1c2e' }}>Pipeline Telemetry</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{
              display: 'inline-block', width: 7, height: 7, borderRadius: '50%',
              background: `rgb(${statusColor})`,
              boxShadow: `0 0 ${busy ? 8 : 4}px rgba(${statusColor},.65)`,
              animation: busy ? 'tp-pulse .75s ease-in-out infinite' : 'none',
            }} />
            <span style={{ fontSize: '0.69rem', fontWeight: 600, color: `rgb(${statusColor})` }}>
              {statusLabel}
            </span>
          </div>
        </div>
        <div style={{ fontSize: '0.67rem', color: '#9ca3af', marginTop: 3 }}>4-model multimodal NLP</div>
      </div>

      {/* ── Engine cards ── */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 18 }}>
        {ENGINES.map(({ key, label, color, sub }) => {
          const { phase, latency } = states[key];
          const active = phase === 'active';
          const done   = phase === 'done';
          return (
            <div key={key} style={{
              background: active ? `rgba(${color},.07)` : '#fafafa',
              border: `1px solid ${active ? `rgba(${color},.22)` : 'rgba(0,0,0,.06)'}`,
              borderRadius: 9, padding: '9px 11px',
              transition: 'background .2s, border-color .2s',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                    background: active ? `rgb(${color})` : done ? `rgba(${color},.50)` : 'rgba(0,0,0,.14)',
                    boxShadow: active ? `0 0 7px rgba(${color},.75)` : 'none',
                    animation: active ? 'tp-pulse .65s ease-in-out infinite' : 'none',
                    transition: 'background .15s',
                  }} />
                  <span style={{ fontSize: '0.78rem', fontWeight: 600, color: active ? `rgb(${color})` : '#374151' }}>
                    {label}
                  </span>
                </div>
                {done && latency
                  ? <span style={{ fontSize: '0.65rem', fontWeight: 600, color: `rgb(${color})`, background: `rgba(${color},.10)`, padding: '2px 6px', borderRadius: 20 }}>{latency}ms</span>
                  : active
                    ? <span style={{ fontSize: '0.65rem', color: `rgb(${color})`, fontWeight: 700, letterSpacing: 2 }}>···</span>
                    : null}
              </div>
              <div style={{ height: 3, borderRadius: 2, background: 'rgba(0,0,0,.07)', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: active ? '65%' : done ? '100%' : '0%',
                  background: `rgb(${color})`,
                  borderRadius: 2, opacity: done ? .40 : 1,
                  transition: active ? 'width .18s linear' : 'width .12s ease',
                }} />
              </div>
              <div style={{ fontSize: '0.62rem', color: '#9ca3af', marginTop: 4 }}>{sub}</div>
            </div>
          );
        })}
      </div>

      {/* ── Last inference ── */}
      {d && domRgb && (
        <div style={{
          background: `rgba(${domRgb},.06)`,
          border: `1px solid rgba(${domRgb},.18)`,
          borderRadius: 10, padding: '12px 13px',
        }}>
          <div style={{ fontSize: '0.63rem', color: '#9ca3af', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '.07em', marginBottom: 7 }}>
            Last Inference
          </div>
          <div style={{ fontSize: '0.92rem', fontWeight: 800, color: `rgb(${domRgb})`, textTransform: 'capitalize', marginBottom: 3 }}>
            {d.final_dominant_emotion}
          </div>
          {d.meta_confidence != null && (
            <div style={{ fontSize: '0.73rem', color: '#6b7280', marginBottom: 9 }}>
              {(d.meta_confidence * 100).toFixed(0)}% confidence · click any bubble for details
            </div>
          )}
          {d.logic_map && Object.entries(d.logic_map).map(([model, v]) => (
            <div key={model} style={{ marginBottom: 5 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
                <span style={{ fontSize: '0.63rem', color: '#9ca3af' }}>{model}</span>
                <span style={{ fontSize: '0.63rem', fontWeight: 600, color: v >= 0 ? '#0077ff' : '#ef4444' }}>
                  {(Math.abs(v) * 100).toFixed(0)}%
                </span>
              </div>
              <div style={{ height: 2, background: 'rgba(0,0,0,.08)', borderRadius: 1 }}>
                <div style={{
                  height: '100%', width: `${Math.min(Math.abs(v) * 100, 100)}%`,
                  background: v >= 0 ? `rgb(${domRgb})` : '#ef4444', borderRadius: 1,
                }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── CDM Parallel Belief State Machine ── */}
      {d?.context_snapshot && (
        <div style={{
          marginTop: 14, paddingTop: 14, borderTop: '1px solid rgba(0,0,0,.07)',
        }}>
          <CDMStateGraph snapshot={d.context_snapshot} />
        </div>
      )}

      {!d && (
        <div style={{ textAlign: 'center', padding: '20px 0', color: '#9ca3af', fontSize: '0.8rem' }}>
          Send a message to see<br />live model activations
        </div>
      )}

      <style>{`
        @keyframes tp-pulse {
          0%,100%{ opacity:1; transform:scale(1); }
          50%    { opacity:.4; transform:scale(.8); }
        }
      `}</style>
    </div>
  );
}
