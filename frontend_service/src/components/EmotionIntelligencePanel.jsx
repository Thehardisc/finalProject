import React, { useEffect, useState } from 'react';
import { EmotionPalette } from './EmotionPalette';

const EMOTION_EMOJI = {
  joy:'😄', sadness:'😢', anger:'😠', fear:'😨', surprise:'😲',
  disgust:'🤢', neutral:'😐', excitement:'🤩', love:'❤️', caring:'🤗',
  gratitude:'🙏', amusement:'😂', admiration:'✨', annoyance:'😒',
  confusion:'😕', curiosity:'🤔', desire:'😍', disappointment:'😞',
  disapproval:'👎', embarrassment:'😳', grief:'😭', nervousness:'😬',
  optimism:'🌟', pride:'😌', realization:'💡', relief:'😅',
  remorse:'😔', approval:'👍', happiness:'😊',
};

function Sparkline({ values, color }) {
  if (values.length < 2) return null;
  const W = 200, H = 40;
  const toY = v => H - ((v + 1) / 2) * H * 0.8 - H * 0.1;
  const toX = i => (i / (values.length - 1)) * W;
  const pathD = values.map((v, i) => `${i === 0 ? 'M' : 'L'}${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' ');
  const areaD = pathD + ` L${W},${H} L0,${H} Z`;
  const last = values[values.length - 1];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none" style={{ display: 'block' }}>
      <defs>
        <linearGradient id={`sg-${color.replace(/,/g,'-')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={`rgb(${color})`} stopOpacity="0.35" />
          <stop offset="100%" stopColor={`rgb(${color})`} stopOpacity="0" />
        </linearGradient>
      </defs>
      <line x1="0" y1={H / 2} x2={W} y2={H / 2} stroke="rgba(255,255,255,0.08)" strokeWidth="1" strokeDasharray="4 3" />
      <path d={areaD} fill={`url(#sg-${color.replace(/,/g,'-')})`} />
      <path d={pathD} fill="none" stroke={`rgb(${color})`} strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx={toX(values.length - 1)} cy={toY(last)} r="4" fill={`rgb(${color})`} style={{ filter: `drop-shadow(0 0 4px rgb(${color}))` }} />
    </svg>
  );
}

export default function EmotionIntelligencePanel({ lastAnalysis, processing, partialModels = new Set(), messages = [] }) {
  const data = lastAnalysis?.data;
  const [animKey, setAnimKey] = useState(0);
  const [prevDom, setPrevDom] = useState(null);

  const dom = data?.final_dominant_emotion?.toLowerCase() || 'neutral';
  const rgb = EmotionPalette[dom] || EmotionPalette.neutral;
  const emoji = EMOTION_EMOJI[dom] || '●';
  const confidence = data?.meta_confidence ?? null;
  const valence = data?.final_valence ?? 0;
  const valencePct = Math.round(((valence + 1) / 2) * 100);

  useEffect(() => {
    if (dom !== prevDom) { setPrevDom(dom); setAnimKey(k => k + 1); }
  }, [dom]);

  const topEmos = [...(data?.bert_emotions || [])].sort((a, b) => b.score - a.score).slice(0, 6);
  const valenceHistory = messages.filter(m => m.analysis?.data?.final_valence != null).map(m => m.analysis.data.final_valence).slice(-20);
  const traj   = data?.lstm_trajectory;
  const snap   = data?.context_snapshot;
  const sarcasm = data?.sarcasm_score ?? data?.llm_sarcasm_score ?? 0;
  const shift  = data?.context_shift;
  const logicMap = data?.logic_map || {};

  const BORDER = 'rgba(255,255,255,0.07)';
  const MUTED  = '#4b5563';
  const TEXT   = '#d1d5db';

  const PIPELINE_STAGES = [
    { key: 'vader',      label: 'VADER' },
    { key: 'basic_bert', label: 'BERT' },
    { key: 'go_emotions',label: 'GoEmo' },
    { key: 'context_engine', label: 'Context' },
    { key: 'meta',       label: 'Meta' },
  ];

  if (!data && !processing) {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 14, background: '#080810', color: MUTED }}>
        <div style={{ fontSize: '3.5rem', opacity: 0.25, filter: 'grayscale(1)' }}>🧠</div>
        <div style={{ fontSize: '0.80rem', fontWeight: 600, textAlign: 'center', lineHeight: 1.6, color: '#374151' }}>
          Send a message to<br />activate emotion analysis
        </div>
      </div>
    );
  }

  if (processing && !data) {
    return (
      <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 20, background: '#080810' }}>
        <div style={{ position: 'relative', width: 60, height: 60 }}>
          <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', border: `2px solid rgba(${rgb},0.2)`, animation: 'eiSpin 2s linear infinite' }} />
          <div style={{ position: 'absolute', inset: 8, borderRadius: '50%', background: `radial-gradient(circle, rgba(${rgb},0.4), transparent)` }} />
        </div>
        <div style={{ fontSize: '0.78rem', fontWeight: 600, color: TEXT }}>Analyzing…</div>
        <div style={{ display: 'flex', gap: 6 }}>
          {PIPELINE_STAGES.map(s => (
            <div key={s.key} style={{
              padding: '3px 8px', borderRadius: 99, fontSize: '0.60rem', fontWeight: 700,
              background: partialModels.has(s.key) ? `rgba(${rgb},0.18)` : 'rgba(255,255,255,0.04)',
              color: partialModels.has(s.key) ? `rgb(${rgb})` : MUTED,
              border: `1px solid ${partialModels.has(s.key) ? `rgba(${rgb},0.35)` : 'transparent'}`,
              transition: 'all 0.3s ease',
            }}>{s.label}</div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div style={{ height: '100%', overflowY: 'auto', background: '#080810', display: 'flex', flexDirection: 'column' }}>

      {/* ── Hero ── */}
      <div key={animKey} style={{
        padding: '22px 18px 18px',
        background: `radial-gradient(ellipse at 50% -10%, rgba(${rgb},0.22) 0%, transparent 65%)`,
        borderBottom: `1px solid ${BORDER}`,
        animation: 'eiFadeIn 0.45s ease both',
        flexShrink: 0,
      }}>
        <div style={{ fontSize: '0.58rem', fontWeight: 700, color: MUTED, textTransform: 'uppercase', letterSpacing: '.12em', marginBottom: 14 }}>
          {processing ? '⚡ Analyzing…' : 'Current Emotion'}
        </div>

        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontSize: '2.4rem', lineHeight: 1, filter: `drop-shadow(0 0 8px rgba(${rgb},0.6))` }}>{emoji}</span>
            <div>
              <div style={{
                fontSize: '1.5rem', fontWeight: 900, textTransform: 'capitalize', lineHeight: 1,
                background: `linear-gradient(135deg, rgb(${rgb}) 0%, rgba(${rgb},0.5) 100%)`,
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
              }}>{dom}</div>
              {confidence != null && (
                <div style={{ fontSize: '0.70rem', color: MUTED, marginTop: 4 }}>
                  {(confidence * 100).toFixed(0)}% confidence · {processing ? 'updating…' : 'live'}
                </div>
              )}
            </div>
          </div>

          <div style={{ position: 'relative', width: 48, height: 48, flexShrink: 0 }}>
            <div style={{ position: 'absolute', inset: -4, borderRadius: '50%', background: `rgba(${rgb},0.10)`, animation: 'eiPulse 2.4s ease-in-out infinite' }} />
            <div style={{ position: 'absolute', inset: 0, borderRadius: '50%', background: `radial-gradient(circle at 35% 35%, rgba(${rgb},0.9), rgba(${rgb},0.35))`, boxShadow: `0 0 18px rgba(${rgb},0.55), 0 0 40px rgba(${rgb},0.18)` }} />
          </div>
        </div>

        {shift?.significance === 'High' && (
          <div style={{ marginTop: 12, padding: '7px 10px', borderRadius: 8, background: 'rgba(239,68,68,0.10)', border: '1px solid rgba(239,68,68,0.22)', fontSize: '0.71rem', color: '#f87171', display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>⚡</span>
            <span>Mood shift: <strong style={{ textTransform: 'capitalize' }}>{shift.from}</strong> → <strong style={{ textTransform: 'capitalize' }}>{shift.to}</strong></span>
          </div>
        )}

        {sarcasm > 0.38 && (
          <div style={{ marginTop: 8, padding: '6px 10px', borderRadius: 8, background: 'rgba(245,158,11,0.10)', border: '1px solid rgba(245,158,11,0.22)', fontSize: '0.71rem', color: '#fbbf24', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span>Sarcasm detected</span>
            <strong>{Math.round(sarcasm * 100)}%</strong>
          </div>
        )}
      </div>

      {/* ── Valence bar + arc ── */}
      <div style={{ padding: '14px 18px', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
          <span style={{ fontSize: '0.58rem', fontWeight: 700, color: MUTED, textTransform: 'uppercase', letterSpacing: '.10em' }}>Valence</span>
          <span style={{ fontSize: '0.74rem', fontWeight: 700, color: valence > 0.15 ? '#4ade80' : valence < -0.15 ? '#f87171' : TEXT }}>
            {valence > 0.15 ? 'Positive' : valence < -0.15 ? 'Negative' : 'Neutral'} {valence.toFixed(2)}
          </span>
        </div>
        <div style={{ position: 'relative', height: 8, borderRadius: 4, overflow: 'hidden', background: 'rgba(255,255,255,0.05)' }}>
          <div style={{ position: 'absolute', inset: 0, left: '50%', borderRadius: '0 4px 4px 0', background: 'linear-gradient(90deg, transparent, rgba(74,222,128,0.25))' }} />
          <div style={{ position: 'absolute', inset: 0, right: '50%', borderRadius: '4px 0 0 4px', background: 'linear-gradient(90deg, rgba(248,113,113,0.25), transparent)' }} />
          <div style={{
            position: 'absolute', top: '50%', transform: 'translate(-50%, -50%)',
            left: `${valencePct}%`,
            width: 14, height: 14, borderRadius: '50%',
            background: `rgb(${rgb})`,
            boxShadow: `0 0 10px rgba(${rgb},0.9), 0 0 20px rgba(${rgb},0.4)`,
            transition: 'left 0.7s cubic-bezier(0.34,1.2,0.64,1)',
            zIndex: 2,
          }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
          <span style={{ fontSize: '0.56rem', color: '#4b5563' }}>Negative</span>
          <span style={{ fontSize: '0.56rem', color: '#4b5563' }}>Positive</span>
        </div>

        {valenceHistory.length >= 3 && (
          <div style={{ marginTop: 12 }}>
            <div style={{ fontSize: '0.56rem', color: MUTED, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 6 }}>
              Conversation Arc · {valenceHistory.length} messages
            </div>
            <Sparkline values={valenceHistory} color={rgb} />
          </div>
        )}
      </div>

      {/* ── Emotion spectrum ── */}
      {topEmos.length > 0 && (
        <div style={{ padding: '14px 18px', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
          <div style={{ fontSize: '0.58rem', fontWeight: 700, color: MUTED, textTransform: 'uppercase', letterSpacing: '.10em', marginBottom: 12 }}>Emotion Spectrum</div>
          {topEmos.map(({ label, score }) => {
            const eRgb = EmotionPalette[label.toLowerCase()] || EmotionPalette.default;
            return (
              <div key={label} style={{ marginBottom: 9 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <span style={{ fontSize: '0.80rem' }}>{EMOTION_EMOJI[label.toLowerCase()] || ''}</span>
                    <span style={{ fontSize: '0.74rem', color: TEXT, textTransform: 'capitalize', fontWeight: 500 }}>{label}</span>
                  </div>
                  <span style={{ fontSize: '0.70rem', fontWeight: 700, color: `rgb(${eRgb})` }}>{(score * 100).toFixed(0)}%</span>
                </div>
                <div style={{ height: 5, borderRadius: 3, background: 'rgba(255,255,255,0.05)' }}>
                  <div style={{
                    height: '100%', width: `${score * 100}%`,
                    background: `linear-gradient(90deg, rgba(${eRgb},0.9), rgba(${eRgb},0.45))`,
                    borderRadius: 3, boxShadow: `0 0 5px rgba(${eRgb},0.35)`,
                    transition: 'width 0.7s cubic-bezier(0.34,1.2,0.64,1)',
                  }} />
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Model contributions ── */}
      {Object.keys(logicMap).length > 0 && (
        <div style={{ padding: '14px 18px', borderBottom: `1px solid ${BORDER}`, flexShrink: 0 }}>
          <div style={{ fontSize: '0.58rem', fontWeight: 700, color: MUTED, textTransform: 'uppercase', letterSpacing: '.10em', marginBottom: 10 }}>Model Contributions</div>
          {Object.entries(logicMap).map(([model, v]) => (
            <div key={model} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 7 }}>
              <span style={{ fontSize: '0.66rem', color: MUTED, width: 72, flexShrink: 0 }}>{model}</span>
              <div style={{ flex: 1, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.05)', overflow: 'hidden' }}>
                <div style={{
                  height: '100%', width: `${Math.min(Math.abs(v) * 100, 100)}%`,
                  background: v >= 0 ? 'linear-gradient(90deg,#3b82f6,#8b5cf6)' : '#ef4444',
                  borderRadius: 2, transition: 'width 0.5s ease',
                }} />
              </div>
              <span style={{ fontSize: '0.68rem', fontWeight: 700, color: v >= 0 ? '#60a5fa' : '#f87171', width: 36, textAlign: 'right', flexShrink: 0 }}>
                {v >= 0 ? '+' : ''}{(v * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}

      {/* ── Intelligence signals ── */}
      <div style={{ padding: '14px 18px', flexShrink: 0 }}>
        <div style={{ fontSize: '0.58rem', fontWeight: 700, color: MUTED, textTransform: 'uppercase', letterSpacing: '.10em', marginBottom: 12 }}>Intelligence Signals</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>

          {traj?.model_available && traj.top_predicted && (
            <div style={{ padding: '10px 12px', borderRadius: 10, background: 'rgba(139,92,246,0.08)', border: '1px solid rgba(139,92,246,0.18)' }}>
              <div style={{ fontSize: '0.58rem', fontWeight: 700, color: '#a78bfa', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 5 }}>Trajectory — Next Predicted</div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: '1.1rem' }}>{EMOTION_EMOJI[traj.top_predicted.toLowerCase()] || '→'}</span>
                <span style={{ fontSize: '0.86rem', fontWeight: 800, color: '#c4b5fd', textTransform: 'capitalize' }}>{traj.top_predicted}</span>
                <span style={{ fontSize: '0.66rem', color: MUTED, marginLeft: 'auto' }}>LSTM prediction</span>
              </div>
            </div>
          )}

          {snap && (
            <div style={{ padding: '10px 12px', borderRadius: 10, background: 'rgba(6,182,212,0.06)', border: '1px solid rgba(6,182,212,0.14)' }}>
              <div style={{ fontSize: '0.58rem', fontWeight: 700, color: '#67e8f9', textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 8 }}>Context Engine</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
                {[
                  { label: 'Volatility', val: `${Math.round((snap.volatility || 0) * 100)}%`, color: (snap.volatility || 0) > 0.6 ? '#f87171' : TEXT },
                  { label: 'Topic Fit', val: `${Math.round((snap.topic_resonance || 0) * 100)}%`, color: TEXT },
                  { label: 'CDM State', val: snap.cdm_current_state || '—', color: snap.cdm_current_state ? '#a78bfa' : MUTED },
                  { label: 'Episodic', val: snap.ce_available ? 'Active' : 'Off', color: snap.ce_available ? '#4ade80' : MUTED },
                ].map(({ label, val, color }) => (
                  <div key={label} style={{ padding: '5px 8px', borderRadius: 6, background: 'rgba(255,255,255,0.03)' }}>
                    <div style={{ fontSize: '0.55rem', color: MUTED, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.06em' }}>{label}</div>
                    <div style={{ fontSize: '0.74rem', fontWeight: 700, color, marginTop: 2 }}>{val}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Pipeline activity */}
          <div style={{ padding: '8px 12px', borderRadius: 8, background: 'rgba(255,255,255,0.02)', border: `1px solid ${BORDER}` }}>
            <div style={{ fontSize: '0.58rem', color: MUTED, fontWeight: 700, textTransform: 'uppercase', letterSpacing: '.08em', marginBottom: 7 }}>ML Pipeline</div>
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {PIPELINE_STAGES.map(s => {
                const active = partialModels.has(s.key) || (data && !processing);
                return (
                  <div key={s.key} style={{
                    padding: '2px 8px', borderRadius: 99, fontSize: '0.60rem', fontWeight: 700,
                    background: active ? `rgba(${rgb},0.14)` : 'rgba(255,255,255,0.04)',
                    color: active ? `rgb(${rgb})` : MUTED,
                    border: `1px solid ${active ? `rgba(${rgb},0.28)` : 'transparent'}`,
                    transition: 'all 0.35s ease',
                    boxShadow: active ? `0 0 6px rgba(${rgb},0.25)` : 'none',
                  }}>{s.label}</div>
                );
              })}
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes eiPulse {
          0%, 100% { transform: scale(1); opacity: 0.7; }
          50% { transform: scale(1.18); opacity: 0.3; }
        }
        @keyframes eiFadeIn {
          from { opacity: 0; transform: translateY(-6px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes eiSpin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}
